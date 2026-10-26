package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// Responses dialect: chat-completions cannot carry reasoning across tool calls, so it replays it.

// responsesProvider is the OpenAI Responses API Provider; fields are read-only during Complete.
type responsesProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	userAgent  string
	store      bool
	headers    map[string]string
	models     modelListCache
}

// respReserved are the Extra keys the typed core always overrides.
var respReserved = set.Of("input", "instructions", "model", "stream", "tools")

// Complete implements Provider over a streaming response.
func (o *responsesProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	return o.Stream(ctx, req, &Collector{Events: ev})
}

// Stream sends the call and writes its nodes into out as they stream.
func (o *responsesProvider) Stream(ctx context.Context, req Request, out CallSink) (*Completion, error) {
	body, err := o.buildBody(req)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(o.baseURL, "/") + "/responses"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("responses: build request: " + err.Error())
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if o.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
	if o.userAgent != "" {
		hreq.Header.Set("User-Agent", o.userAgent)
	}
	for k, v := range o.headers {
		hreq.Header.Set(k, v)
	}
	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("responses: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	// A 2xx that is not an SSE stream is the plain Response object, accepted with Streamed false.
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		raw, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, badRequestErr("responses: read non-streaming response: " + rerr.Error())
		}
		return completionIntoSink(parseResponsesObject(raw))(out, DialectResponses)
	}

	sink := out
	st := newRespStream(sink)
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("responses: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	if st.failure != nil {
		if st.sawData {
			return st.completion(), st.failure
		}
		return nil, st.failure
	}
	return st.completion(), nil
}

// buildBody assembles the JSON request body; include defaults to reasoning.encrypted_content.
func (o *responsesProvider) buildBody(req Request) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.ParamsFor(DialectResponses) {
		if respReserved.Contains(k) {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	if req.System != "" {
		body["instructions"] = req.System
	}
	input, err := respInputItems(req.Messages)
	if err != nil {
		return nil, err
	}
	body["input"] = input
	body["stream"] = true
	body["store"] = o.store
	if _, ok := body["include"]; !ok {
		body["include"] = []string{respIncludeEncryptedReasoning}
	}
	if len(req.Tools) > 0 {
		tools := make([]respTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, respTool{
				Type:        "function",
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.schema(),
			})
		}
		body["tools"] = tools
	}
	if req.MaxTokens > 0 {
		body["max_output_tokens"] = req.MaxTokens
	}
	if req.CacheKey != "" {
		body["prompt_cache_key"] = req.CacheKey
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, badRequestErr("responses: marshal request: " + err.Error())
	}
	return b, nil
}

// respEvent is streamed event. Every Responses event carries its own type
// in the payload, so the SSE `event:` line adds nothing the scanner must read.
type respEvent struct {
	Type     string          `json:"type"`
	Delta    string          `json:"delta,omitempty"`
	ItemID   string          `json:"item_id,omitempty"`
	Item     *respItem       `json:"item,omitempty"`
	Response *respResponse   `json:"response,omitempty"`
	Error    *respEventError `json:"error,omitempty"`
}

// respEventError is the payload of a stream-level `error` event.
type respEventError struct {
	Message string `json:"message"`
	Code    string `json:"code,omitempty"`
	Type    string `json:"type,omitempty"`
}

// respResponse is the Response object, delivered whole by response.completed /
// response.incomplete / response.failed and by a non-streaming call.
type respResponse struct {
	Status            string          `json:"status"`
	Output            []respItem      `json:"output"`
	Usage             *respUsage      `json:"usage"`
	IncompleteDetails *respIncomplete `json:"incomplete_details"`
	Error             *respEventError `json:"error"`
}

// respIncomplete says why a response stopped short.
type respIncomplete struct {
	Reason string `json:"reason"`
}

// respThinkingBlock maps a reasoning item onto a ThinkingBlock.
func respThinkingBlock(item respItem) ThinkingBlock {
	text, raw := item.reasoningText()
	return ThinkingBlock{ID: item.ID, Text: text, Signature: item.EncryptedContent, Raw: raw}
}

// respStopReason normalizes a terminal Response's outcome. A response that
// stopped on the output-token cap says so through incomplete_details, which is
// the only place this API reports it.
func respStopReason(r respResponse, hasCalls bool) string {
	if r.IncompleteDetails != nil && r.IncompleteDetails.Reason == "max_output_tokens" {
		return StopMaxTokens
	}
	if r.IncompleteDetails != nil && r.IncompleteDetails.Reason != "" {
		return r.IncompleteDetails.Reason
	}
	return respStopFromShape(hasCalls)
}

// respStopFromShape infers the stop reason: tools end on tool_use, else end_turn.
func respStopFromShape(hasCalls bool) string {
	if hasCalls {
		return StopToolUse
	}
	return StopEndTurn
}

// respFailure turns a response.failed event into a permanent error, not a transient.
func respFailure(r *respResponse) error {
	if r != nil && r.Error != nil {
		return respEventFailure(r.Error)
	}
	return badRequestErr("responses: the response failed with no reason given")
}

// respEventFailure turns a stream-level error payload into an error.
func respEventFailure(e *respEventError) error {
	if e == nil {
		return badRequestErr("responses: the stream reported an error with no detail")
	}
	msg := e.Message
	if msg == "" {
		msg = "the stream reported an error with no message"
	}
	if e.Code != "" {
		msg += " (" + e.Code + ")"
	}
	return badRequestErr("responses: " + msg)
}

// parseResponsesObject reassembles a plain-JSON Response into a Completion with
// Streamed false -- the transparent acceptance of a server that ignored
// stream:true. Here the items ARE the only source, since no deltas arrived.
func parseResponsesObject(data []byte) (*Completion, error) {
	var r respResponse
	if err := json.Unmarshal(data, &r); err != nil {
		return nil, badRequestErr("responses: decode non-streaming response: " + err.Error())
	}
	if r.Status == "failed" {
		return nil, respFailure(&r)
	}
	comp := &Completion{Message: Message{Role: RoleAssistant}, Streamed: false}
	haveCall := false
	for _, item := range r.Output {
		switch item.Type {
		case respItemMessage:
			for _, c := range item.Content {
				if c.Type == respTextOutput {
					comp.Message.Parts = append(comp.Message.Parts, TextPart{Text: c.Text})
				}
			}
		case respItemFuncCall:
			comp.Message.Parts = append(comp.Message.Parts,
				ToolCallPart{ID: item.CallID, Name: item.Name, Arguments: toolArgs(item.Arguments)})
			haveCall = true
		case respItemReasoning:
			tb := respThinkingBlock(item)
			comp.Message.Parts = append(comp.Message.Parts,
				ThinkingPart{Text: tb.Text, Signature: tb.Signature, ID: tb.ID, Raw: tb.Raw})
		}
	}
	comp.Message.SyncViews()
	comp.StopReason = respStopReason(r, haveCall)
	if r.Usage != nil {
		u := r.Usage.toUsage()
		u.ReasoningTokens = r.Usage.reasoningTokens()
		var envelope struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(data, &envelope); err == nil {
			u.Raw = envelope.Usage
		}
		comp.Usages = []Usage{u}
	}
	return comp, nil
}
