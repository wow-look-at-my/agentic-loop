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
}

// respReserved are the Extra keys the typed core always overrides.
var respReserved = set.Of("input", "instructions", "model", "stream", "tools")

// Complete implements Provider over a streaming response.
func (o *responsesProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
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
		return parseResponsesObject(raw)
	}

	st := &respStream{ev: ev}
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

// respEvent is one streamed event. Every Responses event carries its own type
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

// The streamed event types this dialect acts on. Everything else -- the
// added/done lifecycle chatter around parts, the per-item bookkeeping -- is
// ignored, because the deltas plus the final Response carry the whole result.
const (
	respEvOutputTextDelta = "response.output_text.delta"
	respEvSummaryDelta    = "response.reasoning_summary_text.delta"
	respEvItemDone        = "response.output_item.done"
	respEvCompleted       = "response.completed"
	respEvIncomplete      = "response.incomplete"
	respEvFailed          = "response.failed"
	respEvError           = "error"
)

// respStream accumulates one streamed response.
type respStream struct {
	ev      *StreamEvents
	content strings.Builder
	reason  strings.Builder
	// parts is the output items in the order the API emitted them, which is
	// the order the model produced them: reasoning, the text it wrote, the
	// calls it made. itemText is the text those items carried, so a stream cut
	// before its message item completed can still contribute the deltas it had
	// already delivered.
	parts    []Part
	itemText strings.Builder
	haveCall bool
	usages   []Usage

	stop    string
	failure error
	sawData bool
}

// onData decodes one SSE payload. Unparseable payloads are tolerated silently
// (keep-alive noise), matching the chat-completions dialect. State is
// accumulated BEFORE each emit, so a callback error yields a partial completion
// including the failing delta.
//
// Completed items are read from response.output_item.done rather than
// reassembled from argument fragments: this API sends each item whole when it
// finishes, so there is no accumulator to get wrong. The argument-delta events
// are therefore not decoded at all -- a fragment stream nobody needs is a
// second source of truth for the same bytes.
func (st *respStream) onData(data []byte) error {
	var e respEvent
	if err := json.Unmarshal(data, &e); err != nil {
		return nil
	}
	switch e.Type {
	case respEvOutputTextDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.content.WriteString(e.Delta)
		return st.ev.EmitText(e.Delta)

	case respEvSummaryDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.reason.WriteString(e.Delta)
		return st.ev.EmitReasoning(e.Delta)

	case respEvItemDone:
		if e.Item != nil {
			st.sawData = true
			before := len(st.parts)
			st.addItem(*e.Item)
			// An item is done, which on this dialect is a finished content part, reasoning included.
			return st.ev.EmitParts(st.parts[before:])
		}
		return nil

	case respEvCompleted, respEvIncomplete:
		if e.Response == nil {
			return nil
		}
		st.sawData = true
		return st.finishResponse(data, *e.Response)

	case respEvFailed:
		st.failure = respFailure(e.Response)
		return nil

	case respEvError:
		st.failure = respEventFailure(e.Error)
		return nil
	}
	return nil
}

// addItem records a finished output item, in the position the API gave it. A
// completed message item carries its own text, so that -- not the delta
// accumulator -- is what lands in the parts; taking it from both would double
// the content.
func (st *respStream) addItem(item respItem) {
	switch item.Type {
	case respItemMessage:
		for _, c := range item.Content {
			if c.Text == "" {
				continue
			}
			st.parts = append(st.parts, TextPart{Text: c.Text})
			st.itemText.WriteString(c.Text)
		}
	case respItemFuncCall:
		st.parts = append(st.parts, ToolCallPart{ID: item.CallID, Name: item.Name, Arguments: toolArgs(item.Arguments)})
		st.haveCall = true
	case respItemReasoning:
		tb := respThinkingBlock(item)
		st.parts = append(st.parts, ThinkingPart{Text: tb.Text, Signature: tb.Signature, ID: tb.ID})
	}
}

// finishResponse reads the terminal Response object: the usage snapshot with
// its verbatim wire form, and the stop reason. Items are NOT re-read from
// response.output -- each already arrived as an output_item.done, and taking
// them twice would duplicate every tool call.
func (st *respStream) finishResponse(raw []byte, r respResponse) error {
	st.stop = respStopReason(r, st.haveCall)
	if r.Usage == nil {
		return nil
	}
	u := r.Usage.toUsage()
	var envelope struct {
		Response struct {
			Usage json.RawMessage `json:"usage"`
		} `json:"response"`
	}
	if err := json.Unmarshal(raw, &envelope); err == nil {
		u.Raw = envelope.Response.Usage
	}
	u.ReasoningTokens = r.Usage.reasoningTokens()
	st.usages = append(st.usages, u)
	return st.ev.EmitUsage(u)
}

// completion assembles the final (or partial) result.
func (st *respStream) completion() *Completion {
	parts := st.parts
	// Deltas the message item never got to confirm: a stream cut mid-message
	// still delivered them, and dropping them would discard output the caller
	// already watched arrive.
	if got, seen := st.content.String(), st.itemText.String(); strings.HasPrefix(got, seen) && len(got) > len(seen) {
		parts = append(parts, TextPart{Text: got[len(seen):]})
	}
	if reason := st.reason.String(); reason != "" && !hasThinking(parts) {
		// Summary text with no replayable item behind it is kept, since losing the reasoning is worse.
		parts = append([]Part{ThinkingPart{Text: reason}}, parts...)
	}
	msg := Message{Role: RoleAssistant, Parts: parts}
	msg.SyncViews()
	stop := st.stop
	if stop == "" {
		stop = respStopFromShape(st.haveCall)
	}
	return &Completion{
		Message:    msg,
		Usages:     st.usages,
		Streamed:   true,
		StopReason: stop,
	}
}

// hasThinking reports whether the parts already carry reasoning.
func hasThinking(parts []Part) bool {
	for _, p := range parts {
		if p.Kind() == PartKindThinking || p.Kind() == PartKindRedactedThinking {
			return true
		}
	}
	return false
}

// respThinkingBlock maps a reasoning item onto a ThinkingBlock: the summary
// text, the item id, and the encrypted payload in Signature -- the field whose
// job on every dialect is "the opaque token that must be replayed verbatim".
func respThinkingBlock(item respItem) ThinkingBlock {
	var b strings.Builder
	for _, s := range item.Summary {
		b.WriteString(s.Text)
	}
	return ThinkingBlock{ID: item.ID, Text: b.String(), Signature: item.EncryptedContent}
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

// respFailure turns a response.failed event into a permanent error, not a transient one.
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
				ThinkingPart{Text: tb.Text, Signature: tb.Signature, ID: tb.ID})
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
