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

// The OpenAI Responses API (POST /v1/responses), the third dialect behind the
// Provider seam.
//
// It exists beside the chat-completions dialect rather than replacing it
// because of ONE thing chat-completions structurally cannot do: carry a
// reasoning model's chain of thought across a tool call. On chat-completions
// the reasoning a model produced before calling a tool is gone by the next
// request -- there is no field to put it back in -- so the model re-derives it
// every turn, which costs tokens, breaks the prompt cache at the reasoning
// boundary, and loses the thread of a long investigation. The Responses API
// models the turn as an ordered list of ITEMS, and a reasoning item can be sent
// back verbatim. That is why this dialect asks for
// include: ["reasoning.encrypted_content"] and replays what it gets.
//
// Two deliberate positions:
//
//   - store defaults to FALSE. The API's own default retains every prompt and
//     response server-side, and a library that opted its callers into that
//     silently would be making a privacy decision on their behalf. It is
//     ResponsesConfig.Store, off unless asked.
//   - previous_response_id is never sent. This library's contract is a flat
//     transcript the CALLER owns and can edit, fork, compact or persist; a
//     server-side conversation id would make that transcript a partial lie
//     about what the model is actually seeing. Every call sends its full input.

// responsesProvider is the Provider for the OpenAI Responses API, built by
// NewResponsesProvider. baseURL is the API root including the version segment
// (e.g. "https://api.openai.com/v1"); requests POST to baseURL + "/responses".
//
// The fields are read-only during Complete, so a value is safe for concurrent
// use.
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

	// A 2xx that is not an SSE stream is the plain Response object: the server
	// ignored stream:true, or a proxy buffered it. Accepted transparently and
	// reassembled with Streamed false, exactly as the chat-completions dialect
	// does, so the caller keeps a truthful record of the transport.
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

// buildBody assembles the JSON request body. Extra is merged FIRST so the typed
// core always wins, and its reserved keys are ignored so they cannot break
// routing. The system prompt is `instructions` (its native position on this
// dialect, not a message), the transcript becomes `input` items, MaxTokens > 0
// sets max_output_tokens, and CacheKey rides as prompt_cache_key.
//
// `include: ["reasoning.encrypted_content"]` is the point of the dialect: it is
// what makes the reasoning items replayable on the next turn. A caller may
// override `include` through Extra, but not by accident -- the default is set
// only when Extra carries no `include` key.
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
			// An item is done, which on this dialect is what a finished
			// content part IS -- reasoning included, with the encrypted
			// payload that makes it replayable.
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
		// Summary text with no replayable item behind it (a server that
		// withheld the encrypted payload): keep it, because losing the
		// reasoning from the transcript is worse than not being able to
		// replay it.
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

// respStopFromShape infers the stop reason from what the turn produced. This
// API has no finish_reason field: a turn that asked for tools ends on tool_use,
// anything else on end_turn.
func respStopFromShape(hasCalls bool) string {
	if hasCalls {
		return StopToolUse
	}
	return StopEndTurn
}

// respFailure turns a response.failed event into an error carrying whatever the
// API said. It is a 200-with-a-failure-inside, so there is no status code to
// classify on: it surfaces as a permanent error rather than a transient one,
// because re-sending a request the server accepted and then rejected is how a
// bad request gets charged for ten times.
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
