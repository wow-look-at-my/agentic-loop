package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
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
var respReserved = map[string]bool{
	"input": true, "instructions": true, "model": true, "stream": true, "tools": true,
}

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
	for k, v := range req.Extra {
		if respReserved[k] {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	if req.System != "" {
		body["instructions"] = req.System
	}
	body["input"] = respInputItems(req.Messages)
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
	ev       *StreamEvents
	content  strings.Builder
	reason   strings.Builder
	calls    []ToolCall
	thinking []ThinkingBlock
	usage    *Usage
	usageRaw json.RawMessage

	reasoningTokens *int
	stop            string
	failure         error
	sawData         bool
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
		return st.ev.emitText(e.Delta)

	case respEvSummaryDelta:
		if e.Delta == "" {
			return nil
		}
		st.sawData = true
		st.reason.WriteString(e.Delta)
		return st.ev.emitReasoning(e.Delta)

	case respEvItemDone:
		if e.Item != nil {
			st.sawData = true
			st.addItem(*e.Item)
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

// addItem records a finished output item. Text is skipped here because the
// deltas already carried it -- taking it from both would double the content.
func (st *respStream) addItem(item respItem) {
	switch item.Type {
	case respItemFuncCall:
		st.calls = append(st.calls, ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments})
	case respItemReasoning:
		st.thinking = append(st.thinking, respThinkingBlock(item))
	}
}

// finishResponse reads the terminal Response object: the usage snapshot with
// its verbatim wire form, and the stop reason. Items are NOT re-read from
// response.output -- each already arrived as an output_item.done, and taking
// them twice would duplicate every tool call.
func (st *respStream) finishResponse(raw []byte, r respResponse) error {
	st.stop = respStopReason(r, len(st.calls) > 0)
	if r.Usage == nil {
		return nil
	}
	u := r.Usage.toUsage()
	if merged := mergeUsage(st.usage, &u); merged != st.usage {
		st.usage = merged
		var envelope struct {
			Response struct {
				Usage json.RawMessage `json:"usage"`
			} `json:"response"`
		}
		if err := json.Unmarshal(raw, &envelope); err == nil {
			st.usageRaw = envelope.Response.Usage
		}
		st.reasoningTokens = r.Usage.reasoningTokens()
	}
	return st.ev.emitUsage(*st.usage)
}

// completion assembles the final (or partial) result.
func (st *respStream) completion() *Completion {
	msg := Message{Role: RoleAssistant, Content: st.content.String(), ToolCalls: st.calls}
	msg.Thinking = respThinking(st.thinking, st.reason.String())
	var u Usage
	if st.usage != nil {
		u = floorTotal(*st.usage)
	}
	stop := st.stop
	if stop == "" {
		stop = respStopFromShape(len(st.calls) > 0)
	}
	return &Completion{
		Message:         msg,
		Usage:           u,
		UsageReported:   st.usage != nil,
		RawUsage:        st.usageRaw,
		ReasoningTokens: st.reasoningTokens,
		Streamed:        true,
		StopReason:      stop,
	}
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

// respThinking is the assistant message's reasoning. The blocks carry what can
// be REPLAYED; the streamed summary text is what the caller WATCHED. When the
// blocks already hold summary text the two are the same words, so the streamed
// copy is dropped rather than shown twice; a stream that produced summary text
// but no replayable item (a server that withheld the encrypted payload) still
// yields a text-only block, because losing the reasoning from the transcript
// entirely would be worse than not being able to replay it.
func respThinking(blocks []ThinkingBlock, streamed string) []ThinkingBlock {
	if len(blocks) > 0 {
		return blocks
	}
	if streamed == "" {
		return nil
	}
	return []ThinkingBlock{{Text: streamed}}
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
	var text strings.Builder
	for _, item := range r.Output {
		switch item.Type {
		case respItemMessage:
			for _, c := range item.Content {
				if c.Type == respTextOutput {
					text.WriteString(c.Text)
				}
			}
		case respItemFuncCall:
			comp.Message.ToolCalls = append(comp.Message.ToolCalls,
				ToolCall{ID: item.CallID, Name: item.Name, Arguments: item.Arguments})
		case respItemReasoning:
			comp.Message.Thinking = append(comp.Message.Thinking, respThinkingBlock(item))
		}
	}
	comp.Message.Content = text.String()
	comp.StopReason = respStopReason(r, len(comp.Message.ToolCalls) > 0)
	if r.Usage != nil {
		comp.Usage = floorTotal(r.Usage.toUsage())
		comp.UsageReported = true
		comp.ReasoningTokens = r.Usage.reasoningTokens()
		var envelope struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(data, &envelope); err == nil {
			comp.RawUsage = envelope.Usage
		}
	}
	return comp, nil
}
