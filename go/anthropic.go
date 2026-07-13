package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Anthropic is a Provider for the Anthropic Messages API. BaseURL is the API
// root (empty defaults to "https://api.anthropic.com"); requests POST to
// BaseURL + "/v1/messages" with x-api-key and anthropic-version headers
// (Version empty defaults to "2023-06-01"). Headers are applied after the
// defaults so a caller-supplied header can override them.
//
// Unless DisableCaching is set, every request carries exactly two ephemeral
// prompt-cache breakpoints: a static one on the system block (the cache
// hierarchy is tools → system → messages, so it covers the tools array too)
// and a moving one on the last content block of the last message, so each
// turn cache-hits everything through the previous turn's tail. Both markers
// are applied to the per-request wire structures only — the caller's Messages
// are never mutated, so the stored transcript stays marker-free.
//
// The exported fields are read-only during Complete, so an Anthropic value is
// safe for concurrent use.
type Anthropic struct {
	BaseURL        string
	APIKey         string
	Version        string
	HTTPClient     *http.Client
	UserAgent      string
	DisableCaching bool
	Headers        map[string]string
}

const (
	defaultAnthropicBaseURL = "https://api.anthropic.com"
	defaultAnthropicVersion = "2023-06-01"
)

// anReserved are the Extra keys the typed core always overrides.
var anReserved = map[string]bool{
	"model": true, "max_tokens": true, "stream": true,
	"system": true, "messages": true, "tools": true,
}

// cacheEphemeral is the prompt-cache breakpoint marker. The Messages API
// allows at most 4 breakpoints per request; this provider uses exactly 2.
var cacheEphemeral = map[string]string{"type": "ephemeral"}

// Complete implements Provider over a streaming Messages API call. The
// Messages API requires max_tokens on every request, so a Request without a
// positive MaxTokens fails fast before any I/O.
func (a *Anthropic) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	if req.MaxTokens <= 0 {
		return nil, badRequestErr("anthropic: Request.MaxTokens must be positive (the Messages API requires max_tokens)")
	}
	body, err := a.buildBody(req)
	if err != nil {
		return nil, err
	}
	base := a.BaseURL
	if base == "" {
		base = defaultAnthropicBaseURL
	}
	url := strings.TrimRight(base, "/") + "/v1/messages"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("anthropic: build request: " + err.Error())
	}
	version := a.Version
	if version == "" {
		version = defaultAnthropicVersion
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("anthropic-version", version)
	if a.APIKey != "" {
		hreq.Header.Set("x-api-key", a.APIKey)
	}
	if a.UserAgent != "" {
		hreq.Header.Set("User-Agent", a.UserAgent)
	}
	for k, v := range a.Headers {
		hreq.Header.Set(k, v)
	}
	client := a.HTTPClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	st := &anStream{ev: ev, blocks: map[int]*anBlock{}}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("anthropic: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	return st.completion(), nil
}

// buildBody assembles the Messages API request. Extra passthrough params are
// merged FIRST so the typed core fields always win; reserved keys in Extra
// (model, max_tokens, stream, system, messages, tools) are silently ignored.
// The library does not gate thinking/temperature by model — deciding what to
// send is the caller's job via Extra.
func (a *Anthropic) buildBody(req Request) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.Extra {
		if anReserved[k] {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	body["max_tokens"] = req.MaxTokens
	body["stream"] = true
	if req.System != "" {
		// system is a content-block array because cache_control lives on
		// blocks, not string bodies. The static breakpoint on the (last)
		// system block covers the tools array too via the cache hierarchy. No
		// system prompt → no system field and no static breakpoint (the
		// moving one still covers the whole prefix).
		sys := map[string]any{"type": "text", "text": req.System}
		if !a.DisableCaching {
			sys["cache_control"] = cacheEphemeral
		}
		body["system"] = []map[string]any{sys}
	}
	msgs := anWireMessages(req.Messages)
	if !a.DisableCaching {
		markTranscriptTail(msgs)
	}
	body["messages"] = msgs
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.schema(),
			})
		}
		body["tools"] = tools
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, badRequestErr("anthropic: marshal request: " + err.Error())
	}
	return b, nil
}

// anWireMessages maps the neutral transcript onto Messages API messages,
// building fresh wire structures every call (the caller's Messages are never
// touched). Assistant messages become content-block arrays with thinking
// blocks replayed FIRST (required, or tool-use continuations 400 — signatures
// and redacted payloads replayed verbatim), then the text, then tool_use
// blocks whose input is the PARSED argument object. Consecutive RoleTool
// messages fold into ONE user message of tool_result blocks. Everything else
// (user, and any stray system message — Request.System is the system channel
// on this dialect) rides as a user message with string content.
func anWireMessages(msgs []Message) []map[string]any {
	out := make([]map[string]any, 0, len(msgs))
	for i := 0; i < len(msgs); i++ {
		m := msgs[i]
		switch {
		case m.Role == RoleTool:
			var blocks []map[string]any
			for ; i < len(msgs) && msgs[i].Role == RoleTool; i++ {
				t := msgs[i]
				blocks = append(blocks, map[string]any{
					"type":        "tool_result",
					"tool_use_id": t.ToolCallID,
					"is_error":    t.ToolIsError,
					"content":     t.Content,
				})
			}
			i-- // the outer loop increments past the run's last message
			out = append(out, map[string]any{"role": "user", "content": blocks})
		case m.Role == RoleAssistant:
			out = append(out, map[string]any{"role": "assistant", "content": anAssistantContent(m)})
		default:
			out = append(out, map[string]any{"role": "user", "content": m.Content})
		}
	}
	return out
}

// anAssistantContent builds an assistant message's content blocks in replay
// order: thinking first, then text, then tool_use. A message with no blocks
// at all degrades to string content.
func anAssistantContent(m Message) any {
	blocks := make([]map[string]any, 0, len(m.Thinking)+1+len(m.ToolCalls))
	for _, tb := range m.Thinking {
		if tb.Redacted != "" {
			blocks = append(blocks, map[string]any{"type": "redacted_thinking", "data": tb.Redacted})
			continue
		}
		blocks = append(blocks, map[string]any{"type": "thinking", "thinking": tb.Text, "signature": tb.Signature})
	}
	if m.Content != "" {
		blocks = append(blocks, map[string]any{"type": "text", "text": m.Content})
	}
	for _, tc := range m.ToolCalls {
		blocks = append(blocks, map[string]any{
			"type":  "tool_use",
			"id":    tc.ID,
			"name":  tc.Name,
			"input": parseToolInput(tc.Arguments),
		})
	}
	if len(blocks) == 0 {
		return m.Content
	}
	return blocks
}

// parseToolInput parses a tool call's raw argument JSON into the object the
// Messages API expects; invalid, empty, or non-object arguments become {}.
func parseToolInput(args string) map[string]any {
	s := strings.TrimSpace(args)
	if s == "" {
		return map[string]any{}
	}
	var v any
	if err := json.Unmarshal([]byte(s), &v); err != nil {
		return map[string]any{}
	}
	if obj, ok := v.(map[string]any); ok {
		return obj
	}
	return map[string]any{}
}

// markTranscriptTail places the moving prompt-cache breakpoint on the last
// content block of the last message, so the next request cache-hits
// everything through this one's tail. It operates on the per-request wire
// structures only (freshly built by anWireMessages), which is what keeps the
// caller's transcript marker-free — old moving markers can never accumulate.
// A non-empty string content becomes a one-block array; a non-empty block
// array gets the marker on its last block; empty strings, empty arrays, and
// unrecognized shapes pass through unmarked — caching is an optimization,
// never a correctness requirement. The empty-string case matters: the API
// rejects empty text blocks, and a transcript CAN legitimately end on an
// empty message (Run finalizes a turn cancelled after only tool-call deltas
// as an assistant message with no content), so converting "" into an empty
// marked text block would turn a valid request into a 400.
func markTranscriptTail(msgs []map[string]any) {
	if len(msgs) == 0 {
		return
	}
	last := msgs[len(msgs)-1]
	switch c := last["content"].(type) {
	case string:
		if c == "" {
			return
		}
		last["content"] = []map[string]any{{"type": "text", "text": c, "cache_control": cacheEphemeral}}
	case []map[string]any:
		if len(c) == 0 {
			return
		}
		c[len(c)-1]["cache_control"] = cacheEphemeral
	}
}

// anEvent is one decoded Messages API stream event; the payload's type field
// discriminates, so the SSE event name is not needed.
type anEvent struct {
	Type         string          `json:"type"`
	Index        int             `json:"index"`
	Message      *anMessageStart `json:"message"`
	ContentBlock *anContentBlock `json:"content_block"`
	Delta        *anDelta        `json:"delta"`
	Usage        *anUsage        `json:"usage"`
	Error        *anError        `json:"error"`
}

// anMessageStart is the message envelope of a message_start event.
type anMessageStart struct {
	Usage *anUsage `json:"usage"`
}

// anContentBlock seeds a block at content_block_start: id/name for tool_use,
// thinking/signature for thinking, data for redacted_thinking.
type anContentBlock struct {
	Type      string `json:"type"`
	ID        string `json:"id"`
	Name      string `json:"name"`
	Thinking  string `json:"thinking"`
	Signature string `json:"signature"`
	Data      string `json:"data"`
}

// anDelta carries both content_block_delta payloads (text_delta,
// thinking_delta, signature_delta, input_json_delta) and the message_delta
// stop_reason.
type anDelta struct {
	Type        string `json:"type"`
	Text        string `json:"text"`
	Thinking    string `json:"thinking"`
	Signature   string `json:"signature"`
	PartialJSON string `json:"partial_json"`
	StopReason  string `json:"stop_reason"`
}

// anUsage is the wire usage of message_start / message_delta. input_tokens
// EXCLUDES cached tokens on this dialect; the cache fields are pointers so an
// absent field is distinguishable from an explicit zero.
type anUsage struct {
	InputTokens              int  `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
}

// anError is the payload of an error event; Type discriminates against the
// documented error-type table (the human-readable message stays in the raw
// event JSON, which becomes the APIError body).
type anError struct {
	Type string `json:"type"`
}

// anthropicErrorStatus maps a stream error event's error type onto the HTTP
// status Anthropic documents for it, so an in-stream error classifies for
// retry exactly like its non-2xx counterpart. Unrecognized types map to 500:
// an unknown in-stream failure is a server-side abort, and treating it as
// transient is the safe default.
func anthropicErrorStatus(errType string) int {
	switch errType {
	case "invalid_request_error":
		return 400
	case "authentication_error":
		return 401
	case "permission_error", "billing_error":
		return 403
	case "not_found_error":
		return 404
	case "request_too_large":
		return 413
	case "rate_limit_error":
		return 429
	case "api_error":
		return 500
	case "overloaded_error":
		return 529
	}
	return 500
}

// anBlock accumulates one content block across start/delta/stop events.
type anBlock struct {
	typ       string
	id        string
	name      string
	json      strings.Builder
	thinking  strings.Builder
	signature strings.Builder
	data      string
}

// anStream accumulates one streamed Messages API response.
type anStream struct {
	ev       *StreamEvents
	content  strings.Builder
	thinking []ThinkingBlock
	calls    []ToolCall
	blocks   map[int]*anBlock
	stop     string

	inputTokens  int
	outputTokens int
	cacheRead    *int
	cacheWrite   *int
	haveUsage    bool

	sawData bool
}

// onData decodes one stream payload. Unparseable payloads are tolerated
// silently; ping events are ignored; an error event aborts the stream with
// the server's message.
func (st *anStream) onData(data []byte) error {
	var msg anEvent
	if err := json.Unmarshal(data, &msg); err != nil {
		return nil
	}
	switch msg.Type {
	case "message_start":
		st.sawData = true
		if msg.Message != nil && msg.Message.Usage != nil {
			u := msg.Message.Usage
			st.haveUsage = true
			st.inputTokens = u.InputTokens
			st.cacheRead = clonePtr(u.CacheReadInputTokens)
			st.cacheWrite = clonePtr(u.CacheCreationInputTokens)
			if u.OutputTokens != nil {
				st.outputTokens = *u.OutputTokens
			}
			st.ev.emitUsage(st.currentUsage())
		}
	case "content_block_start":
		st.sawData = true
		if msg.ContentBlock == nil {
			return nil
		}
		b := &anBlock{
			typ:  msg.ContentBlock.Type,
			id:   msg.ContentBlock.ID,
			name: msg.ContentBlock.Name,
			data: msg.ContentBlock.Data,
		}
		b.thinking.WriteString(msg.ContentBlock.Thinking)
		b.signature.WriteString(msg.ContentBlock.Signature)
		st.blocks[msg.Index] = b
	case "content_block_delta":
		st.sawData = true
		if msg.Delta == nil {
			return nil
		}
		switch msg.Delta.Type {
		case "text_delta":
			st.content.WriteString(msg.Delta.Text)
			st.ev.emitText(msg.Delta.Text)
		case "thinking_delta":
			if b := st.blocks[msg.Index]; b != nil {
				b.thinking.WriteString(msg.Delta.Thinking)
			}
			st.ev.emitReasoning(msg.Delta.Thinking)
		case "signature_delta":
			if b := st.blocks[msg.Index]; b != nil {
				b.signature.WriteString(msg.Delta.Signature)
			}
		case "input_json_delta":
			if b := st.blocks[msg.Index]; b != nil {
				b.json.WriteString(msg.Delta.PartialJSON)
			}
		}
	case "content_block_stop":
		st.sawData = true
		b := st.blocks[msg.Index]
		if b == nil {
			return nil
		}
		switch b.typ {
		case "tool_use":
			st.calls = append(st.calls, ToolCall{ID: b.id, Name: b.name, Arguments: b.json.String()})
		case "thinking":
			st.thinking = append(st.thinking, ThinkingBlock{Text: b.thinking.String(), Signature: b.signature.String()})
		case "redacted_thinking":
			st.thinking = append(st.thinking, ThinkingBlock{Redacted: b.data})
		}
	case "message_delta":
		st.sawData = true
		if msg.Delta != nil && msg.Delta.StopReason != "" {
			st.stop = msg.Delta.StopReason
		}
		if msg.Usage != nil && msg.Usage.OutputTokens != nil {
			// Cumulative snapshot: overwrite, never sum.
			st.haveUsage = true
			st.outputTokens = *msg.Usage.OutputTokens
			st.ev.emitUsage(st.currentUsage())
		}
	case "message_stop":
		st.sawData = true
	case "error":
		// The Messages API can reject or abort a request in-stream: an HTTP
		// 200 whose stream carries an error event (overloaded_error arrives
		// this way). Map the event onto the same *APIError a non-2xx response
		// produces — status from the documented error-type table, body = the
		// raw event JSON — so retry classification (IsTransient) and overflow
		// detection work identically on both delivery paths. Deliberately not
		// marked as sawData: when the error is the first thing on the stream,
		// the call stays retryable.
		errType := ""
		if msg.Error != nil {
			errType = msg.Error.Type
		}
		status := anthropicErrorStatus(errType)
		body := string(data)
		return &APIError{
			Status:          status,
			Body:            body,
			ContextOverflow: status == http.StatusBadRequest && contextOverflowRe.MatchString(body),
		}
	}
	return nil
}

// currentUsage builds the normalized cumulative usage. input_tokens excludes
// cached tokens on this dialect, so the full prompt is
// input + cache_read + cache_creation; the tri-state cache pointers are
// passed through as reported.
func (st *anStream) currentUsage() Usage {
	read, write := 0, 0
	if st.cacheRead != nil {
		read = *st.cacheRead
	}
	if st.cacheWrite != nil {
		write = *st.cacheWrite
	}
	u := Usage{
		PromptTokens:     st.inputTokens + read + write,
		CompletionTokens: st.outputTokens,
		CacheReadTokens:  clonePtr(st.cacheRead),
		CacheWriteTokens: clonePtr(st.cacheWrite),
	}
	u.TotalTokens = u.PromptTokens + u.CompletionTokens
	return u
}

// completion assembles the final (or partial) result. Blocks that never saw
// content_block_stop (a turn cut off mid-block) are dropped; accumulated text
// is kept. A missing stop_reason falls back to tool_use when calls were
// assembled, else end_turn.
func (st *anStream) completion() *Completion {
	stop := st.stop
	if stop == "" {
		if len(st.calls) > 0 {
			stop = StopToolUse
		} else {
			stop = StopEndTurn
		}
	}
	msg := Message{
		Role:      RoleAssistant,
		Content:   st.content.String(),
		Thinking:  st.thinking,
		ToolCalls: st.calls,
	}
	var u Usage
	if st.haveUsage {
		u = floorTotal(st.currentUsage())
	}
	return &Completion{Message: msg, Usage: u, StopReason: stop}
}
