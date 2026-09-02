package commonai

import (
	"encoding/json"
	"net/http"
	"strings"
)

// anEvent is decoded Messages API stream event; the payload's type field
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
	Type string `json:"type"`
	ID   string `json:"id"`
	Name string `json:"name"`
	// Text is the block's starting text; streamed blocks fill it by delta, replayed ones carry it.
	Text      string `json:"text"`
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
// absent field is distinguishable from an explicit.
type anUsage struct {
	InputTokens              int  `json:"input_tokens"`
	OutputTokens             *int `json:"output_tokens"`
	CacheReadInputTokens     *int `json:"cache_read_input_tokens"`
	CacheCreationInputTokens *int `json:"cache_creation_input_tokens"`
}

// anError is an error event's payload; Type keys the error table, message in the APIError body.
type anError struct {
	Type string `json:"type"`
}

// anthropicErrorStatus maps a stream error event's error type onto the HTTP
// status Anthropic documents for it, so an in-stream error classifies for
// retry exactly like its non-2xx counterpart. Unrecognized types map to:
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

// anBlock accumulates content block across start/delta/stop events.
type anBlock struct {
	typ       string
	id        string
	name      string
	text      strings.Builder
	json      strings.Builder
	thinking  strings.Builder
	signature strings.Builder
	data      string
}

// part renders content block as the part it is, or nil when the block
// carried nothing worth keeping.
func (b *anBlock) part() Part {
	switch b.typ {
	case "text":
		if s := b.text.String(); s != "" {
			return TextPart{Text: s}
		}
	case "tool_use":
		return ToolCallPart{ID: b.id, Name: b.name, Arguments: toolArgs(b.json.String())}
	case "thinking":
		return ThinkingPart{Text: b.thinking.String(), Signature: b.signature.String()}
	case "redacted_thinking":
		return RedactedThinkingPart{Data: b.data}
	}
	return nil
}

// anStream accumulates streamed Messages API response.
type anStream struct {
	ev *StreamEvents
	// blocks+order preserve the content-block sequence; a mid-stream cut still yields its partial.
	blocks map[int]*anBlock
	order  []int
	stop   string

	inputTokens  int
	outputTokens int
	cacheRead    *int
	cacheWrite   *int
	haveUsage    bool

	sawData bool
}

// blockFor opens the block at index if none announced; a blockless delta's text is still kept.
func (st *anStream) blockFor(index int, typ string) *anBlock {
	if b := st.blocks[index]; b != nil {
		return b
	}
	b := &anBlock{typ: typ}
	st.blocks[index] = b
	st.order = append(st.order, index)
	return b
}

// onData decodes stream payload. Unparseable payloads are tolerated
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
			return st.ev.EmitUsage(st.currentUsage())
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
		b.text.WriteString(msg.ContentBlock.Text)
		if _, seen := st.blocks[msg.Index]; !seen {
			st.order = append(st.order, msg.Index)
		}
		st.blocks[msg.Index] = b
	case "content_block_delta":
		st.sawData = true
		if msg.Delta == nil {
			return nil
		}
		switch msg.Delta.Type {
		case "text_delta":
			st.blockFor(msg.Index, "text").text.WriteString(msg.Delta.Text)
			return st.ev.EmitText(msg.Delta.Text)
		case "thinking_delta":
			st.blockFor(msg.Index, "thinking").thinking.WriteString(msg.Delta.Thinking)
			return st.ev.EmitReasoning(msg.Delta.Thinking)
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
		// The finished block is the only with everything; a thinking signature arrives after its text.
		if b := st.blocks[msg.Index]; b != nil {
			if p := b.part(); p != nil {
				return st.ev.EmitPart(p)
			}
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
			return st.ev.EmitUsage(st.currentUsage())
		}
	case "message_stop":
		st.sawData = true
	case "error":
		// In-stream error events map to *APIError like non-2xx; not sawData, so error-first stays retryable.
		errType := ""
		if msg.Error != nil {
			errType = msg.Error.Type
		}
		return &APIError{
			Status:          anthropicErrorStatus(errType),
			Body:            string(data),
			ContextOverflow: anthropicErrorStatus(errType) == http.StatusBadRequest && contextOverflowRe.MatchString(string(data)),
		}
	}
	return nil
}

// currentUsage builds the normalized cumulative usage. input_tokens excludes
// cached tokens on this layer, so the full prompt is
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

// completion assembles the final (or partial) result from every block, the
// cut off mid-stream included: what a block accumulated before the connection
// dropped is output the caller already watched arrive. A missing stop_reason
// falls back to tool_use when calls were
// assembled, else end_turn. Completion.Timings stays nil on this layer --
// the Messages API has no timings equivalent.
func (st *anStream) completion() *Completion {
	var parts []Part
	haveCall := false
	for _, idx := range st.order {
		b := st.blocks[idx]
		if b == nil {
			continue
		}
		if p := b.part(); p != nil {
			parts = append(parts, p)
			if p.Kind() == PartKindToolCall {
				haveCall = true
			}
		}
	}
	stop := st.stop
	if stop == "" {
		if haveCall {
			stop = StopToolUse
		} else {
			stop = StopEndTurn
		}
	}
	msg := Message{Role: RoleAssistant, Parts: parts}
	msg.SyncViews()
	comp := &Completion{Message: msg, StopReason: stop, Streamed: true}
	if st.haveUsage {
		// Usage fragments (input+cache at start, output at delta) join into; Raw is the wire shape.
		u := st.currentUsage()
		u.Raw = anRawUsageJSON(st.inputTokens, st.outputTokens, st.cacheRead, st.cacheWrite)
		comp.Usages = []Usage{u}
	}
	return comp
}

// anRawUsageJSON builds the Messages-API-shaped usage object the provider's
// usage fragments combine into: input/output tokens plus the tri-state cache
// sibling fields (each included only when reported). Reasoning and cost
// figures never exist on this layer.
func anRawUsageJSON(input, output int, read, write *int) json.RawMessage {
	m := map[string]any{
		"input_tokens":  input,
		"output_tokens": output,
	}
	if read != nil {
		m["cache_read_input_tokens"] = *read
	}
	if write != nil {
		m["cache_creation_input_tokens"] = *write
	}
	b, err := json.Marshal(m)
	if err != nil {
		return nil
	}
	return b
}
