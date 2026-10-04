package commonai

import (
	"encoding/json"
	"net/http"
	"strings"
)

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
	Type string `json:"type"`
	ID   string `json:"id"`
	Name string `json:"name"`
	// Text is the block's starting text. A streamed text block opens empty and
	// fills by delta, but a block replayed from a non-streaming response
	// carries it here.
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
	text      strings.Builder
	json      strings.Builder
	thinking  strings.Builder
	signature strings.Builder
	data      string
}

// part renders one content block as the part it is, or nil when the block
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

// anStream accumulates one streamed Messages API response.
type anStream struct {
	ev *StreamEvents
	// blocks and order together preserve what this layer actually sends: a
	// numbered sequence of content blocks, which is why a reply whose text
	// brackets a thinking block reads the way the model wrote it. A finished
	// block is delivered through OnPart as it stops, and the final parts are
	// built from all of them -- so a stream cut mid-block still yields the
	// partial block it was filling, which OnPart never announced.
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

// blockFor returns the block at index, opening one of the given type if no
// content_block_start announced it. A delta with no block is malformed, but
// the text in it is still what the model said, and a message assembled from
// blocks would otherwise drop it silently -- a wrong answer that reads like a
// right one.
func (st *anStream) blockFor(index int, typ string) *anBlock {
	if b := st.blocks[index]; b != nil {
		return b
	}
	b := &anBlock{typ: typ}
	st.blocks[index] = b
	st.order = append(st.order, index)
	return b
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
		// The block is finished, which is the only moment it carries
		// everything that goes on its element -- a thinking block's signature
		// arrives after its text, and the blocks are the order the message is
		// in.
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
		// The Messages API can reject or abort a request in-stream: an HTTP
		// 200 whose stream carries an error event (overloaded_error arrives
		// this way). Map the event onto the same *APIError a non-2xx response
		// produces -- status from the documented error-type table, body = the
		// raw event JSON -- so retry classification (IsTransient) and overflow
		// detection work identically on both delivery paths. Deliberately not
		// marked as sawData: when the error is the first thing on the stream,
		// the call stays retryable.
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

// completion assembles the final (or partial) result from every block, the one
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
		// The usage report arrives in fragments -- input and cache
		// counts on message_start, output tokens on message_delta -- so the
		// assembled report is one entry, not one per event. Raw is the
		// wire-shaped object a non-streaming response would have carried
		// (input_tokens excludes cached tokens; the cache fields ride as
		// siblings).
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
