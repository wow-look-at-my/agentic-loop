package commonai

import (
	"encoding/json"
	"net/http"
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
	Model string   `json:"model"`
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
// status Anthropic documents for it. An in-stream error classifies for
// retry exactly like its non-2xx counterpart. Unrecognized types map to: an
// unknown in-stream failure is a server-side abort, and treating it as
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

// anStream writes a streamed Messages API response into a CallSink, one node per content block.
type anStream struct {
	out CallSink
	// blocks maps a content-block index to its node; nodes keeps the order the blocks began.
	blocks map[int]IMessage
	nodes  []IMessage
	stop   string

	inputTokens  int
	outputTokens int
	cacheRead    *int
	cacheWrite   *int
	haveUsage    bool
	model        string

	sawData bool
}

func newANStream(out CallSink) *anStream {
	return &anStream{out: out, blocks: map[int]IMessage{}}
}

func (st *anStream) start(index int, n IMessage) {
	st.blocks[index] = n
	st.nodes = append(st.nodes, n)
	st.out.Start(n)
}

// blockFor is the node at index. A delta for a block that was never announced still keeps its text.
func (st *anStream) blockFor(index int, mk func() IMessage) IMessage {
	if n := st.blocks[index]; n != nil {
		return n
	}
	n := mk()
	st.start(index, n)
	return n
}

// startBlock makes the node a content_block_start announces.
func (st *anStream) startBlock(index int, cb *anContentBlock, data []byte) {
	if old := st.blocks[index]; old != nil && !old.Final() {
		st.out.Done(old)
	}
	switch cb.Type {
	case "text":
		n := NewAssistant()
		st.start(index, n)
		if cb.Text != "" {
			st.out.Delta(n, FieldText, cb.Text)
		}
	case "tool_use":
		st.start(index, NewToolCall(CallID(cb.ID), cb.Name))
	case "thinking":
		n := NewThinking(DialectAnthropic)
		n.Signature = NewEncryptedReasoning(cb.Signature)
		st.start(index, n)
		if cb.Thinking != "" {
			st.out.Delta(n, FieldText, cb.Thinking)
		}
	case "redacted_thinking":
		n := &ThinkingMessage{Text: FinishedString(""), Redacted: NewEncryptedReasoning(cb.Data), Dialect: DialectAnthropic}
		st.start(index, n)
	default:
		var raw struct {
			ContentBlock json.RawMessage `json:"content_block"`
		}
		_ = json.Unmarshal(data, &raw)
		st.start(index, &RawBlockMessage{Type: cb.Type, JSON: raw.ContentBlock, Dialect: DialectAnthropic})
	}
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
		if msg.Message != nil && msg.Message.Model != "" {
			st.model = msg.Message.Model
		}
		if msg.Message != nil && msg.Message.Usage != nil {
			u := msg.Message.Usage
			st.haveUsage = true
			st.inputTokens = u.InputTokens
			st.cacheRead = clonePtr(u.CacheReadInputTokens)
			st.cacheWrite = clonePtr(u.CacheCreationInputTokens)
			if u.OutputTokens != nil {
				st.outputTokens = *u.OutputTokens
			}
			st.out.Usage(st.currentUsage())
			return st.out.Err()
		}
	case "content_block_start":
		st.sawData = true
		if msg.ContentBlock == nil {
			return nil
		}
		st.startBlock(msg.Index, msg.ContentBlock, data)
		return st.out.Err()
	case "content_block_delta":
		st.sawData = true
		if msg.Delta == nil {
			return nil
		}
		switch msg.Delta.Type {
		case "text_delta":
			n := st.blockFor(msg.Index, func() IMessage { return NewAssistant() })
			st.out.Delta(n, FieldText, msg.Delta.Text)
		case "thinking_delta":
			n := st.blockFor(msg.Index, func() IMessage { return NewThinking(DialectAnthropic) })
			st.out.Delta(n, FieldText, msg.Delta.Thinking)
		case "signature_delta":
			// signature_delta carries the whole signature: it assigns, never appends.
			if t, ok := st.blocks[msg.Index].(*ThinkingMessage); ok && msg.Delta.Signature != "" {
				t.Signature = NewEncryptedReasoning(msg.Delta.Signature)
			}
		case "input_json_delta":
			if tc, ok := st.blocks[msg.Index].(*ToolCallMessage); ok {
				st.out.Delta(tc, FieldArguments, msg.Delta.PartialJSON)
			}
		}
		return st.out.Err()
	case "content_block_stop":
		st.sawData = true
		// The finished block has everything: a thinking signature arrives after its text.
		if n := st.blocks[msg.Index]; n != nil && !n.Final() {
			st.out.Done(n)
		}
		return st.out.Err()
	case "message_delta":
		st.sawData = true
		if msg.Delta != nil && msg.Delta.StopReason != "" {
			st.stop = msg.Delta.StopReason
		}
		if msg.Usage != nil && msg.Usage.OutputTokens != nil {
			// Cumulative snapshot: overwrite, never sum.
			st.haveUsage = true
			st.outputTokens = *msg.Usage.OutputTokens
			st.out.Usage(st.currentUsage())
			return st.out.Err()
		}
	case "message_stop":
		st.sawData = true
	case "error":
		// In-stream error events map to *APIError like non-2xx; not sawData, so error- stays retryable.
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
// cached tokens on this layer. The full prompt is input + cache_read +
// cache_creation. The tri-state cache pointers are passed through as
// reported.
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

// completion assembles the final (or partial) result from every block. The
// cut off mid-stream included: what a block accumulated before the connection
// dropped is output the caller already watched arrive. A missing stop_reason
// falls back to tool_use when calls were assembled, else end_turn.
// Completion.Timings stays nil on this layer -- the Messages API has no
// timings equivalent.
func (st *anStream) completion() *Completion {
	haveCall := false
	for _, n := range st.nodes {
		if !n.Final() {
			st.out.Done(n)
		}
		if _, ok := n.(*ToolCallMessage); ok {
			haveCall = true
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
	comp := &Completion{Message: MessageFromNodes(st.nodes), StopReason: stop, Streamed: true, Model: st.model}
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
