package commonai

// The chat-completions wire vocabulary on the way OUT: the tool and message
// shapes, the transcript mapping, and the two prompt-cache breakpoints. The
// transport and the stream decoding live in openai.go.

import (
	"encoding/json"
	"strings"
)

// oaTool is the OpenAI wire shape of an advertised tool.
type oaTool struct {
	Type     string         `json:"type"`
	Function oaToolFunction `json:"function"`
}

// oaToolFunction is the function descriptor inside an advertised tool.
type oaToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// oaToolCall is the OpenAI wire shape of a tool call, used both for replaying
// assistant tool_calls and for decoding streamed deltas (where Index keys the
// fragment accumulation).
type oaToolCall struct {
	Index    int            `json:"index,omitempty"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function oaFunctionCall `json:"function"`
}

// oaFunctionCall is the function name and JSON-encoded arguments of a call.
type oaFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// oaMessage is one chat message on the OpenAI wire. Its encoding is owned by
// MarshalJSON, because the content field has a role-dependent presence rule
// the standard omitempty cannot express, and because ContentBlocks (a prompt
// cache_control block array) takes precedence over the plain Content string
// when set. Reasoning carries the replayed gateway-extension reasoning text
// (ReplayReasoning).
type oaMessage struct {
	Role          string
	Content       string
	ContentBlocks []map[string]any
	ToolCalls     []oaToolCall
	ToolCallID    string
	Reasoning     string
}

// MarshalJSON serializes a message for an OpenAI-compatible request. The
// OpenAI spec requires a content field on tool, user, and system messages
// even when empty: a plain `content,omitempty` drops an empty tool result and
// produces {"role":"tool","tool_call_id":...}, which upstreams reject with
// "invalid message content type: <nil>" / a 400 -- failing the whole turn. So
// content is always emitted, except for an assistant message that carries
// tool_calls, where the spec makes content optional and the model originally
// produced none; there an empty content is omitted to match what was
// generated. A non-empty ContentBlocks array is emitted as the content field
// (a block pointer is never "empty" to omitempty, so even an empty string
// content survives -- the same trick as the *string below).
func (m oaMessage) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role       string       `json:"role"`
		Content    any          `json:"content,omitempty"`
		Reasoning  string       `json:"reasoning,omitempty"`
		ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
		ToolCallID string       `json:"tool_call_id,omitempty"`
	}
	w := wire{Role: m.Role, Reasoning: m.Reasoning, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID}
	if !(m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0) {
		if len(m.ContentBlocks) > 0 {
			w.Content = m.ContentBlocks
		} else {
			// A non-nil pointer (even to "") is always emitted by omitempty, so this
			// forces a content field to appear; nil omits it for the assistant-with-
			// tool-calls case above.
			s := m.Content
			w.Content = &s
		}
	}
	return json.Marshal(w)
}

// oaWireMessages maps the neutral transcript onto the OpenAI wire: the system
// prompt (when non-empty) is prepended as a system message, assistant
// tool calls are replayed as tool_calls, and tool results ride as role:"tool"
// messages keyed by tool_call_id. Message.Thinking is not replayed on this
// dialect by default (OpenAI-compatible APIs have no reasoning-replay field) --
// only when replayReasoning is set, so a strict server never sees the unknown
// field -- and Message.ToolIsError has no wire equivalent.
func oaWireMessages(system string, msgs []Message, replayReasoning bool) ([]oaMessage, error) {
	out := make([]oaMessage, 0, len(msgs)+1)
	if system != "" {
		out = append(out, oaMessage{Role: string(RoleSystem), Content: system})
	}
	for _, m := range msgs {
		wm := oaMessage{Role: string(m.Role), Content: m.Content, ToolCallID: m.ToolCallID}
		blocks, err := oaContentBlocks(m)
		if err != nil {
			return nil, err
		}
		if blocks != nil {
			wm.ContentBlocks = blocks
		}
		if replayReasoning && m.Role == RoleAssistant {
			wm.Reasoning = reasoningText(m)
		}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, oaToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: oaFunctionCall{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
		out = append(out, wm)
	}
	return out, nil
}

// oaContentBlocks renders a message as the content ARRAY this dialect takes
// when a message holds more than text, and nil when the plain string form says
// the same thing -- which is what every OpenAI-compatible server accepts,
// including the ones that never implemented the array.
func oaContentBlocks(m Message) ([]map[string]any, error) {
	if !hasImage(m) {
		return nil, nil
	}
	var blocks []map[string]any
	for _, p := range m.EffectiveParts() {
		switch v := p.(type) {
		case TextPart:
			if v.Text != "" {
				blocks = append(blocks, map[string]any{"type": "text", "text": v.Text})
			}
		case ImagePart:
			ref, err := imageRef(DialectOpenAI, v)
			if err != nil {
				return nil, err
			}
			blocks = append(blocks, map[string]any{
				"type":      "image_url",
				"image_url": map[string]any{"url": ref},
			})
		}
	}
	return blocks, nil
}

// reasoningText concatenates an assistant message's accumulated reasoning. On
// the openai dialect the provider always produces a single ThinkingBlock; the
// concatenation keeps replay robust to a multi-block message regardless.
func reasoningText(m Message) string {
	var b strings.Builder
	for _, tb := range m.Thinking {
		if tb.Text != "" {
			b.WriteString(tb.Text)
		}
	}
	return b.String()
}

// oaMarkPromptCache applies the two Anthropic-style ephemeral prompt-cache
// breakpoints to a per-request wire message list in openai shape: the leading
// system message's string content becomes a marked one-block array (the static
// breakpoint) and the last message's content gets the moving marker. It
// operates on the freshly-built wire structures only (the caller's Messages
// are never touched), and empty content passes through unmarked -- an empty
// marked text block is rejected by upstreams and caching is an optimization,
// never a correctness requirement.
func oaMarkPromptCache(msgs []oaMessage) {
	if len(msgs) == 0 {
		return
	}
	msgs[len(msgs)-1] = oaWithMarkedContent(msgs[len(msgs)-1])
	if msgs[0].Role == "system" {
		msgs[0] = oaWithMarkedContent(msgs[0])
	}
}

// oaWithMarkedContent returns a copy of the message whose content carries the
// ephemeral cache breakpoint: a non-empty block array gets the marker on its
// last block, a non-empty string becomes a one-block array, and empty content
// passes through unmarked.
func oaWithMarkedContent(m oaMessage) oaMessage {
	switch {
	case len(m.ContentBlocks) > 0:
		blocks := make([]map[string]any, len(m.ContentBlocks))
		copy(blocks, m.ContentBlocks)
		marked := make(map[string]any, len(blocks[len(blocks)-1])+1)
		for k, v := range blocks[len(blocks)-1] {
			marked[k] = v
		}
		marked["cache_control"] = cacheEphemeral
		blocks[len(blocks)-1] = marked
		m.ContentBlocks = blocks
	case m.Content != "":
		m.ContentBlocks = []map[string]any{{
			"type": "text", "text": m.Content, "cache_control": cacheEphemeral,
		}}
	}
	return m
}
