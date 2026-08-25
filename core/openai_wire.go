package commonai

// Outbound chat-completions wire vocabulary (shapes, mapping, cache); transport in openai.go.

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

// oaFunctionCall has no omitempty on Arguments: a missing arguments field makes Z.AI reject with 400.
type oaFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments"`
}

// oaMessage is one chat message; MarshalJSON owns encoding because content presence is role-dependent.
type oaMessage struct {
	Role             string
	Content          string
	ContentBlocks    []map[string]any
	ToolCalls        []oaToolCall
	ToolCallID       string
	Reasoning        string
	ReasoningDetails []oaReasoningDetail
}

// MarshalJSON always emits content so an empty tool result doesn't 400; assistant tool_calls may omit it.
func (m oaMessage) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role             string              `json:"role"`
		Content          any                 `json:"content,omitempty"`
		Reasoning        string              `json:"reasoning,omitempty"`
		ReasoningDetails []oaReasoningDetail `json:"reasoning_details,omitempty"`
		ToolCalls        []oaToolCall        `json:"tool_calls,omitempty"`
		ToolCallID       string              `json:"tool_call_id,omitempty"`
	}
	w := wire{
		Role: m.Role, Reasoning: m.Reasoning, ReasoningDetails: m.ReasoningDetails,
		ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID,
	}
	if !(m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0) {
		if len(m.ContentBlocks) > 0 {
			w.Content = m.ContentBlocks
		} else {
			// A non-nil pointer forces the content field via omitempty; nil omits it for the tool-call case.
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
// dialect by default (a strict OpenAI-compatible server rejects an unknown
// field) -- only when replayReasoning is set, and then both the flattened
// text (reasoning) and, when captured, the verbatim reasoning_details array
// go out -- a gateway requiring the latter for tool-call continuity (see
// oaReplayReasoningDetails) ignores the former, and a server that only knows
// the former ignores the latter. Message.ToolIsError has no wire equivalent.
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
			wm.ReasoningDetails = oaReplayReasoningDetails(m)
		}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, oaToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: oaFunctionCall{Name: tc.Name, Arguments: toolArgs(tc.Arguments)},
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

// reasoningText concatenates an assistant's thinking, robust to multi-block messages.
func reasoningText(m Message) string {
	var b strings.Builder
	for _, tb := range m.Thinking {
		if tb.Text != "" {
			b.WriteString(tb.Text)
		}
	}
	return b.String()
}

// oaReasoningDetail is one item of an OpenRouter-style reasoning_details
// array. A field this dialect never interprets (Signature, Format, Index) is
// still captured and replayed, since a downstream gateway checks the whole
// item, not the fields this library happens to read.
type oaReasoningDetail struct {
	Type      string  `json:"type"`
	Text      string  `json:"text,omitempty"`
	Summary   string  `json:"summary,omitempty"`
	Data      string  `json:"data,omitempty"`
	Signature *string `json:"signature,omitempty"`
	ID        string  `json:"id,omitempty"`
	Format    string  `json:"format,omitempty"`
	Index     int     `json:"index,omitempty"`
}

// oaReasoningDetailsJSON marshals a captured reasoning_details array for
// storage in a ThinkingBlock's Signature, or "" when none arrived.
func oaReasoningDetailsJSON(details []oaReasoningDetail) string {
	if len(details) == 0 {
		return ""
	}
	b, err := json.Marshal(details)
	if err != nil {
		return ""
	}
	return string(b)
}

// oaReplayReasoningDetails decodes the verbatim reasoning_details array this
// dialect stashed in a Thinking block's Signature. A gateway that requires
// reasoning continuity for a tool call (a reasoning model reachable only
// through its own provider's Responses-shaped API, fronted by an
// OpenAI-compatible chat-completions endpoint) reconstructs that API's
// request from this array; sending back only the flattened text drops the
// item ids it pairs a tool call against, and the request is rejected. A
// block with no Signature, or one holding a different dialect's opaque
// payload (not a JSON array), contributes nothing here.
func oaReplayReasoningDetails(m Message) []oaReasoningDetail {
	for _, tb := range m.Thinking {
		if tb.Signature == "" {
			continue
		}
		var details []oaReasoningDetail
		if err := json.Unmarshal([]byte(tb.Signature), &details); err == nil {
			return details
		}
	}
	return nil
}

// oaMarkPromptCache marks the system (static) and last (moving) messages; empties stay unmarked.
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
