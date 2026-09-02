package commonai

import (
	"encoding/json"
)

// oaNonStream is the non-streaming chat-completions response shape a server
// that ignores stream:true answers with.
type oaNonStream struct {
	Choices []struct {
		Message      oaNonStreamMessage `json:"message"`
		FinishReason string             `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

// oaNonStreamMessage is the assistant message inside a non-streaming choice.
// Content is a pointer because a tool-call-only response carries null.
type oaNonStreamMessage struct {
	Role             string              `json:"role"`
	Content          *string             `json:"content"`
	Reasoning        string              `json:"reasoning"`
	ReasoningDetails []oaReasoningDetail `json:"reasoning_details"`
	ToolCalls        []oaToolCall        `json:"tool_calls"`
}

// parseNonStream reassembles a plain-JSON chat-completions response into a
// Completion with Streamed false -- the transparent acceptance of a server that
// ignored stream:true (or a proxy that buffered it). The stop reason is
// normalized and post-inferred exactly like the streamed path; usage extras
// (reasoning tokens, provider cost, the verbatim usage object) are captured
// when present.
func (o *openaiProvider) parseNonStream(data []byte) (*Completion, error) {
	var resp oaNonStream
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, badRequestErr("openai: decode non-streaming response: " + err.Error())
	}
	if len(resp.Choices) == 0 {
		return nil, badRequestErr("openai: non-streaming response carries no choices")
	}
	first := resp.Choices[0]
	content := ""
	if first.Message.Content != nil {
		content = *first.Message.Content
	}
	var calls []ToolCall
	for _, tc := range first.Message.ToolCalls {
		calls = append(calls, ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Arguments: toolArgs(tc.Function.Arguments),
		})
	}
	details := oaReasoningDetailsJSON(first.Message.ReasoningDetails)
	comp := &Completion{
		Message:    oaAssistantMessage(first.Message.Reasoning, details, content, calls),
		StopReason: normalizeStopReason(first.FinishReason),
		Streamed:   false,
	}
	if len(resp.Usage) > 0 {
		var u oaUsage
		if err := json.Unmarshal(resp.Usage, &u); err == nil {
			usage := u.toUsage()
			usage.Raw = resp.Usage
			usage.ReasoningTokens = u.reasoningTokens()
			usage.CostUsd = u.costUsd()
			comp.Usages = append(comp.Usages, usage)
		}
	}
	if comp.StopReason == "" {
		if len(comp.Message.ToolCalls) > 0 {
			comp.StopReason = StopToolUse
		} else {
			comp.StopReason = StopEndTurn
		}
	}
	return comp, nil
}

// normalizeStopReason maps OpenAI finish reasons onto the normalized
// constants; anything unrecognized passes through raw.
func normalizeStopReason(reason string) string {
	switch reason {
	case "stop":
		return StopEndTurn
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopMaxTokens
	}
	return reason
}

// oaAssistantMessage assembles turn's parts: reasoning (text plus, when
// captured, the verbatim reasoningDetailsJSON in Signature -- see
// oaReasoningDetailsJSON), then content, then tool calls.
func oaAssistantMessage(reasoning, reasoningDetailsJSON, content string, calls []ToolCall) Message {
	var parts []Part
	if reasoning != "" || reasoningDetailsJSON != "" {
		parts = append(parts, ThinkingPart{Text: reasoning, Signature: reasoningDetailsJSON})
	}
	if content != "" {
		parts = append(parts, TextPart{Text: content})
	}
	for _, c := range calls {
		parts = append(parts, ToolCallPart{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	m := Message{Role: RoleAssistant, Parts: parts}
	m.SyncViews()
	return m
}
