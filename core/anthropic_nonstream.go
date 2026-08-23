package commonai

import (
	"encoding/json"
)

// anNonStream is the non-streaming Messages API response shape a server that
// ignores stream:true answers with.
type anNonStream struct {
	Content    []anNonStreamBlock `json:"content"`
	StopReason string             `json:"stop_reason"`
	Usage      *anUsage           `json:"usage"`
}

// anNonStreamBlock is one content block of a non-streaming response.
type anNonStreamBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text"`
	Thinking  string          `json:"thinking"`
	Signature string          `json:"signature"`
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Input     json.RawMessage `json:"input"`
	Data      string          `json:"data"`
}

// parseAnthropicNonStream reassembles a plain-JSON Messages response into a
// Completion with Streamed false. Content blocks map onto the same
// Message fields the stream path builds (text concatenated, thinking and
// redacted_thinking collected, tool_use calls assembled with their input
// re-serialized to the raw argument string), usage normalized identically,
// and the stop reason post-inferred the same way.
func parseAnthropicNonStream(data []byte) (*Completion, error) {
	var resp anNonStream
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, badRequestErr("anthropic: decode non-streaming response: " + err.Error())
	}
	comp := &Completion{
		Message:    Message{Role: RoleAssistant},
		StopReason: resp.StopReason,
		Streamed:   false,
	}
	for _, b := range resp.Content {
		switch b.Type {
		case "text":
			comp.Message.Parts = append(comp.Message.Parts, TextPart{Text: b.Text})
		case "thinking":
			comp.Message.Parts = append(comp.Message.Parts, ThinkingPart{Text: b.Thinking, Signature: b.Signature})
		case "redacted_thinking":
			comp.Message.Parts = append(comp.Message.Parts, RedactedThinkingPart{Data: b.Data})
		case "tool_use":
			comp.Message.Parts = append(comp.Message.Parts, ToolCallPart{
				ID: b.ID, Name: b.Name, Arguments: toolArgs(string(b.Input)),
			})
		}
	}
	comp.Message.SyncViews()
	if resp.Usage != nil {
		u := resp.Usage.toUsage()
		u.Raw = anRawUsageJSON(
			resp.Usage.InputTokens,
			intOrZero(resp.Usage.OutputTokens),
			resp.Usage.CacheReadInputTokens,
			resp.Usage.CacheCreationInputTokens,
		)
		comp.Usages = []Usage{u}
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

// intOrZero dereferences a tri-state int pointer, zero when nil.
func intOrZero(p *int) int {
	if p == nil {
		return 0
	}
	return *p
}

// toUsage normalizes a Messages-API usage object: input_tokens EXCLUDES
// cached tokens, so the full prompt is input + cache_read + cache_creation;
// the tri-state cache pointers pass through as reported.
func (u *anUsage) toUsage() Usage {
	read, write := 0, 0
	if u.CacheReadInputTokens != nil {
		read = *u.CacheReadInputTokens
	}
	if u.CacheCreationInputTokens != nil {
		write = *u.CacheCreationInputTokens
	}
	out := Usage{
		PromptTokens:     u.InputTokens + read + write,
		CompletionTokens: intOrZero(u.OutputTokens),
		CacheReadTokens:  clonePtr(u.CacheReadInputTokens),
		CacheWriteTokens: clonePtr(u.CacheCreationInputTokens),
	}
	out.TotalTokens = out.PromptTokens + out.CompletionTokens
	return out
}
