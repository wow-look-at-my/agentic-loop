package agentic

import (
	"context"
	"strings"
	"time"

	"github.com/wow-look-at-my/go-containers/set"
)

// maxSharedContextRunes caps a rendered parent-context transcript so an enormous
// history can't defeat the very purpose of a sub-agent (a *clean* context). When
// over the cap the oldest part is dropped, keeping the most recent (most
// task-relevant) tail.
const maxSharedContextRunes = 200_000

// subagentSummaryTimeout bounds the one extra model call made when
// share_context=summary, so a slow summary can't wedge the (gate-held) turn.
const subagentSummaryTimeout = 2 * time.Minute

// RenderTranscript renders parent-conversation messages into a readable,
// role-labeled transcript suitable for handing to a sub-agent as background
// context. Tool calls are summarized inline; the result is rune-capped. It is
// deliberately plain text (not replayed messages) so any subset — even one that
// would split a tool_call/tool_result pair — is always well-formed.
func RenderTranscript(msgs []Message) string {
	var b strings.Builder
	for _, m := range msgs {
		b.WriteString(transcriptLabel(m.Role))
		b.WriteString(":\n")
		if c := strings.TrimSpace(m.Content); c != "" {
			b.WriteString(c)
			b.WriteString("\n")
		}
		for _, tc := range m.ToolCalls {
			b.WriteString("[requested tool ")
			b.WriteString(tc.Name)
			if a := strings.TrimSpace(tc.Arguments); a != "" {
				b.WriteString(" with ")
				b.WriteString(a)
			}
			b.WriteString("]\n")
		}
		b.WriteString("\n")
	}
	return capRunesTail(strings.TrimSpace(b.String()), maxSharedContextRunes)
}

// transcriptLabel is the display label for a role in a rendered transcript.
func transcriptLabel(role Role) string {
	switch role {
	case RoleSystem:
		return "System"
	case RoleUser:
		return "User"
	case RoleAssistant:
		return "Assistant"
	case RoleTool:
		return "Tool result"
	case "":
		return "Message"
	default:
		r := string(role)
		return strings.ToUpper(r[:1]) + r[1:]
	}
}

// capRunesTail returns s unchanged when within maxRunes, else the last
// maxRunes runes prefixed with a truncation marker (keeping the most recent
// context).
func capRunesTail(s string, maxRunes int) string {
	if maxRunes <= 0 {
		return s
	}
	r := []rune(s)
	if len(r) <= maxRunes {
		return s
	}
	return "[... earlier shared context truncated ...]\n\n" + strings.TrimSpace(string(r[len(r)-maxRunes:]))
}

// SelectLastN returns the last n messages of msgs (all of them when n exceeds
// the length, nil when n <= 0).
func SelectLastN(msgs []Message, n int) []Message {
	if n <= 0 || len(msgs) == 0 {
		return nil
	}
	if n >= len(msgs) {
		return msgs
	}
	return msgs[len(msgs)-n:]
}

// SelectByEndIndices returns the messages at the given 1-based-from-the-end
// indices (1 = most recent), in chronological order, de-duplicated. Indices that
// fall outside the range are ignored.
func SelectByEndIndices(msgs []Message, indices []int) []Message {
	chosen := set.New[int](len(indices))
	for _, idx := range indices {
		if idx < 1 {
			continue
		}
		pos := len(msgs) - idx
		if pos >= 0 && pos < len(msgs) {
			chosen.Add(pos)
		}
	}
	var out []Message
	for i, m := range msgs {
		if chosen.Contains(i) {
			out = append(out, m)
		}
	}
	return out
}

// contextSummarySystemPrompt primes the model for the share_context=summary
// briefing call. Ported verbatim from the source application.
const contextSummarySystemPrompt = "You condense a conversation into a briefing for a sub-agent that will carry out a related task. " +
	"Capture the salient facts, decisions, constraints, and any specific identifiers, names, or values the sub-agent will need to act correctly. " +
	"Be faithful and concise; omit pleasantries and meta-commentary. Output only the briefing."

// generateContextSummary asks the same model to summarize a parent
// conversation into a briefing a sub-agent can use: one bounded
// (subagentSummaryTimeout), tool-less call with no retry, via OneShot. An
// empty transcript yields an empty summary and a nil completion, with no model
// call. The completion is returned rather than dropped because this briefing is
// part of what the sub-agent run spent, and it rides up with the sub-run's own
// usages.
func generateContextSummary(ctx context.Context, p Provider, model string, msgs []Message, maxTokens int, extra map[string]any) (string, *Completion, error) {
	transcript := RenderTranscript(msgs)
	if strings.TrimSpace(transcript) == "" {
		return "", nil, nil
	}
	comp, err := OneShot(ctx, p, Request{
		Model:  model,
		System: contextSummarySystemPrompt,
		Messages: []Message{
			{Role: RoleUser, Content: "Conversation to brief the sub-agent on:\n\n" + transcript},
		},
		MaxTokens: maxTokens,
		Extra:     extra,
	}, subagentSummaryTimeout)
	if err != nil {
		return "", comp, err
	}
	return strings.TrimSpace(comp.Message.Content), comp, nil
}

// composeSubagentTask folds an optional shared-context block into the
// orchestrator's prompt as a single, clearly delimited task message. With no
// block it returns the prompt unchanged (the isolated default).
func composeSubagentTask(block, prompt string) string {
	if strings.TrimSpace(block) == "" {
		return prompt
	}
	var b strings.Builder
	b.WriteString("Context shared from the parent conversation (background only):\n\n")
	b.WriteString(block)
	b.WriteString("\n\n----------------------------------------\n\nYour task:\n\n")
	b.WriteString(prompt)
	return b.String()
}

const subagentPreviewMaxRunes = 160

// subagentPreview flattens whitespace and truncates s to a single short line for
// the activity strip, so a giant argument blob or a long tool result (a whole
// file body) never floods the progress view -- only the model's distilled final
// report is ever shown in full.
func subagentPreview(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	r := []rune(s)
	if len(r) > subagentPreviewMaxRunes {
		return string(r[:subagentPreviewMaxRunes-3]) + "..."
	}
	return s
}
