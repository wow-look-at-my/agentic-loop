package loop

import (
	"fmt"

	"github.com/wow-look-at-my/go-containers/event"
)

// A turn's output did not reach the next call. The model then starts that
// call without what it already worked out.
type StateLossEvent struct {
	event.Args
	// Turn is the call that started without the earlier output.
	Turn int
	// Local is true when the transcript itself lost the output.
	Local bool
	// PromptTokens is what the provider reported for Turn.
	PromptTokens    int
	MinPromptTokens int
	Detail          string
}

// Thresholds for the provider-side check. A replay of N characters is at
// least N/stateCharsPerToken tokens in every tokenizer in use.
const (
	stateCharsPerToken = 6
	stateMinMissing    = 128
)

// stateCheck compares each call with the call before it. It only checks a
// tool continuation: the output, then nothing but tool results.
type stateCheck struct {
	prev    *Completion
	sentLen int
}

// observe runs after each main call: msgs is what the call sent, comp what it returned.
func (s *stateCheck) observe(turn int, msgs []Message, comp *Completion) *StateLossEvent {
	prev, sentLen := s.prev, s.sentLen
	s.prev, s.sentLen = nil, len(msgs)
	if comp != nil && len(comp.Message.ToolCalls) > 0 {
		s.prev = comp
	}
	if prev == nil || len(msgs) <= sentLen {
		return nil
	}
	recorded := msgs[sentLen]
	if recorded.Role != RoleAssistant {
		return nil
	}
	toolChars := 0
	for _, m := range msgs[sentLen+1:] {
		if m.Role != RoleTool {
			return nil
		}
		toolChars += len(m.Content)
	}
	if lost := lostLocally(prev.Message, recorded); lost != "" {
		return &StateLossEvent{Turn: turn, Local: true, Detail: lost}
	}
	if comp == nil || !comp.UsageReported || !prev.UsageReported || prev.Usage.PromptTokens == 0 {
		return nil
	}
	out := min(prev.Usage.CompletionTokens, outputChars(recorded)/stateCharsPerToken)
	floor := prev.Usage.PromptTokens + out + toolChars/stateCharsPerToken
	missing := floor - comp.Usage.PromptTokens
	if missing < stateMinMissing {
		return nil
	}
	return &StateLossEvent{
		Turn: turn, PromptTokens: comp.Usage.PromptTokens, MinPromptTokens: floor,
		Detail: fmt.Sprintf("the endpoint counted %d prompt tokens, but the previous call's %d plus its "+
			"output and tool results is at least %d: about %d tokens of the transcript did not reach the model",
			comp.Usage.PromptTokens, prev.Usage.PromptTokens, floor, missing),
	}
}

// recordStateLoss reports a loss and builds the run's last message and error.
// A model that lost its earlier output only looks busy, so the run ends.
func recordStateLoss(cfg *Config, res *Result, comp *Completion, id MessageID, loss StateLossEvent) (Message, error) {
	res.StateLosses = append(res.StateLosses, loss)
	cfg.Events.emitStateLoss(loss)
	lost := comp.Message
	lost.ToolCalls = nil
	if id != "" {
		lost.ID = string(id)
	}
	return lost, fmt.Errorf("%w: turn %d: %s", ErrStateLost, loss.Turn, loss.Detail)
}

// outputChars is the length of what a turn recorded: reasoning, text and tool arguments.
func outputChars(m Message) int {
	n := len(m.Content)
	for _, tb := range m.Thinking {
		n += len(tb.Text)
	}
	for _, tc := range m.ToolCalls {
		n += len(tc.Arguments)
	}
	return n
}

// lostLocally names what the recorded turn is missing from the completion, or "".
func lostLocally(got, recorded Message) string {
	if reasoningLen(recorded) < reasoningLen(got) {
		return fmt.Sprintf("the transcript kept %d of the %d reasoning characters the model produced",
			reasoningLen(recorded), reasoningLen(got))
	}
	if len(recorded.Content) < len(got.Content) {
		return fmt.Sprintf("the transcript kept %d of the %d text characters the model produced",
			len(recorded.Content), len(got.Content))
	}
	return ""
}

func reasoningLen(m Message) int {
	n := 0
	for _, tb := range m.Thinking {
		n += len(tb.Text) + len(tb.Signature) + len(tb.Redacted)
	}
	return n
}
