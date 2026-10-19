package loop

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// thinkingComp is a tool-calling turn whose reasoning is many characters, billed as multiple tokens.
func thinkingComp(prompt int) *Completion {
	return &Completion{
		Message: Message{
			Role:      RoleAssistant,
			Thinking:  []ThinkingBlock{{Text: strings.Repeat("reason ", 343)[:2400]}},
			ToolCalls: []ToolCall{{ID: "c1", Name: "alpha", Arguments: "{}"}},
		},
		Usage:         Usage{PromptTokens: prompt, CompletionTokens: 600, TotalTokens: prompt + 600},
		UsageReported: true,
		StopReason:    StopToolUse,
	}
}

func answerComp(prompt int) *Completion {
	c := assistantComp("done")
	c.Usage = Usage{PromptTokens: prompt, CompletionTokens: 5, TotalTokens: prompt + 5}
	c.UsageReported = true
	return c
}

func runStateCheck(t *testing.T, second *Completion) (*Result, []StateLossEvent, error) {
	t.Helper()
	provider := &scriptProvider{steps: []scriptStep{{comp: thinkingComp(1000)}, {comp: second}}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	var seen []StateLossEvent
	events := Events{}
	keep(t, &events.OnStateLoss, func(ev StateLossEvent) error { seen = append(seen, ev); return nil })
	res, err := runTurn(context.Background(), Config{
		Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events,
	}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	return res, seen, err
}

func TestStateCheckFaithfulReplayIsQuiet(t *testing.T) {
	res, seen, err := runStateCheck(t, answerComp(1610))
	require.NoError(t, err)
	assert.Empty(t, seen)
	assert.Empty(t, res.StateLosses)
	assert.Equal(t, "done", res.Final.Content)
}

func TestStateCheckCatchesEndpointThatDropsReasoning(t *testing.T) {
	// The prompt grew by the tool result only: the endpoint never saw the reasoning.
	res, seen, err := runStateCheck(t, answerComp(1010))
	require.NoError(t, err, "a state loss is reported loudly, but the run goes on: it can still recover")
	assert.Equal(t, "done", res.Final.Content)
	require.Len(t, seen, 1)
	assert.Equal(t, SeverityNotice, seen[0].Severity, "one loss is a notice")
	ev := seen[0]
	assert.False(t, ev.Local)
	assert.Equal(t, 2, ev.Turn)
	assert.Equal(t, 1010, ev.PromptTokens)
	assert.GreaterOrEqual(t, ev.MinPromptTokens, 1400)
	assert.Contains(t, ev.Detail, "did not reach the model")
	require.Len(t, res.StateLosses, 1)
	assert.Equal(t, ev.Detail, res.StateLosses[0].Detail)
}

func TestStateCheckSkipsUnreportedUsage(t *testing.T) {
	c := answerComp(1010)
	c.UsageReported = false
	_, seen, err := runStateCheck(t, c)
	require.NoError(t, err)
	assert.Empty(t, seen, "no usage means no evidence either way")
}

func TestStateCheckCatchesTranscriptThatDropsReasoning(t *testing.T) {
	var s stateCheck
	first := thinkingComp(1000)
	sent := []Message{{Role: RoleUser, Content: "go"}}
	assert.Nil(t, s.observe(1, sent, first))

	stripped := first.Message
	stripped.Thinking = nil
	next := append(append([]Message{}, sent...), stripped, Message{Role: RoleTool, Content: "r", ToolCallID: "c1"})
	ev := s.observe(2, next, answerComp(5000))
	require.NotNil(t, ev)
	assert.True(t, ev.Local)
	assert.Contains(t, ev.Detail, "kept 0 of the 2400 reasoning characters")
}

func TestStateCheckCatchesTranscriptThatDropsText(t *testing.T) {
	var s stateCheck
	first := thinkingComp(1000)
	first.Message.Content = "visible"
	sent := []Message{{Role: RoleUser, Content: "go"}}
	s.observe(1, sent, first)
	short := first.Message
	short.Content = ""
	next := append(append([]Message{}, sent...), short, Message{Role: RoleTool, Content: "r", ToolCallID: "c1"})
	ev := s.observe(2, next, answerComp(5000))
	require.NotNil(t, ev)
	assert.True(t, ev.Local)
	assert.Contains(t, ev.Detail, "text characters")
}

func TestStateCheckIgnoresNewUserMessage(t *testing.T) {
	var s stateCheck
	first := thinkingComp(1000)
	sent := []Message{{Role: RoleUser, Content: "go"}}
	s.observe(1, sent, first)
	// A template may drop old reasoning at a new user message, so this is no evidence.
	next := append(append([]Message{}, sent...), first.Message,
		Message{Role: RoleTool, Content: "r", ToolCallID: "c1"}, Message{Role: RoleUser, Content: "more"})
	assert.Nil(t, s.observe(2, next, answerComp(1010)))
}

func TestStateCheckIgnoresTurnWithoutToolCalls(t *testing.T) {
	var s stateCheck
	sent := []Message{{Role: RoleUser, Content: "go"}}
	s.observe(1, sent, answerComp(1000))
	next := append(append([]Message{}, sent...), Message{Role: RoleAssistant, Content: "done"})
	assert.Nil(t, s.observe(2, next, answerComp(10)))
}
