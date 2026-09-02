package loop

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mintingEvents records every row the loop mints and every finalize it emits,
// the way a persisting host sees them.
type mintingEvents struct {
	events    Events
	minted    []MessageID
	finalized []FinalizeAssistantEvent
	mintCb    func(AssistantMessageEvent) error
	finCb     func(FinalizeAssistantEvent) error
}

func newMintingEvents() *mintingEvents {
	m := &mintingEvents{}
	m.mintCb = func(ev AssistantMessageEvent) error {
		id := MessageID("row_" + string(rune('a'+len(m.minted))))
		m.minted = append(m.minted, id)
		*ev.ID = id
		return nil
	}
	m.finCb = func(ev FinalizeAssistantEvent) error {
		m.finalized = append(m.finalized, ev)
		return nil
	}
	m.events.OnAssistantMessage.Subscribe(&m.mintCb)
	m.events.OnFinalizeAssistant.Subscribe(&m.finCb)
	return m
}

// A stalled turn's wrap-up answer lands on the row the host minted for the
// turn: one mint, one finalize, the same id, the answer as its content.
func TestRunWrapUpAnswerFinalizesTheMintedRow(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "only thoughts"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("synthesized report")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	m := newMintingEvents()
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &m.events},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "task"}}})
	require.NoError(t, err)

	require.Len(t, m.minted, 1, "the wrap-up call reuses the turn's row rather than minting a second")
	require.Len(t, m.finalized, 1, "a minted row is finalized exactly once")
	assert.Equal(t, m.minted[0], m.finalized[0].ID)
	assert.Equal(t, "complete", m.finalized[0].Status)
	assert.Equal(t, "synthesized report", m.finalized[0].Msg.Content)
	assert.Equal(t, string(m.minted[0]), res.Final.ID, "the answer carries the host's id like every other answer")
}

// The wrap-up answer is a stop boundary: the stop hook is asked, and what it
// queues takes another turn.
func TestRunWrapUpAnswerAsksTheStopHook(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "only thoughts"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("synthesized report")},
		{comp: assistantComp("and the follow-up")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	q := &MessageQueue{}
	m := newMintingEvents()
	asked := 0
	stopCb := func(StopEvent) error {
		asked++
		if asked == 1 {
			q.Queue(SystemMessage{Message{Role: RoleUser, Content: "not done yet"}})
		}
		return nil
	}
	m.events.OnStop.Subscribe(&stopCb)
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &m.events, Messages: q},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "task"}}})
	require.NoError(t, err)

	assert.Equal(t, 2, asked, "asked after the wrap-up answer and again after the follow-up")
	assert.Equal(t, "and the follow-up", res.Final.Content)
	require.Len(t, m.minted, 2)
	require.Len(t, m.finalized, 2)
	assert.Equal(t, "synthesized report", m.finalized[0].Msg.Content)
	assert.Equal(t, "and the follow-up", m.finalized[1].Msg.Content)
	// user, wrap-up instruction, answer, queued message, follow-up.
	require.Len(t, res.Messages, 5)
	assert.Equal(t, "not done yet", res.Messages[3].Content)
}

// A turn that ends on reasoning alone (no tools to wrap up with) is still a
// stop boundary, and the row is finalized on the fallback text.
func TestRunReasoningOnlyAnswerAsksTheStopHook(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "the reasoning"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("now with words")},
	}}
	q := &MessageQueue{}
	m := newMintingEvents()
	asked := 0
	stopCb := func(StopEvent) error {
		asked++
		if asked == 1 {
			q.Queue(SystemMessage{Message{Role: RoleUser, Content: "say it out loud"}})
		}
		return nil
	}
	m.events.OnStop.Subscribe(&stopCb)
	res, err := Run(context.Background(), Config{Provider: provider, Events: &m.events, Messages: q}, Request{Model: "m"})
	require.NoError(t, err)

	assert.Equal(t, 2, asked)
	assert.Equal(t, "now with words", res.Final.Content)
	require.Len(t, m.finalized, 2)
	assert.Equal(t, "the reasoning", m.finalized[0].Msg.Content, "the fallback text is what the host stores")
}

// A capped final turn with sub-agents still out records the answer once, with
// the delivery trailing it, and finalizes the row even when the answer is
// only reasoning.
func TestRunCappedFinalTurnRecordsTheAnswerOnce(t *testing.T) {
	for _, tc := range []struct {
		name    string
		comp    *Completion
		content string
	}{
		{"written answer", assistantComp("launched one; out of turns"), "launched one; out of turns"},
		{"reasoning only", &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "hmm"}}}, StopReason: StopEndTurn}, "hmm"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			provider := &scriptProvider{steps: []scriptStep{{comp: tc.comp}}}
			runs := NewSubagentRuns(nil)
			runs.Launch("call_a", "audit auth", "look")
			runs.Complete("call_a", "no defects found", false, nil)
			runs.Launch("call_b", "audit db", "look")
			m := newMintingEvents()
			res, err := Run(context.Background(), Config{Provider: provider, Subagents: runs, MaxTurns: 1, Events: &m.events}, Request{Model: "m"})
			require.NoError(t, err)

			require.Len(t, m.finalized, 1, "the row is finalized once, whatever the answer was")
			assert.Equal(t, "complete", m.finalized[0].Status)
			assert.Equal(t, tc.content, m.finalized[0].Msg.Content)
			assert.Equal(t, tc.content, res.Final.Content)

			var assistants []Message
			var deliveries []Message
			for _, msg := range res.Messages {
				switch {
				case msg.Role == RoleAssistant:
					assistants = append(assistants, msg)
				case msg.Kind == SubagentReportKind:
					deliveries = append(deliveries, msg)
				}
			}
			require.Len(t, assistants, 1, "the answer appears once in the transcript")
			require.Len(t, deliveries, 1)
			assert.Contains(t, deliveries[0].Content, "no defects found")
			assert.Contains(t, deliveries[0].Content, "1 sub-agent was still running when this turn hit its limit")
			assert.Zero(t, runs.Pending(), "the abandoned run is cancelled")
		})
	}
}

// A message queued while the wrap-up call runs is not lost: it takes the
// next turn rather than coming back undelivered.
func TestRunMessageQueuedDuringWrapUpIsDelivered(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "only thoughts"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("synthesized report"), emit: func(*StreamEvents) {
			q.Queue(UserMessage{Message{Role: RoleUser, Content: "also, check the tests"}})
		}},
		{comp: assistantComp("tests checked")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	res, err := Run(ctx, Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Messages: q}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Empty(t, res.Undelivered)
	assert.Equal(t, "tests checked", res.Final.Content)
	assert.Equal(t, 3, res.Turns)
}
