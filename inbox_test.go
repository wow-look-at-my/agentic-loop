package agentic

import (
	"context"
	"testing"

	"github.com/stretchr/testify/require"
)

// queueInbox is a test Inbox backed by a slice. Receive pops from the front;
// ok=false when empty.
type queueInbox struct {
	msgs []Message
}

func (q *queueInbox) Receive(context.Context) (Message, bool) {
	if len(q.msgs) == 0 {
		return Message{}, false
	}
	m := q.msgs[0]
	q.msgs = q.msgs[1:]
	return m, true
}

func (q *queueInbox) push(m Message) { q.msgs = append(q.msgs, m) }

// TestRunDrainsInboxBeforeEachTurn verifies that a message queued before the
// run starts is appended to the transcript and answered by the model.
func TestRunDrainsInboxBeforeEachTurn(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("first answer")},
		{comp: assistantComp("second answer")},
	}}
	// The inbox holds one message before the run; the model answers it on its
	// first call. No message is dropped.
	inbox := &queueInbox{}
	inbox.push(Message{Role: RoleUser, Content: "queued notice"})

	cfg := Config{Provider: provider, Inbox: inbox}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	// The model saw both the original user message and the queued notice.
	require.Len(t, provider.reqs, 2)
	got := provider.reqs[0].Messages
	require.Equal(t, "go", got[len(got)-2].Content)
	require.Equal(t, "queued notice", got[len(got)-1].Content)
	require.Equal(t, "second answer", res.Final.Content)
	require.Empty(t, inbox.msgs, "inbox must be drained")
}

// TestRunDrainsInboxAtFinish verifies the at-least-once contract: a message
// that arrives DURING the final turn is still answered before the run ends.
func TestRunDrainsInboxAtFinish(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		// First turn: model answers, asks for no tools. The inbox is empty at
		// the top of this turn, so the run would end -- but a message lands
		// during the turn.
		{comp: assistantComp("done")},
		// finish() drains, finds the message, and loops: the model answers it.
		{comp: assistantComp("answered the notice")},
	}}
	inbox := &queueInbox{}

	cfg := Config{
		Provider: provider,
		Inbox:    inbox,
		// A hook that pushes a message DURING the first turn, simulating an
		// external event arriving mid-run.
		Events: Events{OnTurnEnd: func(_ int, _ *Completion, _ error) error {
			if len(provider.reqs) == 1 {
				inbox.push(Message{Role: RoleUser, Content: "late notice"})
			}
			return nil
		}},
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	// The run did not end after the first turn: it drained the inbox, found
	// the late message, and answered it.
	require.Len(t, provider.reqs, 2)
	require.Equal(t, "answered the notice", res.Final.Content)
	require.Empty(t, inbox.msgs)
}

// TestRunNoInboxIsNoop verifies a nil Inbox changes nothing about the loop.
func TestRunNoInboxIsNoop(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("done")},
	}}
	res, err := Run(context.Background(), Config{Provider: provider}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	require.Len(t, provider.reqs, 1)
	require.Equal(t, "done", res.Final.Content)
}
