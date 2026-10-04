package loop

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A message queued while the model works is delivered at the top of the next
// turn -- the model asked for a tool, so a turn boundary is coming anyway.
func TestQueuedMessageIsDeliveredAtTheNextTurn(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	exec.execute = func(context.Context, ToolCall) (ToolResult, error) {
		assert.True(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "actually, stop"}}))
		return ToolResult{Content: "ran"}, nil
	}
	res, err := Run(context.Background(), Config{
		Provider: provider,
		Tools:    exec.registry(),
		Messages: q,
	}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Final.Content)
	assert.Contains(t, contents(res.Messages), "actually, stop")
	assert.Empty(t, res.Undelivered)
}

// The bug this file exists for: a message queued when the model has already
// answered must START ANOTHER TURN, every time -- not per run.
func TestQueuedMessageStartsAnotherTurnEveryTime(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("first")},
		{comp: assistantComp("second")},
		{comp: assistantComp("third")},
	}}
	turns := 0
	cfg := Config{Provider: provider, Messages: q}
	cfg.TurnHook = func(int) {
		turns++
		if turns <= 2 {
			assert.True(t, q.Queue(SystemMessage{Message{Role: RoleUser, Kind: "ci_status_change", Content: "CI went red"}}))
		}
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "third", res.Final.Content)
	assert.Equal(t, 3, res.Turns, "each queued notice takes another turn")
	assert.Equal(t, 2, count(contents(res.Messages), "CI went red"))
	assert.Empty(t, res.Undelivered)
}

// System messages precede user messages queued in the same gap.
func TestQueuedSystemMessagesPrecedeUserMessages(t *testing.T) {
	q := &MessageQueue{}
	require.True(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "from the user"}}))
	require.True(t, q.Queue(SystemMessage{Message{Role: RoleUser, Content: "from the watcher"}}))
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
	res, err := Run(context.Background(), Config{
		Provider: provider, Messages: q,
	}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	got := contents(res.Messages)
	require.Equal(t, []string{"go", "from the watcher", "from the user", "ok"}, got)
}

// A queued message reaches the model even when the answering turn produced no
// content: the wrap-up call is skipped in favour of delivering it.
func TestQueuedMessageContinuesAStalledTurn(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("")},
		{comp: assistantComp("answered")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	cfg := Config{Provider: provider, Tools: exec.registry(), Messages: q}
	cfg.TurnHook = func(turn int) {
		if turn == 1 {
			assert.True(t, q.Queue(SystemMessage{Message{Role: RoleUser, Content: "CI went red"}}))
		}
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "answered", res.Final.Content)
	assert.Contains(t, contents(res.Messages), "CI went red")
	assert.Equal(t, 2, res.Turns, "no wrap-up call is spent when a message is waiting")
}

// Run closes the queue on the way out, so a producer that misses the run is
// told so instead of leaving a message nothing will read.
func TestQueueAfterTheRunEndsIsRefused(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}
	_, err := Run(context.Background(), Config{
		Provider: provider, Messages: q,
	}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.False(t, q.Queue(SystemMessage{Message{Role: RoleUser, Content: "too late"}}))
	assert.False(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "too late"}}))
	assert.True(t, q.Closed())
}

// A run that ends on an error hands back what it never delivered rather than
// dropping it: the producer believes the model saw those messages.
func TestUndeliveredMessagesComeBackWhenTheRunFails(t *testing.T) {
	q := &MessageQueue{}
	boom := errors.New("upstream gave up")
	// Both messages are queued DURING the call that then fails, so the run
	// ends with no turn boundary left to drain them at.
	provider := &scriptProvider{steps: []scriptStep{{
		emit: func(*StreamEvents) {
			assert.True(t, q.Queue(SystemMessage{Message{Role: RoleUser, Content: "CI went red"}}))
			assert.True(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "and stop"}}))
		},
		err: boom,
	}}}
	res, err := Run(context.Background(), Config{
		Provider: provider, Messages: q,
	}, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.ErrorIs(t, err, boom)
	require.NotNil(t, res)
	require.Equal(t, []string{"CI went red", "and stop"}, contents(res.Undelivered))
}

// A host cap still bounds the run: the queued message is handed back rather
// than delivered, and the model's own answer stays the final.
func TestQueuedMessageDoesNotOutrunAHostCap(t *testing.T) {
	q := &MessageQueue{}
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("capped answer")}}}
	cfg := Config{Provider: provider, Messages: q, MaxTurns: 1}
	cfg.TurnHook = func(int) {
		assert.True(t, q.Queue(SystemMessage{Message{Role: RoleUser, Content: "CI went red"}}))
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "capped answer", res.Final.Content)
	assert.Equal(t, 1, res.Turns)
	assert.Equal(t, []string{"CI went red"}, contents(res.Undelivered))
}

// Drain stable-partitions system ahead of user, each kind keeping its own
// arrival order, regardless of how the kinds interleaved going in.
func TestDrainStablePartitionsSystemBeforeUser(t *testing.T) {
	q := &MessageQueue{}
	q.Queue(UserMessage{Message{Content: "u1"}})
	q.Queue(SystemMessage{Message{Content: "s1"}})
	q.Queue(UserMessage{Message{Content: "u2"}})
	q.Queue(SystemMessage{Message{Content: "s2"}})
	assert.Equal(t, []string{"s1", "s2", "u1", "u2"}, contents(q.Drain()))
}

// A nil queue accepts nothing: there is no run behind it to deliver anything.
func TestNilQueueAcceptsNothing(t *testing.T) {
	var q *MessageQueue
	assert.False(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "x"}}))
	assert.True(t, q.Closed())
	assert.Zero(t, q.Len())
	assert.Nil(t, q.Drain())
	assert.Nil(t, q.Close())
}

// Close is idempotent and drains only.
func TestCloseReturnsUndrainedMessagesOnce(t *testing.T) {
	q := &MessageQueue{}
	require.True(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "a"}}))
	assert.Equal(t, []string{"a"}, contents(q.Close()))
	assert.Nil(t, q.Close())
	assert.False(t, q.Queue(UserMessage{Message{Role: RoleUser, Content: "b"}}))
}

// Producers queue from their own goroutines; the loop drains from its own.
func TestQueueIsSafeForConcurrentProducers(t *testing.T) {
	q := &MessageQueue{}
	var wg sync.WaitGroup
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			q.Queue(UserMessage{Message{Role: RoleUser, Content: "n"}})
		}()
	}
	wg.Wait()
	assert.Len(t, q.Drain(), 50)
}

func contents(msgs []Message) []string {
	out := make([]string, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, m.Content)
	}
	return out
}

func count(all []string, want string) int {
	n := 0
	for _, s := range all {
		if s == want {
			n++
		}
	}
	return n
}
