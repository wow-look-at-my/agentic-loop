package loop

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

func waitTurn(t *testing.T, turn ITurn) *Result {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	require.NoError(t, turn.Wait(ctx))
	return turn.Result()
}

func TestSessionKeepsOneConversationAcrossSends(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("first answer")},
		{comp: assistantComp("second answer")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	var added []IMessage
	events := &Events{}
	keep(t, &events.OnNode, func(ev NodeEvent) error {
		if ev.Added {
			added = append(added, ev.Node)
		}
		return nil
	})
	s, err := NewSession(SessionConfig{
		Loop:    Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: events},
		Request: Request{Model: "m"},
	})
	require.NoError(t, err)

	res := waitTurn(t, s.Send(context.Background(), Message{Role: RoleUser, Content: "one"}))
	assert.Equal(t, "first answer", res.Final.Content)
	res = waitTurn(t, s.Send(context.Background(), Message{Role: RoleUser, Content: "two"}))
	assert.Equal(t, "second answer", res.Final.Content)

	require.Len(t, provider.reqs, 3)
	assert.Len(t, provider.reqs[2].Messages, 5, "the second Send carries the first turn: user, call, result, answer, user")
	msgs := s.Messages()
	require.Len(t, msgs, 6)
	assert.Equal(t, "second answer", msgs[5].Content)
	assert.Len(t, s.Path(), len(added), "every node was announced as it joined")
	st := s.State()
	assert.Equal(t, 3, st.Turns)
	assert.Len(t, st.Usages, 3)
	assert.Equal(t, len(s.Path()), len(s.Nodes()))
	assert.Equal(t, s.Leaf(), s.Node(s.Leaf().Base().ID()))
	assert.Equal(t, "m", s.Config().Request.Model)
}

func TestSessionBranchesAndRefusesWhatItCannotDo(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("a")},
		{comp: assistantComp("b")},
	}}
	s, err := NewSession(SessionConfig{Loop: Config{Provider: provider}, Request: Request{Model: "m"}})
	require.NoError(t, err)
	waitTurn(t, s.Send(context.Background(), Message{Role: RoleUser, Content: "q"}))
	first := s.Path()
	require.Len(t, first, 2)

	// A regenerate is a sibling of the answer, under the same user turn.
	require.NoError(t, s.SetLeaf(first[0]))
	waitTurn(t, s.Send(context.Background()))
	second := s.Path()
	require.Len(t, second, 2)
	assert.Same(t, first[0], second[0])
	assert.NotSame(t, first[1], second[1])
	assert.Len(t, first[0].Base().Children(), 2)
	assert.Len(t, s.Nodes(), 3, "nothing left the tree")

	edit := commonai.NewUserText("edited")
	require.NoError(t, s.Append(edit, nil))
	assert.ErrorIs(t, s.SetLeaf(commonai.NewUserText("stranger")), commonai.ErrNotInTree)

	assert.True(t, s.Queue(UserMessage{Message{Role: RoleUser, Content: "later"}}))
	_, err = s.Close(context.Background()).Wait(context.Background())
	require.NoError(t, err)
	assert.ErrorIs(t, s.Send(context.Background()).Err(), ErrSessionClosed)
	assert.False(t, s.Queue(UserMessage{Message{Role: RoleUser, Content: "too late"}}))

	_, err = NewSession(SessionConfig{})
	assert.Error(t, err)
}

func TestSessionRefusesASecondSendWhileATurnRuns(t *testing.T) {
	release := make(chan struct{})
	provider := &gatedProvider{release: release, started: make(chan struct{})}
	s, err := NewSession(SessionConfig{Loop: Config{Provider: provider}, Request: Request{Model: "m"}})
	require.NoError(t, err)
	first := s.Send(context.Background(), Message{Role: RoleUser, Content: "q"})
	<-provider.started
	assert.ErrorIs(t, s.Send(context.Background()).Err(), ErrSessionBusy)
	assert.True(t, s.Queue(UserMessage{Message{Role: RoleUser, Content: "while running"}}))
	_, _, ok := first.(turnHandle).f.Result()
	assert.False(t, ok, "the first turn is still running")
	close(release)
	res := waitTurn(t, first)
	assert.Equal(t, "late", res.Messages[len(res.Messages)-1].Content)
	assert.Len(t, provider.reqs, 2, "the queued message took a second call")
}

// gatedProvider holds its first call until release closes, then answers every call.
type gatedProvider struct {
	release chan struct{}
	started chan struct{}
	reqs    []Request
}

func (p *gatedProvider) Complete(_ context.Context, req Request, _ *StreamEvents) (*Completion, error) {
	p.reqs = append(p.reqs, req)
	if len(p.reqs) == 1 {
		close(p.started)
		<-p.release
		return assistantComp("early"), nil
	}
	return assistantComp("late"), nil
}

func TestFutureSettlesOnce(t *testing.T) {
	f := newFuture[int](nil)
	_, _, ok := f.Result()
	assert.False(t, ok)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := f.Wait(ctx)
	assert.ErrorIs(t, err, context.Canceled)
	f.settle(1, nil)
	f.settle(2, nil)
	v, err := f.Wait(context.Background())
	require.NoError(t, err)
	assert.Equal(t, 1, v)
	f.Cancel()
}
