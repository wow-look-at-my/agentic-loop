package loop

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Part A extension 1: public per-turn hooks on Events
// (OnTurnBegin / OnTurnEnd; the internal turnHook stays untouched)
// ---------------------------------------------------------------------------

func TestOnTurnBeginNumberedTurnsAndReqMutation(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	var begins []int
	events := Events{}
	// Wind-down injection: append a notice to THIS call's request only,
	// on a fresh copy (the TS `[...messages, notice]` shape) so the
	// stored transcript is never aliased.
	turnBeginCb := func(ev TurnBeginEvent) error {
		turn, req := ev.Turn, ev.Req
		begins = append(begins, turn)
		msg := Message{Role: RoleUser, Content: fmt.Sprintf("notice-%d", turn)}
		req.Messages = append(append([]Message{}, req.Messages...), msg)
		return nil
	}
	events.OnTurnBegin.Subscribe(&turnBeginCb)
	cfg := Config{
		Provider: provider,
		Tools:    exec.registry(),
		Approver: allowAll,
		Events:   &events,
	}
	res, err := Run(context.Background(), cfg, Request{
		Model:    "m",
		System:   "sys",
		Messages: []Message{{Role: RoleUser, Content: "go"}},
	})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Final.Content)

	// Numbered 1..2, in order.
	assert.Equal(t, []int{1, 2}, begins)
	// The mutation reached the provider's per-call request -- and only that call.
	require.Len(t, provider.reqs, 2)
	last := provider.reqs[0].Messages[len(provider.reqs[0].Messages)-1]
	assert.Equal(t, "notice-1", last.Content, "the injected notice rode this call's request")
	assert.Equal(t, RoleUser, provider.reqs[0].Messages[len(provider.reqs[0].Messages)-1].Role)
	assert.Equal(t, "notice-2", provider.reqs[1].Messages[len(provider.reqs[1].Messages)-1].Content)
	// The persistent transcript never carried the notices.
	require.Len(t, res.Messages, 4)
	assert.Equal(t, "go", res.Messages[0].Content)
	assert.NotContains(t, fmt.Sprint(res.Messages), "notice-")
}

func TestOnTurnEndReceivesCompletionAndError(t *testing.T) {
	modelErr := errors.New("model failed")
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("working", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{err: modelErr},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	var turns []int
	var comps []*Completion
	var errs []error
	events := Events{}
	turnEndCb := func(ev TurnEndEvent) error {
		turn, comp, err := ev.Turn, ev.Comp, ev.Err
		turns = append(turns, turn)
		comps = append(comps, comp)
		errs = append(errs, err)
		return nil
	}
	events.OnTurnEnd.Subscribe(&turnEndCb)
	cfg := Config{
		Provider: provider,
		Tools:    exec.registry(),
		Approver: allowAll,
		Events:   &events,
	}
	req := Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}}
	res, err := Run(context.Background(), cfg, req)
	require.Error(t, err)

	// First call succeeded (comp non-nil, err nil); the second failed with the
	// model's error (comp nil -- nothing was produced).
	assert.Equal(t, []int{1, 2}, turns)
	require.NotNil(t, comps[0])
	assert.NoError(t, errs[0])
	assert.Equal(t, "working", comps[0].Message.Content)
	assert.Nil(t, comps[1], "comp may be nil when err != nil")
	assert.ErrorIs(t, errs[1], modelErr)
	assert.ErrorIs(t, err, modelErr)
	require.NotNil(t, res)
}

// A call that fails before producing any completion must still finalize as
// "cancelled" when its error is a context cancellation, matching the
// classification the mid-stream partial-completion path already applies.
// Before the fix, a nil completion always finalized "error" regardless of
// cause, so an outbound call torn down before it streamed a single byte
// (e.g. "openai: Post ...: context canceled") persisted as a permanent
// failure instead of the graceful cancellation it actually was.
func TestOnFinalizeAssistantClassifiesNilCompletionCancellation(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want string
	}{
		{"context canceled", fmt.Errorf("post: %w", context.Canceled), "cancelled"},
		{"deadline exceeded", fmt.Errorf("post: %w", context.DeadlineExceeded), "cancelled"},
		{"plain error", errors.New("boom"), "error"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			provider := &scriptProvider{steps: []scriptStep{{err: tc.err}}}
			var statuses []string
			events := Events{}
			finalizeCb := func(ev FinalizeAssistantEvent) error {
				statuses = append(statuses, ev.Status)
				return nil
			}
			events.OnFinalizeAssistant.Subscribe(&finalizeCb)
			cfg := Config{Provider: provider, Events: &events}
			_, err := Run(context.Background(), cfg, Request{
				Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}},
			})
			require.Error(t, err)
			require.Len(t, statuses, 1)
			assert.Equal(t, tc.want, statuses[0])
		})
	}
}

func TestOnTurnBeginErrorAbortsBeforeTheCall(t *testing.T) {
	sentinel := errors.New("begin abort")
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("never")},
	}}
	events := Events{}
	turnBeginCb := func(ev TurnBeginEvent) error { return sentinel }
	events.OnTurnBegin.Subscribe(&turnBeginCb)
	cfg := Config{
		Provider: provider,
		Events:   &events,
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel, "the callback sentinel is reachable via errors.Is")
	assert.False(t, IsTransient(err), "a callback error is never transient")
	require.NotNil(t, res)
	assert.Equal(t, 0, res.Turns, "the aborted call never counted a turn")
	assert.Empty(t, provider.reqs, "the provider was never called")
}

func TestOnTurnEndErrorAbortsAfterTheCall(t *testing.T) {
	sentinel := errors.New("end abort")
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("partial answer")},
	}}
	events := Events{}
	turnEndCb := func(ev TurnEndEvent) error { return sentinel }
	events.OnTurnEnd.Subscribe(&turnEndCb)
	cfg := Config{
		Provider: provider,
		Events:   &events,
	}
	res, err := Run(context.Background(), cfg, Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}},
	})
	require.Error(t, err)
	assert.ErrorIs(t, err, sentinel)
	require.NotNil(t, res)
	assert.Equal(t, 1, res.Turns, "the call happened before the sink failed")
	require.Len(t, res.Messages, 2)
	assert.Equal(t, "partial answer", res.Messages[1].Content,
		"the completed data is kept, like a mid-stream break")
}

func TestWrapUpFiresAsOnePastTheStalledTurn(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "only thoughts"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("synthesized report")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	var begins, ends []int
	events := Events{}
	turnBeginCb := func(ev TurnBeginEvent) error { begins = append(begins, ev.Turn); return nil }
	turnEndCb := func(ev TurnEndEvent) error { ends = append(ends, ev.Turn); return nil }
	events.OnTurnBegin.Subscribe(&turnBeginCb)
	events.OnTurnEnd.Subscribe(&turnEndCb)
	cfg := Config{
		Provider: provider,
		Tools:    exec.registry(),
		Approver: allowAll,
		Events:   &events,
	}
	res, err := Run(context.Background(), cfg, Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "task"}},
	})
	require.NoError(t, err)
	// Turn 1 stalled, so the wrap-up is turn 2 -- the call it actually is. It
	// used to be numbered maxTurns+1, naming a turn that never ran.
	assert.Equal(t, []int{1, 2}, begins)
	assert.Equal(t, []int{1, 2}, ends)
	assert.Equal(t, "synthesized report", res.Final.Content)
	assert.Equal(t, 2, res.Turns)
}

func TestInternalTurnHookUntouchedByPublicHooks(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	var internal, begins []int
	events := Events{}
	turnBeginCb := func(ev TurnBeginEvent) error { begins = append(begins, ev.Turn); return nil }
	events.OnTurnBegin.Subscribe(&turnBeginCb)
	cfg := Config{
		Provider: provider,
		Tools:    exec.registry(),
		Approver: allowAll,
		Events:   &events,
	}
	cfg.TurnHook = func(turn int) { internal = append(internal, turn) }
	res, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Final.Content)
	// Both fire once per numbered turn, in order -- the subagent telemetry seam
	// (turnHook) is byte-for-byte unaffected by the new public hooks.
	assert.Equal(t, []int{1, 2}, internal)
	assert.Equal(t, []int{1, 2}, begins)
}
