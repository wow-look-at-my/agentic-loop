package loop

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errSink is the sentinel a failing callback returns; abort paths keep it reachable via errors.Is.
var errSink = errors.New("sink closed")

func TestRunCallbackErrorSingleRequestAndPartialResult(t *testing.T) {
	// Delivery is marked before the callback runs, so a first-delta failure still blocks a re-attempt.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"part","tool_calls":[{"index":0,"id":"c1","function":{"name":"x","arguments":"{}"}}]}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"never"}}]}` + "\n\ndata: [DONE]\n\n"))
		fl.Flush()
	}))
	defer srv.Close()

	cfg := Config{
		Provider: oaProvider(t, srv.URL),
		Events: &Events{StreamEvents: StreamEvents{
			OnText: func(string) error { return errSink },
		}},
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	assert.Equal(t, int32(1), hits.Load(), "a failing first callback still blocks the re-attempt")
	require.NotNil(t, res, "partial Result returned alongside the error")
	require.Len(t, res.Messages, 2)
	assert.Equal(t, "part", res.Messages[1].Content)
	assert.Nil(t, res.Messages[1].ToolCalls, "never-executed calls are cleared from the finalized turn")
	assert.Equal(t, 1, res.Turns)
}

func TestRunOnToolCallErrorAbortsBatch(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{
			Role:     RoleAssistant,
			Content:  "checking",
			Thinking: []ThinkingBlock{{Text: "hmm"}},
			ToolCalls: []ToolCall{
				{ID: "c1", Name: "alpha", Arguments: "{}"},
				{ID: "c2", Name: "beta", Arguments: "{}"},
			},
		}, StopReason: StopToolUse}},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}, {Name: "beta"}}}
	fails := 0
	events := Events{}
	toolCallCb := func(ev ToolCallEvent) error {
		c := ev.Call
		if c.Name == "beta" {
			fails++
			return errSink
		}
		return nil
	}
	events.OnToolCall.Subscribe(&toolCallCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	assert.False(t, IsTransient(err))
	require.NotNil(t, res)
	assert.Equal(t, 1, fails)

	// alpha ran before the abort, but the batch is cleared: no orphan tool calls remain.
	assert.Len(t, exec.executed, 1)
	require.Len(t, res.Messages, 2)
	final := res.Messages[1]
	assert.Equal(t, "checking", final.Content)
	assert.Equal(t, "hmm", final.Thinking[0].Text)
	assert.Nil(t, final.ToolCalls)
	assert.Equal(t, final, res.Final)
}

func TestRunOnToolResultErrorAbortsBatch(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	events := Events{}
	toolResultCb := func(ev ToolResultEvent) error { return errSink }
	events.OnToolResult.Subscribe(&toolResultCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	require.NotNil(t, res)
	assert.Len(t, exec.executed, 1, "the tool DID run; only its delivery failed")
	require.Len(t, res.Messages, 2, "the executed result is dropped so the transcript stays replayable")
	assert.Nil(t, res.Messages[1].ToolCalls)
	assert.Equal(t, RoleAssistant, res.Messages[1].Role)
}

// A hook rewrites a call's arguments: the rewrite is what the Approver judges
// and what the tool executes, while the transcript keeps recording what the
// MODEL asked for. Both facts matter, so neither is overwritten by the other.
func TestRunOnToolCallRewritesWhatExecutes(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "bash", Arguments: `{"cmd":"rm -rf /"}`})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "bash"}}}
	var judged []string
	approver := approverFunc(func(_ context.Context, c ToolCall) (Approval, error) {
		judged = append(judged, c.Arguments)
		return Approval{OK: true}, nil
	})
	events := Events{}
	toolCallCb := func(ev ToolCallEvent) error {
		c := ev.Call
		c.Arguments = `{"cmd":"ls"}`
		return nil
	}
	events.OnToolCall.Subscribe(&toolCallCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: approver, Events: &events}

	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	require.Len(t, exec.executed, 1)
	assert.Equal(t, `{"cmd":"ls"}`, exec.executed[0].Arguments, "the tool receives the rewritten bytes")
	assert.Equal(t, []string{`{"cmd":"ls"}`}, judged,
		"the approver decides on what will actually run, not on what the model asked for")

	assistant := res.Messages[1]
	require.Len(t, assistant.ToolCalls, 1)
	assert.Equal(t, `{"cmd":"rm -rf /"}`, assistant.ToolCalls[0].Arguments,
		"the transcript records the model's own request; a mutation never rewrites history")
	assert.Equal(t, "c1", res.Messages[2].ToolCallID, "the result still answers the id the model minted")
}

// A rewritten id must never become the id the tool result answers: the
// transcript's assistant message carries the model's, and a mismatch is an
// orphan tool call no upstream will replay.
func TestRunOnToolCallCannotOrphanTheResult(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	toolCallCb := func(ev ToolCallEvent) error { c := ev.Call; c.ID = "hijacked"; return nil }
	events.OnToolCall.Subscribe(&toolCallCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Events: &events}

	res, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "c1", res.Messages[0].ToolCalls[0].ID)
	assert.Equal(t, "c1", res.Messages[1].ToolCallID)
}

// The tool returned one thing and the transcript recorded another: dedup
// replaced the repeat with a marker. OnToolResult reports both, so a host that
// persists the transcript stores what the model actually saw instead of
// re-deriving it by diffing Result.Messages afterwards.
func TestRunOnToolResultCarriesTheRecordedMessage(t *testing.T) {
	const fullOutput = "the huge status diff"
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "status", Arguments: "{}"})},
		{comp: assistantComp("", ToolCall{ID: "c2", Name: "status", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := identicalExec([]ToolDecl{{Name: "status", Readonly: true}}, fullOutput)
	var results []ToolResult
	var recorded []Message
	events := Events{}
	toolResultCb := func(ev ToolResultEvent) error {
		r, m := ev.Result, ev.Recorded
		results = append(results, r)
		recorded = append(recorded, m)
		return nil
	}
	events.OnToolResult.Subscribe(&toolResultCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Events: &events}

	res, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.NoError(t, err)

	require.Len(t, results, 2)
	require.Len(t, recorded, 2)
	assert.Equal(t, fullOutput, results[0].Content)
	assert.Equal(t, fullOutput, recorded[0].Content, "the first occurrence is recorded whole")

	assert.Equal(t, fullOutput, results[1].Content, "the tool's own result is reported unchanged")
	assert.Contains(t, recorded[1].Content, UnchangedPrefix, "what the model saw was the marker")
	assert.Equal(t, "c2", recorded[1].ToolCallID)
	assert.False(t, recorded[1].ToolIsError)

	// And it is the transcript's own entry, not an approximation of it.
	msgs := toolMessages(t, res)
	require.Len(t, msgs, 2)
	assert.Equal(t, msgs[0], recorded[0])
	assert.Equal(t, msgs[1], recorded[1])
}

// A refused call reports the denial as the recorded message too: a host
// persisting the transcript needs the refusal in it, not a gap.
func TestRunOnToolResultCarriesADeniedMessage(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "danger", Arguments: "{}"})},
		{comp: assistantComp("ok")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "danger"}}}
	var recorded []Message
	events := Events{}
	toolResultCb := func(ev ToolResultEvent) error {
		m := ev.Recorded
		recorded = append(recorded, m)
		return nil
	}
	events.OnToolResult.Subscribe(&toolResultCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Events: &events}

	_, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.NoError(t, err)
	require.Len(t, recorded, 1)
	assert.Equal(t, DeniedMessage, recorded[0].Content)
	assert.True(t, recorded[0].ToolIsError)
}

func TestWrapCallbackErrIdempotentAndTransparent(t *testing.T) {
	assert.NoError(t, wrapCallbackErr(nil))
	w := wrapCallbackErr(errSink)
	assert.ErrorIs(t, w, errSink)
	assert.Equal(t, errSink.Error(), w.Error(), "the marker adds no text of its own")
	assert.Same(t, w, wrapCallbackErr(w), "already-marked errors are not re-wrapped")
	assert.False(t, IsTransient(w))
}
