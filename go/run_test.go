package agentic

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// scriptStep is one scripted provider response.
type scriptStep struct {
	comp *Completion
	err  error
	// emit, when set, fires stream events before returning (to exercise
	// delivery tracking).
	emit func(ev *StreamEvents)
}

// scriptProvider replays scripted responses and records every request.
type scriptProvider struct {
	steps []scriptStep
	reqs  []Request
}

func (p *scriptProvider) Complete(_ context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	p.reqs = append(p.reqs, req)
	if len(p.steps) == 0 {
		return nil, errors.New("script exhausted")
	}
	step := p.steps[0]
	p.steps = p.steps[1:]
	if step.emit != nil {
		step.emit(ev)
	}
	return step.comp, step.err
}

func assistantComp(content string, calls ...ToolCall) *Completion {
	stop := StopEndTurn
	if len(calls) > 0 {
		stop = StopToolUse
	}
	return &Completion{
		Message:    Message{Role: RoleAssistant, Content: content, ToolCalls: calls},
		Usage:      Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		StopReason: stop,
	}
}

// noSleep makes retries instantaneous in tests.
var noSleep = RetryPolicy{MaxAttempts: 4, BaseDelay: time.Millisecond,
	Sleep: func(context.Context, time.Duration) error { return nil }}

func TestRunMultiTurnToolLoop(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: `{"a":1}`}, ToolCall{ID: "c2", Name: "beta", Arguments: "{}"})},
		{comp: assistantComp("all done")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha", Readonly: true}, {Name: "beta"}}}
	var calls []ToolCall
	var results []ToolResult
	cfg := Config{
		Provider: provider,
		Tools:    exec,
		Retry:    &noSleep,
		Events: Events{
			OnToolCall:   func(c ToolCall) { calls = append(calls, c) },
			OnToolResult: func(_ ToolCall, r ToolResult) { results = append(results, r) },
		},
	}
	req := Request{Model: "m", System: "sys", Messages: []Message{{Role: RoleUser, Content: "go"}},
		Tools: []Tool{{Name: "ignored"}}}

	res, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	assert.Equal(t, 2, res.Turns)
	require.Len(t, res.Usages, 2, "one usage entry per model call, in order, never summed")

	// Transcript: user, assistant(calls), tool, tool, assistant(final).
	require.Len(t, res.Messages, 5)
	assert.Equal(t, RoleUser, res.Messages[0].Role)
	assert.Equal(t, RoleAssistant, res.Messages[1].Role)
	require.Len(t, res.Messages[1].ToolCalls, 2)
	assert.Equal(t, Message{Role: RoleTool, Content: "ran alpha", ToolCallID: "c1"}, res.Messages[2])
	assert.Equal(t, Message{Role: RoleTool, Content: "ran beta", ToolCallID: "c2"}, res.Messages[3])
	assert.Equal(t, "all done", res.Messages[4].Content)
	assert.Equal(t, "all done", res.Final.Content)

	// Events fired in call order.
	require.Len(t, calls, 2)
	assert.Equal(t, "alpha", calls[0].Name)
	assert.Equal(t, "beta", calls[1].Name)
	require.Len(t, results, 2)
	assert.Equal(t, "ran alpha", results[0].Content)

	// The loop advertises the executor's tools, overriding req.Tools; the
	// second request replays the assistant tool calls and the tool results.
	require.Len(t, provider.reqs, 2)
	require.Len(t, provider.reqs[0].Tools, 2)
	assert.Equal(t, "alpha", provider.reqs[0].Tools[0].Name)
	assert.Equal(t, "sys", provider.reqs[0].System)
	second := provider.reqs[1]
	require.Len(t, second.Messages, 4)
	assert.Equal(t, RoleTool, second.Messages[2].Role)

	// Executed against the executor with full call payloads.
	require.Len(t, exec.executed, 2)
	assert.Equal(t, `{"a":1}`, exec.executed[0].Arguments)
}

func TestRunExecuteErrorBecomesTeachingResult(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "explode", Arguments: "{}"})},
		{comp: assistantComp("recovered")},
	}}
	exec := &fakeExec{
		tools:   []Tool{{Name: "explode"}},
		execute: func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, errors.New("boom") },
	}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err, "the loop never aborts on tool failure")
	toolMsg := res.Messages[1]
	assert.Equal(t, RoleTool, toolMsg.Role)
	assert.Equal(t, "tool execution failed: boom", toolMsg.Content)
	assert.True(t, toolMsg.ToolIsError)
	assert.Equal(t, "recovered", res.Final.Content)
}

func TestRunApprovalDeny(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "danger", Arguments: "{}"})},
		{comp: assistantComp("understood")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "danger"}}, ask: map[string]bool{"danger": true}}
	approver := approverFunc(func(context.Context, ToolCall) (bool, error) { return false, nil })

	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Approver: approver, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	toolMsg := res.Messages[1]
	assert.Equal(t, DeniedMessage, toolMsg.Content, "exact denial text recorded as the tool result")
	assert.True(t, toolMsg.ToolIsError)
	assert.Empty(t, exec.executed, "denied call never executes")
	assert.Equal(t, "understood", res.Final.Content, "the loop continues so the model can react")
}

type approverFunc func(ctx context.Context, call ToolCall) (bool, error)

func (f approverFunc) Ask(ctx context.Context, call ToolCall) (bool, error) { return f(ctx, call) }

func TestRunApprovalAllow(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "danger", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "danger"}}, ask: map[string]bool{"danger": true}}
	approver := approverFunc(func(context.Context, ToolCall) (bool, error) { return true, nil })
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Approver: approver, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Len(t, exec.executed, 1)
	assert.Equal(t, "ran danger", res.Messages[1].Content)
}

func TestRunNilApproverDeniesGatedCalls(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "danger", Arguments: "{}"})},
		{comp: assistantComp("ok")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "danger"}}, ask: map[string]bool{"danger": true}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, DeniedMessage, res.Messages[1].Content, "no approver means gated calls fail closed")
	assert.Empty(t, exec.executed)
}

func TestRunApprovalAskError(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{
			Role:    RoleAssistant,
			Content: "let me check",
			Thinking: []ThinkingBlock{{Text: "hmm"}},
			ToolCalls: []ToolCall{
				{ID: "c1", Name: "safe", Arguments: "{}"},
				{ID: "c2", Name: "danger", Arguments: "{}"},
			},
		}, StopReason: StopToolUse}},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "safe"}, {Name: "danger"}}, ask: map[string]bool{"danger": true}}
	interrupted := errors.New("stream closed")
	approver := approverFunc(func(context.Context, ToolCall) (bool, error) { return false, interrupted })

	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Approver: approver, Retry: &noSleep},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, interrupted)
	require.NotNil(t, res, "partial Result returned alongside the error")

	// The first (ungated) call executed and its result was appended, then the
	// gated call's Ask failed: the batch is cleared — the assistant message
	// keeps content and reasoning but loses its tool calls, and the executed
	// result is dropped from the transcript so no orphans remain.
	require.Len(t, res.Messages, 2)
	final := res.Messages[1]
	assert.Equal(t, "let me check", final.Content)
	assert.Equal(t, "hmm", final.Thinking[0].Text)
	assert.Nil(t, final.ToolCalls, "tool calls cleared")
	assert.Equal(t, final, res.Final)
	assert.Len(t, exec.executed, 1, "the first call did run before the interruption")
	assert.Equal(t, 1, res.Turns)
}

func TestRunFinalTurnWithholdsTools(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("forced answer")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, MaxTurns: 2, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	require.Len(t, provider.reqs, 2)
	assert.NotEmpty(t, provider.reqs[0].Tools)
	assert.Empty(t, provider.reqs[1].Tools, "tools withheld on the final permitted turn")
	assert.Equal(t, "forced answer", res.Final.Content)
	assert.Equal(t, 2, res.Turns)
}

func TestRunStallFallbackSynthesizes(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "only thoughts"}}}, StopReason: StopEndTurn}},
		{comp: assistantComp("synthesized report")},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Retry: &noSleep},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "task"}}})
	require.NoError(t, err)

	require.Len(t, provider.reqs, 2)
	wrapReq := provider.reqs[1]
	assert.Empty(t, wrapReq.Tools, "the wrap-up turn is tool-less")
	last := wrapReq.Messages[len(wrapReq.Messages)-1]
	assert.Equal(t, RoleUser, last.Role)
	assert.Equal(t, wrapUpInstruction, last.Content)
	// The stalled assistant message is NOT in the wrap-up request.
	require.Len(t, wrapReq.Messages, 2)

	assert.Equal(t, "synthesized report", res.Final.Content)
	require.Len(t, res.Messages, 3, "user, wrap-up instruction, final answer")
	assert.Equal(t, wrapUpInstruction, res.Messages[1].Content)
	assert.Equal(t, 2, res.Turns)
	assert.Len(t, res.Usages, 2)
}

func TestRunStallFallbackToReasoning(t *testing.T) {
	stalled := &Completion{Message: Message{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "the reasoning"}}}, StopReason: StopEndTurn}
	emptyAgain := &Completion{Message: Message{Role: RoleAssistant}, StopReason: StopEndTurn}
	provider := &scriptProvider{steps: []scriptStep{{comp: stalled}, {comp: emptyAgain}}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "the reasoning", res.Final.Content, "reasoning is the fallback answer")
}

func TestRunStallPlaceholderWithoutTools(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant}, StopReason: StopEndTurn}},
	}}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	require.Len(t, provider.reqs, 1, "no wrap-up turn without an executor")
	assert.Equal(t, noOutputPlaceholder, res.Final.Content)
}

func TestRunHallucinatedCallWithoutExecutor(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "ghost", Arguments: "{}"})},
		{comp: assistantComp("sorry")},
	}}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	require.Len(t, provider.reqs, 2)
	assert.Empty(t, provider.reqs[0].Tools, "nil Tools advertises nothing")
	toolMsg := res.Messages[1]
	assert.Equal(t, "unknown tool: ghost", toolMsg.Content, "a hallucinated call gets a teaching error")
	assert.True(t, toolMsg.ToolIsError)
	assert.Equal(t, "sorry", res.Final.Content)
}

func TestRunMaxTurnsDefault(t *testing.T) {
	// A model that always requests tools: DefaultMaxTurns caps the loop, the
	// last turn withholds tools, and the scripted final text ends it.
	steps := make([]scriptStep, 0, DefaultMaxTurns)
	for i := 0; i < DefaultMaxTurns-1; i++ {
		steps = append(steps, scriptStep{comp: assistantComp("", ToolCall{ID: "c", Name: "alpha", Arguments: "{}"})})
	}
	steps = append(steps, scriptStep{comp: assistantComp("capped")})
	provider := &scriptProvider{steps: steps}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, DefaultMaxTurns, res.Turns)
	assert.Empty(t, provider.reqs[DefaultMaxTurns-1].Tools)
	assert.Equal(t, "capped", res.Final.Content)
	assert.Len(t, res.Usages, DefaultMaxTurns)
}

func TestRunLastTurnDanglingCallsCleared(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("answer with dangling call", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	res, err := Run(context.Background(), Config{Provider: provider, Tools: exec, MaxTurns: 1, Retry: &noSleep}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Nil(t, res.Final.ToolCalls, "never-executed calls do not survive into the final transcript")
	assert.Empty(t, exec.executed)
}

func TestRunRetriesTransientModelFailure(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{err: &APIError{Status: 503, Body: "unavailable"}},
		{comp: assistantComp("after retry")},
	}}
	slept := 0
	retry := RetryPolicy{MaxAttempts: 3, Sleep: func(context.Context, time.Duration) error { slept++; return nil }}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &retry}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "after retry", res.Final.Content)
	assert.Equal(t, 1, slept)
	assert.Equal(t, 1, res.Turns, "a retried call counts as one turn")
	assert.Len(t, res.Usages, 1)
}

func TestRunNoRetryAfterPartialStream(t *testing.T) {
	partial := &Completion{Message: Message{Role: RoleAssistant, Content: "part", ToolCalls: []ToolCall{{ID: "c", Name: "x"}}},
		Usage: Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4}}
	netErr := errors.New("connection reset mid-stream")
	provider := &scriptProvider{steps: []scriptStep{
		{comp: partial, err: netErr, emit: func(ev *StreamEvents) { ev.emitText("part") }},
	}}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &noSleep},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, netErr)
	require.NotNil(t, res)
	require.Len(t, provider.reqs, 1, "no re-attempt once data streamed")
	require.Len(t, res.Messages, 2)
	assert.Equal(t, "part", res.Messages[1].Content)
	assert.Nil(t, res.Messages[1].ToolCalls, "assembled calls dropped on a broken turn")
	assert.Equal(t, "part", res.Final.Content)
	require.Len(t, res.Usages, 1, "the partial call's usage snapshot is kept")
	assert.Equal(t, 4, res.Usages[0].TotalTokens)
}

func TestRunNoRetryWhenEventsDeliveredWithoutCompletion(t *testing.T) {
	netErr := errors.New("reset")
	provider := &scriptProvider{steps: []scriptStep{
		{err: netErr, emit: func(ev *StreamEvents) { ev.emitText("leaked") }},
	}}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &noSleep}, Request{Model: "m"})
	require.Error(t, err)
	require.NotNil(t, res)
	assert.Len(t, provider.reqs, 1, "delivered events block the retry even without a partial completion")
}

func TestRunPermanentFailureSurfaces(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{err: &APIError{Status: 400, Body: "prompt is too long", ContextOverflow: true}},
	}}
	res, err := Run(context.Background(), Config{Provider: provider, Retry: &noSleep},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.True(t, IsContextOverflow(err))
	require.NotNil(t, res)
	assert.Len(t, provider.reqs, 1, "permanent errors are not retried")
	assert.Len(t, res.Messages, 1, "transcript unchanged")
}

func TestRunRequiresProvider(t *testing.T) {
	res, err := Run(context.Background(), Config{}, Request{Model: "m"})
	assert.Nil(t, res)
	require.Error(t, err)
}

func TestRunStreamEventsForwarded(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("hi"), emit: func(ev *StreamEvents) {
			ev.emitText("hi")
			ev.emitReasoning("hmm")
			ev.emitUsage(Usage{PromptTokens: 1})
			ev.emitProgress(PromptProgress{Processed: 1, Total: 2})
		}},
	}}
	var text, reasoning string
	var gotUsage, gotProgress bool
	cfg := Config{Provider: provider, Retry: &noSleep, Events: Events{StreamEvents: StreamEvents{
		OnText:      func(s string) { text += s },
		OnReasoning: func(s string) { reasoning += s },
		OnUsage:     func(Usage) { gotUsage = true },
		OnProgress:  func(PromptProgress) { gotProgress = true },
	}}}
	_, err := Run(context.Background(), cfg, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "hi", text)
	assert.Equal(t, "hmm", reasoning)
	assert.True(t, gotUsage)
	assert.True(t, gotProgress)
}
