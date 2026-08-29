package goal_test

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"github.com/wow-look-at-my/agentic-loop/goal"
)

// answers replies with each text in turn, then repeats the last one.
func answers(texts ...string) goal.Judge {
	calls := 0
	return func(context.Context, string, string) (*agentic.Completion, error) {
		i := calls
		calls++
		if i >= len(texts) {
			i = len(texts) - 1
		}
		return &agentic.Completion{Message: agentic.Message{Content: texts[i]}}, nil
	}
}

func window(entries ...goal.Entry) func() ([]goal.Entry, error) {
	return func() ([]goal.Entry, error) { return entries, nil }
}

func TestParseVerdictReadsAFencedObject(t *testing.T) {
	v, err := goal.ParseVerdict("```json\n{\"met\":true,\"reason\":\"go test passed\"}\n```")
	require.NoError(t, err)
	assert.True(t, v.Met)
	assert.Equal(t, "go test passed", v.Reason)
}

func TestParseVerdictRefusesNothingProseAndAnEmptyReason(t *testing.T) {
	_, err := goal.ParseVerdict("   ")
	require.Error(t, err)

	_, err = goal.ParseVerdict("I think it passed")
	require.Error(t, err)

	_, err = goal.ParseVerdict(`{"met":false,"reason":"  "}`)
	require.Error(t, err, "a block carrying no reason is a refusal nobody can act on")
}

func TestRenderWindowSaysWhenThereIsNoTranscript(t *testing.T) {
	assert.Equal(t,
		"(no transcript yet: no work has been recorded since the goal was set)",
		goal.RenderWindow(nil, goal.EvalTokens))
}

func TestRenderWindowLabelsEachKindAndDropsEmptyText(t *testing.T) {
	out := goal.RenderWindow([]goal.Entry{
		{Kind: goal.EntryUser, Text: "fix the tests"},
		{Kind: goal.EntryAssistant, Text: "on it"},
		{Kind: goal.EntryToolCall, Text: `run_tests {"pkg":"./..."}`},
		{Kind: goal.EntryToolResult, Text: "2 failed"},
		{Kind: goal.EntrySubagent, Text: "read auth.go"},
		{Kind: goal.EntryError, Text: "exit 1"},
		{Kind: goal.EntryAssistant, Text: "   "},
	}, goal.EvalTokens)

	assert.Contains(t, out, "user: fix the tests")
	assert.Contains(t, out, "assistant: on it")
	assert.Contains(t, out, `tool call: run_tests {"pkg":"./..."}`)
	assert.Contains(t, out, "tool result: 2 failed")
	assert.Contains(t, out, "subagent: read auth.go")
	assert.Contains(t, out, "error: exit 1")
	assert.NotContains(t, out, goal.OmittedMarker)
}

func TestRenderWindowKeepsTheNewestAndSaysItTruncated(t *testing.T) {
	entries := []goal.Entry{
		{Kind: goal.EntryUser, Text: strings.Repeat("old ", 100)},
		{Kind: goal.EntryAssistant, Text: "newest"},
	}
	out := goal.RenderWindow(entries, 8)
	assert.True(t, strings.HasPrefix(out, goal.OmittedMarker+"\n"))
	assert.Contains(t, out, "assistant: newest")
	assert.NotContains(t, out, "old old")
}

func TestRenderWindowTruncatesALongToolResultInTheMiddle(t *testing.T) {
	long := strings.Repeat("x", 5000)
	out := goal.RenderWindow([]goal.Entry{{Kind: goal.EntryToolResult, Text: long}}, goal.EvalTokens)
	assert.Contains(t, out, "runes elided")
	assert.Less(t, len(out), len(long), "both ends are kept and the middle goes")
}

func TestEntriesFromMessagesMapsTheTranscript(t *testing.T) {
	entries := goal.EntriesFromMessages([]agentic.Message{
		{Role: agentic.RoleUser, Content: "fix it"},
		{Role: agentic.RoleAssistant, Content: "looking", ToolCalls: []agentic.ToolCall{{Name: "grep", Arguments: `{"q":"x"}`}}},
		{Role: agentic.RoleTool, Content: "3 hits"},
		{Role: agentic.RoleTool, Content: "boom", ToolIsError: true},
	})
	require.Len(t, entries, 5)
	assert.Equal(t, goal.Entry{Kind: goal.EntryUser, Text: "fix it"}, entries[0])
	assert.Equal(t, goal.Entry{Kind: goal.EntryAssistant, Text: "looking"}, entries[1])
	assert.Equal(t, goal.Entry{Kind: goal.EntryToolCall, Text: `grep {"q":"x"}`}, entries[2])
	assert.Equal(t, goal.Entry{Kind: goal.EntryToolResult, Text: "3 hits"}, entries[3])
	assert.Equal(t, goal.Entry{Kind: goal.EntryError, Text: "boom"}, entries[4])
}

func TestEvaluateBlocksAndCarriesTheDirective(t *testing.T) {
	state := &goal.State{Condition: "tests pass"}
	e := &goal.Evaluator{
		State:  state,
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "2 failed"}),
		Judge:  answers(`{"met":false,"impossible":false,"reason":"two of four still fail"}`),
	}
	v := e.Evaluate(context.Background())
	assert.Equal(t, goal.Blocked, v.Outcome)
	assert.Equal(t, "two of four still fail", v.Reason)
	assert.Equal(t, goal.Directive("tests pass", "two of four still fail"), v.Directive)
	assert.Equal(t, 1, state.Iterations)
	assert.Equal(t, 1, state.ReasonRun)
}

func TestEvaluateMeetsAndReportsImpossible(t *testing.T) {
	met := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
		Judge:  answers(`{"met":true,"reason":"go test reported ok"}`),
	}
	assert.Equal(t, goal.Met, met.Evaluate(context.Background()).Outcome)

	impossible := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "no tests"}),
		Judge:  answers(`{"met":false,"impossible":true,"reason":"the repo has no test suite"}`),
	}
	v := impossible.Evaluate(context.Background())
	assert.Equal(t, goal.Failed, v.Outcome)
	assert.Equal(t, "the repo has no test suite", v.Reason)
}

func TestEvaluateFailsOnTheThirdIdenticalReason(t *testing.T) {
	state := &goal.State{Condition: "tests pass"}
	e := &goal.Evaluator{
		State:  state,
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "2 failed"}),
		Judge:  answers(`{"met":false,"reason":"two of four still fail"}`),
	}
	assert.Equal(t, goal.Blocked, e.Evaluate(context.Background()).Outcome)
	assert.Equal(t, goal.Blocked, e.Evaluate(context.Background()).Outcome)

	v := e.Evaluate(context.Background())
	assert.Equal(t, goal.Failed, v.Outcome)
	assert.Contains(t, v.Reason, "the same thing 3 times running")
	assert.Contains(t, v.Reason, "two of four still fail")
}

func TestEvaluateRetriesOnceOnAnUnparseableAnswer(t *testing.T) {
	e := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
		Judge:  answers("I think so", `{"met":true,"reason":"go test reported ok"}`),
	}
	assert.Equal(t, goal.Met, e.Evaluate(context.Background()).Outcome)
}

func TestEvaluateSuspendsAfterTwoUnusableAnswers(t *testing.T) {
	state := &goal.State{Condition: "tests pass"}
	e := &goal.Evaluator{
		State:  state,
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
		Judge:  answers("I think so"),
	}
	v := e.Evaluate(context.Background())
	assert.Equal(t, goal.Suspended, v.Outcome)
	assert.Equal(t, "evaluator returned unparseable output twice", v.Reason)
	assert.True(t, state.Suspended)
	assert.Equal(t, 0, state.Iterations, "a failure to evaluate is not a verdict on the condition")

	// A suspended goal blocks nothing and makes no further call.
	e.Judge = func(context.Context, string, string) (*agentic.Completion, error) {
		t.Fatal("a suspended goal must not call the evaluator")
		return nil, nil
	}
	assert.Equal(t, goal.Permitted, e.Evaluate(context.Background()).Outcome)
}

func TestEvaluateSuspendsWhenTheCallFailsAndNamesIt(t *testing.T) {
	e := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
		Judge: func(context.Context, string, string) (*agentic.Completion, error) {
			return nil, errors.New("dial tcp: refused")
		},
	}
	v := e.Evaluate(context.Background())
	assert.Equal(t, goal.Suspended, v.Outcome)
	assert.Contains(t, v.Reason, "evaluator call failed (dial tcp: refused)")
}

func TestEvaluateSuspendsWhenTheTranscriptCannotBeRead(t *testing.T) {
	e := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: func() ([]goal.Entry, error) { return nil, errors.New("database is locked") },
		Judge:  answers(`{"met":true,"reason":"x"}`),
	}
	v := e.Evaluate(context.Background())
	assert.Equal(t, goal.Suspended, v.Outcome)
	assert.Contains(t, v.Reason, "database is locked")
}

func TestEvaluatePermitsACancelledRunWithoutACall(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	e := &goal.Evaluator{
		State:  &goal.State{Condition: "tests pass"},
		Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
		Judge: func(context.Context, string, string) (*agentic.Completion, error) {
			t.Fatal("a cancelled run is never evaluated")
			return nil, nil
		},
	}
	v := e.Evaluate(ctx)
	assert.Equal(t, goal.Permitted, v.Outcome)
	assert.Equal(t, "the run was cancelled", v.Reason)
}

func TestEvaluateWithNoStateOrNoWiringPermitsAndSaysSo(t *testing.T) {
	empty := &goal.Evaluator{}
	assert.Equal(t, goal.Permitted, empty.Evaluate(context.Background()).Outcome)

	unwired := &goal.Evaluator{State: &goal.State{Condition: "tests pass"}}
	v := unwired.Evaluate(context.Background())
	assert.Equal(t, goal.Suspended, v.Outcome)
	assert.Contains(t, v.Reason, "goal mode is not wired up")
}

func TestOutcomeStrings(t *testing.T) {
	assert.Equal(t, "permitted", goal.Permitted.String())
	assert.Equal(t, "blocked", goal.Blocked.String())
	assert.Equal(t, "met", goal.Met.String())
	assert.Equal(t, "failed", goal.Failed.String())
	assert.Equal(t, "suspended", goal.Suspended.String())
	assert.Equal(t, "outcome(?)", goal.Outcome(9).String())
}

func TestEvalUserCarriesTheConditionAndTheWindow(t *testing.T) {
	assert.Equal(t, "Condition: tests pass\n\nTranscript:\nuser: go",
		goal.EvalUser("tests pass", "user: go"))
}
