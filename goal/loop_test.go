package goal_test

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"github.com/wow-look-at-my/agentic-loop/client"
	"github.com/wow-look-at-my/agentic-loop/goal"
)

// replies answers each call with the next text, recording what it was sent.
type replies struct {
	texts []string
	calls int
	seen  []agentic.Request
}

func (r *replies) Complete(_ context.Context, req agentic.Request, _ *agentic.StreamEvents) (*agentic.Completion, error) {
	r.seen = append(r.seen, req)
	i := r.calls
	r.calls++
	if i >= len(r.texts) {
		i = len(r.texts) - 1
	}
	return &agentic.Completion{Message: agentic.Message{Role: agentic.RoleAssistant, Content: r.texts[i]}}, nil
}

func TestStopListenerBlocksThenLetsTheRunEnd(t *testing.T) {
	provider := &replies{texts: []string{"I think I am done", "now the tests pass"}}
	state := &goal.State{Condition: "tests pass"}
	queue := &agentic.MessageQueue{}
	var reported []goal.Outcome

	listener := &goal.StopListener{
		Evaluator: &goal.Evaluator{
			State:  state,
			Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "2 failed"}),
			Judge: answers(
				`{"met":false,"reason":"two of four still fail"}`,
				`{"met":true,"reason":"go test reported ok"}`,
			),
		},
		Report: func(v goal.Verdicts) { reported = append(reported, v.Outcome) },
	}
	events := agentic.Events{}
	listener.Attach(context.Background(), &events, queue)

	res, err := agentic.Run(context.Background(), agentic.Config{
		Provider: provider,
		Events:   &events,
		Messages: queue,
	}, agentic.Request{Model: "m", Messages: []agentic.Message{{Role: agentic.RoleUser, Content: "go"}}})
	require.NoError(t, err)

	assert.Equal(t, []goal.Outcome{goal.Blocked, goal.Met}, reported)
	assert.Equal(t, "now the tests pass", res.Final.Content)
	assert.Equal(t, 2, provider.calls, "a blocked stop takes the loop round again")

	// The directive reached the model as a message of its own kind.
	var directive *agentic.Message
	for i := range res.Messages {
		if res.Messages[i].Kind == goal.DirectiveKind {
			directive = &res.Messages[i]
		}
	}
	require.NotNil(t, directive, "the blocked stop's directive is in the transcript")
	assert.Equal(t, agentic.RoleUser, directive.Role)
	assert.Contains(t, directive.Content, "[goal not met] two of four still fail")
	assert.Equal(t, 2, state.Iterations)
}

func TestStopListenerPermittedOutcomesQueueNothing(t *testing.T) {
	provider := &replies{texts: []string{"done"}}
	queue := &agentic.MessageQueue{}
	listener := &goal.StopListener{
		Evaluator: &goal.Evaluator{
			State:  &goal.State{Condition: "tests pass"},
			Window: window(goal.Entry{Kind: goal.EntryToolResult, Text: "ok"}),
			Judge:  answers("not json at all"),
		},
	}
	events := agentic.Events{}
	listener.Attach(context.Background(), &events, queue)

	res, err := agentic.Run(context.Background(), agentic.Config{
		Provider: provider,
		Events:   &events,
		Messages: queue,
	}, agentic.Request{Model: "m", Messages: []agentic.Message{{Role: agentic.RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, 1, provider.calls, "a suspended goal fails open: the run ends")
	assert.Equal(t, "done", res.Final.Content)
}

func TestStopListenerWithNoEvaluatorDoesNothing(t *testing.T) {
	provider := &replies{texts: []string{"done"}}
	queue := &agentic.MessageQueue{}
	events := agentic.Events{}
	(&goal.StopListener{}).Attach(context.Background(), &events, queue)

	_, err := agentic.Run(context.Background(), agentic.Config{
		Provider: provider,
		Events:   &events,
		Messages: queue,
	}, agentic.Request{Model: "m", Messages: []agentic.Message{{Role: agentic.RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, 1, provider.calls)
}

func TestOneShotJudgeAsksWithTheEvaluatorsOwnPrompt(t *testing.T) {
	provider := &replies{texts: []string{`{"met":true,"reason":"ok"}`}}
	judge := goal.OneShotJudge(provider, agentic.Request{
		Model:       "m",
		System:      "you are a coding agent",
		SystemParts: []client.Part{client.TextPart{Text: "you are a coding agent"}},
		Messages:    []agentic.Message{{Role: agentic.RoleUser, Content: "the whole conversation"}},
	})

	comp, err := judge(context.Background(), goal.EvalSystem, goal.EvalUser("tests pass", "user: go"))
	require.NoError(t, err)
	assert.Equal(t, `{"met":true,"reason":"ok"}`, comp.Message.Content)

	require.Len(t, provider.seen, 1)
	sent := provider.seen[0]
	assert.Equal(t, goal.EvalSystem, sent.System)
	assert.Nil(t, sent.SystemParts, "the host's own system prompt would otherwise outrank the evaluator's")
	assert.Equal(t, goal.EvalMaxTokens, sent.MaxTokens)
	assert.Nil(t, sent.Tools, "the evaluator judges; it never calls a tool")
	require.Len(t, sent.Messages, 1, "the window is the whole of what it reads")
	assert.Equal(t, goal.EvalUser("tests pass", "user: go"), sent.Messages[0].Content)
}
