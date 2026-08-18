package agentic

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunOnStopInjectsOnceAndContinues(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("first")},
		{comp: assistantComp("second")},
	}}
	calls := 0
	cfg := Config{
		Provider: provider,
		Events: Events{OnStop: func(turn int, comp *Completion) (Message, bool) {
			calls++
			assert.Equal(t, 1, turn)
			assert.Equal(t, "first", comp.Message.Content)
			return Message{Role: RoleUser, Content: "push staged work"}, true
		}},
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "second", res.Final.Content)
	assert.Equal(t, 2, res.Turns)
	assert.Equal(t, 1, calls)
	require.Len(t, res.Messages, 4)
	assert.Equal(t, "first", res.Messages[1].Content)
	assert.Equal(t, "push staged work", res.Messages[2].Content)
	assert.Equal(t, "second", res.Messages[3].Content)
}

func TestRunOnStopAbsentFinishesNormally(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}
	res, err := Run(context.Background(), Config{Provider: provider}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Final.Content)
	assert.Equal(t, 1, res.Turns)
}
