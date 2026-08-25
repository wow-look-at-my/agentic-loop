package loop

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRunOnStopInjectsAndContinues(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("first")},
		{comp: assistantComp("second")},
	}}
	sys := &MessageQueue{}
	events := Events{}
	// A host re-arms while its policy says to; the loop asks every time, the host decides when to stop.
	asked := 0
	stopCb := func(ev StopEvent) error {
		asked++
		if asked == 1 {
			sys.Queue(Message{Role: RoleUser, Content: "push staged work"})
		}
		return nil
	}
	events.OnStop.Subscribe(&stopCb)
	cfg := Config{
		Provider:       provider,
		Events:         &events,
		SystemMessages: sys,
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, "second", res.Final.Content)
	assert.Equal(t, 2, res.Turns)
	assert.Equal(t, 2, asked, "the hook is asked at every stop, not once per run")
	require.Len(t, res.Messages, 4)
	assert.Equal(t, "first", res.Messages[1].Content)
	assert.Equal(t, "push staged work", res.Messages[2].Content)
	assert.Equal(t, "second", res.Messages[3].Content)
}

// A host that re-arms on every verdict keeps being asked. Nothing in the loop
// counts the asks: a policy that judges each answer needs every boundary, and
// what ends a run that will not settle is the host's own guard.
func TestRunOnStopIsAskedAtEveryBoundary(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("first")},
		{comp: assistantComp("second")},
		{comp: assistantComp("third")},
	}}
	sys := &MessageQueue{}
	events := Events{}
	asked := 0
	stopCb := func(ev StopEvent) error {
		asked++
		if asked < 3 {
			sys.Queue(Message{Role: RoleUser, Content: "not done yet"})
		}
		return nil
	}
	events.OnStop.Subscribe(&stopCb)
	cfg := Config{Provider: provider, Events: &events, SystemMessages: sys}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	assert.Equal(t, 3, asked)
	assert.Equal(t, "third", res.Final.Content)
}

func TestRunOnStopAbsentFinishesNormally(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}
	res, err := Run(context.Background(), Config{Provider: provider}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "done", res.Final.Content)
	assert.Equal(t, 1, res.Turns)
}
