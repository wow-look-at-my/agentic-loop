package loop

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stepClock returns a clock that hands out the given instants in order, then
// repeats the last one.
func stepClock(times ...time.Time) func() time.Time {
	i := 0
	return func() time.Time {
		t := times[i]
		if i < len(times)-1 {
			i++
		}
		return t
	}
}

func TestFormatElapsedRendersTwoUnits(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{500 * time.Millisecond, "less than a second"},
		{time.Second, "1 second"},
		{45 * time.Second, "45 seconds"},
		{time.Minute, "1 minute"},
		{90 * time.Second, "1 minute 30 seconds"},
		{2*time.Hour + 14*time.Minute, "2 hours 14 minutes"},
		{2*time.Hour + 14*time.Minute + 9*time.Second, "2 hours 14 minutes"},
		{25 * time.Hour, "1 day 1 hour"},
		{72 * time.Hour, "3 days"},
		{3*24*time.Hour + 20*time.Second, "3 days 20 seconds"},
	}
	for _, c := range cases {
		assert.Equal(t, c.want, FormatElapsed(c.d), c.d.String())
	}
}

func TestFormatElapsedNoticeSaysItIsNotTheUser(t *testing.T) {
	assert.Equal(t,
		"[automated notice -- time since the previous request in this conversation:"+
			" 2 hours 14 minutes; this is not a message from the user]",
		FormatElapsedNotice(2*time.Hour+14*time.Minute))
}

func TestRunWithoutElapsedTimeSaysNothing(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}
	res, err := Run(context.Background(), Config{Provider: provider},
		Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)
	require.Len(t, provider.reqs, 1)
	assert.Len(t, provider.reqs[0].Messages, 1, "the option is off: nothing is added")
	assert.Len(t, res.Messages, 2)
}

func TestRunElapsedNoticeRidesEveryCall(t *testing.T) {
	base := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	cfg := Config{
		Provider: provider,
		Tools:    exec.registry(),
		Approver: allowAll,
		ElapsedTime: &ElapsedTime{
			Since: base.Add(-2 * time.Hour),
			Now:   stepClock(base, base.Add(30*time.Second)),
		},
	}
	res, err := Run(context.Background(), cfg, Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	require.Len(t, provider.reqs, 2)
	first := provider.reqs[0].Messages
	require.Len(t, first, 2)
	assert.Equal(t, Message{Role: RoleUser, Kind: ElapsedKind, Content: FormatElapsedNotice(2 * time.Hour)}, first[1])

	// The second call measures from the FIRST call, not from Since.
	second := provider.reqs[1].Messages
	require.NotEmpty(t, second)
	assert.Equal(t, Message{Role: RoleUser, Kind: ElapsedKind, Content: FormatElapsedNotice(30 * time.Second)},
		second[len(second)-1])

	// The notice is per-call: the durable transcript never holds one.
	for _, m := range res.Messages {
		assert.NotEqual(t, ElapsedKind, m.Kind, "a stored notice is false the moment it is replayed")
	}
	// Neither does the transcript the second call replayed.
	for _, m := range second[:len(second)-1] {
		assert.NotEqual(t, ElapsedKind, m.Kind)
	}
}

func TestRunElapsedSaysNothingOnAnUnseededFirstCall(t *testing.T) {
	base := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	cfg := Config{
		Provider:    provider,
		Tools:       exec.registry(),
		Approver:    allowAll,
		ElapsedTime: &ElapsedTime{Now: stepClock(base, base.Add(5*time.Second))},
	}
	_, err := Run(context.Background(), cfg, Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	require.Len(t, provider.reqs, 2)
	assert.Len(t, provider.reqs[0].Messages, 1, "nothing to measure from yet")
	second := provider.reqs[1].Messages
	assert.Equal(t, FormatElapsedNotice(5*time.Second), second[len(second)-1].Content)
}

func TestElapsedTrackerReportsZeroWhenTheClockGoesBackwards(t *testing.T) {
	base := time.Date(2026, 3, 1, 12, 0, 0, 0, time.UTC)
	tr := newElapsedTracker(&ElapsedTime{Since: base, Now: stepClock(base.Add(-time.Hour))})
	assert.Equal(t, FormatElapsedNotice(0), tr.mark())
}

func TestElapsedTrackerIsNilWhenTheOptionIsOff(t *testing.T) {
	assert.Nil(t, newElapsedTracker(nil))
	assert.Empty(t, (*elapsedTracker)(nil).mark())
	assert.Nil(t, elapsedMessages(nil, ""))
}
