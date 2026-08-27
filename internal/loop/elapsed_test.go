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

// noon is the instant the notices in these tests are rendered at.
var noon = time.Date(2026, 8, 26, 3, 14, 0, 0, time.UTC)

func TestFormatElapsedIsTwoUnitsAndTerse(t *testing.T) {
	cases := []struct {
		d    time.Duration
		want string
	}{
		{500 * time.Millisecond, "<1sec"},
		{time.Second, "1sec"},
		{45 * time.Second, "45secs"},
		{time.Minute, "1min"},
		{90 * time.Second, "1min 30secs"},
		{2*time.Hour + 14*time.Minute, "2hrs 14mins"},
		{2*time.Hour + 14*time.Minute + 9*time.Second, "2hrs 14mins"},
		{25 * time.Hour, "1d 1hr"},
		{47 * time.Hour, "1d 23hrs"},
		{72 * time.Hour, "3d"},
		{3*24*time.Hour + 20*time.Second, "3d 20secs"},
	}
	for _, c := range cases {
		assert.Equal(t, c.want, FormatElapsed(c.d), c.d.String())
	}
}

func TestFormatElapsedNoticeLeadsWithTheClock(t *testing.T) {
	assert.Equal(t, "Current time is 3:14 AM on 8/26/2026, 1d 23hrs have passed",
		FormatElapsedNotice(noon, noon.Add(-47*time.Hour)))
}

func TestFormatElapsedNoticeStatesTheTimeAloneWithNothingToMeasure(t *testing.T) {
	// Nothing has been asked yet, and a clock that went backwards has no gap to
	// report either. The current time is still worth stating.
	assert.Equal(t, "Current time is 3:14 AM on 8/26/2026", FormatElapsedNotice(noon, time.Time{}))
	assert.Equal(t, "Current time is 3:14 AM on 8/26/2026", FormatElapsedNotice(noon, noon.Add(time.Hour)))
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
			Since: noon.Add(-2 * time.Hour),
			Now:   stepClock(noon, noon.Add(30*time.Second)),
		},
	}
	res, err := Run(context.Background(), cfg, Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	require.Len(t, provider.reqs, 2)
	first := provider.reqs[0].Messages
	require.Len(t, first, 2)
	assert.Equal(t, Message{Role: RoleUser, Kind: ElapsedKind,
		Content: "Current time is 3:14 AM on 8/26/2026, 2hrs have passed"}, first[1])

	// The second call measures from the FIRST call, not from Since.
	second := provider.reqs[1].Messages
	require.NotEmpty(t, second)
	assert.Equal(t, Message{Role: RoleUser, Kind: ElapsedKind,
		Content: "Current time is 3:14 AM on 8/26/2026, 30secs have passed"}, second[len(second)-1])

	// The notice is per-call: the durable transcript never holds one.
	for _, m := range res.Messages {
		assert.NotEqual(t, ElapsedKind, m.Kind, "a stored notice is false the moment it is replayed")
	}
	// Neither does the transcript the second call replayed.
	for _, m := range second[:len(second)-1] {
		assert.NotEqual(t, ElapsedKind, m.Kind)
	}
}

func TestRunElapsedStatesTheTimeOnAnUnseededFirstCall(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	cfg := Config{
		Provider:    provider,
		Tools:       exec.registry(),
		Approver:    allowAll,
		ElapsedTime: &ElapsedTime{Now: stepClock(noon, noon.Add(5*time.Second))},
	}
	_, err := Run(context.Background(), cfg, Request{Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.NoError(t, err)

	require.Len(t, provider.reqs, 2)
	firstMsgs := provider.reqs[0].Messages
	require.Len(t, firstMsgs, 2)
	assert.Equal(t, "Current time is 3:14 AM on 8/26/2026", firstMsgs[1].Content,
		"nothing to measure from yet, so no gap is invented")

	second := provider.reqs[1].Messages
	assert.Equal(t, "Current time is 3:14 AM on 8/26/2026, 5secs have passed",
		second[len(second)-1].Content)
}

func TestElapsedTrackerIsNilWhenTheOptionIsOff(t *testing.T) {
	assert.Nil(t, newElapsedTracker(nil))
	assert.Empty(t, (*elapsedTracker)(nil).mark())
	assert.Nil(t, elapsedMessages(nil, ""))
}
