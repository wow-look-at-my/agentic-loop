package loop

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// updateLog records the lifecycle callback, which fires from the launched
// goroutines while the turn is still running.
type updateLog struct {
	mu  sync.Mutex
	ups []SubagentUpdate
}

func (l *updateLog) record(u SubagentUpdate) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ups = append(l.ups, u)
}

func (l *updateLog) states() []SubagentState {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]SubagentState, 0, len(l.ups))
	for _, u := range l.ups {
		out = append(out, u.State)
	}
	return out
}

func TestSubagentRunsLifecycleEmitsUpdates(t *testing.T) {
	log := &updateLog{}
	runs := NewSubagentRuns(log.record)

	runs.Launch("call_a", "audit auth", "look at the middleware")
	assert.Equal(t, 1, runs.Pending())
	assert.Equal(t, 1, runs.Running())
	runs.MarkRunning("call_a")
	runs.Complete("call_a", "the middleware is fine", false, nil)

	// Still pending: a report nobody has delivered yet is not done with.
	assert.Equal(t, 1, runs.Pending())
	assert.Zero(t, runs.Running())
	assert.Equal(t, []SubagentState{SubagentQueued, SubagentRunning, SubagentDone}, log.states())

	reports, err := runs.Collect(context.Background())
	require.NoError(t, err)
	require.Len(t, reports, 1)
	assert.Equal(t, "the middleware is fine", reports[0].Text)
	assert.Equal(t, "audit auth", reports[0].Label)
	assert.False(t, reports[0].IsError)
	assert.Zero(t, runs.Pending(), "a delivered report is finished with")
}

func TestSubagentRunsCollectDrainsEveryReadyReport(t *testing.T) {
	runs := NewSubagentRuns(nil)
	for _, id := range []string{"a", "b", "c"} {
		runs.Launch(id, id, "p")
	}
	runs.Complete("a", "ra", false, nil)
	runs.Complete("b", "rb", true, nil)

	// Two finished together, so they cost ONE delivery, not two.
	reports, err := runs.Collect(context.Background())
	require.NoError(t, err)
	require.Len(t, reports, 2)
	assert.Equal(t, "ra", reports[0].Text)
	assert.True(t, reports[1].IsError)
	assert.Equal(t, 1, runs.Pending(), "the third is still out")
}

func TestSubagentRunsCollectWaitsForTheNextReport(t *testing.T) {
	runs := NewSubagentRuns(nil)
	runs.Launch("slow", "dig", "p")
	go func() {
		time.Sleep(20 * time.Millisecond)
		runs.Complete("slow", "found it", false, nil)
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reports, err := runs.Collect(ctx)
	require.NoError(t, err)
	require.Len(t, reports, 1)
	assert.Equal(t, "found it", reports[0].Text)
}

func TestSubagentRunsCollectHonoursCancellation(t *testing.T) {
	runs := NewSubagentRuns(nil)
	runs.Launch("never", "hangs", "p")
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err := runs.Collect(ctx)
	require.ErrorIs(t, err, context.Canceled)
}

func TestSubagentRunsCollectReturnsNothingWhenIdle(t *testing.T) {
	runs := NewSubagentRuns(nil)
	reports, err := runs.Collect(context.Background())
	require.NoError(t, err)
	assert.Empty(t, reports, "an idle registry must not block")
}

// A loop may consult a registry it was never given -- a run with no
// asynchronous sub-agents has none -- so every question has a nil answer.
func TestNilSubagentRunsIsIdle(t *testing.T) {
	var runs *SubagentRuns
	assert.Zero(t, runs.Pending())
	assert.Zero(t, runs.Running())
	assert.Empty(t, runs.Take())
	assert.Zero(t, runs.CancelRemaining())
	reports, err := runs.Collect(context.Background())
	require.NoError(t, err)
	assert.Empty(t, reports)
}

func TestSubagentRunsIgnoreUnknownAndDuplicateCalls(t *testing.T) {
	runs := NewSubagentRuns(nil)
	runs.Launch("a", "one", "p")
	runs.Launch("a", "again", "p") // one tool call executes once
	runs.MarkRunning("ghost")
	runs.Complete("ghost", "from nowhere", false, nil)

	// One launch, one outstanding run, and nothing ready: the ghost report was
	// not adopted (Collect would block on the real run, which is the point).
	assert.Equal(t, 1, runs.Pending())
	assert.Empty(t, runs.Take())
	assert.Equal(t, 1, runs.Running())
}

// A backend that assigns no tool_call id must still get one run per launch:
// two sub-agents sharing an id would report over each other.
func TestSubagentRunsMintsDistinctIDs(t *testing.T) {
	runs := NewSubagentRuns(nil)
	first, second := runs.nextID(), runs.nextID()
	assert.NotEqual(t, first, second)
	runs.Launch(first, "one", "p")
	runs.Launch(second, "two", "p")
	assert.Equal(t, 2, runs.Pending())
}

func TestSubagentRunsCancelRemainingIsVisible(t *testing.T) {
	log := &updateLog{}
	runs := NewSubagentRuns(log.record)
	runs.Launch("a", "one", "p")
	runs.Launch("b", "two", "p")
	runs.Complete("a", "done", false, nil)

	assert.Len(t, runs.Take(), 1)
	assert.Equal(t, 1, runs.CancelRemaining())
	assert.Zero(t, runs.Pending())

	var abandoned int
	for _, st := range log.states() {
		if st == SubagentAbandoned {
			abandoned++
		}
	}
	assert.Equal(t, 1, abandoned, "an abandoned run must be reported, never dropped quietly")
}

// --- the delivered text -----------------------------------------------------

func TestFormatSubagentDeliveryNamesTheTaskAndWhatIsLeft(t *testing.T) {
	text := FormatSubagentDelivery([]SubagentReport{
		{CallID: "a", Label: "audit auth", Text: "the middleware is fine"},
	}, 2, 0)

	assert.Contains(t, text, "automated delivery", "the model must never read this as the user speaking")
	assert.Contains(t, text, `"audit auth"`)
	assert.Contains(t, text, "the middleware is fine")
	assert.Contains(t, text, "2 sub-agents are still running")
	assert.NotContains(t, text, "abandoned")
}

func TestFormatSubagentDeliveryFlagsFailuresAndEmptyReports(t *testing.T) {
	text := FormatSubagentDelivery([]SubagentReport{
		{CallID: "a", Label: "", Text: "  ", IsError: true},
	}, 0, 0)
	assert.Contains(t, text, "FAILED")
	assert.Contains(t, text, "(no description given)")
	assert.Contains(t, text, "(the sub-agent returned no report)")
	assert.Contains(t, text, "No sub-agents are still running.")
}

func TestFormatSubagentDeliverySaysWhatWasAbandoned(t *testing.T) {
	text := FormatSubagentDelivery(nil, 0, 3)
	assert.Contains(t, text, "3 sub-agents were still running")
	assert.Contains(t, text, "their work is lost")
}

// A turn that would END while a sub-agent is out instead waits for it and
// hands the report to the model -- the promise the receipt made.
func TestRunDeliversAnOutstandingSubagentReport(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("launched one; waiting")},
		{comp: assistantComp("the auth middleware is fine")},
	}}
	runs := NewSubagentRuns(nil)
	runs.Launch("call_a", "audit auth", "look")
	go func() {
		time.Sleep(20 * time.Millisecond)
		runs.Complete("call_a", "no defects found", false, nil)
	}()

	res, err := Run(context.Background(), Config{Provider: provider, Subagents: runs}, Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "the auth middleware is fine", res.Final.Content)
	assert.Equal(t, 2, res.Turns, "the delivery must earn the model another turn")

	var delivered string
	for _, m := range res.Messages {
		if m.Role == RoleUser && len(m.Content) > 0 {
			delivered = m.Content
		}
	}
	assert.Contains(t, delivered, "no defects found")
	assert.Contains(t, delivered, "No sub-agents are still running.")
}
