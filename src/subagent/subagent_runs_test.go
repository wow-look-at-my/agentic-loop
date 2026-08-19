package subagent

import (
	"context"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
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
	runs.Complete("call_a", "the middleware is fine", false)

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
	runs.Complete("a", "ra", false)
	runs.Complete("b", "rb", true)

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
		runs.Complete("slow", "found it", false)
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
	runs.Complete("ghost", "from nowhere", false)

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
	runs.Complete("a", "done", false)

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

// --- asynchronous launches --------------------------------------------------

// With a registry configured, run_subagent hands back a receipt rather than an
// answer -- which is what lets a model fan several out in one turn -- and the
// report arrives through the registry.
func TestSubagentToolLaunchesAsynchronously(t *testing.T) {
	provider := &scriptProvider{Steps: []scriptStep{{Comp: assistantComp("no defects found")}}}
	runs := NewSubagentRuns(nil)
	tool := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Runs: runs})

	res, err := tool.Execute(agentic.WithToolCallID(context.Background(), "call_1"),
		[]byte(`{"description":"audit auth","prompt":"look"}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "Sub-agent launched: audit auth")
	assert.NotContains(t, res.Content, "no defects found", "a launch is not an answer")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reports, cerr := runs.Collect(ctx)
	require.NoError(t, cerr)
	require.Len(t, reports, 1)
	assert.Equal(t, "no defects found", reports[0].Text)
	assert.Equal(t, "call_1", reports[0].CallID)
	assert.Equal(t, "audit auth", reports[0].Label)
}

// A misused argument still teaches the model -- one delivery later, as the
// report, instead of as the launch's result.
func TestAsyncSubagentMisuseArrivesAsTheReport(t *testing.T) {
	runs := NewSubagentRuns(nil)
	tool := NewSubagentTool(SubagentConfig{Provider: &scriptProvider{}, Model: "m", Runs: runs})

	res, err := tool.Execute(agentic.WithToolCallID(context.Background(), "call_1"), []byte(`{"prompt":"  "}`))
	require.NoError(t, err)
	assert.False(t, res.IsError, "the launch itself succeeded")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reports, cerr := runs.Collect(ctx)
	require.NoError(t, cerr)
	require.Len(t, reports, 1)
	assert.True(t, reports[0].IsError)
	assert.Contains(t, reports[0].Text, "non-empty prompt")
}

// A turn that would END while a sub-agent is out instead waits for it and
// hands the report to the model -- the promise the receipt made.
func TestRunDeliversAnOutstandingSubagentReport(t *testing.T) {
	provider := &scriptProvider{Steps: []scriptStep{
		{Comp: assistantComp("launched one; waiting")},
		{Comp: assistantComp("the auth middleware is fine")},
	}}
	runs := NewSubagentRuns(nil)
	runs.Launch("call_a", "audit auth", "look")
	go func() {
		time.Sleep(20 * time.Millisecond)
		runs.Complete("call_a", "no defects found", false)
	}()

	res, err := agentic.Run(context.Background(), agentic.Config{Provider: provider, Subagents: runs}, agentic.Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "the auth middleware is fine", res.Final.Content)
	assert.Equal(t, 2, res.Turns, "the delivery must earn the model another turn")

	var delivered string
	for _, m := range res.Messages {
		if m.Role == agentic.RoleUser && len(m.Content) > 0 {
			delivered = m.Content
		}
	}
	assert.Contains(t, delivered, "no defects found")
	assert.Contains(t, delivered, "No sub-agents are still running.")
}
