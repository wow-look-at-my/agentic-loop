package subagent

import (
	"context"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentic "github.com/wow-look-at-my/agentic-loop"
)

// updateLog records the lifecycle callback, which fires from the launched
// goroutines while the turn is still running.
type updateLog struct {
	mu  sync.Mutex
	ups []agentic.SubagentUpdate
}

func (l *updateLog) record(u agentic.SubagentUpdate) {
	l.mu.Lock()
	defer l.mu.Unlock()
	l.ups = append(l.ups, u)
}

// --- asynchronous launches --------------------------------------------------

// With a registry configured, run_subagent hands back a receipt rather than an
// answer -- which is what lets a model fan several out in turn -- and the
// report arrives through the registry.
func TestSubagentToolLaunchesAsynchronously(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("no defects found")}}}
	runs := agentic.NewSubagentRuns(nil)
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

// What the sub-agent spent rides up with its report, so a host that watches
// only lifecycle still gets the total. A sub-agent answers in text; nothing
// else in the result carries a number, and money that was never emitted cannot
// be recovered after the run.
func TestAsyncSubagentReportCarriesItsUsages(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", agentic.ToolCall{ID: "s1", Name: "Repo__read", Arguments: "{}"})},
		{comp: assistantComp("no defects found")},
	}}
	log := &updateLog{}
	runs := agentic.NewSubagentRuns(log.record)
	tool := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Tools: subParentExec().registry(), Runs: runs,
	})

	_, err := tool.Execute(agentic.WithToolCallID(context.Background(), "call_1"), []byte(`{"prompt":"look"}`))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reports, cerr := runs.Collect(ctx)
	require.NoError(t, cerr)
	require.Len(t, reports, 1)
	require.Len(t, reports[0].Usages, 2, "one entry per nested model call, in order, never summed")
	assert.Equal(t, 15, reports[0].Usages[0].TotalTokens)

	// The lifecycle update carries the same numbers.
	terminal := log.ups[len(log.ups)-1]
	assert.Equal(t, agentic.SubagentDone, terminal.State)
	assert.Equal(t, reports[0].Usages, terminal.Usages)
}

// The share_context=summary briefing is a model call the sub-agent's own run
// made, so it is charged to that run rather than vanishing.
func TestAsyncSubagentUsagesIncludeTheContextBriefing(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("a briefing")}, // the share_context=summary call
		{comp: assistantComp("no defects found")},
	}}
	runs := agentic.NewSubagentRuns(nil)
	tool := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Runs: runs,
		ParentMessages: []agentic.Message{{Role: agentic.RoleUser, Content: "the parent asked something"}},
	})

	_, err := tool.Execute(agentic.WithToolCallID(context.Background(), "call_1"),
		[]byte(`{"prompt":"look","share_context":"summary"}`))
	require.NoError(t, err)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	reports, cerr := runs.Collect(ctx)
	require.NoError(t, cerr)
	require.Len(t, reports, 1)
	assert.Len(t, reports[0].Usages, 2, "the briefing call plus the run's one turn")
}

// A misused argument still teaches the model -- delivery later, as the
// report, instead of as the launch's result.
func TestAsyncSubagentMisuseArrivesAsTheReport(t *testing.T) {
	runs := agentic.NewSubagentRuns(nil)
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
