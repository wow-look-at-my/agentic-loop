package loop

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func windowRun(t *testing.T, reported int, cfg Config) (*Result, []ContextWindowEvent, error) {
	t.Helper()
	provider := &limitedProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{{comp: usageComp("answer", 10)}}},
		limits:         Limits{ContextWindow: reported},
	}
	events := Events{}
	windows := windowEvents(t, &events)
	cfg.Provider, cfg.Events = provider, &events
	res, err := Run(context.Background(), cfg, Request{Model: "m", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}})
	return res, *windows, err
}

func TestFallbackIsIgnoredWhenTheServerReportsAWindow(t *testing.T) {
	_, windows, err := windowRun(t, 50000, Config{ContextWindowFallback: 10000})
	require.NoError(t, err)
	assert.Equal(t, []ContextWindowEvent{{Window: 50000, Reported: 50000, FromModelList: true}}, windows)
}

func TestFallbackIsUsedWhenTheServerReportsNone(t *testing.T) {
	_, windows, err := windowRun(t, 0, Config{ContextWindowFallback: 10000})
	require.NoError(t, err)
	require.Len(t, windows, 1)
	assert.Equal(t, 10000, windows[0].Window)
	assert.True(t, windows[0].FromFallback)
	assert.Contains(t, windows[0].Warning, "publishes no context window")
	assert.NoError(t, windows[0].Err)
}

func TestOverrideShrinksAReportedWindow(t *testing.T) {
	_, windows, err := windowRun(t, 50000, Config{ContextWindowOverride: 20000})
	require.NoError(t, err)
	assert.Equal(t, []ContextWindowEvent{{Window: 20000, Reported: 50000, Overridden: true}}, windows)
}

func TestOverrideEqualToTheReportedWindowIsAllowed(t *testing.T) {
	_, windows, err := windowRun(t, 50000, Config{ContextWindowOverride: 50000})
	require.NoError(t, err)
	assert.Equal(t, 50000, windows[0].Window)
}

func TestOverrideLargerThanTheReportedWindowFails(t *testing.T) {
	res, windows, err := windowRun(t, 50000, Config{ContextWindowOverride: 60000})
	require.ErrorIs(t, err, ErrContextWindowConfig)
	assert.Contains(t, err.Error(), "60000, larger than the 50000 tokens the server reports")
	assert.Nil(t, res, "the run fails before any model call")
	assert.Empty(t, windows)
}

func TestOverrideWithoutAReportedWindowWarns(t *testing.T) {
	_, windows, err := windowRun(t, 0, Config{ContextWindowOverride: 20000})
	require.NoError(t, err)
	require.Len(t, windows, 1)
	assert.Equal(t, 20000, windows[0].Window)
	assert.True(t, windows[0].Overridden)
	assert.Contains(t, windows[0].Warning, "nothing checks it")
}

func TestFallbackAndOverrideTogetherFail(t *testing.T) {
	res, windows, err := windowRun(t, 50000, Config{ContextWindowFallback: 10000, ContextWindowOverride: 20000})
	require.ErrorIs(t, err, ErrContextWindowConfig)
	assert.Contains(t, err.Error(), "both set")
	assert.Nil(t, res)
	assert.Empty(t, windows)
}

func TestFallbackAndOverrideTogetherFailWithoutAutoCompact(t *testing.T) {
	_, err := Run(context.Background(), Config{
		Provider: &scriptProvider{}, ContextWindowFallback: 1, ContextWindowOverride: 1,
	}, Request{Model: "m"})
	require.ErrorIs(t, err, ErrContextWindowConfig)
}

func TestNegativeWindowSettingsFail(t *testing.T) {
	_, err := Run(context.Background(), Config{Provider: &scriptProvider{}, ContextWindowOverride: -1}, Request{Model: "m"})
	require.ErrorIs(t, err, ErrContextWindowConfig)
}

// bigResultRun answers one tool call with a result of multiple chars, then answers.
func bigResultRun(t *testing.T, cfg Config, autoCompact float64) (*Result, bool, error) {
	t.Helper()
	provider := &limitedProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{
			{comp: usageComp("working", 1000, ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
			{comp: assistantComp("the summary")},
			{comp: usageComp("done", 2000)},
		}},
		limits: Limits{ContextWindow: 10000},
	}
	exec := &fakeExec{
		tools:   []ToolDecl{{Name: "alpha", Readonly: true}},
		results: map[string]ToolResult{"alpha": {Content: strings.Repeat("x", 30000)}},
	}
	events := Events{}
	var compacted bool
	keep(t, &events.OnCompaction, func(CompactionEvent) error { compacted = true; return nil })
	cfg.Provider, cfg.Events, cfg.Tools, cfg.Approver = provider, &events, exec.registry(), allowAll
	res, err := Run(context.Background(), cfg, Request{Model: "m", AutoCompact: autoCompact, Messages: []Message{{Role: RoleUser, Content: "go"}}})
	return res, compacted, err
}

func TestARequestOverTheWindowIsCompactedFirst(t *testing.T) {
	res, compacted, err := bigResultRun(t, Config{}, 0.8)
	require.NoError(t, err)
	assert.True(t, compacted)
	assert.Equal(t, "done", res.Final.Content)
}

func TestARequestOverTheOverrideFailsWithoutAutoCompact(t *testing.T) {
	res, compacted, err := bigResultRun(t, Config{ContextWindowOverride: 8000}, 0)
	require.ErrorIs(t, err, ErrContextWindowExceeded)
	assert.Contains(t, err.Error(), "the window is 8000")
	assert.False(t, compacted)
	require.NotNil(t, res)
	assert.NotEmpty(t, res.Messages, "the transcript so far comes back with the error")
}

// servedProvider reports a window only for the model a response names.
type servedProvider struct {
	*scriptProvider
	window int
}

func (p *servedProvider) ModelLimits(_ context.Context, model string) (Limits, error) {
	if model == "vendor/served" {
		return Limits{ContextWindow: p.window}, nil
	}
	return Limits{}, nil
}

func servedRun(t *testing.T, window int, cfg Config) ([]ContextWindowEvent, error) {
	t.Helper()
	first := usageComp("working", 10, ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})
	first.Model = "vendor/served"
	provider := &servedProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{{comp: first}, {comp: usageComp("done", 10)}}},
		window:         window,
	}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	windows := windowEvents(t, &events)
	cfg.Provider, cfg.Events, cfg.Tools, cfg.Approver = provider, &events, exec.registry(), allowAll
	_, err := Run(context.Background(), cfg, Request{Model: "alias", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}})
	return *windows, err
}

func TestAServedWindowReplacesTheFallback(t *testing.T) {
	windows, err := servedRun(t, 50000, Config{ContextWindowFallback: 10000})
	require.NoError(t, err)
	require.Len(t, windows, 2)
	assert.True(t, windows[0].FromFallback)
	assert.Equal(t, ContextWindowEvent{Window: 50000, Reported: 50000, FromModelList: true, Served: "vendor/served"}, windows[1])
}

func TestAServedWindowSmallerThanTheOverrideFails(t *testing.T) {
	_, err := servedRun(t, 50000, Config{ContextWindowOverride: 60000})
	require.ErrorIs(t, err, ErrContextWindowConfig)
	assert.Contains(t, err.Error(), `for "vendor/served"`)
}
