package loop

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// aliasProvider is a scriptProvider whose model list knows the served model and not the alias the run asks for.
type aliasProvider struct {
	*scriptProvider
	limits map[string]Limits
	asked  []string
}

func (p *aliasProvider) ModelLimits(_ context.Context, model string) (Limits, error) {
	p.asked = append(p.asked, model)
	return p.limits[model], nil
}

func servedComp(content string, served string, calls ...ToolCall) *Completion {
	comp := usageComp(content, 8000, calls...)
	comp.Model = served
	return comp
}

// The gateway resolves "auto-alias" to a model its list names. The first response says which, and the run
// compacts against that model's window from then on.
func TestRunResolvesTheWindowFromTheModelTheResponseNames(t *testing.T) {
	provider := &aliasProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{
			{comp: servedComp("working", "vendor/served", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
			{comp: assistantComp("the summary")},
			{comp: servedComp("done", "vendor/served")},
		}},
		limits: map[string]Limits{"vendor/served": {ContextWindow: 10000}},
	}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	var compacted bool
	cb := func(ev CompactionEvent) error { compacted = true; return nil }
	keep(t, &events.OnCompaction, cb)
	windows := windowEvents(t, &events)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events}
	req := Request{Model: "auto-alias", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	assert.Equal(t, []string{"auto-alias", "vendor/served"}, provider.asked, "the alias first, then the model the response named, once")
	require.Len(t, *windows, 2)
	assert.Error(t, (*windows)[0].Err, "the alias publishes no window")
	assert.Equal(t, ContextWindowEvent{Window: 10000, FromModelList: true, Served: "vendor/served"}, (*windows)[1])
	assert.True(t, compacted, "8000 prompt tokens is 80% of the served model's 10000")
}

// A response that names the requested model, or none, adds no lookup.
func TestARequestedModelTheResponseEchoesIsNotLookedUpTwice(t *testing.T) {
	provider := &aliasProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{
			{comp: servedComp("one", "m", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
			{comp: servedComp("two", "")},
		}},
		limits: map[string]Limits{},
	}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	windows := windowEvents(t, &events)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events}
	req := Request{Model: "m", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	assert.Equal(t, []string{"m"}, provider.asked)
	require.Len(t, *windows, 1)
	assert.Error(t, (*windows)[0].Err)
}

// A served model the list does not limit is reported once, with the name, and never asked again.
func TestAServedModelWithNoPublishedWindowIsReportedOnce(t *testing.T) {
	provider := &aliasProvider{
		scriptProvider: &scriptProvider{steps: []scriptStep{
			{comp: servedComp("one", "vendor/unknown", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
			{comp: servedComp("two", "vendor/unknown")},
		}},
		limits: map[string]Limits{},
	}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	windows := windowEvents(t, &events)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll, Events: &events}
	req := Request{Model: "auto-alias", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	assert.Equal(t, []string{"auto-alias", "vendor/unknown"}, provider.asked)
	require.Len(t, *windows, 2)
	assert.Equal(t, "vendor/unknown", (*windows)[1].Served)
	assert.ErrorContains(t, (*windows)[1].Err, "vendor/unknown")
	assert.Zero(t, (*windows)[1].Window)
}
