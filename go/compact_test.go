package agentic

import (
	"context"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCompact(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("  the detailed recap  ")},
	}}
	history := []Message{
		{Role: RoleUser, Content: "first"},
		{Role: RoleAssistant, Content: "reply"},
	}
	req := Request{Model: "m", System: "you compact", Messages: history,
		Tools: []ToolDecl{{Name: "should-be-stripped"}}}

	res, err := Compact(context.Background(), provider, req)
	require.NoError(t, err)

	require.Len(t, provider.reqs, 1)
	sent := provider.reqs[0]
	assert.Empty(t, sent.Tools, "the summarize call sends NO tools")
	assert.Equal(t, "you compact", sent.System)
	require.Len(t, sent.Messages, 3)
	assert.Equal(t, history[0], sent.Messages[0])
	assert.Equal(t, history[1], sent.Messages[1])
	assert.Equal(t, Message{Role: RoleUser, Content: CompactRequestText}, sent.Messages[2],
		"the request text rides as the trailing user message")

	assert.Equal(t, "the detailed recap", res.Summary, "trimmed")
	require.Len(t, res.Messages, 2)
	assert.Equal(t, Message{Role: RoleUser, Content: CompactRequestText}, res.Messages[0])
	assert.Equal(t, Message{Role: RoleAssistant, Content: "the detailed recap"}, res.Messages[1])
	require.NotNil(t, res.Completion, "the summarize call's whole completion, not a projection of it")
	assert.Equal(t, 15, res.Completion.Usage.TotalTokens)

	// The caller's request was not mutated.
	require.Len(t, req.Messages, 2)
	require.Len(t, req.Tools, 1)
}

func TestCompactEmptySummary(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("   ")}}}
	res, err := Compact(context.Background(), provider, Request{Model: "m"})
	assert.Nil(t, res)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "empty summary")
}

func TestCompactProviderError(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 500, Body: "boom"}}}}
	res, err := Compact(context.Background(), provider, Request{Model: "m"})
	assert.Nil(t, res)
	require.Error(t, err)

	_, err = Compact(context.Background(), nil, Request{Model: "m"})
	require.Error(t, err)
}

func TestOneShot(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("  My Title  ")}}}
	comp, err := OneShot(context.Background(), provider, Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "name this"}},
		Tools: []ToolDecl{{Name: "stripped"}},
	}, 0)
	require.NoError(t, err)
	require.NotNil(t, comp)
	assert.Equal(t, "My Title", strings.TrimSpace(comp.Message.Content))
	assert.Equal(t, 15, comp.Usage.TotalTokens)
	require.Len(t, provider.reqs, 1)
	assert.Empty(t, provider.reqs[0].Tools, "OneShot strips tools")
}

// The reason the signature is the whole Completion: a Usage return cannot say
// whether the upstream reported usage at all, and it drops the dollar figure
// the upstream did report.
func TestOneShotSurfacesWhatAUsageReturnCannot(t *testing.T) {
	cost := 0.0042
	silent := &Completion{Message: Message{Role: RoleAssistant, Content: "done"}}
	priced := &Completion{
		Message: Message{Role: RoleAssistant, Content: "done"},
		Usage:   Usage{TotalTokens: 11}, UsageReported: true, CostUsd: &cost,
	}
	provider := &scriptProvider{steps: []scriptStep{{comp: silent}, {comp: priced}}}

	comp, err := OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.NoError(t, err)
	assert.False(t, comp.UsageReported, "all-zero usage and no usage are different facts")
	assert.Nil(t, comp.CostUsd)

	comp, err = OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.NoError(t, err)
	assert.True(t, comp.UsageReported)
	require.NotNil(t, comp.CostUsd, "a provider-reported dollar cost survives")
	assert.InDelta(t, cost, *comp.CostUsd, 1e-9)
}

// blockingProvider blocks until its context is done.
type blockingProvider struct{}

func (blockingProvider) Complete(ctx context.Context, _ Request, _ *StreamEvents) (*Completion, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestOneShotTimeout(t *testing.T) {
	start := time.Now()
	comp, err := OneShot(context.Background(), blockingProvider{}, Request{Model: "m"}, 20*time.Millisecond)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Nil(t, comp, "nothing arrived, so there is no completion to report")
	assert.Less(t, time.Since(start), 5*time.Second)
	assert.False(t, IsTransient(err), "a deadline expiry is never transient")

	_, err = OneShot(context.Background(), nil, Request{}, 0)
	require.Error(t, err)
}

func TestOneShotSingleAttempt(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 503, Body: "unavailable"}}}}
	_, err := OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.Error(t, err)
	assert.Len(t, provider.reqs, 1, "OneShot never retries")
}

func TestOneShotPartialError(t *testing.T) {
	partial := &Completion{Message: Message{Role: RoleAssistant, Content: "par"}, Usage: Usage{TotalTokens: 3}}
	provider := &scriptProvider{steps: []scriptStep{{comp: partial, err: context.Canceled}}}
	comp, err := OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.Error(t, err)
	require.NotNil(t, comp, "the partial completion rides alongside the error")
	assert.Equal(t, 3, comp.Usage.TotalTokens, "partial usage still reported")
	assert.Equal(t, "par", comp.Message.Content)
}
