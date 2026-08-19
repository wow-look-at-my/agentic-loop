package agentic

import (
	"context"
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
	assert.Equal(t, 15, res.Usage.TotalTokens)

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
	text, usage, err := OneShot(context.Background(), provider, Request{
		Model: "m", Messages: []Message{{Role: RoleUser, Content: "name this"}},
		Tools: []ToolDecl{{Name: "stripped"}},
	}, 0)
	require.NoError(t, err)
	assert.Equal(t, "My Title", text, "trimmed final text")
	assert.Equal(t, 15, usage.TotalTokens)
	require.Len(t, provider.reqs, 1)
	assert.Empty(t, provider.reqs[0].Tools, "OneShot strips tools")
}

// blockingProvider blocks until its context is done.
type blockingProvider struct{}

func (blockingProvider) Complete(ctx context.Context, _ Request, _ *StreamEvents) (*Completion, error) {
	<-ctx.Done()
	return nil, ctx.Err()
}

func TestOneShotTimeout(t *testing.T) {
	start := time.Now()
	text, _, err := OneShot(context.Background(), blockingProvider{}, Request{Model: "m"}, 20*time.Millisecond)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.DeadlineExceeded)
	assert.Empty(t, text)
	assert.Less(t, time.Since(start), 5*time.Second)
	assert.False(t, IsTransient(err), "a deadline expiry is never transient")

	_, _, err = OneShot(context.Background(), nil, Request{}, 0)
	require.Error(t, err)
}

func TestOneShotSingleAttempt(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 503, Body: "unavailable"}}}}
	_, _, err := OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.Error(t, err)
	assert.Len(t, provider.reqs, 1, "OneShot never retries")
}

func TestOneShotPartialError(t *testing.T) {
	partial := &Completion{Message: Message{Role: RoleAssistant, Content: "par"}, Usage: Usage{TotalTokens: 3}}
	provider := &scriptProvider{steps: []scriptStep{{comp: partial, err: context.Canceled}}}
	text, usage, err := OneShot(context.Background(), provider, Request{Model: "m"}, 0)
	require.Error(t, err)
	assert.Empty(t, text)
	assert.Equal(t, 3, usage.TotalTokens, "partial usage still reported")
}
