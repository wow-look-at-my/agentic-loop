package agentic

import (
	"context"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOpenAITimingsDecodeLastWins(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"Hel"}}],"timings":{"prompt_n":5,"prompt_ms":100.5,"predicted_n":1,"predicted_ms":10}}`,
		`{"choices":[{"delta":{"content":"lo"}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"stop"}],"timings":{"prompt_n":5,"prompt_ms":100.5,"predicted_n":2,"predicted_ms":25.25},"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	var snaps []Timings
	ev := &StreamEvents{OnTimings: func(tm Timings) error { snaps = append(snaps, tm); return nil }}
	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}}, ev)
	require.NoError(t, err)

	require.Len(t, snaps, 2, "OnTimings fires per reported snapshot")
	assert.Equal(t, Timings{PromptN: 5, PromptMS: 100.5, PredictedN: 1, PredictedMS: 10}, snaps[0],
		"wire-faithful field decode (prompt_n/prompt_ms/predicted_n/predicted_ms)")

	require.NotNil(t, comp.Timings, "reported timings surface on the completion")
	assert.Equal(t, Timings{PromptN: 5, PromptMS: 100.5, PredictedN: 2, PredictedMS: 25.25}, *comp.Timings,
		"the LAST snapshot wins")
	assert.True(t, comp.UsageReported)
	assert.Equal(t, 7, comp.Usage.TotalTokens)
	assert.Equal(t, "Hello", comp.Message.Content)
}

func TestOpenAITimingsAbsentStaysNil(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	fired := false
	ev := &StreamEvents{OnTimings: func(Timings) error { fired = true; return nil }}
	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, ev)
	require.NoError(t, err)
	assert.Nil(t, comp.Timings, "nil means the provider never reported timings — the library synthesizes nothing")
	assert.False(t, fired)
	assert.False(t, comp.UsageReported, "no usage snapshot arrived either")
	assert.Equal(t, Usage{}, comp.Usage)
}

func TestOpenAIUsageReportedDistinguishesZero(t *testing.T) {
	// An upstream that reports an all-zero usage snapshot is distinguishable
	// from one that reports none: UsageReported is true, Usage is zero.
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`,
		`{"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.True(t, comp.UsageReported, "an explicit all-zero snapshot IS a report")
	assert.Equal(t, Usage{}, comp.Usage)
}

func TestAnthropicNeverReportsTimings(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()

	fired := false
	ev := &StreamEvents{OnTimings: func(Timings) error { fired = true; return nil }}
	p := anProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, ev)
	require.NoError(t, err)
	assert.False(t, fired, "the Anthropic dialect never fires OnTimings")
	assert.Nil(t, comp.Timings)
	assert.True(t, comp.UsageReported, "message_start carried usage")
}
