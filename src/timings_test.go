package commonai

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

	require.NotNil(t, lastTimings(comp), "reported timings surface on the completion")
	assert.Equal(t, Timings{PromptN: 5, PromptMS: 100.5, PredictedN: 2, PredictedMS: 25.25}, *lastTimings(comp),
		"the LAST snapshot wins")
	assert.True(t, (len(comp.Usages) > 0))
	assert.Equal(t, 7, firstUsage(comp).TotalTokens)
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
	assert.Nil(t, lastTimings(comp), "nil means the provider never reported timings — the library synthesizes nothing")
	assert.False(t, fired)
	assert.False(t, (len(comp.Usages) > 0), "no usage snapshot arrived either")
	assert.Equal(t, Usage{}, firstUsage(comp))
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
	require.Len(t, comp.Usages, 1, "an explicit all-zero snapshot IS a report")
	u := comp.Usages[0]
	assert.Zero(t, u.PromptTokens)
	assert.Zero(t, u.CompletionTokens)
	assert.Zero(t, u.TotalTokens)
	assert.Nil(t, u.CacheReadTokens, "zero counters are not a cache report")
	assert.JSONEq(t, `{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}`, string(u.Raw))
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
	assert.Nil(t, lastTimings(comp))
	assert.True(t, (len(comp.Usages) > 0), "message_start carried usage")
}
