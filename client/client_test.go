package client

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func intPtr(v int) *int { return &v }

// coreProvider is a fake at the format level, so a test can say exactly what
// the provider reported and check what the fold made of it.
type coreProvider struct {
	comp *commonai.Completion
	err  error
	reqs []Request
}

func (p *coreProvider) Complete(_ context.Context, req Request, _ *StreamEvents) (*commonai.Completion, error) {
	p.reqs = append(p.reqs, req)
	return p.comp, p.err
}

func TestFoldTakesTheNewestSnapshotAndNeverSums(t *testing.T) {
	// The xAI shape: a cumulative usage object on every chunk. Summing them
	// would multiply the real counts by the chunk count.
	inner := &coreProvider{comp: &commonai.Completion{
		Message: NewMessage(RoleAssistant, TextPart{Text: "hi"}),
		Usages: []Usage{
			{PromptTokens: 10, CompletionTokens: 1, TotalTokens: 11},
			{PromptTokens: 10, CompletionTokens: 4, TotalTokens: 14},
			{PromptTokens: 10, CompletionTokens: 9, TotalTokens: 19},
		},
	}}

	comp, err := up(inner).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.True(t, comp.UsageReported)
	assert.Equal(t, Usage{PromptTokens: 10, CompletionTokens: 9, TotalTokens: 19}, comp.Usage)
}

func TestFoldDiscardsARegressingSnapshot(t *testing.T) {
	// A final chunk that zeroes usage reports strictly less evidence than one
	// already seen, so it loses.
	inner := &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{
			{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
			{},
		},
	}}
	comp, err := up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.Equal(t, 150, comp.Usage.TotalTokens)

	// Equal evidence lets the later one win: it may carry richer cache detail.
	inner = &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{
			{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150},
			{PromptTokens: 100, CompletionTokens: 50, TotalTokens: 150, CacheReadTokens: intPtr(80)},
		},
	}}
	comp, err = up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	require.NotNil(t, comp.Usage.CacheReadTokens)
	assert.Equal(t, 80, *comp.Usage.CacheReadTokens)
}

func TestFoldFloorsTheTotalButKeepsASurplus(t *testing.T) {
	inner := &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 0}},
	}}
	comp, err := up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.Equal(t, 15, comp.Usage.TotalTokens, "an omitted total is floored at the parts")

	// xAI reports total = prompt + completion + reasoning, and that surplus is
	// real spend: flattening it would hide what the reasoning cost.
	inner = &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 40, ReasoningTokens: intPtr(25)}},
	}}
	comp, err = up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.Equal(t, 40, comp.Usage.TotalTokens)
	require.NotNil(t, comp.ReasoningTokens)
	assert.Equal(t, 25, *comp.ReasoningTokens)
}

// Reporting nothing and reporting zeros are different facts, and a caller
// deciding whether to bill a turn needs to tell them apart.
func TestUsageReportedDistinguishesSilence(t *testing.T) {
	silent := &coreProvider{comp: &commonai.Completion{}}
	comp, err := up(silent).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.False(t, comp.UsageReported)
	assert.Equal(t, Usage{}, comp.Usage)

	zeros := &coreProvider{comp: &commonai.Completion{Usages: []Usage{{}}}}
	comp, err = up(zeros).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.True(t, comp.UsageReported)
	assert.Equal(t, Usage{}, comp.Usage)
}

func TestFoldKeepsTheProviderExtras(t *testing.T) {
	cost := 0.0021
	raw := json.RawMessage(`{"prompt_tokens":7,"cost":0.0021}`)
	inner := &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{{
			PromptTokens: 7, CompletionTokens: 1, TotalTokens: 8,
			Raw: raw, CostUsd: &cost, ReasoningTokens: intPtr(3),
		}},
		Timings: []Timings{
			{PromptN: 1, PromptMS: 1},
			{PromptN: 12, PromptMS: 8.4, PredictedN: 45, PredictedMS: 611.2},
		},
	}}
	comp, err := up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.JSONEq(t, string(raw), string(comp.RawUsage))
	require.NotNil(t, comp.CostUsd)
	assert.InDelta(t, 0.0021, *comp.CostUsd, 1e-9)
	require.NotNil(t, comp.Timings)
	assert.Equal(t, 12, comp.Timings.PromptN, "the last snapshot wins")
}

// The tri-state cache counts are copied, so a caller cannot reach back through
// the returned Completion and change what the provider said.
func TestFoldClonesTriStatePointers(t *testing.T) {
	reported := intPtr(5)
	inner := &coreProvider{comp: &commonai.Completion{
		Usages: []Usage{{PromptTokens: 9, CacheReadTokens: reported}},
	}}
	comp, err := up(inner).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	require.NotNil(t, comp.Usage.CacheReadTokens)
	*comp.Usage.CacheReadTokens = 999
	assert.Equal(t, 5, *reported)
}

// A failed call that already streamed hands back its partial completion, and
// the fold must not turn that into a nil: the layers above read a non-nil
// completion as "this attempt streamed, do not re-send it".
func TestFoldKeepsAPartialCompletion(t *testing.T) {
	inner := &coreProvider{
		comp: &commonai.Completion{Message: NewMessage(RoleAssistant, TextPart{Text: "half"})},
		err:  &APIError{Status: 503},
	}
	comp, err := up(inner).Complete(context.Background(), Request{}, nil)
	require.Error(t, err)
	require.NotNil(t, comp)
	assert.Equal(t, "half", comp.Message.Content)

	failed := &coreProvider{err: &APIError{Status: 500}}
	comp, err = up(failed).Complete(context.Background(), Request{}, nil)
	require.Error(t, err)
	assert.Nil(t, comp, "nothing streamed, so there is no partial to keep")
}

// A provider built here goes back down to the format layer as itself, not as
// a stand-in derived from the folded numbers.
func TestAdaptersRoundTripWithoutASubstitute(t *testing.T) {
	inner := &coreProvider{comp: &commonai.Completion{}}
	assert.Same(t, commonai.Provider(inner), down(up(inner)))

	caller := &scriptProvider{steps: []scriptStep{{comp: &Completion{}}}}
	assert.Same(t, Provider(caller), up(down(caller)))
}

// A caller's own Provider only ever had the folded figure, so handing it to a
// format-level decorator reports that figure as the call's one report.
func TestDownAdapterReportsTheFoldedFigure(t *testing.T) {
	caller := &scriptProvider{steps: []scriptStep{{comp: &Completion{
		UsageReported: true,
		Usage:         Usage{PromptTokens: 3, CompletionTokens: 1, TotalTokens: 4},
		Timings:       &Timings{PromptN: 2},
	}}}}
	comp, err := down(caller).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	require.Len(t, comp.Usages, 1)
	assert.Equal(t, 4, comp.Usages[0].TotalTokens)
	require.Len(t, comp.Timings, 1)

	silent := &scriptProvider{steps: []scriptStep{{comp: &Completion{}}}}
	comp, err = down(silent).Complete(context.Background(), Request{}, nil)
	require.NoError(t, err)
	assert.Empty(t, comp.Usages, "a caller that reported nothing still reports nothing")
}

func TestConstructorsRequireABaseURL(t *testing.T) {
	_, err := NewOpenAIProvider(OpenAIConfig{})
	require.Error(t, err)
	assert.False(t, IsTransient(err), "a missing base URL cannot be fixed by trying again")

	_, err = NewAnthropicProvider(AnthropicConfig{})
	require.Error(t, err)
	_, err = NewResponsesProvider(ResponsesConfig{})
	require.Error(t, err)
}

// Retry is on without being asked for, and a call that fails transiently
// before streaming anything is ridden out rather than surfaced.
func TestConstructedProviderRetriesByDefault(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		if hits < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{
		BaseURL: srv.URL,
		Retry:   &RetryPolicy{MaxAttempts: 5, Sleep: func(context.Context, time.Duration) error { return nil }},
	}})
	require.NoError(t, err)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	assert.Equal(t, 3, hits)
}

func TestRateLimiterGatesTheProvidersRequests(t *testing.T) {
	var hits int
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		hits++
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"}}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{
		BaseURL:     srv.URL,
		RateLimiter: NewRateLimiter(120_000), // 0.5ms apart: gated, not slow
	}})
	require.NoError(t, err)
	for range 3 {
		_, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
		require.NoError(t, err)
	}
	assert.Equal(t, 3, hits)
}

// The stripper drops the parameter the upstream named and re-sends once, and
// the folded Completion still comes back on the other side.
func TestParamStripperRetriesWithoutTheRejectedParam(t *testing.T) {
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)

		bodies = append(bodies, string(body))
		if strings.Contains(string(body), "reasoning_effort") {
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"unsupported parameter: reasoning_effort"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	base, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{
		BaseURL: srv.URL,
		Retry:   &RetryPolicy{MaxAttempts: 1},
	}})
	require.NoError(t, err)

	p := NewParamStripper(base)
	req := Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}
	comp, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	require.Len(t, bodies, 2)
	assert.Contains(t, bodies[0], "reasoning_effort")
	assert.NotContains(t, bodies[1], "reasoning_effort")

	// The strip is remembered, so the next call does not have to fail first.
	_, err = p.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	require.Len(t, bodies, 3)
	assert.NotContains(t, bodies[2], "reasoning_effort")

	// The caller's own map is untouched.
	assert.Equal(t, "high", req.Extra["reasoning_effort"])
}

// scriptProvider is a caller-supplied Provider: it speaks this package's
// Completion and knows nothing about the format layer.
type scriptStep struct {
	comp *Completion
	err  error
}

type scriptProvider struct {
	steps []scriptStep
	reqs  []Request
}

func (p *scriptProvider) Complete(_ context.Context, req Request, _ *StreamEvents) (*Completion, error) {
	p.reqs = append(p.reqs, req)
	if len(p.steps) == 0 {
		return nil, errors.New("script exhausted")
	}
	step := p.steps[0]
	p.steps = p.steps[1:]
	return step.comp, step.err
}
