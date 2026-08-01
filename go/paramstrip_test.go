package agentic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRejectedParamName(t *testing.T) {
	cases := []struct {
		body string
		want string
	}{
		{"Model grok-build-0.1 does not support parameter reasoningEffort.", "reasoningEffort"},
		{`Unsupported parameter: 'reasoning_effort' is not supported with this model.`, "reasoning_effort"},
		{`unknown field "reasoning_effort"`, "reasoning_effort"},
		{"unrecognized request argument: reasoning_effort", "reasoning_effort"},
		{"unsupported parameter `top_k`.", "top_k"},
	}
	for _, tc := range cases {
		name, ok := rejectedParamName(tc.body)
		require.True(t, ok, tc.body)
		assert.Equal(t, tc.want, name, tc.body)
	}

	_, ok := rejectedParamName("some other 400 entirely")
	assert.False(t, ok)
}

func TestNormalizeParamName(t *testing.T) {
	assert.Equal(t, "reasoningeffort", normalizeParamName("reasoningEffort"))
	assert.Equal(t, "reasoningeffort", normalizeParamName("reasoning_effort"))
}

func TestParamStripperStripsAndRetries(t *testing.T) {
	inner := &scriptProvider{steps: []scriptStep{
		{err: &APIError{Status: 400, Body: "Model x does not support parameter reasoningEffort."}},
		{comp: assistantComp("ok")},
		{comp: assistantComp("ok again")},
	}}
	s := NewParamStripper(inner)
	req := Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high", "num_ctx": 4096}}

	comp, err := s.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	require.Len(t, inner.reqs, 2, "retried exactly once")
	assert.Contains(t, inner.reqs[0].Extra, "reasoning_effort")
	assert.NotContains(t, inner.reqs[1].Extra, "reasoning_effort", "the camelCase report matched the snake_case key")
	assert.Contains(t, inner.reqs[1].Extra, "num_ctx", "only the rejected key is dropped")

	// The strip is remembered: the next call drops the key up front, with a
	// single inner call.
	_, err = s.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	require.Len(t, inner.reqs, 3)
	assert.NotContains(t, inner.reqs[2].Extra, "reasoning_effort")

	// The caller's own map was never mutated.
	assert.Contains(t, req.Extra, "reasoning_effort")
}

func TestParamStripperForwardsRetryEvents(t *testing.T) {
	// The stripper is composed ABOVE the provider, so its delivery probe sits
	// between the caller's events and the retry layer. Rebuilding the events
	// without OnRetry silently swallowed every retry notification — the caller
	// saw a stream that hung for the whole backoff with no explanation.
	inner := &scriptProvider{steps: []scriptStep{
		{err: &APIError{Status: 503, Body: "unavailable"}},
		{comp: assistantComp("recovered")},
	}}
	var seen []RetryAttempt
	ev := &StreamEvents{OnRetry: func(a RetryAttempt) error {
		seen = append(seen, a)
		return nil
	}}

	p := NewParamStripper(newProvider(inner, &noSleep))
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, ev)
	require.NoError(t, err)
	assert.Equal(t, "recovered", comp.Message.Content)
	require.Len(t, seen, 1, "the retry reaches the caller through the stripper's probe")
	assert.Equal(t, 1, seen[0].Attempt)

	// And it still does not count as delivery: the call was re-sent, which a
	// "streamed something" verdict would have prevented.
	assert.Len(t, inner.reqs, 2)
}

func TestParamStripperNoMatchNoRetry(t *testing.T) {
	apiErr := &APIError{Status: 400, Body: "unsupported parameter: something_else"}
	inner := &scriptProvider{steps: []scriptStep{{err: apiErr}}}
	s := NewParamStripper(inner)
	_, err := s.Complete(context.Background(), Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, apiErr)
	assert.Len(t, inner.reqs, 1, "an error naming a param not in Extra never retries")
}

func TestParamStripperUnparseableErrorNoRetry(t *testing.T) {
	apiErr := &APIError{Status: 400, Body: "totally different failure"}
	inner := &scriptProvider{steps: []scriptStep{{err: apiErr}}}
	s := NewParamStripper(inner)
	_, err := s.Complete(context.Background(), Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}, nil)
	require.Error(t, err)
	assert.Len(t, inner.reqs, 1)
}

func TestParamStripperNeverOnCancel(t *testing.T) {
	inner := &scriptProvider{steps: []scriptStep{{err: context.Canceled}}}
	s := NewParamStripper(inner)
	_, err := s.Complete(context.Background(), Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}, nil)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Len(t, inner.reqs, 1)
}

func TestParamStripperNeverAfterDelivery(t *testing.T) {
	apiErr := &APIError{Status: 400, Body: "unsupported parameter: reasoning_effort"}
	// A provider that streamed returns its partial completion with the error
	// (Provider contract) — that is what marks the call unsafe to re-send.
	partial := &Completion{Message: Message{Role: RoleAssistant, Content: "half a token"}}
	inner := &scriptProvider{steps: []scriptStep{
		{comp: partial, err: apiErr, emit: func(ev *StreamEvents) { _ = ev.emitText("half a token") }},
	}}
	s := NewParamStripper(inner)
	_, err := s.Complete(context.Background(), Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}, nil)
	require.Error(t, err)
	assert.Len(t, inner.reqs, 1, "a call that already streamed is never re-sent")
}

func TestParamStripperSecondFailureSurfaces(t *testing.T) {
	first := &APIError{Status: 400, Body: "unsupported parameter: reasoning_effort"}
	second := &APIError{Status: 400, Body: "still broken"}
	inner := &scriptProvider{steps: []scriptStep{{err: first}, {err: second}}}
	s := NewParamStripper(inner)
	_, err := s.Complete(context.Background(), Request{Model: "m", Extra: map[string]any{"reasoning_effort": "high"}}, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, second, "at most one strip-retry; the second failure surfaces")
	assert.Len(t, inner.reqs, 2)
}

func TestParamStripperOverOpenAIProvider(t *testing.T) {
	hits := 0
	var secondBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits == 1 {
			http.Error(w, `{"error":{"message":"Model grok-x does not support parameter reasoningEffort."}}`, http.StatusBadRequest)
			return
		}
		var buf [8192]byte
		n, _ := r.Body.Read(buf[:])
		secondBody = append([]byte(nil), buf[:n]...)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}` + "\n\n" + "data: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := NewParamStripper(oaProvider(t, srv.URL))
	comp, err := p.Complete(context.Background(), Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "q"}},
		Extra:    map[string]any{"reasoning_effort": "high"},
	}, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	assert.Equal(t, 2, hits)
	assert.NotContains(t, string(secondBody), "reasoning_effort")
}
