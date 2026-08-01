package agentic

import (
	"context"
	"errors"
	"net"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIsTransient(t *testing.T) {
	cases := []struct {
		name string
		err  error
		want bool
	}{
		{"nil", nil, false},
		{"408", &APIError{Status: 408}, true},
		{"429", &APIError{Status: 429}, true},
		{"500", &APIError{Status: 500}, true},
		{"524", &APIError{Status: 524}, true},
		{"400", &APIError{Status: 400}, false},
		{"400 overflow", &APIError{Status: 400, ContextOverflow: true}, false},
		{"401", &APIError{Status: 401}, false},
		{"404", &APIError{Status: 404}, false},
		{"canceled", context.Canceled, false},
		{"deadline", context.DeadlineExceeded, false},
		{"wrapped canceled", &net.OpError{Op: "read", Err: context.Canceled}, false},
		{"network", errors.New("connection reset by peer"), true},
		{"wrapped api error", wrapErr(&APIError{Status: 503}), true},
		{"request error", badRequestErr("bad config"), false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, IsTransient(tc.err))
		})
	}
}

func wrapErr(err error) error {
	return &net.OpError{Op: "http", Err: err}
}

func TestIsContextOverflow(t *testing.T) {
	assert.True(t, IsContextOverflow(&APIError{Status: 400, ContextOverflow: true}))
	assert.False(t, IsContextOverflow(&APIError{Status: 400}))
	assert.False(t, IsContextOverflow(errors.New("nope")))
	assert.False(t, IsContextOverflow(nil))
}

func TestContextOverflowPatterns(t *testing.T) {
	positives := []string{
		"prompt is too long: 210000 tokens > 200000 maximum",
		"Prompt too long for this model",
		"This model's maximum context length is 8192 tokens",
		"context length exceeded",
		"the context window is full",
		"too many tokens in the request",
		"input exceeds the maximum context",
		"request exceeds token limit",
	}
	for _, s := range positives {
		assert.True(t, contextOverflowRe.MatchString(s), s)
	}
	negatives := []string{
		"invalid api key",
		"unsupported parameter: reasoning_effort",
		"model not found",
	}
	for _, s := range negatives {
		assert.False(t, contextOverflowRe.MatchString(s), s)
	}
}

func TestRetryPolicyDo(t *testing.T) {
	t.Run("transient then success", func(t *testing.T) {
		var delays []time.Duration
		p := RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond,
			Sleep: func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			if calls < 3 {
				return &APIError{Status: 429, Body: "slow down"}
			}
			return nil
		})
		require.NoError(t, err)
		assert.Equal(t, 3, calls)
		assert.Equal(t, []time.Duration{500 * time.Millisecond, 1 * time.Second}, delays,
			"exponential backoff: base * 2^(attempt-1), no jitter")
	})

	t.Run("permanent error fails fast", func(t *testing.T) {
		p := RetryPolicy{Sleep: func(context.Context, time.Duration) error { return nil }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			return &APIError{Status: 400, Body: "bad request"}
		})
		require.Error(t, err)
		assert.Equal(t, 1, calls)
	})

	t.Run("attempts exhausted returns last error", func(t *testing.T) {
		p := RetryPolicy{MaxAttempts: 2, Sleep: func(context.Context, time.Duration) error { return nil }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			return &APIError{Status: 503}
		})
		var ae *APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, 503, ae.Status)
		assert.Equal(t, 2, calls)
	})

	t.Run("cancelled sleep stops retrying", func(t *testing.T) {
		p := RetryPolicy{Sleep: func(context.Context, time.Duration) error { return context.Canceled }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			return &APIError{Status: 503, Body: "unavailable"}
		})
		var ae *APIError
		require.ErrorAs(t, err, &ae, "the fn error surfaces, not the sleep error")
		assert.Equal(t, 1, calls)
	})

	t.Run("zero value defaults", func(t *testing.T) {
		var p RetryPolicy
		assert.Equal(t, defaultAttempts, p.attempts())
		assert.Equal(t, 500*time.Millisecond, p.base())
		assert.Equal(t, 500*time.Millisecond, p.delay(1))
		assert.Equal(t, 2*time.Second, p.delay(3))
	})

	t.Run("default sleep honors context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		var p RetryPolicy
		err := p.sleep(ctx, time.Hour)
		assert.ErrorIs(t, err, context.Canceled)
	})
}

func TestDefaultRetryValues(t *testing.T) {
	assert.Equal(t, 10, DefaultRetry.MaxAttempts, "ten attempts, matching Claude Code")
	assert.Equal(t, 500*time.Millisecond, DefaultRetry.BaseDelay)
}

func TestRetryingProvider(t *testing.T) {
	t.Run("transient nothing-streamed failure retries", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503, Body: "unavailable"}},
			{err: &APIError{Status: 429, Body: "slow down"}},
			{comp: assistantComp("recovered")},
		}}
		var delays []time.Duration
		policy := RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond,
			Sleep: func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }}

		comp, err := newProvider(inner, &policy).
			Complete(context.Background(), Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Equal(t, "recovered", comp.Message.Content)
		assert.Len(t, inner.reqs, 3)
		assert.Equal(t, []time.Duration{500 * time.Millisecond, 1 * time.Second}, delays,
			"exponential backoff: base * 2^(attempt-1), no jitter")
	})

	t.Run("every retry is announced before its backoff", func(t *testing.T) {
		// Ten attempts of uncapped backoff is minutes of silence otherwise;
		// the caller has to be able to show what failed and what it is waiting
		// on. One event per retry, none for the attempt that succeeds.
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503, Body: "unavailable"}},
			{err: &APIError{Status: 429, Body: "slow down"}},
			{comp: assistantComp("recovered")},
		}}
		var seen []RetryAttempt
		ev := &StreamEvents{OnRetry: func(a RetryAttempt) error {
			seen = append(seen, a)
			return nil
		}}
		policy := RetryPolicy{MaxAttempts: 5, BaseDelay: time.Second,
			Sleep: func(context.Context, time.Duration) error { return nil }}

		_, err := newProvider(inner, &policy).
			Complete(context.Background(), Request{Model: "m"}, ev)
		require.NoError(t, err)
		require.Len(t, seen, 2, "one per retry, none for the successful attempt")

		assert.Equal(t, 1, seen[0].Attempt)
		assert.Equal(t, 5, seen[0].Of)
		assert.Equal(t, time.Second, seen[0].Delay)
		var ae *APIError
		require.ErrorAs(t, seen[0].Err, &ae, "the caller can render why it failed")
		assert.Equal(t, 503, ae.Status)

		assert.Equal(t, 2, seen[1].Attempt)
		assert.Equal(t, 2*time.Second, seen[1].Delay, "the delay actually about to be waited")
	})

	t.Run("no retry event when nothing is retried", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 400}}}}
		fired := 0
		ev := &StreamEvents{OnRetry: func(RetryAttempt) error { fired++; return nil }}
		_, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, ev)
		require.Error(t, err)
		assert.Zero(t, fired, "a permanent failure is not a retry")
	})

	t.Run("OnRetry error stops the retrying", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503}},
			{comp: assistantComp("never reached")},
		}}
		gaveUp := errors.New("client went away")
		ev := &StreamEvents{OnRetry: func(RetryAttempt) error { return gaveUp }}
		_, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, ev)
		require.ErrorIs(t, err, gaveUp, "the caller's error surfaces, not the upstream's")
		assert.False(t, IsTransient(err), "a callback error is never transient")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("an announced retry does not count as streamed", func(t *testing.T) {
		// OnRetry fires outside the probe, so notifying about attempt N cannot
		// make attempt N+1 look unsafe to send.
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503}},
			{err: &APIError{Status: 503}},
			{comp: assistantComp("recovered")},
		}}
		ev := &StreamEvents{OnRetry: func(RetryAttempt) error { return nil }}
		comp, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, ev)
		require.NoError(t, err)
		assert.Equal(t, "recovered", comp.Message.Content)
		assert.Len(t, inner.reqs, 3)
	})

	t.Run("permanent error surfaces immediately", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 400, Body: "bad request"}},
			{comp: assistantComp("never reached")},
		}}
		_, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("context overflow is never retried", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 400, Body: "prompt is too long", ContextOverflow: true}},
		}}
		_, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.True(t, IsContextOverflow(err), "the flag survives the decorator")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("partial completion is not retried", func(t *testing.T) {
		partial := &Completion{Message: Message{Role: RoleAssistant, Content: "half"}}
		inner := &scriptProvider{steps: []scriptStep{
			{comp: partial, err: &APIError{Status: 503}},
			{comp: assistantComp("never reached")},
		}}
		comp, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, nil)
		require.Error(t, err)
		require.NotNil(t, comp)
		assert.Equal(t, "half", comp.Message.Content, "the partial rides alongside the error")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("a streamed delta blocks the retry", func(t *testing.T) {
		// Once a delta reached the sink, re-sending would duplicate it — even
		// though the failure itself is transient. A provider that streamed
		// says so by returning its partial completion alongside the error.
		var got []string
		partial := &Completion{Message: Message{Role: RoleAssistant, Content: "tok"}}
		inner := &scriptProvider{steps: []scriptStep{
			{emit: func(ev *StreamEvents) { _ = ev.emitText("tok") }, comp: partial, err: &APIError{Status: 503}},
			{comp: assistantComp("never reached")},
		}}
		ev := &StreamEvents{OnText: func(s string) error { got = append(got, s); return nil }}

		_, err := newProvider(inner, &noSleep).
			Complete(context.Background(), Request{Model: "m"}, ev)
		require.Error(t, err)
		assert.Len(t, inner.reqs, 1)
		assert.Equal(t, []string{"tok"}, got, "the caller's callbacks reach it unwrapped")
	})

	t.Run("attempts exhausted returns the last error", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503, Body: "one"}},
			{err: &APIError{Status: 503, Body: "two"}},
		}}
		policy := RetryPolicy{MaxAttempts: 2, Sleep: func(context.Context, time.Duration) error { return nil }}
		_, err := newProvider(inner, &policy).
			Complete(context.Background(), Request{Model: "m"}, nil)
		var ae *APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, "two", ae.Body)
		assert.Len(t, inner.reqs, 2)
	})

	t.Run("cancelled sleep stops retrying", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &APIError{Status: 503, Body: "unavailable"}},
			{comp: assistantComp("never reached")},
		}}
		policy := RetryPolicy{Sleep: func(context.Context, time.Duration) error { return context.Canceled }}
		_, err := newProvider(inner, &policy).
			Complete(context.Background(), Request{Model: "m"}, nil)
		var ae *APIError
		require.ErrorAs(t, err, &ae, "the call's error surfaces, not the sleep error")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("zero-value policy uses the DefaultRetry values", func(t *testing.T) {
		steps := make([]scriptStep, DefaultRetry.MaxAttempts)
		for i := range steps {
			steps[i] = scriptStep{err: &APIError{Status: 503}}
		}
		inner := &scriptProvider{steps: steps}
		var delays []time.Duration
		policy := RetryPolicy{Sleep: func(_ context.Context, d time.Duration) error {
			delays = append(delays, d)
			return nil
		}}
		_, err := newProvider(inner, &policy).
			Complete(context.Background(), Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.Len(t, inner.reqs, DefaultRetry.MaxAttempts)
		// Uncapped doubling from the 500ms base, one delay per retry.
		assert.Equal(t, []time.Duration{
			500 * time.Millisecond, 1 * time.Second, 2 * time.Second, 4 * time.Second,
			8 * time.Second, 16 * time.Second, 32 * time.Second, 64 * time.Second,
			128 * time.Second,
		}, delays)
	})
}
