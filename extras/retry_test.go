package extras

import (
	"context"
	"errors"
	"testing"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRetryPolicyDo(t *testing.T) {
	t.Run("transient then success", func(t *testing.T) {
		var delays []time.Duration
		p := RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond,
			Sleep: func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			if calls < 3 {
				return &commonai.APIError{Status: 429, Body: "slow down"}
			}
			return nil
		})
		require.NoError(t, err)
		assert.Equal(t, 3, calls)
		assert.Equal(t, []time.Duration{500 * time.Millisecond, 1 * time.Second}, delays,
			"exponential backoff: base * 2^(attempt-1), no jitter")
	})

	t.Run("permanent error fails fast", func(t *testing.T) {
		calls := 0
		err := noSleep.Do(context.Background(), func() error {
			calls++
			return &commonai.APIError{Status: 400, Body: "bad request"}
		})
		require.Error(t, err)
		assert.Equal(t, 1, calls)
	})

	t.Run("attempts exhausted returns last error", func(t *testing.T) {
		p := RetryPolicy{MaxAttempts: 2, Sleep: noSleep.Sleep}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			return &commonai.APIError{Status: 503}
		})
		var ae *commonai.APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, 503, ae.Status)
		assert.Equal(t, 2, calls)
	})

	t.Run("cancelled sleep stops retrying", func(t *testing.T) {
		p := RetryPolicy{Sleep: func(context.Context, time.Duration) error { return context.Canceled }}
		calls := 0
		err := p.Do(context.Background(), func() error {
			calls++
			return &commonai.APIError{Status: 503, Body: "unavailable"}
		})
		var ae *commonai.APIError
		require.ErrorAs(t, err, &ae, "the fn error surfaces, not the sleep error")
		assert.Equal(t, 1, calls)
	})

	t.Run("zero value defaults", func(t *testing.T) {
		var p RetryPolicy
		assert.Equal(t, defaultAttempts, p.Attempts())
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
			{err: &commonai.APIError{Status: 503, Body: "unavailable"}},
			{err: &commonai.APIError{Status: 429, Body: "slow down"}},
			{comp: assistantComp("recovered")},
		}}
		var delays []time.Duration
		policy := RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond,
			Sleep: func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }}

		comp, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Equal(t, "recovered", comp.Message.Content)
		assert.Len(t, inner.reqs, 3)
		assert.Equal(t, []time.Duration{500 * time.Millisecond, 1 * time.Second}, delays,
			"exponential backoff: base * 2^(attempt-1), no jitter")
	})

	t.Run("every retry is announced before its backoff", func(t *testing.T) {
		// attempts of uncapped backoff is minutes of silence otherwise;
		// the caller has to be able to show what failed and what it is waiting
		// on. event per retry, none for the attempt that succeeds.
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 503, Body: "unavailable"}},
			{err: &commonai.APIError{Status: 429, Body: "slow down"}},
			{comp: assistantComp("recovered")},
		}}
		var seen []commonai.RetryAttempt
		ev := &commonai.StreamEvents{OnRetry: func(a commonai.RetryAttempt) error {
			seen = append(seen, a)
			return nil
		}}
		policy := RetryPolicy{MaxAttempts: 5, BaseDelay: time.Second, Sleep: noSleep.Sleep}

		_, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, ev)
		require.NoError(t, err)
		require.Len(t, seen, 2, "one per retry, none for the successful attempt")

		assert.Equal(t, 1, seen[0].Attempt)
		assert.Equal(t, 5, seen[0].Of)
		assert.Equal(t, time.Second, seen[0].Delay)
		var ae *commonai.APIError
		require.ErrorAs(t, seen[0].Err, &ae, "the caller can render why it failed")
		assert.Equal(t, 503, ae.Status)

		assert.Equal(t, 2, seen[1].Attempt)
		assert.Equal(t, 2*time.Second, seen[1].Delay, "the delay actually about to be waited")
	})

	t.Run("no retry event when nothing is retried", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{{err: &commonai.APIError{Status: 400}}}}
		fired := 0
		ev := &commonai.StreamEvents{OnRetry: func(commonai.RetryAttempt) error { fired++; return nil }}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, ev)
		require.Error(t, err)
		assert.Zero(t, fired, "a permanent failure is not a retry")
	})

	t.Run("OnRetry error stops the retrying", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 503}},
			{comp: assistantComp("never reached")},
		}}
		gaveUp := errors.New("client went away")
		ev := &commonai.StreamEvents{OnRetry: func(commonai.RetryAttempt) error { return gaveUp }}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, ev)
		require.ErrorIs(t, err, gaveUp, "the caller's error surfaces, not the upstream's")
		assert.False(t, commonai.IsTransient(err), "a callback error is never transient")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("permanent error surfaces immediately", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 400, Body: "bad request"}},
			{comp: assistantComp("never reached")},
		}}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("context overflow is never retried", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 400, Body: "prompt is too long", ContextOverflow: true}},
		}}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.True(t, commonai.IsContextOverflow(err), "the flag survives the decorator")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("partial completion is not retried", func(t *testing.T) {
		partial := &commonai.Completion{
			Message: commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "half"}),
		}
		inner := &scriptProvider{steps: []scriptStep{
			{comp: partial, err: &commonai.APIError{Status: 503}},
			{comp: assistantComp("never reached")},
		}}
		comp, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.Error(t, err)
		require.NotNil(t, comp)
		assert.Equal(t, "half", comp.Message.Content, "the partial rides alongside the error")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("a streamed delta blocks the retry", func(t *testing.T) {
		// a delta reached the sink, re-sending would duplicate it; the provider returns its partial completion.
		var got []string
		partial := &commonai.Completion{
			Message: commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "tok"}),
		}
		inner := &scriptProvider{steps: []scriptStep{
			{
				emit: func(ev *commonai.StreamEvents) { _ = ev.OnText("tok") },
				comp: partial,
				err:  &commonai.APIError{Status: 503},
			},
			{comp: assistantComp("never reached")},
		}}
		ev := &commonai.StreamEvents{OnText: func(s string) error { got = append(got, s); return nil }}

		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, ev)
		require.Error(t, err)
		assert.Len(t, inner.reqs, 1)
		assert.Equal(t, []string{"tok"}, got, "the caller's callbacks reach it unwrapped")
	})

	t.Run("attempts exhausted returns the last error", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 503, Body: "one"}},
			{err: &commonai.APIError{Status: 503, Body: "two"}},
		}}
		policy := RetryPolicy{MaxAttempts: 2, Sleep: noSleep.Sleep}
		_, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		var ae *commonai.APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, "two", ae.Body)
		assert.Len(t, inner.reqs, 2)
	})

	t.Run("cancelled sleep stops retrying", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{err: &commonai.APIError{Status: 503, Body: "unavailable"}},
			{comp: assistantComp("never reached")},
		}}
		policy := RetryPolicy{Sleep: func(context.Context, time.Duration) error { return context.Canceled }}
		_, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		var ae *commonai.APIError
		require.ErrorAs(t, err, &ae, "the call's error surfaces, not the sleep error")
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("one attempt means no wrapper at all", func(t *testing.T) {
		inner := &scriptProvider{}
		policy := RetryPolicy{MaxAttempts: 1}
		assert.Same(t, commonai.Provider(inner), Retrying(inner, &policy),
			"retry off is the provider itself, not a decorator that never retries")
	})

	t.Run("zero-value policy uses the DefaultRetry values", func(t *testing.T) {
		steps := make([]scriptStep, DefaultRetry.MaxAttempts)
		for i := range steps {
			steps[i] = scriptStep{err: &commonai.APIError{Status: 503}}
		}
		inner := &scriptProvider{steps: steps}
		var delays []time.Duration
		policy := RetryPolicy{Sleep: func(_ context.Context, d time.Duration) error {
			delays = append(delays, d)
			return nil
		}}
		_, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.Len(t, inner.reqs, DefaultRetry.MaxAttempts)
		// Uncapped doubling from the 500ms base, delay per retry.
		assert.Equal(t, []time.Duration{
			500 * time.Millisecond, 1 * time.Second, 2 * time.Second, 4 * time.Second,
			8 * time.Second, 16 * time.Second, 32 * time.Second, 64 * time.Second,
			128 * time.Second,
		}, delays)
	})

	t.Run("a nil policy is the default one", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		comp, err := Retrying(inner, nil).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Equal(t, "ok", comp.Message.Content)
	})
}

func TestRetryingProviderEmptyCompletion(t *testing.T) {
	t.Run("a genuinely empty completion retries", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{
			{comp: emptyComp()},
			{comp: emptyComp()},
			{comp: assistantComp("recovered")},
		}}
		var delays []time.Duration
		policy := RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond,
			Sleep: func(_ context.Context, d time.Duration) error { delays = append(delays, d); return nil }}

		comp, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Equal(t, "recovered", comp.Message.Content)
		assert.Len(t, inner.reqs, 3)
		assert.Equal(t, []time.Duration{500 * time.Millisecond, 1 * time.Second}, delays)
	})

	t.Run("the retry is announced with a synthetic error naming the cause", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{{comp: emptyComp()}, {comp: assistantComp("ok")}}}
		var seen []commonai.RetryAttempt
		ev := &commonai.StreamEvents{OnRetry: func(a commonai.RetryAttempt) error { seen = append(seen, a); return nil }}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, ev)
		require.NoError(t, err)
		require.Len(t, seen, 1)
		require.Error(t, seen[0].Err, "the caller can render why it retried, not just that it did")
	})

	t.Run("attempts exhausted on empty returns the empty completion, no error", func(t *testing.T) {
		inner := &scriptProvider{steps: []scriptStep{{comp: emptyComp()}, {comp: emptyComp()}}}
		policy := RetryPolicy{MaxAttempts: 2, Sleep: noSleep.Sleep}
		comp, err := Retrying(inner, &policy).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err, "an empty answer is not a transport failure")
		require.NotNil(t, comp)
		assert.Empty(t, comp.Message.Content)
		assert.Len(t, inner.reqs, 2)
	})

	t.Run("a tool-call-only completion is not empty and is not retried", func(t *testing.T) {
		toolOnly := &commonai.Completion{
			Message: commonai.NewMessage(commonai.RoleAssistant,
				commonai.ToolCallPart{ID: "c1", Name: "grep", Arguments: "{}"}),
		}
		inner := &scriptProvider{steps: []scriptStep{
			{comp: toolOnly},
			{comp: assistantComp("never reached")},
		}}
		comp, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Len(t, comp.Message.ToolCalls, 1)
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("a thinking-only completion is not empty and is not retried", func(t *testing.T) {
		thinkingOnly := &commonai.Completion{
			Message: commonai.NewMessage(commonai.RoleAssistant, commonai.ThinkingPart{Text: "hmm"}),
		}
		inner := &scriptProvider{steps: []scriptStep{
			{comp: thinkingOnly},
			{comp: assistantComp("never reached")},
		}}
		comp, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.NoError(t, err)
		assert.Equal(t, "hmm", comp.Message.Thinking[0].Text)
		assert.Len(t, inner.reqs, 1)
	})

	t.Run("an empty completion alongside an error is not double-counted as retryable", func(t *testing.T) {
		// A permanent error still wins even though the completion is also empty:
		// the error path is authoritative, and retrying it is already covered by IsTransient.
		inner := &scriptProvider{steps: []scriptStep{
			{comp: emptyComp(), err: &commonai.APIError{Status: 400, Body: "bad request"}},
			{comp: assistantComp("never reached")},
		}}
		_, err := Retrying(inner, &noSleep).
			Complete(context.Background(), commonai.Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.Len(t, inner.reqs, 1)
	})
}
