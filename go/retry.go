package agentic

import (
	"context"
	"time"
)

// RetryPolicy is exponential-backoff retry for transient failures. The
// zero-value fields default at use time to 4 attempts (1 try + 3 retries) and
// a 500ms base delay; the delay before retry n is BaseDelay × 2^(n−1) with no
// jitter and no cap. Sleep, when nil, uses a context-aware timer; inject it in
// tests to skip real waiting.
type RetryPolicy struct {
	MaxAttempts int
	BaseDelay   time.Duration
	Sleep       func(context.Context, time.Duration) error
}

// DefaultRetry is the default policy: 4 attempts, 500ms base delay.
var DefaultRetry = RetryPolicy{MaxAttempts: 4, BaseDelay: 500 * time.Millisecond}

// attempts returns the effective attempt cap.
func (p RetryPolicy) attempts() int {
	if p.MaxAttempts > 0 {
		return p.MaxAttempts
	}
	return 4
}

// base returns the effective base delay.
func (p RetryPolicy) base() time.Duration {
	if p.BaseDelay > 0 {
		return p.BaseDelay
	}
	return 500 * time.Millisecond
}

// delay is the backoff before retrying after the given 1-based attempt.
func (p RetryPolicy) delay(attempt int) time.Duration {
	return p.base() << (attempt - 1)
}

// sleep waits the given duration via the injected Sleep or a context-aware
// timer.
func (p RetryPolicy) sleep(ctx context.Context, d time.Duration) error {
	if p.Sleep != nil {
		return p.Sleep(ctx, d)
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Do runs fn up to MaxAttempts times, retrying only failures IsTransient
// reports as retryable. Permanent errors (other 4xx, context cancellation,
// context overflow) surface immediately; the final attempt's error surfaces
// regardless. A sleep interrupted by context cancellation stops retrying and
// returns the last fn error.
func (p RetryPolicy) Do(ctx context.Context, fn func() error) error {
	n := p.attempts()
	for attempt := 1; ; attempt++ {
		err := fn()
		if err == nil {
			return nil
		}
		if attempt >= n || !IsTransient(err) {
			return err
		}
		if serr := p.sleep(ctx, p.delay(attempt)); serr != nil {
			return err
		}
	}
}

// retryComplete runs one model call with retry. A retry happens ONLY when the
// failed attempt streamed nothing — no partial completion, no delivered stream
// event — and the error is transient: once a delta reached the caller's sink,
// re-sending would duplicate it.
func retryComplete(ctx context.Context, p Provider, policy RetryPolicy, req Request, ev *StreamEvents) (*Completion, error) {
	attempts := policy.attempts()
	if attempts <= 1 {
		// Retry is off: skip the probe wrapper, which would allocate a closure
		// per callback to answer a question nothing asks.
		return p.Complete(ctx, req, ev)
	}
	var comp *Completion
	var err error
	for attempt := 1; ; attempt++ {
		delivered := false
		comp, err = p.Complete(ctx, req, probeEvents(ev, &delivered))
		if err == nil || comp != nil || delivered || !IsTransient(err) || attempt >= attempts {
			break
		}
		if serr := policy.sleep(ctx, policy.delay(attempt)); serr != nil {
			break
		}
	}
	return comp, err
}

// retryingProvider is the Provider-decorator form of retryComplete.
type retryingProvider struct {
	inner  Provider
	policy RetryPolicy
}

// newProvider finishes a dialect implementation into the Provider callers
// hold: it gives it the library's standard retry behavior, so a transient
// failure (408, 429, 5xx, transport errors) is re-attempted per the policy —
// but ONLY when the attempt streamed nothing, so a caller's sink never sees
// the same delta twice. Permanent failures — other 4xx, context overflow,
// cancellation, and errors the caller's own stream callbacks returned —
// surface immediately.
//
// A nil policy means DefaultRetry, because retry is not something a caller
// opts into: a retry you have to remember to enable is one that silently
// isn't there. A policy capped at one attempt turns it off and returns the
// dialect provider unwrapped.
//
// Both dialect constructors end here, so every Provider the library builds
// retries. See ProviderConfig.Retry.
func newProvider(dialect Provider, policy *RetryPolicy) Provider {
	resolved := DefaultRetry
	if policy != nil {
		resolved = *policy
	}
	if resolved.attempts() <= 1 {
		return dialect
	}
	return &retryingProvider{inner: dialect, policy: resolved}
}

// Complete implements Provider.
func (r *retryingProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	return retryComplete(ctx, r.inner, r.policy, req, ev)
}
