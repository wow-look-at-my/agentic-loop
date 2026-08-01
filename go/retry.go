package agentic

import (
	"context"
	"time"
)

// RetryPolicy is exponential-backoff retry for transient failures. The
// zero-value fields default at use time to 10 attempts (1 try + 9 retries)
// and a 500ms base delay; the delay before retry n is BaseDelay × 2^(n−1)
// with no jitter and no cap. Sleep, when nil, uses a context-aware timer;
// inject it in tests to skip real waiting.
type RetryPolicy struct {
	MaxAttempts int
	BaseDelay   time.Duration
	Sleep       func(context.Context, time.Duration) error
}

// defaultAttempts is the attempt cap applied when a policy does not set one.
// Ten, matching Claude Code: a transient upstream should be ridden out, not
// surfaced to the user as a failed turn after three tries.
const defaultAttempts = 10

// DefaultRetry is the default policy: 10 attempts, 500ms base delay.
var DefaultRetry = RetryPolicy{MaxAttempts: defaultAttempts, BaseDelay: 500 * time.Millisecond}

// attempts returns the effective attempt cap.
func (p RetryPolicy) attempts() int {
	if p.MaxAttempts > 0 {
		return p.MaxAttempts
	}
	return defaultAttempts
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
//
// Every retry is announced through StreamEvents.OnRetry BEFORE the backoff, so
// a waiting caller can show what failed and what is being waited on rather
// than sitting silent through minutes of backoff.
func retryComplete(ctx context.Context, p Provider, policy RetryPolicy, req Request, ev *StreamEvents) (*Completion, error) {
	attempts := policy.attempts()
	var comp *Completion
	var err error
	for attempt := 1; ; attempt++ {
		delivered := false
		comp, err = p.Complete(ctx, req, probeEvents(ev, &delivered))
		if err == nil || comp != nil || delivered || !IsTransient(err) || attempt >= attempts {
			break
		}
		delay := policy.delay(attempt)
		if cberr := ev.emitRetry(RetryAttempt{
			Attempt: attempt, Of: attempts, Delay: delay, Err: err,
		}); cberr != nil {
			// The caller pulled the plug on retrying (a dead sink, a UI that
			// gave up). Surface their error, not the upstream's.
			return comp, cberr
		}
		if serr := policy.sleep(ctx, delay); serr != nil {
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
// hold: it gives it the library's retry behavior, so a transient failure
// (408, 429, 5xx, transport errors) is re-attempted per the policy — but ONLY
// when the attempt streamed nothing, so a caller's sink never sees the same
// delta twice. Permanent failures — other 4xx, context overflow,
// cancellation, and errors the caller's own stream callbacks returned —
// surface immediately.
//
// A nil policy means DefaultRetry. A policy capped at one attempt returns the
// dialect provider unwrapped: retry is off, and the probe wrapper would be
// pure overhead.
//
// Both dialect constructors end here, so retry is not something a caller opts
// into: a retry you have to remember to enable is one that silently isn't
// there. ProviderConfig.Retry is the library's ONE retry knob.
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
