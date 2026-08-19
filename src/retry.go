package extras

import (
	"context"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
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

// Attempts returns the effective attempt cap.
func (p RetryPolicy) Attempts() int {
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
	n := p.Attempts()
	for attempt := 1; ; attempt++ {
		err := fn()
		if err == nil {
			return nil
		}
		if attempt >= n || !commonai.IsTransient(err) {
			return err
		}
		if serr := p.sleep(ctx, p.delay(attempt)); serr != nil {
			return err
		}
	}
}

// retryComplete runs one model call with retry. A retry happens ONLY when the
// failed attempt streamed nothing — signalled by a nil completion, per the
// Provider contract — and the error is transient: once a delta reached the
// caller's sink, re-sending would duplicate it.
//
// Every retry is announced through StreamEvents.OnRetry BEFORE the backoff, so
// a waiting caller can show what failed and what is being waited on rather
// than sitting silent through minutes of backoff.
func retryComplete(ctx context.Context, p commonai.Provider, policy RetryPolicy, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	attempts := policy.Attempts()
	var comp *commonai.Completion
	var err error
	for attempt := 1; ; attempt++ {
		comp, err = p.Complete(ctx, req, ev)
		// comp != nil IS "this attempt streamed something": a Provider must
		// return the partial completion once data has arrived, so re-sending
		// would duplicate what the caller already saw.
		if err == nil || comp != nil || !commonai.IsTransient(err) || attempt >= attempts {
			break
		}
		delay := policy.delay(attempt)
		if cberr := ev.EmitRetry(commonai.RetryAttempt{
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
	inner  commonai.Provider
	policy RetryPolicy
}

// Retrying gives a provider the library's retry behavior, so a transient
// failure (408, 429, 5xx, transport errors) is re-attempted per the policy —
// but ONLY when the attempt streamed nothing, so a caller's sink never sees
// the same delta twice. Permanent failures — other 4xx, context overflow,
// cancellation, and errors the caller's own stream callbacks returned —
// surface immediately.
//
// A nil policy means DefaultRetry. A policy capped at one attempt returns the
// provider unwrapped: retry is off, and the wrapper would be pure overhead.
func Retrying(inner commonai.Provider, policy *RetryPolicy) commonai.Provider {
	resolved := DefaultRetry
	if policy != nil {
		resolved = *policy
	}
	if resolved.Attempts() <= 1 {
		return inner
	}
	return &retryingProvider{inner: inner, policy: resolved}
}

// Complete implements commonai.Provider.
func (r *retryingProvider) Complete(ctx context.Context, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	return retryComplete(ctx, r.inner, r.policy, req, ev)
}
