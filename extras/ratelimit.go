// ratelimit.go implements a fixed-rate request limiter, a provider-side
// throttle for staying under an upstream's per-minute request cap (e.g.
// Novita's requests/minute). A RateLimiter spaces request STARTS evenly --
// at most n per minute -- so a hard provider cap is never exceeded.
//
// The limiter belongs to the Provider, like retry: making a call happen
// without tripping the upstream's rate limit is "how the call gets made", the
// layer's job. It is wired in as an http.RoundTripper on the provider's
// client, so EVERY request it sends -- its transient-failure retries included,
// which ride the same client -- passes through the gate.
//
// Only request starts are counted. A call that has begun may take as long as
// it needs, so slow calls never push the average over the limit, and because
// consecutive starts are at least interval apart, no 60- window can
// contain more than the configured number of started requests.
package extras

import (
	"context"
	"net/http"
	"sync"
	"time"
)

// RateLimiter gates request starts to at most every interval -- a fixed-rate gate of minute / n for n per minute.
type RateLimiter struct {
	mu       sync.Mutex
	interval time.Duration
	// next is the earliest time the next request may start.
	next time.Time
	// now and sleep are seams for tests; nil means the real clock.
	now   func() time.Time
	sleep func(context.Context, time.Duration) error
}

// NewRateLimiter returns a RateLimiter permitting at most n request starts per minute, n clamped to >=.
func NewRateLimiter(n int) *RateLimiter {
	if n < 1 {
		n = 1
	}
	return &RateLimiter{interval: time.Minute / time.Duration(n)}
}

// Wait blocks until the next request may start, or until ctx is done (it then
// returns ctx.Err(), which the transport surfaces like any other canceled
// request). The caller is admitted immediately; each later caller waits
// until interval has passed since the previous start.
func (l *RateLimiter) Wait(ctx context.Context) error {
	l.mu.Lock()
	now := l.now
	if now == nil {
		now = time.Now
	}
	t := now()
	if l.next.IsZero() || !t.Before(l.next) {
		// The gate has opened: admit now; starting from the current time (not chasing missed slots) preserves the average.
		l.next = t.Add(l.interval)
		l.mu.Unlock()
		return nil
	}
	d := l.next.Sub(t)
	l.next = l.next.Add(l.interval)
	l.mu.Unlock()
	return l.sleepFor(ctx, d)
}

// sleepFor waits the interval via the injected seam or a context-aware timer.
func (l *RateLimiter) sleepFor(ctx context.Context, d time.Duration) error {
	if l.sleep != nil {
		return l.sleep(ctx, d)
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

// rateLimitedTransport waits on the limiter before delegating, so every request the http.Client sends is gated.
type rateLimitedTransport struct {
	base    http.RoundTripper
	limiter *RateLimiter
}

func (rt *rateLimitedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	if err := rt.limiter.Wait(req.Context()); err != nil {
		return nil, err
	}
	return rt.base.RoundTrip(req)
}

// RateLimitedClient returns a copy of base with a limiter-wrapped transport; a nil limiter returns base unchanged.
func RateLimitedClient(base *http.Client, limiter *RateLimiter) *http.Client {
	if limiter == nil {
		return base
	}
	if base == nil {
		base = http.DefaultClient
	}
	transport := base.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}
	out := *base
	out.Transport = &rateLimitedTransport{base: transport, limiter: limiter}
	return &out
}
