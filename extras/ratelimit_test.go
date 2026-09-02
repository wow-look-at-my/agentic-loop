// ratelimit_test.go pins the fixed-rate limiter and its transport wrapper:
// request starts are spaced at least interval apart (so n per minute is a
// hard ceiling), the caller is admitted immediately, a slow previous
// call lets the gate catch up instead of blocking forever, context
// cancellation aborts a wait, and the transport gates every request it
// forwards. All timing is injected (a fake clock), so the suite is hermetic
// and fast.
package extras

import (
	"context"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeClock is a mutable time source for the limiter's injected now/sleep
// seams.
type fakeClock struct {
	mu  sync.Mutex
	now time.Time
	// waits records the durations the limiter asked to sleep, in order.
	waits []time.Duration
}

func newFakeClock(t0 time.Time) *fakeClock { return &fakeClock{now: t0} }

func (c *fakeClock) current() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.now
}

// advance moves the clock forward by d.
func (c *fakeClock) advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.now = c.now.Add(d)
}

// sleep is the injected sleep: it records the requested wait and advances the
// clock by it, standing in for the real timer.
func (c *fakeClock) sleep(_ context.Context, d time.Duration) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.waits = append(c.waits, d)
	c.now = c.now.Add(d)
	return nil
}

// rateLimiter builds a RateLimiter over the fake clock with the given
// interval.
func rateLimiter(clock *fakeClock, interval time.Duration) *RateLimiter {
	return &RateLimiter{
		interval: interval,
		now:      clock.current,
		sleep:    clock.sleep,
	}
}

func TestNewRateLimiterClampsSubOne(t *testing.T) {
	require.NotPanics(t, func() { NewRateLimiter(0) })
	require.NotPanics(t, func() { NewRateLimiter(-3) })
	// NewRateLimiter() spaces requests 5ms apart ( minute /).
	require.Equal(t, 5*time.Millisecond, NewRateLimiter(12000).interval)
}

func TestRateLimiterSpacesStartsAtLeastOneInterval(t *testing.T) {
	clock := newFakeClock(time.Unix(0, 0))
	l := rateLimiter(clock, 10*time.Millisecond)

	// request starts, as fast as the gate allows.
	started := make([]time.Time, 0, 5)
	for i := 0; i < 5; i++ {
		require.NoError(t, l.Wait(context.Background()))
		started = append(started, clock.current())
	}
	// The starts immediately; every following start is >= interval after the previous.
	assert.Equal(t, time.Unix(0, 0), started[0])
	for i := 1; i < len(started); i++ {
		assert.GreaterOrEqual(t, started[i].Sub(started[i-1]), 10*time.Millisecond)
	}
	// of the waited (the was admitted instantly).
	require.Len(t, clock.waits, 4)
}

func TestRateLimiterCatchUpAfterSlowCall(t *testing.T) {
	clock := newFakeClock(time.Unix(0, 0))
	l := rateLimiter(clock, 10*time.Millisecond)

	require.NoError(t, l.Wait(context.Background())) // start at t=
	require.NoError(t, l.Wait(context.Background())) // waits 10ms, start at t=10ms

	// The call is slow; the next start is admitted immediately rather than held to a missed schedule.
	clock.advance(90 * time.Millisecond)
	require.NoError(t, l.Wait(context.Background()))
	assert.Equal(t, time.Unix(0, 0).Add(100*time.Millisecond), clock.current())
	require.Len(t, clock.waits, 1)
}

func TestRateLimiterWaitHonorsContextCancellation(t *testing.T) {
	clock := newFakeClock(time.Unix(0, 0))
	// An injected sleep that actually honors ctx, standing in for the real
	// context-aware timer.
	l := &RateLimiter{
		interval: 10 * time.Millisecond,
		now:      clock.current,
		sleep: func(ctx context.Context, _ time.Duration) error {
			select {
			case <-ctx.Done():
				return ctx.Err()
			default:
				return nil
			}
		},
	}
	require.NoError(t, l.Wait(context.Background())) // gate closes until t=10ms

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	err := l.Wait(ctx)
	require.Error(t, err)
	assert.True(t, errors.Is(err, context.Canceled))
}

// countingTransport is a roundTripper stub that counts the requests it
// forwards.
type countingTransport struct {
	mu    sync.Mutex
	calls int
}

func (c *countingTransport) RoundTrip(_ *http.Request) (*http.Response, error) {
	c.mu.Lock()
	c.calls++
	c.mu.Unlock()
	return &http.Response{StatusCode: http.StatusOK, Request: &http.Request{}}, nil
}

func TestRateLimitedTransportGatesEveryRequest(t *testing.T) {
	clock := newFakeClock(time.Unix(0, 0))
	l := rateLimiter(clock, 5*time.Millisecond)
	base := &countingTransport{}
	rt := &rateLimitedTransport{base: base, limiter: l}

	client := &http.Client{Transport: rt}
	for i := 0; i < 3; i++ {
		req, err := http.NewRequest(http.MethodPost, "https://example.invalid/chat/completions", nil)
		require.NoError(t, err)
		resp, err := client.Do(req)
		require.NoError(t, err)
		require.Equal(t, http.StatusOK, resp.StatusCode)
	}
	// All reached the base transport...
	base.mu.Lock()
	assert.Equal(t, 3, base.calls)
	base.mu.Unlock()
	//...but the and waited interval each before being let through.
	require.Len(t, clock.waits, 2)
	for _, d := range clock.waits {
		assert.Equal(t, 5*time.Millisecond, d)
	}
}

func TestRateLimitedClientWrapsTheBaseTransport(t *testing.T) {
	clock := newFakeClock(time.Unix(0, 0))
	l := rateLimiter(clock, 5*time.Millisecond)
	base := &http.Client{Timeout: time.Minute}

	client := RateLimitedClient(base, l)
	require.NotNil(t, client)
	// The wrap is additive: the caller's other settings survive.
	assert.Equal(t, time.Minute, client.Timeout)
	rt, ok := client.Transport.(*rateLimitedTransport)
	require.True(t, ok)
	assert.Same(t, l, rt.limiter)
	assert.NotNil(t, rt.base)

	// A nil client wraps the default client's transport; a nil transport wraps the default transport.
	def := RateLimitedClient(nil, l)
	rt, ok = def.Transport.(*rateLimitedTransport)
	require.True(t, ok)
	require.NotNil(t, rt.base)

	noTransport := RateLimitedClient(&http.Client{}, l)
	rt, ok = noTransport.Transport.(*rateLimitedTransport)
	require.True(t, ok)
	require.NotNil(t, rt.base)

	// A nil limiter leaves the base client untouched (the caller that wants no limiter passes none).
	require.Same(t, base, RateLimitedClient(base, nil))
	require.Nil(t, RateLimitedClient(nil, nil))
}
