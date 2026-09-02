package loop

import "context"

// Gate is a semaphore bounding concurrent sub-agents; a nil *Gate imposes no limit.
type Gate struct {
	ch chan struct{}
}

// NewGate returns a Gate that permits at most n concurrent holders. n < is
// clamped to — a Gate always limits; use a nil Gate for "no limit".
func NewGate(n int) *Gate {
	if n < 1 {
		n = 1
	}
	return &Gate{ch: make(chan struct{}, n)}
}

// Acquire blocks until a slot is free or ctx is done. On success it returns a
// release func that must be called (typically deferred) to free the slot. A
// nil Gate acquires immediately.
func (g *Gate) Acquire(ctx context.Context) (release func(), err error) {
	if g == nil {
		return func() {}, nil
	}
	select {
	case g.ch <- struct{}{}:
		return func() { <-g.ch }, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}
