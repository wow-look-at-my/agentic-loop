package loop

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestGateNilAcquiresImmediately(t *testing.T) {
	var g *Gate
	release, err := g.Acquire(context.Background())
	require.NoError(t, err)
	release() // must be callable
}

func TestGateCapacity(t *testing.T) {
	g := NewGate(2)
	r1, err := g.Acquire(context.Background())
	require.NoError(t, err)
	r2, err := g.Acquire(context.Background())
	require.NoError(t, err, "capacity 2 admits two holders")

	// The acquire must block: prove it by cancelling.
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	r3, err := g.Acquire(ctx)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Nil(t, r3)

	r1()
	r4, err := g.Acquire(context.Background())
	require.NoError(t, err, "a released slot is reusable")
	r4()
	r2()
}

func TestGateClampsToOne(t *testing.T) {
	g := NewGate(0)
	r1, err := g.Acquire(context.Background())
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = g.Acquire(ctx)
	require.Error(t, err, "n < 1 clamps to a capacity-1 gate, not an unlimited one")
	r1()
}
