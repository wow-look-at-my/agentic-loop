package client

import (
	"context"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/extras"
)

// Provider executes streaming call; on mid-stream failure return the partial *Completion with the error.
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}

// upAdapter presents a format-level provider as a Go-level, folding usage on the way out.
type upAdapter struct{ inner commonai.Provider }

func (a *upAdapter) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	comp, err := a.inner.Complete(ctx, req, ev)
	return fold(comp), err
}

// downAdapter wraps a caller's own Provider in a decorator speaking the format's Completion.
type downAdapter struct{ inner Provider }

func (a *downAdapter) Complete(ctx context.Context, req Request, ev *StreamEvents) (*commonai.Completion, error) {
	comp, err := a.inner.Complete(ctx, req, ev)
	return unfold(comp), err
}

// up presents a format-level provider as a Go-level.
func up(p commonai.Provider) Provider {
	if a, ok := p.(*downAdapter); ok {
		return a.inner
	}
	return &upAdapter{inner: p}
}

// down presents a Go-level provider as a format-level, unwrapping rather
// than re-deriving when it was to begin with.
func down(p Provider) commonai.Provider {
	if a, ok := p.(*upAdapter); ok {
		return a.inner
	}
	return &downAdapter{inner: p}
}

// Unwrap is the format-level Provider behind p, for a caller handing it to something speaking the format's Completion.
func Unwrap(p Provider) commonai.Provider { return down(p) }

// NewParamStripper retries a failed-before-streaming call, removing a named parameter from Extra.
func NewParamStripper(p Provider) Provider {
	return up(commonai.NewParamStripper(down(p)))
}

// Retrying gives a caller-written Provider retry; re-sends only when nothing streamed, nil policy means DefaultRetry.
func Retrying(p Provider, policy *RetryPolicy) Provider {
	return up(extras.Retrying(down(p), policy))
}
