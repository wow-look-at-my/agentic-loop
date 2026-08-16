package client

import (
	"context"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
	"github.com/wow-look-at-my/agentic-loop/go/extras"
)

// Provider executes one streaming model call. Implementations stream under
// the hood and deliver deltas through ev (which may be nil). Build one with
// the wire dialect's constructor -- NewOpenAIProvider, NewAnthropicProvider or
// NewResponsesProvider.
//
// On a mid-stream failure or cancellation AFTER data has arrived -- including
// a stream callback returning an error -- Complete MUST return the PARTIAL
// *Completion alongside the error, both non-nil, so the caller can keep the
// partial content, reasoning, and the last usage snapshot. Before any data
// (connection errors, non-2xx responses), the completion MUST be nil.
//
// That rule is load-bearing, not just convenient: a non-nil completion is how
// the layers above tell that a failed call already streamed, and therefore
// that re-sending it would duplicate output the caller has seen. Retry and
// NewParamStripper both read it, and neither watches the callbacks to
// second-guess it. An implementation that emits deltas and then returns a nil
// completion will be re-sent.
//
// Providers must be safe for concurrent use by multiple goroutines.
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}

// upAdapter presents a format-level provider as a Go-level one, folding each
// call's usage reports on the way out. It keeps the wrapped provider so
// [down] can hand back the original rather than a re-derived stand-in.
type upAdapter struct{ inner commonai.Provider }

func (a *upAdapter) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	comp, err := a.inner.Complete(ctx, req, ev)
	return fold(comp), err
}

// downAdapter is the other direction, for wrapping a caller's own Provider in
// a decorator that speaks the format's Completion.
type downAdapter struct{ inner Provider }

func (a *downAdapter) Complete(ctx context.Context, req Request, ev *StreamEvents) (*commonai.Completion, error) {
	comp, err := a.inner.Complete(ctx, req, ev)
	return unfold(comp), err
}

// up presents a format-level provider as a Go-level one.
func up(p commonai.Provider) Provider {
	if a, ok := p.(*downAdapter); ok {
		return a.inner
	}
	return &upAdapter{inner: p}
}

// down presents a Go-level provider as a format-level one, unwrapping rather
// than re-deriving when it was one to begin with.
func down(p Provider) commonai.Provider {
	if a, ok := p.(*upAdapter); ok {
		return a.inner
	}
	return &downAdapter{inner: p}
}

// Unwrap is the format-level Provider behind p, for a caller handing it to
// something that speaks the format's own Completion -- a transport, whose
// documents have to say what the provider reported rather than what this
// package folded that into. A provider built here comes back as itself.
func Unwrap(p Provider) commonai.Provider { return down(p) }

// NewParamStripper wraps a Provider with rejected-parameter recovery: when a
// call fails before anything streamed and the error text names a parameter
// present in the request's Extra (matched by normalized form, so
// reasoningEffort matches reasoning_effort), that key is removed and the call
// is retried ONCE. The strip is remembered, so subsequent calls through the
// same stripper drop the key up front, without mutating the caller's Extra
// map. A context cancellation is never treated as a parameter problem, and a
// call that already streamed (a non-nil completion) is never retried.
func NewParamStripper(p Provider) Provider {
	return up(commonai.NewParamStripper(down(p)))
}

// Retrying gives p the same transient-failure retry behavior the dialect
// constructors apply, for a Provider a caller wrote themselves -- a mock, a
// router, a cache, a decorator. A nil policy means DefaultRetry.
//
// A call is re-sent only when the failed attempt streamed nothing, which the
// Provider contract states as a nil completion. That condition is why this is
// a wrapper and not something a caller loops by hand: a retry that misreads it
// duplicates output the caller has already seen.
func Retrying(p Provider, policy *RetryPolicy) Provider {
	return up(extras.Retrying(down(p), policy))
}
