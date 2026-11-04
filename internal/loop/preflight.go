// preflight.go answers a tool call from its streamed arguments, before the rest arrives.

package loop

import (
	"context"
	"errors"
)

// Preflighter is a Tool that can refuse a call before its arguments finish
// arriving. Run asks it on every argument delta until it decides.
type Preflighter interface {
	Tool
	// Preflight judges a call from the arguments streamed so far.
	Preflight(ctx context.Context, partial string) (refusal *ToolResult, decided bool)
}

// PreflightFunc is a Preflight as a function.
type PreflightFunc func(ctx context.Context, partial string) (refusal *ToolResult, decided bool)

// WithPreflight gives t a Preflight. The tool keeps its declaration and Execute.
func WithPreflight(t Tool, fn PreflightFunc) Tool {
	return preflightTool{Tool: t, fn: fn}
}

type preflightTool struct {
	Tool
	fn PreflightFunc
}

func (t preflightTool) Preflight(ctx context.Context, partial string) (*ToolResult, bool) {
	return t.fn(ctx, partial)
}

// errPreflightStop is the sink error that ends a model call on a refusal.
var errPreflightStop = errors.New("agentic: a tool refused the call from its streamed arguments")

// preflightFor is the sink's judge for one call: the turn's advertised tools, by name. A turn that
// advertises nothing never asks, since no call of it runs.
func preflightFor(tools Tools, advertised []ToolDecl) preflightFn {
	if len(advertised) == 0 {
		return nil
	}
	return func(ctx context.Context, name, partial string) (*ToolResult, bool) {
		if name == "" {
			// The name has not arrived; the next delta may carry it.
			return nil, false
		}
		t, ok := tools.Find(name)
		if !ok {
			return nil, true
		}
		pf, ok := t.(Preflighter)
		if !ok {
			return nil, true
		}
		return pf.Preflight(ctx, partial)
	}
}
