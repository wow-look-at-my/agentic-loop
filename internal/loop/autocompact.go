package loop

import (
	"context"
	"errors"
	"fmt"
)

// resolveContextWindow fills Config.ContextWindow from the provider's model list when the host left it unset, and
// reports the window the run will compact against. A run that does not auto-compact asks nothing.
func resolveContextWindow(ctx context.Context, cfg *Config, req Request) {
	if req.AutoCompact <= 0 {
		return
	}
	if cfg.ContextWindow > 0 {
		cfg.Events.emitContextWindow(ContextWindowEvent{Window: cfg.ContextWindow})
		return
	}
	l, ok, err := ModelLimitsOf(ctx, cfg.Provider, req.Model)
	switch {
	case !ok:
		err = errors.New("agentic: no Config.ContextWindow, and the provider has no model list to read one from")
	case err != nil:
		err = fmt.Errorf("agentic: reading the context window for %q: %w", req.Model, err)
	case l.ContextWindow <= 0:
		err = fmt.Errorf("agentic: the model list publishes no context window for %q", req.Model)
	}
	if err != nil {
		cfg.Events.emitContextWindow(ContextWindowEvent{Err: err})
		return
	}
	cfg.ContextWindow = l.ContextWindow
	cfg.Events.emitContextWindow(ContextWindowEvent{Window: l.ContextWindow, FromModelList: true})
}

// compactHere summarizes the transcript when the last turn reached the AutoCompact
// fraction, returning the replacement and whether it replaced anything. A failure
// is non-fatal: the run continues on the transcript it already had.
//
// The host's stored row id lands on the summary message, so the next assistant row
// hangs off the summary instead of detaching from the conversation tree.
func compactHere(ctx context.Context, cfg *Config, req Request, transcript []Message,
	last *Completion, res *Result, deduper *OutputDeduper) ([]Message, bool) {
	if !shouldCompact(req, *cfg, last) {
		return nil, false
	}
	cr, err := Compact(ctx, cfg.Provider, Request{Model: req.Model, System: req.System, Messages: transcript})
	if err != nil {
		return nil, false
	}
	if cr.Completion != nil {
		res.Usages = append(res.Usages, cr.Completion.Usage)
	}
	if deduper != nil {
		deduper.Reset()
	}
	hostID := cfg.Events.emitCompaction(CompactionEvent{
		Summary:    cr.Summary,
		Messages:   cr.Messages,
		Completion: cr.Completion,
	})
	next := cr.Messages
	if hostID != "" && len(next) > 0 {
		next[len(next)-1].ID = string(hostID)
	}
	res.Messages = next
	return next, true
}

// shouldCompact auto-compacts when prompt tokens reach AutoCompact * ContextWindow.
func shouldCompact(req Request, cfg Config, comp *Completion) bool {
	if req.AutoCompact <= 0 || cfg.ContextWindow <= 0 || comp == nil {
		return false
	}
	if !comp.UsageReported {
		return false
	}
	threshold := int(req.AutoCompact * float64(cfg.ContextWindow))
	return comp.Usage.PromptTokens >= threshold
}
