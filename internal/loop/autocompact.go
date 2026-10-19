package loop

import (
	"context"
)

// compactHere compacts when the last turn reached the AutoCompact fraction of the window.
func compactHere(ctx context.Context, cfg *Config, req Request, s *session,
	last *Completion, res *Result, deduper *OutputDeduper) bool {
	if !shouldCompact(req, *cfg, last) {
		return false
	}
	return compactNow(ctx, cfg, req, s, res, deduper)
}

// compactNow summarizes the window and appends the summary, which starts the window from then on.
// Nothing before it leaves the tree. A failure is non-fatal: the run continues on the window it had.
// The host's stored row id lands on the summary, so the next assistant row hangs off it.
func compactNow(ctx context.Context, cfg *Config, req Request, s *session,
	res *Result, deduper *OutputDeduper) bool {
	cr, err := Compact(ctx, cfg.Provider, Request{Model: req.Model, System: req.System, Messages: s.messages()})
	if err != nil {
		return false
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
	s.compact(cr.Summary, string(hostID))
	res.Messages = s.messages()
	return true
}

// shouldCompact auto-compacts when prompt tokens reach AutoCompact times the run's window.
func shouldCompact(req Request, cfg Config, comp *Completion) bool {
	if req.AutoCompact <= 0 || cfg.window <= 0 || comp == nil {
		return false
	}
	if !comp.UsageReported {
		return false
	}
	threshold := int(req.AutoCompact * float64(cfg.window))
	return comp.Usage.PromptTokens >= threshold
}
