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
// Nothing before it leaves the tree. A failure does not end the run: OnCompactionFailed reports it,
// and the run continues on the window it had.
// The host's stored row id lands on the summary, so the next assistant row hangs off it.
func compactNow(ctx context.Context, cfg *Config, req Request, s *session,
	res *Result, deduper *OutputDeduper) bool {
	cr, err := compactToFit(ctx, cfg.Provider, req, s.messages(), res)
	if err != nil {
		cfg.Events.emitCompactionFailed(CompactionFailedEvent{Err: err})
		return false
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

// compactToFit summarizes msgs. A transcript too large for one summary call is summarized
// in parts: the older half first, and its summary then stands in for that half.
func compactToFit(ctx context.Context, p Provider, req Request, msgs []Message, res *Result) (*CompactResult, error) {
	for {
		cr, err := Compact(ctx, p, Request{Model: req.Model, System: req.System, Messages: msgs})
		if err == nil {
			if cr.Completion != nil {
				res.Usages = append(res.Usages, cr.Completion.Usage)
			}
			return cr, nil
		}
		cut := splitPoint(msgs)
		if !IsContextOverflow(err) || cut <= 0 {
			return nil, err
		}
		older, err := compactToFit(ctx, p, req, msgs[:cut], res)
		if err != nil {
			return nil, err
		}
		msgs = append(append([]Message{}, older.Messages...), msgs[cut:]...)
	}
}

// splitPoint is the user message nearest the middle of msgs, so a tool call and its result stay
// on the same side. Zero means msgs has no such point.
func splitPoint(msgs []Message) int {
	best := 0
	for i := 1; i < len(msgs); i++ {
		if msgs[i].Role != RoleUser {
			continue
		}
		if best == 0 || abs(i-len(msgs)/2) < abs(best-len(msgs)/2) {
			best = i
		}
	}
	return best
}

func abs(n int) int {
	if n < 0 {
		return -n
	}
	return n
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
