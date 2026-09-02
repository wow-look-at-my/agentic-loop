package client

import (
	"encoding/json"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// model call's outcome: message, folded usage, stop reason; usage fields are tri-state, never zero-filled.
type Completion struct {
	Message         Message
	Usage           Usage
	UsageReported   bool
	Timings         *Timings
	RawUsage        json.RawMessage
	ReasoningTokens *int
	CostUsd         *float64
	Streamed        bool
	StopReason      string
}

// fold turns what the provider SAID into the single figure a caller bills
// against. Core keeps every usage report in the order it arrived, because that
// is the honest record of the call; which to believe is a policy, and this
// is it: the newest snapshot with at least as much evidence wins, snapshots are
// never summed, and the total is floored at prompt+completion.
func fold(c *commonai.Completion) *Completion {
	if c == nil {
		return nil
	}
	out := &Completion{
		Message:    c.Message,
		Streamed:   c.Streamed,
		StopReason: c.StopReason,
	}
	if n := len(c.Timings); n > 0 {
		t := c.Timings[n-1]
		out.Timings = &t
	}
	var merged *Usage
	for i := range c.Usages {
		u := c.Usages[i]
		merged = mergeUsage(merged, &u)
	}
	if merged == nil {
		return out
	}
	out.UsageReported = true
	out.Usage = floorTotal(Usage{
		PromptTokens:     merged.PromptTokens,
		CompletionTokens: merged.CompletionTokens,
		TotalTokens:      merged.TotalTokens,
		CacheReadTokens:  clonePtr(merged.CacheReadTokens),
		CacheWriteTokens: clonePtr(merged.CacheWriteTokens),
	})
	out.RawUsage = merged.Raw
	out.ReasoningTokens = clonePtr(merged.ReasoningTokens)
	if merged.CostUsd != nil {
		v := *merged.CostUsd
		out.CostUsd = &v
	}
	return out
}

// unfold is fold's inverse, for handing a caller-supplied Provider back down to
// a layer that speaks the format's own Completion. It reports the folded figure
// as the call's single usage report, which is all such a provider ever had: the
// individual snapshots were already gone before this package saw it.
func unfold(c *Completion) *commonai.Completion {
	if c == nil {
		return nil
	}
	out := &commonai.Completion{
		Message:    c.Message,
		Streamed:   c.Streamed,
		StopReason: c.StopReason,
	}
	if hasUsage(c) {
		u := c.Usage
		u.Raw = c.RawUsage
		u.ReasoningTokens = clonePtr(c.ReasoningTokens)
		u.CostUsd = c.CostUsd
		out.Usages = []Usage{u}
	}
	if c.Timings != nil {
		out.Timings = []Timings{*c.Timings}
	}
	return out
}

// hasUsage reports whether a completion carries anything a usage report would say.
func hasUsage(c *Completion) bool {
	return c.UsageReported ||
		usageEvidence(&c.Usage) > 0 ||
		c.Usage.CacheReadTokens != nil || c.Usage.CacheWriteTokens != nil ||
		len(c.RawUsage) > 0 || c.ReasoningTokens != nil || c.CostUsd != nil
}

// mergeUsage folds a streamed snapshot in: newest wins, snapshots never summed, regressing ones discarded.
func mergeUsage(prev, next *Usage) *Usage {
	if next == nil {
		return prev
	}
	if prev == nil || usageEvidence(next) >= usageEvidence(prev) {
		return next
	}
	return prev
}

// usageEvidence is the comparable size of a snapshot: max of total_tokens and prompt+completion.
func usageEvidence(u *Usage) int {
	e := u.PromptTokens + u.CompletionTokens
	if u.TotalTokens > e {
		e = u.TotalTokens
	}
	return e
}

// floorTotal floors TotalTokens at prompt+completion, preserving any genuine surplus.
func floorTotal(u Usage) Usage {
	if pc := u.PromptTokens + u.CompletionTokens; u.TotalTokens < pc {
		u.TotalTokens = pc
	}
	return u
}

// clonePtr copies a tri-state int pointer so snapshots never alias.
func clonePtr(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
