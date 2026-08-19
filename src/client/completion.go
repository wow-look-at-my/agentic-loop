package client

import (
	"encoding/json"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
)

// Completion is the outcome of one model call: the assembled assistant
// message, the call's final (merged, total-floored) usage, and the normalized
// stop reason.
//
// UsageReported is true iff the provider reported at least one usage snapshot
// during the call. Usage is a value type, so a caller reading only the
// returned Completion could not otherwise distinguish an upstream that
// reported all-zero usage from one that reported none at all (common on local
// OpenAI-compatible servers) -- check UsageReported before persisting or
// displaying Usage.
//
// Timings is the last provider-reported timings snapshot (llama.cpp-style
// upstreams attach one per chunk; the last one wins), or nil when the
// provider never reported timings -- a tri-state like the Usage cache fields.
// The Anthropic dialect never sets it.
//
// RawUsage is the provider's usage object verbatim (the raw JSON the upstream
// sent, or, for the Anthropic dialect where usage arrives as fragments, the
// merged wire-shaped object), for logging and for extracting provider extras --
// reasoning-token and dollar-cost figures -- the normalized Usage drops.
// ReasoningTokens is the openai
// usage.completion_tokens_details.reasoning_tokens figure when present
// (tri-state; Anthropic never reports one). CostUsd is the provider-reported
// dollar cost from usage.cost or usage.estimated_cost when present
// (OpenRouter/Requesty/DeepInfra style; tri-state). Each stays nil unless the
// upstream reported it -- never zero-fill or estimate.
//
// Streamed records whether the response actually arrived as an SSE stream. A
// server that ignored stream:true and answered with one JSON body is accepted
// transparently, and this is how a caller can tell.
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
// is the honest record of the call; which one to believe is a policy, and this
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

// hasUsage reports whether a completion carries anything a usage report would
// say. UsageReported is the answer when it is set, and the rest is for a
// caller-supplied Provider that filled in the numbers and not the flag: the
// flag exists to separate reported zeros from reported nothing, and dropping
// counts because the flag disagrees with them would lose real numbers to a
// bookkeeping field.
func hasUsage(c *Completion) bool {
	return c.UsageReported ||
		usageEvidence(&c.Usage) > 0 ||
		c.Usage.CacheReadTokens != nil || c.Usage.CacheWriteTokens != nil ||
		len(c.RawUsage) > 0 || c.ReasoningTokens != nil || c.CostUsd != nil
}

// mergeUsage folds one streamed usage snapshot into the running view of a
// call's usage. OpenAI-compatible upstreams differ here: OpenAI itself emits a
// single final usage chunk (stream_options.include_usage), while others (xAI)
// attach a usage object to EVERY chunk carrying the cumulative-so-far counts.
// Both are monotonic snapshots of the same call, so the newest snapshot wins
// and snapshots are NEVER summed — summing cumulative snapshots would multiply
// the real counts by the chunk count. The one guard is against a regressing
// snapshot (a final chunk that zeroes or truncates usage): a snapshot
// reporting strictly less evidence than one already seen is discarded. Equal
// evidence lets the LATER snapshot win (it may carry richer cache detail).
func mergeUsage(prev, next *Usage) *Usage {
	if next == nil {
		return prev
	}
	if prev == nil || usageEvidence(next) >= usageEvidence(prev) {
		return next
	}
	return prev
}

// usageEvidence is the comparable size of a usage snapshot: the larger of
// total_tokens and prompt+completion (upstreams disagree on whether total
// includes separately-counted reasoning tokens).
func usageEvidence(u *Usage) int {
	e := u.PromptTokens + u.CompletionTokens
	if u.TotalTokens > e {
		e = u.TotalTokens
	}
	return e
}

// floorTotal normalizes a finalized usage: TotalTokens is floored at
// prompt+completion, since some upstreams omit total_tokens or report it
// smaller than the parts. A genuine surplus is preserved — xAI reports
// total = prompt + completion + reasoning tokens — so reasoning spend stays
// visible. Applied only when a call finalizes; streamed snapshots are merged
// untouched.
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
