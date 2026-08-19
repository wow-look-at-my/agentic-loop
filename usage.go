package agentic

// Usage is the normalized token accounting for one model call.
//
// PromptTokens is always the FULL prompt, cached tokens included: OpenAI's
// prompt_tokens already includes cached tokens, while Anthropic's input_tokens
// EXCLUDES them — the Anthropic provider therefore reports
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens here so
// the two dialects agree.
//
// CacheReadTokens and CacheWriteTokens are TRI-STATE: nil means the provider
// reported no cache information at all (common on local OpenAI-compatible
// servers), while a non-nil value — including an explicit 0 — is a real
// server-reported count. Never zero-fill or estimate these.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CacheReadTokens  *int
	CacheWriteTokens *int
}

// CachedTokens returns the number of prompt tokens served from the provider's
// prompt cache, or 0 when the provider reported none (or no cache info at
// all). The count is clamped to PromptTokens so a malformed upstream can't
// report nonsense, and floored at 0.
func (u Usage) CachedTokens() int {
	cached := 0
	if u.CacheReadTokens != nil {
		cached = *u.CacheReadTokens
	}
	if u.PromptTokens > 0 && cached > u.PromptTokens {
		cached = u.PromptTokens
	}
	if cached < 0 {
		cached = 0
	}
	return cached
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

// intPtr returns a pointer to v, for building tri-state cache counts.
func intPtr(v int) *int { return &v }

// clonePtr copies a tri-state int pointer so snapshots never alias.
func clonePtr(p *int) *int {
	if p == nil {
		return nil
	}
	v := *p
	return &v
}
