package commonai

import "encoding/json"

// Usage is ONE usage report from a provider, normalized in name only: the
// numbers are the ones the provider sent, and nothing here merges, sums, or
// floors anything. A call can produce several of these (see Completion.Usages).
//
// PromptTokens is always the FULL prompt, cached tokens included: OpenAI's
// prompt_tokens already includes cached tokens, while Anthropic's input_tokens
// EXCLUDES them -- the Anthropic dialect therefore reports
// input_tokens + cache_read_input_tokens + cache_creation_input_tokens here so
// the two dialects mean the same thing by the same field. That is a naming
// correction, not an invented total.
//
// CacheReadTokens and CacheWriteTokens are TRI-STATE: nil means the provider
// reported no cache information at all (common on local OpenAI-compatible
// servers), while a non-nil value -- including an explicit 0 -- is a real
// server-reported count. Never zero-fill or estimate these.
//
// Raw is the provider's usage object verbatim, and ReasoningTokens / CostUsd
// are the two extras worth naming: the openai
// usage.completion_tokens_details.reasoning_tokens figure, and the dollar cost
// from usage.cost or usage.estimated_cost (OpenRouter/Requesty/DeepInfra
// style). Both are tri-state and stay nil unless the upstream reported them.
type Usage struct {
	PromptTokens     int
	CompletionTokens int
	TotalTokens      int
	CacheReadTokens  *int
	CacheWriteTokens *int
	Raw              json.RawMessage
	ReasoningTokens  *int
	CostUsd          *float64
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
