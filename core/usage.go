package commonai

import "encoding/json"

// Usage is provider report; PromptTokens is always the full prompt, cached tokens included.
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

// CachedTokens returns prompt tokens served from the provider's cache, clamped to PromptTokens.
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
