package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The arithmetic, with the three errors that cost real money.
func TestCostDoesNotDoubleCountCachedTokens(t *testing.T) {
	r := Rates{Prompt: 0.000015, Completion: 0.000075, CacheRead: 0.0000015, CacheWrite: 0.00001875}
	read, write := 36000, 0
	u := Usage{PromptTokens: 40000, CompletionTokens: 0, CacheReadTokens: &read, CacheWriteTokens: &write}

	// 4000 uncached at 15/MTok + 36000 cached at 1.5/MTok.
	assert.InDelta(t, 4000*0.000015+36000*0.0000015, r.Cost(u), 1e-9)

	naive := 40000*0.000015 + 36000*0.0000015
	assert.Less(t, r.Cost(u), naive/5,
		"pricing the whole prompt and then adding the cache term is ~5.7x too high")
}

func TestCacheWritesAreNotChargedTwice(t *testing.T) {
	r := Rates{Prompt: 0.000015, Completion: 0.000075, CacheRead: 0.0000015, CacheWrite: 0.00001875}
	read, write := 0, 10000
	u := Usage{PromptTokens: 12000, CacheReadTokens: &read, CacheWriteTokens: &write}

	assert.InDelta(t, 2000*0.000015+10000*0.00001875, r.Cost(u), 1e-9)
}

// Reasoning tokens are inside CompletionTokens on both dialects, so there is no
// reasoning term. This test is the guard on that: Cost takes a Usage, which has
// no reasoning field, so a future edit cannot add the term without changing the
// signature and tripping this.
func TestCostPricesCompletionOnceWhateverTheThinkingWas(t *testing.T) {
	r := Rates{Prompt: 0.000001, Completion: 0.00001}
	u := Usage{PromptTokens: 100, CompletionTokens: 4100}
	assert.InDelta(t, 100*0.000001+4100*0.00001, r.Cost(u), 1e-9)
}

// A provider reporting more cached tokens than prompt tokens is inconsistent.
// The uncached term clamps rather than going negative, and Anomalous is how a
// host learns it clamped instead of quietly billing less than the invoice.
func TestAnInconsistentUsageClampsAndSaysSo(t *testing.T) {
	r := Rates{Prompt: 0.000015, Completion: 0.000075, CacheRead: 0.0000015}
	read, write := 15100, 0
	u := Usage{PromptTokens: 12400, CacheReadTokens: &read, CacheWriteTokens: &write}

	assert.InDelta(t, 15100*0.0000015, r.Cost(u), 1e-9)
	assert.True(t, Anomalous(u))

	ok := Usage{PromptTokens: 12400, CacheReadTokens: &write, CacheWriteTokens: &write}
	assert.False(t, Anomalous(ok))
}
