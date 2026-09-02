package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The arithmetic, with the errors that cost real money.
func TestCostDoesNotDoubleCountCachedTokens(t *testing.T) {
	r := Rates{Prompt: 0.000015, Completion: 0.000075, CacheRead: 0.0000015, CacheWrite: 0.00001875}
	read, write := 36000, 0
	u := Usage{PromptTokens: 40000, CompletionTokens: 0, CacheReadTokens: &read, CacheWriteTokens: &write}

	// uncached at /MTok + cached at /MTok.
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

// Reasoning tokens are inside CompletionTokens on both dialects, so Cost has no reasoning term.
func TestCostPricesCompletionOnceWhateverTheThinkingWas(t *testing.T) {
	r := Rates{Prompt: 0.000001, Completion: 0.00001}
	u := Usage{PromptTokens: 100, CompletionTokens: 4100}
	assert.InDelta(t, 100*0.000001+4100*0.00001, r.Cost(u), 1e-9)
}

// crof.ai's own /v1/models reports pricing already in USD per tokens
// ("" for a model its pricing page prices at $/M), not per token like
// OpenRouter. Reading "" as USD per token billed a real turn a
// millionfold high. A value that would cross maxPlausiblePerTokenUSD under
// the per-token reading is rescaled instead of taken literally.
func TestParseRateRescalesAnAlreadyPerMillionValue(t *testing.T) {
	v, ok := parseRate("0.35")
	require.True(t, ok)
	assert.InDelta(t, 0.00000035, v, 1e-12)

	v, ok = parseRate("0.01")
	require.True(t, ok)
	assert.InDelta(t, 0.00000001, v, 1e-12)

	// A genuine per-token price near the boundary still passes through as-is.
	v, ok = parseRate("0.0006")
	require.True(t, ok)
	assert.InDelta(t, 0.0006, v, 1e-12)
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
