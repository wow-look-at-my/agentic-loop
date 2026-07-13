package agentic

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMergeUsage(t *testing.T) {
	a := &Usage{PromptTokens: 10, CompletionTokens: 2, TotalTokens: 12}
	b := &Usage{PromptTokens: 10, CompletionTokens: 6, TotalTokens: 16}
	zero := &Usage{}

	cases := []struct {
		name string
		prev *Usage
		next *Usage
		want *Usage
	}{
		{"nil nil", nil, nil, nil},
		{"nil next wins", nil, a, a},
		{"nil prev kept", a, nil, a},
		{"newer snapshot wins", a, b, b},
		{"regressing snapshot discarded", b, zero, b},
		{"regressing to smaller discarded", b, a, b},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Same(t, tc.want, mergeUsage(tc.prev, tc.next))
		})
	}

	t.Run("equal evidence later snapshot wins", func(t *testing.T) {
		read := 8
		richer := &Usage{PromptTokens: 10, CompletionTokens: 6, TotalTokens: 16, CacheReadTokens: &read}
		assert.Same(t, richer, mergeUsage(b, richer))
	})

	t.Run("evidence is max of total and parts", func(t *testing.T) {
		// total omits reasoning tokens; prompt+completion is the evidence.
		parts := &Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 5}
		assert.Equal(t, 30, usageEvidence(parts))
		// total includes reasoning surplus; total is the evidence.
		surplus := &Usage{PromptTokens: 10, CompletionTokens: 20, TotalTokens: 40}
		assert.Equal(t, 40, usageEvidence(surplus))
		// a snapshot whose total omits reasoning still beats a smaller one.
		assert.Same(t, parts, mergeUsage(a, parts))
	})
}

func TestFloorTotal(t *testing.T) {
	floored := floorTotal(Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 0})
	assert.Equal(t, 15, floored.TotalTokens)

	surplus := floorTotal(Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 30})
	assert.Equal(t, 30, surplus.TotalTokens, "a genuine surplus (reasoning tokens) is preserved")
}

func decodeOAUsage(t *testing.T, raw string) Usage {
	t.Helper()
	var u oaUsage
	require.NoError(t, json.Unmarshal([]byte(raw), &u))
	return u.toUsage()
}

func TestOpenAIUsageDecodeCachedTokens(t *testing.T) {
	t.Run("openai prompt_tokens_details", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,
			"prompt_tokens_details":{"cached_tokens":80}}`)
		require.NotNil(t, u.CacheReadTokens)
		assert.Equal(t, 80, *u.CacheReadTokens)
		assert.Equal(t, 80, u.CachedTokens())
		require.NotNil(t, u.CacheWriteTokens, "cache info reported: write is an explicit 0, not unknown")
		assert.Equal(t, 0, *u.CacheWriteTokens)
	})

	t.Run("deepseek prompt_cache_hit_tokens", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,
			"prompt_cache_hit_tokens":64,"prompt_cache_miss_tokens":36}`)
		require.NotNil(t, u.CacheReadTokens)
		assert.Equal(t, 64, *u.CacheReadTokens)
		assert.Equal(t, 64, u.CachedTokens())
	})

	t.Run("anthropic-compat cache_read_input_tokens", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,
			"cache_read_input_tokens":50}`)
		require.NotNil(t, u.CacheReadTokens)
		assert.Equal(t, 50, *u.CacheReadTokens)
	})

	t.Run("largest signal wins", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,
			"prompt_tokens_details":{"cached_tokens":70},"prompt_cache_hit_tokens":30,"cache_read_input_tokens":5}`)
		require.NotNil(t, u.CacheReadTokens)
		assert.Equal(t, 70, *u.CacheReadTokens)
	})

	t.Run("no cache info is tri-state nil", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110}`)
		assert.Nil(t, u.CacheReadTokens)
		assert.Nil(t, u.CacheWriteTokens)
		assert.Equal(t, 0, u.CachedTokens())
	})

	t.Run("explicit zero is a real report", func(t *testing.T) {
		u := decodeOAUsage(t, `{"prompt_tokens":100,"completion_tokens":10,"total_tokens":110,
			"prompt_tokens_details":{"cached_tokens":0}}`)
		require.NotNil(t, u.CacheReadTokens)
		assert.Equal(t, 0, *u.CacheReadTokens)
	})
}

func TestCachedTokensClampAndFloor(t *testing.T) {
	clamped := Usage{PromptTokens: 10, CacheReadTokens: intPtr(999)}
	assert.Equal(t, 10, clamped.CachedTokens(), "clamped to PromptTokens")

	unclamped := Usage{PromptTokens: 0, CacheReadTokens: intPtr(999)}
	assert.Equal(t, 999, unclamped.CachedTokens(), "no clamp when PromptTokens is unknown")

	negative := Usage{PromptTokens: 10, CacheReadTokens: intPtr(-3)}
	assert.Equal(t, 0, negative.CachedTokens())

	assert.Equal(t, 0, Usage{}.CachedTokens())
}

func TestClonePtr(t *testing.T) {
	assert.Nil(t, clonePtr(nil))
	v := 7
	c := clonePtr(&v)
	require.NotNil(t, c)
	assert.Equal(t, 7, *c)
	v = 9
	assert.Equal(t, 7, *c, "clone does not alias")
}
