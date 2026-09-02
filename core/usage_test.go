package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCachedTokens(t *testing.T) {
	cases := []struct {
		name string
		u    Usage
		want int
	}{
		{"no cache info at all", Usage{PromptTokens: 100}, 0},
		{"reported zero", Usage{PromptTokens: 100, CacheReadTokens: intPtr(0)}, 0},
		{"reported", Usage{PromptTokens: 100, CacheReadTokens: intPtr(40)}, 40},
		{"clamped to the prompt", Usage{PromptTokens: 10, CacheReadTokens: intPtr(999)}, 10},
		{"floored at zero", Usage{PromptTokens: 10, CacheReadTokens: intPtr(-5)}, 0},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			assert.Equal(t, tc.want, tc.u.CachedTokens())
		})
	}
}

// nil and are different answers to "how many tokens came from cache", and
// only of them is a number the provider actually sent.
func TestCacheCountsAreTriState(t *testing.T) {
	assert.Nil(t, Usage{}.CacheReadTokens)
	assert.NotNil(t, Usage{CacheReadTokens: intPtr(0)}.CacheReadTokens)
}

func TestClonePtr(t *testing.T) {
	assert.Nil(t, clonePtr(nil))
	v := 7
	c := clonePtr(&v)
	v = 9
	assert.Equal(t, 7, *c, "a cloned count never aliases the source")
}
