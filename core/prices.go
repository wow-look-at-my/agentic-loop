package commonai

import (
	"encoding/json"
	"strconv"
	"strings"
)

// Rates: USD PER TOKEN, not per; is a real price, absence = no entry.
type Rates struct {
	Prompt     float64
	Completion float64
	CacheRead  float64
	CacheWrite float64
	// CacheWrite1h is the -hour cache-write tier, published but not used by Cost (no tier in the usage tokens).
	CacheWrite1h float64
}

// Cost prices model call in USD; cached tokens are already in PromptTokens, so cache terms bill separately.
func (r Rates) Cost(u Usage) float64 {
	read, write := cacheCounts(u)
	uncached := u.PromptTokens - read - write
	if uncached < 0 {
		uncached = 0
	}
	return float64(uncached)*r.Prompt +
		float64(read)*r.CacheRead +
		float64(write)*r.CacheWrite +
		float64(u.CompletionTokens)*r.Completion
}

// Anomalous reports more cached tokens than prompt tokens; Cost clamps, and this is how a host finds out.
func Anomalous(u Usage) bool {
	read, write := cacheCounts(u)
	return read+write > u.PromptTokens
}

func cacheCounts(u Usage) (read, write int) {
	if u.CacheReadTokens != nil {
		read = *u.CacheReadTokens
	}
	if u.CacheWriteTokens != nil {
		write = *u.CacheWriteTokens
	}
	if read < 0 {
		read = 0
	}
	if write < 0 {
		write = 0
	}
	return read, write
}

// modelListPricing is the pricing block a model list publishes per model. Every
// field is a string in the wire format — OpenRouter sends "", not
// — so each is decoded as and parsed here.
type modelListPricing struct {
	Prompt             string          `json:"prompt"`
	Completion         string          `json:"completion"`
	InputCacheRead     string          `json:"input_cache_read"`
	InputCacheWrite    string          `json:"input_cache_write"`
	CacheWrite1h       string          `json:"input_cache_write_1h"`
	AltCacheRead       string          `json:"cache_read"`
	AltCacheWrite      string          `json:"cache_write"`
	AltInput           string          `json:"input"`
	AltOutput          string          `json:"output"`
	CurrencyIrrelevant json.RawMessage `json:"currency"`
}

// ratesOf converts pricing block. The return is false when the block
// carried no usable number at all, which keeps an empty `"pricing": {}` out of
// the map instead of turning it into a free model.
//
// Cache rates fall back to the prompt rate when the block omits them, because
// an omitted cache rate on a provider that charges for cache reads is a low
// number, and the prompt rate is the price we know the tokens could have
// cost. A host that wants the exact figure sets it in config.
func ratesOf(p modelListPricing) (Rates, bool) {
	var r Rates
	var any bool
	set := func(dst *float64, raw ...string) {
		for _, s := range raw {
			if v, ok := parseRate(s); ok {
				*dst = v
				any = true
				return
			}
		}
	}
	set(&r.Prompt, p.Prompt, p.AltInput)
	set(&r.Completion, p.Completion, p.AltOutput)
	if !any {
		return Rates{}, false
	}
	r.CacheRead, r.CacheWrite = r.Prompt, r.Prompt
	set(&r.CacheRead, p.InputCacheRead, p.AltCacheRead)
	set(&r.CacheWrite, p.InputCacheWrite, p.AltCacheWrite)
	r.CacheWrite1h = r.CacheWrite
	set(&r.CacheWrite1h, p.CacheWrite1h)
	return r, true
}

// maxPlausiblePerTokenUSD bounds a real per-token price, telling a per-token value from already per.
const maxPlausiblePerTokenUSD = 0.001

// parseRate reads a published rate in USD per token; a value over maxPlausiblePerTokenUSD is per and rescaled.
func parseRate(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v < 0 {
		return 0, false
	}
	if v > maxPlausiblePerTokenUSD {
		v /= 1e6
	}
	return v, true
}
