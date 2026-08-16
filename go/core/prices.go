package commonai

import (
	"encoding/json"
	"strconv"
	"strings"
)

// Rates is what one model charges, in USD PER TOKEN.
//
// Per token, not per million, because that is the unit the model list publishes
// and converting on the way in means the conversion happens once, here, instead
// of at every arithmetic site.
//
// A rate of 0 is a real price: a free model publishes zeros, and a model that
// publishes nothing has no Rates at all. That is the whole reason
// ModelList.Prices is a map — absence has to stay absence, so a host renders an
// em dash instead of claiming a call was free.
type Rates struct {
	Prompt     float64
	Completion float64
	CacheRead  float64
	CacheWrite float64
	// CacheWrite1h is the one-hour cache-write tier, published by the model
	// list and NOT used by Cost: Usage.CacheWriteTokens is a single integer
	// with no tier in it, so nothing here can tell the two apart. The library
	// places only five-minute breakpoints, so CacheWrite is the right rate for
	// every call it makes. The field is carried rather than dropped because a
	// host that grows one-hour breakpoints needs it, and inventing it later
	// from a multiplier is the guess this avoids.
	CacheWrite1h float64
}

// Cost prices one model call in USD.
//
// Three things the obvious formula gets wrong, each worth real money:
//
//   - PromptTokens ALREADY CONTAINS the cached tokens. Every dialect normalizes
//     to that. Pricing the whole prompt at Prompt and then adding the cache
//     terms bills the cached tokens twice, and cache reads are routinely 60-90%
//     of a long session's prompt.
//   - Cache-write tokens are prompt tokens charged at the WRITE rate, not at
//     the write rate on top of the input rate.
//   - Reasoning tokens are already inside CompletionTokens on both dialects, so
//     they are not a term here at all. Adding one roughly doubles the price of
//     a reasoning-heavy turn.
//
// A provider that reports more cached tokens than prompt tokens is
// inconsistent. The uncached term clamps at zero rather than going negative;
// Anomalous reports the same condition so a host can say so out loud instead of
// quietly billing less than the invoice.
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

// Anomalous reports whether a usage record prices something it cannot: more
// cached tokens than there were prompt tokens. Cost clamps; this is how a host
// finds out it clamped.
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
// field is a string in the wire format — OpenRouter sends "0.000015", not
// 0.000015 — so each is decoded as one and parsed here.
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

// ratesOf converts one pricing block. The second return is false when the block
// carried no usable number at all, which keeps an empty `"pricing": {}` out of
// the map instead of turning it into a free model.
//
// Cache rates fall back to the prompt rate when the block omits them, because
// an omitted cache rate on a provider that charges for cache reads is a low
// number, and the prompt rate is the one price we know the tokens could have
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

// parseRate reads one published rate. A negative one is refused rather than
// clamped: it is a document nobody should be billing from.
func parseRate(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil || v < 0 {
		return 0, false
	}
	return v, true
}
