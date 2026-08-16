package commonai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
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
// publishes nothing has no Rates at all. That is the whole reason PricesOf
// returns a map rather than a value per model — absence has to stay absence, so
// a host renders an em dash instead of claiming a call was free.
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

// PricesOfModelList reads per-model rates out of a model-list document, keyed by
// model id. A model that publishes no pricing block is ABSENT from the result
// rather than present with zeros.
//
// It is a separate function from the fetch so a host that already has the bytes
// — the same document DialectOfModelList reads — pays for one request, not two.
// A document that does not parse yields an empty map and no error: this is a
// display input, and a host that cannot price a call renders an em dash, which
// is the same answer a malformed document deserves.
func PricesOfModelList(body []byte) map[string]Rates {
	var doc struct {
		Data []struct {
			ID      string            `json:"id"`
			Name    string            `json:"name"`
			Pricing *modelListPricing `json:"pricing"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return map[string]Rates{}
	}
	out := make(map[string]Rates, len(doc.Data))
	for _, m := range doc.Data {
		id := strings.TrimSpace(m.ID)
		if id == "" || m.Pricing == nil {
			continue
		}
		if r, ok := ratesOf(*m.Pricing); ok {
			out[id] = r
		}
	}
	return out
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

// FetchPrices asks an endpoint's model list what its models cost.
//
// It returns the rates and the raw document, so a caller that also wants the
// dialect reads it off these bytes with DialectOfModelList rather than making
// the request twice.
//
// An endpoint that publishes no pricing answers with an empty map and no error:
// most do not, that is not a failure, and a host's answer to it is an em dash
// plus whatever the user configured — never a manufactured number.
func FetchPrices(ctx context.Context, cfg ProviderConfig) (map[string]Rates, []byte, error) {
	base := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if base == "" {
		return nil, nil, fmt.Errorf("fetching prices: no base URL")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return nil, nil, fmt.Errorf("fetching prices: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if cfg.APIKey != "" {
		// Both credential forms, for the same reason DetectDialect sends both:
		// which server is answering is exactly what is not yet known.
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		req.Header.Set("x-api-key", cfg.APIKey)
	}
	req.Header.Set("anthropic-version", defaultAnthropicVersion)
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := hc.Do(req)
	if err != nil {
		return nil, nil, fmt.Errorf("fetching prices: %w", err)
	}
	defer resp.Body.Close()
	body, _, err := readCapped(resp.Body, detectMaxBytes)
	if err != nil {
		return nil, nil, fmt.Errorf("fetching prices: reading the model list: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, body, fmt.Errorf("fetching prices: the model list answered %d", resp.StatusCode)
	}
	return PricesOfModelList(body), body, nil
}
