package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// openRouterModelList is the shape an aggregator publishes: rates as STRINGS,
// in USD per token, alongside the model list a host fetches anyway.
const openRouterModelList = `{
  "object": "list",
  "data": [
    {
      "id": "anthropic/claude-opus-4-6",
      "object": "model",
      "pricing": {
        "prompt": "0.000015",
        "completion": "0.000075",
        "input_cache_read": "0.0000015",
        "input_cache_write": "0.00001875",
        "input_cache_write_1h": "0.00003"
      }
    },
    {
      "id": "openai/gpt-5",
      "pricing": {"prompt": "0.00000125", "completion": "0.00001"}
    },
    {
      "id": "some/free-model",
      "pricing": {"prompt": "0", "completion": "0"}
    },
    {"id": "local/llama", "pricing": {}},
    {"id": "no-pricing-at-all"}
  ]
}`

func TestPricesOfModelListReadsPublishedRates(t *testing.T) {
	prices := PricesOfModelList([]byte(openRouterModelList))

	opus, ok := prices["anthropic/claude-opus-4-6"]
	require.True(t, ok)
	assert.InDelta(t, 0.000015, opus.Prompt, 1e-12)
	assert.InDelta(t, 0.000075, opus.Completion, 1e-12)
	assert.InDelta(t, 0.0000015, opus.CacheRead, 1e-12)
	assert.InDelta(t, 0.00001875, opus.CacheWrite, 1e-12)
	assert.InDelta(t, 0.00003, opus.CacheWrite1h, 1e-12)
}

// A block with no cache rates still prices a call: the cache terms fall back to
// the prompt rate, which is what those tokens could otherwise have cost.
func TestAMissingCacheRateFallsBackToThePromptRate(t *testing.T) {
	r := PricesOfModelList([]byte(openRouterModelList))["openai/gpt-5"]
	assert.InDelta(t, 0.00000125, r.CacheRead, 1e-12)
	assert.InDelta(t, 0.00000125, r.CacheWrite, 1e-12)
	assert.InDelta(t, 0.00000125, r.CacheWrite1h, 1e-12)
}

// Zero is a price and absence is not. A free model is priced at zero; a model
// that published nothing is missing, so a host renders an em dash rather than
// claiming the call was free.
func TestAFreeModelIsPricedAndAnUnpricedOneIsAbsent(t *testing.T) {
	prices := PricesOfModelList([]byte(openRouterModelList))

	free, ok := prices["some/free-model"]
	require.True(t, ok, "a published zero is a price")
	assert.Zero(t, free.Cost(Usage{PromptTokens: 1000, CompletionTokens: 1000}))

	_, ok = prices["local/llama"]
	assert.False(t, ok, "an empty pricing block is not a free model")
	_, ok = prices["no-pricing-at-all"]
	assert.False(t, ok)
}

func TestPricesOfModelListSurvivesGarbage(t *testing.T) {
	assert.Empty(t, PricesOfModelList(nil))
	assert.Empty(t, PricesOfModelList([]byte("not json")))
	assert.Empty(t, PricesOfModelList([]byte(`{"data":[{"id":"x","pricing":{"prompt":"nonsense"}}]}`)))
	assert.Empty(t, PricesOfModelList([]byte(`{"data":[{"id":"x","pricing":{"prompt":"-1"}}]}`)),
		"a negative rate is a document nobody should bill from")
}

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

func TestFetchPricesReadsTheEndpointsModelList(t *testing.T) {
	var gotAuth, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotPath = r.Header.Get("Authorization"), r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(openRouterModelList))
	}))
	defer srv.Close()

	prices, body, err := FetchPrices(context.Background(), ProviderConfig{
		BaseURL: srv.URL, APIKey: "sk-test", HTTPClient: srv.Client(),
	})
	require.NoError(t, err)
	assert.Equal(t, "/v1/models", gotPath)
	assert.Equal(t, "Bearer sk-test", gotAuth)
	assert.Contains(t, prices, "anthropic/claude-opus-4-6")

	// The same bytes answer the dialect, so a host that wants both pays for one
	// request rather than two.
	assert.Equal(t, DialectOpenAI, DialectOfModelList(body))
}

func TestFetchPricesReportsAnEndpointThatRefuses(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()

	_, _, err := FetchPrices(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401")

	_, _, err = FetchPrices(context.Background(), ProviderConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no base URL")
}

// An endpoint that publishes no pricing is not a failure. Most do not.
func TestAnEndpointWithNoPricingIsNotAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-5","object":"model"}]}`))
	}))
	defer srv.Close()

	prices, _, err := FetchPrices(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.NoError(t, err)
	assert.Empty(t, prices)
}
