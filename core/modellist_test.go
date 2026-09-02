package commonai

import (
	"context"
	"fmt"
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

func decoded(t *testing.T, body string) *ModelList {
	t.Helper()
	list, err := DecodeModelList([]byte(body))
	require.NoError(t, err)
	return list
}

func TestDecodeModelListReadsPublishedRates(t *testing.T) {
	list := decoded(t, openRouterModelList)

	opus, ok := list.Prices["anthropic/claude-opus-4-6"]
	require.True(t, ok)
	assert.InDelta(t, 0.000015, opus.Prompt, 1e-12)
	assert.InDelta(t, 0.000075, opus.Completion, 1e-12)
	assert.InDelta(t, 0.0000015, opus.CacheRead, 1e-12)
	assert.InDelta(t, 0.00001875, opus.CacheWrite, 1e-12)
	assert.InDelta(t, 0.00003, opus.CacheWrite1h, 1e-12)
}

// document, decode: the dialect and the rates come out of the same pass
// rather than functions parsing the same bytes.
func TestDecodeModelListNamesTheDialectFromTheSamePass(t *testing.T) {
	assert.Equal(t, DialectOpenAI, decoded(t, openRouterModelList).Dialect)

	// The ENVELOPE decides, because a list with no models at all still identifies its server.
	assert.Equal(t, DialectOpenAI, decoded(t, `{"object":"list","data":[]}`).Dialect)
	assert.Equal(t, DialectAnthropic, decoded(t, `{"has_more":false,"data":[]}`).Dialect)

	// With no envelope at all, the items answer.
	assert.Equal(t, DialectAnthropic, decoded(t, `{"data":[{"type":"model"}]}`).Dialect)
	assert.Equal(t, DialectOpenAI, decoded(t, `{"data":[{"object":"model"}]}`).Dialect)

	assert.Equal(t, DialectAuto, decoded(t, `{"models":["a"]}`).Dialect,
		"a document of neither shape places neither dialect, and that is not an error here")

	// An endpoint serving /v1/responses answers the same model list as that
	// does not, so detection cannot ever name it. Using it is a choice.
	assert.Equal(t, DialectOpenAI,
		decoded(t, `{"object":"list","data":[{"id":"gpt-x","object":"model"}]}`).Dialect,
		"detection never returns DialectResponses")
}

// A block with no cache rates still prices a call: the cache terms fall back to
// the prompt rate, which is what those tokens could otherwise have cost.
func TestAMissingCacheRateFallsBackToThePromptRate(t *testing.T) {
	r := decoded(t, openRouterModelList).Prices["openai/gpt-5"]
	assert.InDelta(t, 0.00000125, r.CacheRead, 1e-12)
	assert.InDelta(t, 0.00000125, r.CacheWrite, 1e-12)
	assert.InDelta(t, 0.00000125, r.CacheWrite1h, 1e-12)
}

// is a price and absence is not: a free model is priced at, a silent is missing.
func TestAFreeModelIsPricedAndAnUnpricedOneIsAbsent(t *testing.T) {
	list := decoded(t, openRouterModelList)

	free, ok := list.Prices["some/free-model"]
	require.True(t, ok, "a published zero is a price")
	assert.Zero(t, free.Cost(Usage{PromptTokens: 1000, CompletionTokens: 1000}))

	_, ok = list.Prices["local/llama"]
	assert.False(t, ok, "an empty pricing block is not a free model")
	_, ok = list.Prices["no-pricing-at-all"]
	assert.False(t, ok)
}

// A document that will not parse is an ERROR, never an empty result; a misconfigured URL must not look cheap.
func TestADocumentThatWillNotParseIsAnError(t *testing.T) {
	for _, body := range []string{"", "not json", "<html>not json</html>"} {
		_, err := DecodeModelList([]byte(body))
		require.Error(t, err, body)
		assert.Contains(t, err.Error(), "not JSON", body)
	}
}

// A rate that parses to nothing usable leaves the model unpriced rather than
// priced at. That is not a malformed document — the envelope is fine.
func TestAnUnusableRateLeavesTheModelUnpriced(t *testing.T) {
	list := decoded(t, `{"object":"list","data":[{"id":"x","pricing":{"prompt":"nonsense"}}]}`)
	assert.Empty(t, list.Prices)

	list = decoded(t, `{"object":"list","data":[{"id":"x","pricing":{"prompt":"-1"}}]}`)
	assert.Empty(t, list.Prices, "a negative rate is a document nobody should bill from")
}

// crofAIModelList: crof.ai rates are strings, already USD per tokens, not per token.
const crofAIModelList = `{
  "object": "list",
  "data": [
    {
      "id": "deepseek-v4-pro-0813",
      "object": "model",
      "pricing": {"prompt": "0.35", "completion": "0.80", "cache_prompt": "0.01"}
    }
  ]
}`

func TestDecodeModelListRescalesAnAlreadyPerMillionRate(t *testing.T) {
	r, ok := decoded(t, crofAIModelList).Prices["deepseek-v4-pro-0813"]
	require.True(t, ok)
	assert.InDelta(t, 0.00000035, r.Prompt, 1e-12)
	assert.InDelta(t, 0.0000008, r.Completion, 1e-12)
}

func TestFetchModelListReadsTheEndpoint(t *testing.T) {
	var gotAuth, gotKey, gotVersion, gotPath string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth, gotKey = r.Header.Get("Authorization"), r.Header.Get("x-api-key")
		gotVersion, gotPath = r.Header.Get("anthropic-version"), r.URL.Path
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(openRouterModelList))
	}))
	defer srv.Close()

	list, err := FetchModelList(context.Background(), ProviderConfig{
		BaseURL: srv.URL, APIKey: "sk-test", HTTPClient: srv.Client(),
	})
	require.NoError(t, err)
	assert.Equal(t, "/v1/models", gotPath)
	assert.Contains(t, list.Prices, "anthropic/claude-opus-4-6")
	assert.Equal(t, DialectOpenAI, list.Dialect, "one request answers both")

	// Both credential forms, because which server is answering is exactly what
	// this request exists to find out.
	assert.Equal(t, "Bearer sk-test", gotAuth)
	assert.Equal(t, "sk-test", gotKey)
	assert.Equal(t, defaultAnthropicVersion, gotVersion)
}

// The chat dialects disagree about a trailing /v1, so the model list accepts both spellings.
func TestTheModelListURLAcceptsEitherBaseSpelling(t *testing.T) {
	for _, suffix := range []string{"", "/v1", "/v1/"} {
		var gotPath string
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			gotPath = r.URL.Path
			_, _ = w.Write([]byte(`{"object":"list","data":[]}`))
		}))

		_, err := FetchModelList(context.Background(), ProviderConfig{
			BaseURL: srv.URL + suffix, HTTPClient: srv.Client(),
		})
		require.NoError(t, err, suffix)
		assert.Equal(t, "/v1/models", gotPath, "base %q", srv.URL+suffix)
		srv.Close()
	}

	// A path that merely CONTAINS /v1 keeps it: only the trailing is the
	// dialects' disagreement.
	url, err := modelListURL("https://gw.example.com/v1/openai")
	require.NoError(t, err)
	assert.Equal(t, "https://gw.example.com/v1/openai/v1/models", url)
}

func TestFetchModelListReportsAnEndpointThatRefuses(t *testing.T) {
	for _, tc := range []struct{ status, want int }{
		{http.StatusUnauthorized, 401},
		{http.StatusNotFound, 404},
	} {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
			w.WriteHeader(tc.status)
		}))

		_, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
		require.Error(t, err)
		assert.Contains(t, err.Error(), fmt.Sprint(tc.want))
		assert.Contains(t, err.Error(), "/v1/models", "the URL that refused is the thing to check")
		srv.Close()
	}

	_, err := FetchModelList(context.Background(), ProviderConfig{})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "no base URL")
}

// An endpoint that publishes no pricing is not a failure. Most do not.
func TestAnEndpointWithNoPricingIsNotAnError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"gpt-5","object":"model"}]}`))
	}))
	defer srv.Close()

	list, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.NoError(t, err)
	assert.Empty(t, list.Prices)
	assert.Equal(t, DialectOpenAI, list.Dialect)
}

// anthropicModelList is what api.anthropic.com actually answers: type/id/
// display_name/created_at, and no pricing anywhere. It is the endpoint a host's
// cost column has to survive, since nothing it sends can price a call.
const anthropicModelList = `{
  "data": [
    {"type": "model", "id": "claude-opus-4-6", "display_name": "Claude Opus 4.6",
     "created_at": "2026-02-05T00:00:00Z"},
    {"type": "model", "id": "claude-sonnet-4-6", "display_name": "Claude Sonnet 4.6",
     "created_at": "2026-02-05T00:00:00Z"}
  ],
  "has_more": false,
  "first_id": "claude-opus-4-6",
  "last_id": "claude-sonnet-4-6"
}`

// The Anthropic model list names its dialect and prices nothing, and BOTH of
// those are the answer rather than a failure: a host prices the call from
// config and renders an em dash when there is none.
func TestTheAnthropicModelListNamesItsDialectAndPricesNothing(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(anthropicModelList))
	}))
	defer srv.Close()

	list, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.NoError(t, err)
	assert.Equal(t, DialectAnthropic, list.Dialect)
	assert.Empty(t, list.Prices, "anthropic publishes no rates, and absence is not zero")
}

// Pricing is read off the ITEMS, so an Anthropic-shaped list that does carry
// rates — a gateway in front of the Messages API — is priced. Gating the price
// parse on the openai envelope would silently unprice every such endpoint.
func TestPricingIsReadWhateverEnvelopeCarriesIt(t *testing.T) {
	list := decoded(t, `{
	  "has_more": false,
	  "data": [
	    {"type": "model", "id": "claude-opus-4-6",
	     "pricing": {"prompt": "0.000015", "completion": "0.000075"}}
	  ]
	}`)
	assert.Equal(t, DialectAnthropic, list.Dialect)
	r, ok := list.Prices["claude-opus-4-6"]
	require.True(t, ok)
	assert.InDelta(t, 0.000015, r.Prompt, 1e-12)
	assert.InDelta(t, 0.000075, r.Completion, 1e-12)
}
