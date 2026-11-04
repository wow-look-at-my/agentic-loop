package commonai

import _ "embed"

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

// A block with no cache rates still prices a call. The cache terms fall back to
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
	assert.Equal(t, DialectOpenAI, list.Dialect)

	// The OpenAI form only: a gateway reads anthropic-version as a choice of list.
	assert.Equal(t, "Bearer sk-test", gotAuth)
	assert.Empty(t, gotKey)
	assert.Empty(t, gotVersion)
}

// A gateway that speaks both answers a shorter, renamed list when it sees
// anthropic-version. The fetch must get the full list.
func TestFetchModelListGetsTheOpenAIListFromADualGateway(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-version") != "" {
			_, _ = w.Write([]byte(`{"data":[{"id":"anthropic/x/a[1m]","type":"model"}],"has_more":false}`))
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"x/a","endpoints":["/api/v1/chat/completions"]},{"id":"x/b"}]}`))
	}))
	defer srv.Close()
	list, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, APIKey: "k", HTTPClient: srv.Client()})
	require.NoError(t, err)
	assert.Equal(t, []Dialect{DialectOpenAI}, list.Endpoints["x/a"])
	assert.NotContains(t, list.Endpoints, "anthropic/x/a[1m]")
}

// Anthropic refuses a Bearer token, so the fetch re-asks with x-api-key and reads every page.
func TestFetchModelListFallsBackToAnthropicAndReadsEveryPage(t *testing.T) {
	var queries []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("x-api-key") != "k" || r.Header.Get("anthropic-version") == "" {
			w.WriteHeader(http.StatusUnauthorized)
			return
		}
		queries = append(queries, r.URL.RawQuery)
		if r.URL.Query().Get("after_id") == "" {
			_, _ = w.Write([]byte(`{"data":[{"id":"claude-a","type":"model","max_tokens":10}],"has_more":true,"last_id":"claude-a"}`))
			return
		}
		_, _ = w.Write([]byte(`{"data":[{"id":"claude-b","type":"model","max_tokens":20}],"has_more":false,"last_id":"claude-b"}`))
	}))
	defer srv.Close()
	list, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, APIKey: "k", HTTPClient: srv.Client()})
	require.NoError(t, err)
	assert.Equal(t, DialectAnthropic, list.Dialect)
	assert.Equal(t, 10, list.Limits["claude-a"].MaxOutput)
	assert.Equal(t, 20, list.Limits["claude-b"].MaxOutput, "the second page is read")
	assert.Equal(t, []string{"limit=1000", "limit=1000&after_id=claude-a"}, queries)
}

// A list that never stops paging is an error, never a silent cut.
func TestFetchModelListFailsOnAnEndlessPageWalk(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte(`{"data":[],"has_more":true,"last_id":"z"}`))
	}))
	defer srv.Close()
	_, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "still has more")
}

// Both forms refused is an error that names both answers.
func TestFetchModelListNamesBothRefusals(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-version") != "" {
			w.WriteHeader(http.StatusForbidden)
			return
		}
		w.WriteHeader(http.StatusUnauthorized)
	}))
	defer srv.Close()
	_, err := FetchModelList(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "401 to a Bearer token (no body) and 403 to x-api-key (no body)")
}

// OpenRouter advertises the paths that serve each model; a batch-only model advertises none a dialect speaks.
func TestDecodeModelListReadsAdvertisedEndpoints(t *testing.T) {
	list, err := DecodeModelList([]byte(testDecodeModelListReadsAdvertisedEndpointsJson))
	require.NoError(t, err)
	assert.Equal(t, []Dialect{DialectOpenAI, DialectCompletions, DialectAnthropic, DialectResponses}, list.Endpoints["p/all"])
	got, ok := list.EndpointsFor("p/batch")
	assert.True(t, ok)
	assert.Empty(t, got, "advertised, but on nothing a dialect speaks")
	_, ok = list.EndpointsFor("p/none")
	assert.False(t, ok, "a model that advertises nothing is absent")
	got, ok = list.EndpointsFor("all")
	assert.True(t, ok, "a gateway-stripped id resolves by suffix")
	assert.Len(t, got, 4)
	_, ok = (*ModelList)(nil).EndpointsFor("x")
	assert.False(t, ok)
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

	// A path that merely CONTAINS /v1 keeps it: only a trailing /v1 is the disagreement.
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

// The Anthropic model list names its dialect and prices nothing, and BOTH of
// those are the answer rather than a failure. A host prices the call from
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
	list := decoded(t, anthropicPricedModelList)
	assert.Equal(t, DialectAnthropic, list.Dialect)
	r, ok := list.Prices["claude-opus-4-6"]
	require.True(t, ok)
	assert.InDelta(t, 0.000015, r.Prompt, 1e-12)
	assert.InDelta(t, 0.000075, r.Completion, 1e-12)
}

//go:embed testdata/test-decode-model-list-reads-advertised-endpoints.json
var testDecodeModelListReadsAdvertisedEndpointsJson string
