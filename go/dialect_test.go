package agentic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The tell is the SHAPE of the model list, so an Anthropic-compatible gateway
// on a domain no hostname rule would recognize is still identified.
func TestDetectDialectReadsTheModelList(t *testing.T) {
	cases := []struct {
		name string
		body string
		want Dialect
	}{
		{"openai envelope", jsonMust(jsonObj{
			"object": "list",
			"data":   jsonArr{jsonObj{"id": "gpt-x", "object": "model"}},
		}), DialectOpenAI},
		{"anthropic envelope", jsonMust(jsonObj{
			"data":     jsonArr{jsonObj{"type": "model", "id": "claude-x"}},
			"has_more": false,
		}), DialectAnthropic},
		// An empty list still identifies its server: the envelope is the tell.
		{"empty openai list", jsonMust(jsonObj{"object": "list", "data": jsonArr{}}), DialectOpenAI},
		{"empty anthropic list", jsonMust(jsonObj{"data": jsonArr{}, "has_more": false}), DialectAnthropic},
		// No envelope at all, so the items answer.
		{"bare anthropic items", jsonMust(jsonObj{"data": jsonArr{jsonObj{"type": "model"}}}), DialectAnthropic},
		{"bare openai items", jsonMust(jsonObj{"data": jsonArr{jsonObj{"object": "model"}}}), DialectOpenAI},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var got *http.Request
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				got = r
				w.Header().Set("Content-Type", "application/json")
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			d, err := DetectDialect(context.Background(), ProviderConfig{
				BaseURL: srv.URL, APIKey: "k", HTTPClient: srv.Client(),
			})
			require.NoError(t, err)
			assert.Equal(t, tc.want, d)

			require.NotNil(t, got)
			assert.Equal(t, "/v1/models", got.URL.Path)
			// Both credential forms ride along, because which server is
			// answering is the very thing being established.
			assert.Equal(t, "Bearer k", got.Header.Get("Authorization"))
			assert.Equal(t, "k", got.Header.Get("x-api-key"))
			assert.Equal(t, defaultAnthropicVersion, got.Header.Get("anthropic-version"))
		})
	}
}

// The shape test is exported for a host that already fetches model lists, so
// reading the tell costs it no request -- and so the rule is declared once.
func TestDialectOfModelListIsTheOneShapeTest(t *testing.T) {
	assert.Equal(t, DialectOpenAI, DialectOfModelList([]byte(jsonMust(jsonObj{
		"object": "list", "data": jsonArr{},
	}))))
	assert.Equal(t, DialectAnthropic, DialectOfModelList([]byte(jsonMust(jsonObj{
		"data": jsonArr{jsonObj{"type": "model"}},
	}))))
	assert.Equal(t, DialectAuto, DialectOfModelList([]byte(`<html>not json</html>`)),
		"no answer, rather than a guess")
	assert.Equal(t, DialectAuto, DialectOfModelList([]byte(jsonMust(jsonObj{
		"models": jsonArr{"a"},
	}))), "a document of neither shape places neither dialect")

	// An endpoint serving /v1/responses answers the same model list as one
	// that does not, so detection cannot ever name it. Using it is a choice.
	assert.Equal(t, DialectOpenAI, DialectOfModelList([]byte(jsonMust(jsonObj{
		"object": "list", "data": jsonArr{jsonObj{"id": "gpt-x", "object": "model"}},
	}))), "detection never returns DialectResponses")
}

// An unanswered question answers DialectAuto and an error, never a guess: a
// wrong dialect does not degrade, it breaks chat outright.
func TestDetectDialectRefusesToGuess(t *testing.T) {
	cases := []struct {
		name   string
		status int
		body   string
		reason string
	}{
		{"unauthorized", http.StatusUnauthorized, `{"error":{"message":"no key"}}`, "answered 401"},
		{"not found", http.StatusNotFound, `{}`, "answered 404"},
		{"not json", http.StatusOK, `<html>hello</html>`, "matches neither dialect"},
		{"neither shape", http.StatusOK, jsonMust(jsonObj{"models": jsonArr{"a"}}), "matches neither dialect"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer srv.Close()

			d, err := DetectDialect(context.Background(), ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()})
			assert.Equal(t, DialectAuto, d)
			require.Error(t, err)
			assert.Contains(t, err.Error(), tc.reason)
		})
	}

	_, err := DetectDialect(context.Background(), ProviderConfig{})
	require.Error(t, err, "no base URL is a caller mistake, not a detection result")
	assert.Contains(t, err.Error(), "no base URL")
}

// The vocabulary is the library's, so a host's UI does not carry a second copy
// that goes stale when a dialect is added.
func TestDialectVocabulary(t *testing.T) {
	assert.Equal(t, []Dialect{DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses}, Dialects())
	for _, d := range Dialects() {
		assert.True(t, d.Valid())
		assert.NotEmpty(t, d.Label())
	}
	assert.False(t, Dialect("cohere").Valid(), "a dialect this library cannot speak is not valid")
	assert.Equal(t, "detect", DialectAuto.Label())
}
