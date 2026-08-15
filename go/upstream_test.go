package agentic

import (
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/common-ai-api/go/client"
)

// Fake upstreams for the tests that drive the loop over a real provider.
//
// The dialects themselves are tested in common-ai-api, against these same
// shapes. What is exercised here is the loop's behavior when a call goes over
// HTTP and comes back streamed, retried, or broken -- which needs a server, not
// a stub.

// sseHandler serves the given data payloads as an SSE stream, terminated by
// [DONE], capturing the request body and headers.
type sseHandler struct {
	payloads    []string
	contentType string
	body        []byte
	header      http.Header
	hits        int
}

func (h *sseHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits++
	h.body, _ = io.ReadAll(r.Body)
	h.header = r.Header.Clone()
	ct := h.contentType
	if ct == "" {
		ct = "text/event-stream"
	}
	w.Header().Set("Content-Type", ct)
	fl := w.(http.Flusher)
	for _, p := range h.payloads {
		_, _ = w.Write([]byte("data: " + p + "\n\n"))
		fl.Flush()
	}
	_, _ = w.Write([]byte("data: [DONE]\n\n"))
	fl.Flush()
}

// anSSEHandler serves an Anthropic event stream: named events with their own
// data payloads, and no [DONE] terminator.
type anSSEHandler struct {
	events [][2]string // {event name, data payload}
	body   []byte
	header http.Header
	hits   int
}

func (h *anSSEHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.hits++
	h.body, _ = io.ReadAll(r.Body)
	h.header = r.Header.Clone()
	w.Header().Set("Content-Type", "text/event-stream")
	fl := w.(http.Flusher)
	for _, ev := range h.events {
		_, _ = w.Write([]byte("event: " + ev[0] + "\ndata: " + ev[1] + "\n\n"))
		fl.Flush()
	}
}

// newProvider gives a stub Provider the retry behavior a constructed one has,
// for the tests that drive Run over a failure the provider is supposed to
// absorb.
func newProvider(p Provider, policy *RetryPolicy) Provider {
	return client.Retrying(p, policy)
}

// mustOpenAI builds a Provider via NewOpenAIProvider, failing the test on error.
func mustOpenAI(t *testing.T, cfg OpenAIConfig) Provider {
	t.Helper()
	p, err := NewOpenAIProvider(cfg)
	require.NoError(t, err)
	return p
}

// mustAnthropic builds a Provider via NewAnthropicProvider, failing the test on
// error.
func mustAnthropic(t *testing.T, cfg AnthropicConfig) Provider {
	t.Helper()
	p, err := NewAnthropicProvider(cfg)
	require.NoError(t, err)
	return p
}

// oaProvider is shorthand for an OpenAI-dialect test provider. Providers retry
// by default, so tests inject the fast no-sleep policy -- otherwise every
// retrying test would wait out real exponential backoff.
func oaProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return oaProviderRetry(t, baseURL, retryTestPolicy(4))
}

// oaProviderRetry is oaProvider with an explicit retry policy.
func oaProviderRetry(t *testing.T, baseURL string, retry *RetryPolicy) Provider {
	t.Helper()
	return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retry}})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retryTestPolicy(4)}})
}
