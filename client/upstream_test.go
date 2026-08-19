package client

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// Fake upstreams, for the tests that drive a constructed provider over HTTP
// and assert on the folded Completion -- which is what a Go caller holds, and
// the one thing the core package cannot report on.

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

// minimalAnEvents is a bare valid stream: one text block.
func minimalAnEvents(text string) [][2]string {
	body, err := json.Marshal(map[string]any{
		"type": "content_block_delta", "index": 0,
		"delta": map[string]any{"type": "text_delta", "text": text},
	})
	if err != nil {
		panic(err)
	}
	return [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", string(body)},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

// bodyMap decodes a captured request body as a JSON object.
func bodyMap(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	require.NoError(t, json.Unmarshal(body, &m))
	return m
}

// retryTestPolicy retries fast with no real sleeping.
func retryTestPolicy(attempts int) *RetryPolicy {
	return &RetryPolicy{MaxAttempts: attempts, BaseDelay: time.Millisecond,
		Sleep: func(context.Context, time.Duration) error { return nil }}
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
	return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retryTestPolicy(4)}})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retryTestPolicy(4)}})
}
