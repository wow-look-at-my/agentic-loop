package loop

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// retryTestPolicy retries fast with no real sleeping.
func retryTestPolicy(attempts int) *RetryPolicy {
	return &RetryPolicy{MaxAttempts: attempts, BaseDelay: time.Millisecond,
		Sleep: func(context.Context, time.Duration) error { return nil }}
}

func okSSE(w http.ResponseWriter, text string) {
	w.Header().Set("Content-Type", "text/event-stream")
	chunk := jsonMust(jsonObj{"choices": jsonArr{jsonObj{"delta": jsonObj{"content": text}, "finish_reason": "stop"}}})
	_, _ = w.Write([]byte("data: " + chunk + "\n\ndata: [DONE]\n\n"))
}

func TestRunOpenAI429ThenSuccess(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}
		okSSE(w, "recovered")
	}))
	defer srv.Close()

	res, err := Run(context.Background(),
		Config{Provider: oaProvider(t, srv.URL)},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.NoError(t, err)
	assert.Equal(t, "recovered", res.Final.Content)
	assert.Equal(t, int32(2), hits.Load())
	assert.Equal(t, 1, res.Turns, "the retried call is one turn")
}

func TestProviderRetriesWithNoPolicyConfigured(t *testing.T) {
	// End to end over HTTP with NOTHING configured — no Retry field, no
	// wrapper, no Run. This is the guarantee: you cannot forget to enable it.
	// Deliberately runs the real DefaultRetry, so it pays one 500ms backoff.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		if hits.Add(1) == 1 {
			http.Error(w, "rate limited", http.StatusTooManyRequests)
			return
		}
		okSSE(w, "recovered")
	}))
	defer srv.Close()

	p := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "recovered", comp.Message.Content)
	assert.Equal(t, int32(2), hits.Load())
}

func TestRunOpenAI524Retried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) == 1 {
			w.WriteHeader(524)
			return
		}
		okSSE(w, "after 524")
	}))
	defer srv.Close()

	res, err := Run(context.Background(),
		Config{Provider: oaProvider(t, srv.URL)},
		Request{Model: "m"})
	require.NoError(t, err)
	assert.Equal(t, "after 524", res.Final.Content)
	assert.Equal(t, int32(2), hits.Load())
}

func TestRunOpenAIOverflowNotRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		http.Error(w, "prompt is too long: 900000 tokens", http.StatusBadRequest)
	}))
	defer srv.Close()

	res, err := Run(context.Background(),
		Config{Provider: oaProvider(t, srv.URL)},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.True(t, IsContextOverflow(err))
	assert.Equal(t, int32(1), hits.Load(), "a context-overflow 400 is never retried")
	require.NotNil(t, res)
	assert.Len(t, res.Messages, 1)
}

func TestRunAnthropicInStreamOverloadRetried(t *testing.T) {
	// An overloaded_error delivered IN-STREAM (HTTP 200 + error event, before
	// any data) maps to a 529 APIError and is retried like any 5xx.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if hits.Add(1) == 1 {
			_, _ = w.Write([]byte("event: error\ndata: " +
				`{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}` + "\n\n"))
			return
		}
		_, _ = w.Write([]byte("event: content_block_delta\ndata: " +
			`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"recovered"}}` + "\n\n" +
			"event: message_delta\ndata: " +
			`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}` + "\n\n"))
	}))
	defer srv.Close()

	res, err := Run(context.Background(),
		Config{Provider: anProvider(t, srv.URL)},
		Request{Model: "m", MaxTokens: 64, Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.NoError(t, err)
	assert.Equal(t, "recovered", res.Final.Content)
	assert.Equal(t, int32(2), hits.Load(), "the in-stream overload was retried once, then succeeded")
}

// countingFailRT is a RoundTripper that always fails, counting attempts.
type countingFailRT struct{ calls atomic.Int32 }

func (rt *countingFailRT) RoundTrip(*http.Request) (*http.Response, error) {
	rt.calls.Add(1)
	return nil, errors.New("dial tcp: connection refused")
}

func TestRunOpenAINetworkErrorRetried(t *testing.T) {
	rt := &countingFailRT{}
	p := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: "http://placeholder.invalid", HTTPClient: &http.Client{Transport: rt}, Retry: retryTestPolicy(3)}})
	_, err := Run(context.Background(), Config{Provider: p}, Request{Model: "m"})
	require.Error(t, err)
	assert.Equal(t, int32(3), rt.calls.Load(), "network errors are retried up to the attempt cap")
}

func TestRunOpenAIPartialStreamNotRetried(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"half"}}]}` + "\n\n"))
		fl.Flush()
		panic(http.ErrAbortHandler)
	}))
	defer srv.Close()

	res, err := Run(context.Background(),
		Config{Provider: oaProvider(t, srv.URL)},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.Equal(t, int32(1), hits.Load(), "no re-attempt after a partial stream")
	require.NotNil(t, res)
	assert.Equal(t, "half", res.Final.Content, "the partial assistant message is finalized into the transcript")
	require.Len(t, res.Messages, 2)
}

func TestRunEndToEndToolRoundTripOverOpenAI(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		if hits.Add(1) == 1 {
			_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"echo","arguments":"{\"s\":\"hi\"}"}}]},"finish_reason":"tool_calls"}]}` + "\n\ndata: [DONE]\n\n"))
			return
		}
		okSSE(w, "echoed")
	}))
	defer srv.Close()

	exec := &fakeExec{tools: []ToolDecl{{Name: "echo"}}}
	res, err := Run(context.Background(),
		Config{Provider: oaProvider(t, srv.URL), Tools: exec.registry(), Approver: allowAll},
		Request{Model: "m", System: "sys", Messages: []Message{{Role: RoleUser, Content: "start"}}})
	require.NoError(t, err)
	assert.Equal(t, int32(2), hits.Load())
	assert.Equal(t, "echoed", res.Final.Content)
	require.Len(t, exec.executed, 1)
	assert.Equal(t, ToolCall{ID: "call_1", Name: "echo", Arguments: `{"s":"hi"}`}, exec.executed[0])
	assert.Equal(t, 2, res.Turns)
	assert.Len(t, res.Usages, 2)
}
