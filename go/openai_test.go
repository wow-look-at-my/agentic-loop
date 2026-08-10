package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

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

func bodyMap(t *testing.T, body []byte) map[string]any {
	t.Helper()
	var m map[string]any
	require.NoError(t, json.Unmarshal(body, &m))
	return m
}

func TestOpenAIRequestBody(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := mustOpenAI(t, OpenAIConfig{
		ProviderConfig: ProviderConfig{
			BaseURL:   srv.URL + "/", // trailing slash must not double up
			APIKey:    "test-key",
			UserAgent: "agentic-test/1.0",
			Headers:   map[string]string{"X-Custom": "yes"},
		},
		SelfHosted: true,
	})
	req := Request{
		Model:  "test-model",
		System: "be helpful",
		Messages: []Message{
			{Role: RoleUser, Content: "hi"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "srch", Arguments: `{"q":"x"}`}}},
			{Role: RoleTool, Content: "", ToolCallID: "call_1"},
			{Role: RoleAssistant, Content: "found it", ToolCalls: []ToolCall{{ID: "call_2", Name: "srch", Arguments: "{}"}}},
		},
		Tools: []ToolDecl{
			{Name: "srch", Description: "search", InputSchema: json.RawMessage(`{"type":"object","properties":{"q":{"type":"string"}}}`)},
			{Name: "noschema"},
		},
		MaxTokens: 2048,
		Extra:     map[string]any{"reasoning_effort": "high", "model": "evil", "stream": false},
		CacheKey:  "conv-42",
	}
	comp, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	assert.Equal(t, StopEndTurn, comp.StopReason)

	body := bodyMap(t, h.body)
	assert.Equal(t, "test-model", body["model"], "reserved Extra keys cannot override the core")
	assert.Equal(t, true, body["stream"])
	assert.Equal(t, "high", body["reasoning_effort"], "Extra passthrough")
	assert.Equal(t, float64(2048), body["max_tokens"])
	assert.Equal(t, "conv-42", body["prompt_cache_key"])
	assert.Equal(t, true, body["cache_prompt"], "SelfHosted adds cache_prompt")
	assert.Equal(t, map[string]any{"include_usage": true}, body["stream_options"], "defaulted when Extra has none")

	msgs := body["messages"].([]any)
	require.Len(t, msgs, 5)
	sys := msgs[0].(map[string]any)
	assert.Equal(t, "system", sys["role"])
	assert.Equal(t, "be helpful", sys["content"])

	toolcallMsg := msgs[2].(map[string]any)
	assert.Equal(t, "assistant", toolcallMsg["role"])
	_, hasContent := toolcallMsg["content"]
	assert.False(t, hasContent, "assistant with tool_calls and empty content omits the content field")
	tcs := toolcallMsg["tool_calls"].([]any)
	tc0 := tcs[0].(map[string]any)
	assert.Equal(t, "call_1", tc0["id"])
	assert.Equal(t, "function", tc0["type"])
	fn := tc0["function"].(map[string]any)
	assert.Equal(t, "srch", fn["name"])
	assert.Equal(t, `{"q":"x"}`, fn["arguments"])

	toolMsg := msgs[3].(map[string]any)
	assert.Equal(t, "tool", toolMsg["role"])
	assert.Equal(t, "call_1", toolMsg["tool_call_id"])
	content, hasContent := toolMsg["content"]
	assert.True(t, hasContent, "an empty tool result still serializes content")
	assert.Equal(t, "", content)

	// Assistant WITH content and tool calls keeps its content.
	mixed := msgs[4].(map[string]any)
	assert.Equal(t, "found it", mixed["content"])

	tools := body["tools"].([]any)
	require.Len(t, tools, 2)
	tool0 := tools[0].(map[string]any)
	assert.Equal(t, "function", tool0["type"])
	f0 := tool0["function"].(map[string]any)
	assert.Equal(t, "srch", f0["name"])
	assert.Equal(t, "search", f0["description"])
	assert.Equal(t, "object", f0["parameters"].(map[string]any)["type"])
	f1 := tools[1].(map[string]any)["function"].(map[string]any)
	assert.Equal(t, map[string]any{"type": "object"}, f1["parameters"], "nil schema defaults to an empty object schema")

	assert.Equal(t, "Bearer test-key", h.header.Get("Authorization"))
	assert.Equal(t, "agentic-test/1.0", h.header.Get("User-Agent"))
	assert.Equal(t, "yes", h.header.Get("X-Custom"))
	assert.Equal(t, "application/json", h.header.Get("Content-Type"))
}

func TestOpenAIMessageMarshalPin(t *testing.T) {
	b, err := json.Marshal(oaMessage{Role: "tool", Content: "", ToolCallID: "call_1"})
	require.NoError(t, err)
	assert.JSONEq(t, `{"role":"tool","content":"","tool_call_id":"call_1"}`, string(b))
}

func TestOpenAIRequestBodyDefaults(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"x"}}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := oaProvider(t, srv.URL)
	req := Request{
		Model:    "m",
		Messages: []Message{{Role: RoleUser, Content: "q"}},
		Extra:    map[string]any{"stream_options": map[string]any{"include_usage": false}},
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	assert.Equal(t, map[string]any{"include_usage": false}, body["stream_options"],
		"a caller-supplied stream_options survives verbatim")
	_, hasMax := body["max_tokens"]
	assert.False(t, hasMax, "MaxTokens 0 omits the field")
	_, hasCachePrompt := body["cache_prompt"]
	assert.False(t, hasCachePrompt, "cache_prompt only when SelfHosted")
	_, hasCacheKey := body["prompt_cache_key"]
	assert.False(t, hasCacheKey)
	_, hasTools := body["tools"]
	assert.False(t, hasTools, "no tools field when none advertised")
	assert.Empty(t, h.header.Get("Authorization"), "no bearer without an APIKey")
}

func TestOpenAIStreamDecode(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"prompt_progress":{"total":100,"cache":20,"processed":50,"time_ms":123}}`,
		`not-json keep-alive noise`,
		`{"choices":[{"delta":{"role":"assistant"}}],"usage":{"prompt_tokens":10,"completion_tokens":0,"total_tokens":10}}`,
		`{"choices":[{"delta":{"content":"Hel"}}],"usage":{"prompt_tokens":10,"completion_tokens":1,"total_tokens":11}}`,
		`{"choices":[{"delta":{"reasoning_content":"think1","reasoning":"shadowed"}}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}`,
		`{"choices":[{"delta":{"reasoning":"think2"}}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13}}`,
		`{"choices":[{"delta":{"content":"lo"}}],"usage":{"prompt_tokens":10,"completion_tokens":4,"total_tokens":14}}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_a","type":"function","function":{"name":"srch","arguments":"{\"q\":"}}]}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"x\"}"}},{"index":1,"id":"call_b","function":{"name":"other","arguments":"{}"}}]}}]}`,
		`{"choices":[{"delta":{},"finish_reason":"tool_calls"}]}`,
		`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":30,"prompt_tokens_details":{"cached_tokens":8}}}`,
		`{"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	var texts, reasonings []string
	var usages []Usage
	var progresses []PromptProgress
	ev := &StreamEvents{
		OnText:      func(s string) error { texts = append(texts, s); return nil },
		OnReasoning: func(s string) error { reasonings = append(reasonings, s); return nil },
		OnUsage:     func(u Usage) error { usages = append(usages, u); return nil },
		OnProgress:  func(p PromptProgress) error { progresses = append(progresses, p); return nil },
	}
	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}}, ev)
	require.NoError(t, err)

	assert.Equal(t, "Hello", comp.Message.Content)
	assert.Equal(t, []string{"Hel", "lo"}, texts)
	assert.Equal(t, []string{"think1", "think2"}, reasonings, "reasoning_content wins over reasoning; both field names decoded")
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "think1think2", comp.Message.Thinking[0].Text)

	require.Len(t, comp.Message.ToolCalls, 2)
	assert.Equal(t, ToolCall{ID: "call_a", Name: "srch", Arguments: `{"q":"x"}`}, comp.Message.ToolCalls[0],
		"split argument fragments concatenate")
	assert.Equal(t, ToolCall{ID: "call_b", Name: "other", Arguments: "{}"}, comp.Message.ToolCalls[1])
	assert.Equal(t, StopToolUse, comp.StopReason)

	require.Len(t, progresses, 1)
	assert.Equal(t, PromptProgress{Processed: 50, Total: 100, Cache: 20, TimeMS: 123}, progresses[0])

	// Cumulative snapshots merged newest-wins, never summed; the regressing
	// final snapshot is discarded.
	assert.Equal(t, 10, comp.Usage.PromptTokens)
	assert.Equal(t, 5, comp.Usage.CompletionTokens)
	assert.Equal(t, 30, comp.Usage.TotalTokens, "reasoning surplus preserved")
	require.NotNil(t, comp.Usage.CacheReadTokens)
	assert.Equal(t, 8, *comp.Usage.CacheReadTokens)
	require.NotEmpty(t, usages)
	last := usages[len(usages)-1]
	assert.Equal(t, 30, last.TotalTokens, "OnUsage saw the merged view; regression discarded")
	for i := 1; i < len(usages); i++ {
		assert.GreaterOrEqual(t, usageEvidence(&usages[i]), usageEvidence(&usages[i-1]), "merged usage view is monotonic")
	}
}

func TestOpenAIStopReasonMapping(t *testing.T) {
	cases := []struct {
		finish string
		want   string
	}{
		{"stop", StopEndTurn},
		{"tool_calls", StopToolUse},
		{"length", StopMaxTokens},
		{"content_filter", "content_filter"},
	}
	for _, tc := range cases {
		t.Run(tc.finish, func(t *testing.T) {
			h := &sseHandler{payloads: []string{
				`{"choices":[{"delta":{"content":"x"},"finish_reason":"` + tc.finish + `"}]}`,
			}}
			srv := httptest.NewServer(h)
			defer srv.Close()
			p := oaProvider(t, srv.URL)
			comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
			require.NoError(t, err)
			assert.Equal(t, tc.want, comp.StopReason)
		})
	}
}

func TestOpenAINonOK(t *testing.T) {
	t.Run("400 overflow flagged", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, `{"error":{"message":"This model's maximum context length is 8192 tokens"}}`, http.StatusBadRequest)
		}))
		defer srv.Close()
		p := oaProvider(t, srv.URL)
		comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
		assert.Nil(t, comp)
		require.Error(t, err)
		var ae *APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, 400, ae.Status)
		assert.True(t, ae.ContextOverflow)
		assert.True(t, IsContextOverflow(err))
		assert.False(t, IsTransient(err))
		assert.Contains(t, err.Error(), "maximum context length", "body embedded in the error text")
	})

	t.Run("503 transient", func(t *testing.T) {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(http.StatusServiceUnavailable)
		}))
		defer srv.Close()
		p := oaProvider(t, srv.URL)
		_, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
		require.Error(t, err)
		assert.True(t, IsTransient(err))
		var ae *APIError
		require.ErrorAs(t, err, &ae)
		assert.Equal(t, "503 Service Unavailable", ae.Body, "empty body falls back to the status text")
	})
}

func TestOpenAIPartialOnMidStreamDeath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"par"}}],"usage":{"prompt_tokens":4,"completion_tokens":1,"total_tokens":5}}` + "\n\n"))
		fl.Flush()
		panic(http.ErrAbortHandler) // kill the connection mid-stream
	}))
	defer srv.Close()

	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.Error(t, err)
	require.NotNil(t, comp, "partial completion returned alongside the error")
	assert.Equal(t, "par", comp.Message.Content)
	assert.Equal(t, 5, comp.Usage.TotalTokens, "last usage snapshot kept")
}

func TestOpenAIPartialOnCancel(t *testing.T) {
	release := make(chan struct{})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"partial "}}]}` + "\n\n"))
		fl.Flush()
		<-release
	}))
	defer srv.Close()
	defer close(release)

	// Cancel from the client side, the moment the first delta is delivered —
	// deterministic, unlike signaling from the handler (the server may flush
	// before the client has read anything).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	ev := &StreamEvents{OnText: func(string) error { cancel(); return nil }}

	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(ctx, Request{Model: "m"}, ev)
	require.Error(t, err)
	assert.ErrorIs(t, err, context.Canceled)
	require.NotNil(t, comp)
	assert.Equal(t, "partial ", comp.Message.Content)
	assert.False(t, IsTransient(err), "cancellation is never transient")
}

func TestOpenAICleanEOFWithoutDone(t *testing.T) {
	// A stream that ends without [DONE] is still a clean completion.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"done"},"finish_reason":"stop"}]}` + "\n\n"))
	}))
	defer srv.Close()
	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "done", comp.Message.Content)
}

func TestOpenAINetworkError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close() // connection refused from now on
	p := oaProvider(t, url)
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	assert.Nil(t, comp)
	require.Error(t, err)
	assert.True(t, IsTransient(err))
	assert.False(t, errors.Is(err, context.Canceled))
}
