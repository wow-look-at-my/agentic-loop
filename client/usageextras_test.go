package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Part A extension: provider-reported extras on Completion
// (RawUsage verbatim, ReasoningTokens, CostUsd)
// ---------------------------------------------------------------------------

func TestOpenAIStreamUsageExtras(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}}]}`,
		`{"usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":3},"cost":0.00012,"vendor_note":"survives"}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := oaProvider(t, srv.URL)

	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.True(t, comp.Streamed)
	assert.Equal(t, 15, comp.Usage.TotalTokens)
	require.NotNil(t, comp.ReasoningTokens)
	assert.Equal(t, 3, *comp.ReasoningTokens, "completion_tokens_details.reasoning_tokens")
	require.NotNil(t, comp.CostUsd)
	assert.Equal(t, 0.00012, *comp.CostUsd, "usage.cost")
	require.NotNil(t, comp.RawUsage)
	// Verbatim: provider extras the normalized Usage drops survive in the raw object.
	assert.Contains(t, string(comp.RawUsage), "vendor_note")
	assert.Contains(t, string(comp.RawUsage), "reasoning_tokens")
	assert.JSONEq(t,
		`{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15,"completion_tokens_details":{"reasoning_tokens":3},"cost":0.00012,"vendor_note":"survives"}`,
		string(comp.RawUsage))
}

func TestOpenAIEstimatedCostAndCostPrecedence(t *testing.T) {
	// DeepInfra-style estimated_cost is honored when cost is absent...
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"x"}}]}`,
		`{"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"estimated_cost":0.005}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	require.NotNil(t, comp.CostUsd)
	assert.Equal(t, 0.005, *comp.CostUsd)

	//...and cost wins when both are present.
	h2 := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"x"}}]}`,
		`{"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2,"cost":0.001,"estimated_cost":0.009}}`,
	}}
	srv2 := httptest.NewServer(h2)
	defer srv2.Close()
	comp, err = oaProvider(t, srv2.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	require.NotNil(t, comp.CostUsd)
	assert.Equal(t, 0.001, *comp.CostUsd)
}

func TestOpenAIUsageExtrasTriStateAbsent(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}`,
		`{"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Nil(t, comp.ReasoningTokens, "no completion_tokens_details -> nil, never zero")
	assert.Nil(t, comp.CostUsd, "no cost field -> nil")
	assert.JSONEq(t, `{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}`, string(comp.RawUsage))
}

func TestOpenAIPlainJSONUsageExtras(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"choices": [{"message": {"role": "assistant", "content": "plain", "reasoning": "plain thinking"}, "finish_reason": "stop"}],
			"usage": {"prompt_tokens": 12, "completion_tokens": 7, "total_tokens": 19,
				"completion_tokens_details": {"reasoning_tokens": 2}, "cost": 0.0003, "extra": 1}
		}`))
	}))
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.False(t, comp.Streamed)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "plain thinking", comp.Message.Thinking[0].Text)
	require.NotNil(t, comp.ReasoningTokens)
	assert.Equal(t, 2, *comp.ReasoningTokens)
	require.NotNil(t, comp.CostUsd)
	assert.Equal(t, 0.0003, *comp.CostUsd)
	require.NotNil(t, comp.RawUsage)
	assert.JSONEq(t,
		`{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19,"completion_tokens_details":{"reasoning_tokens":2},"cost":0.0003,"extra":1}`,
		string(comp.RawUsage))
}

func TestAnthropicRawUsageSynthesizedWireShape(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := anProvider(t, srv.URL).Complete(
		context.Background(),
		Request{Model: "m", MaxTokens: 256},
		nil,
	)
	require.NoError(t, err)
	assert.True(t, comp.Streamed)
	// message_start {input_tokens:3, output_tokens:1} + message_delta {output_tokens:2}.
	assert.JSONEq(t, `{"input_tokens":3,"output_tokens":2}`, string(comp.RawUsage))
	assert.Equal(t, 5, comp.Usage.TotalTokens)
	assert.Nil(t, comp.ReasoningTokens, "anthropic never reports reasoning tokens")
	assert.Nil(t, comp.CostUsd, "anthropic never reports a cost")
}

func TestAnthropicRawUsageCacheSiblings(t *testing.T) {
	h := &anSSEHandler{events: [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":100,"cache_read_input_tokens":40,"cache_creation_input_tokens":10}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":25}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := anProvider(t, srv.URL).Complete(
		context.Background(),
		Request{Model: "m", MaxTokens: 256},
		nil,
	)
	require.NoError(t, err)
	// input_tokens EXCLUDES cached tokens; the cache fields are siblings.
	assert.JSONEq(t,
		`{"cache_creation_input_tokens":10,"cache_read_input_tokens":40,"input_tokens":100,"output_tokens":25}`,
		string(comp.RawUsage))
	// Normalized usage: full prompt = input + read + creation.
	assert.Equal(t, 150, comp.Usage.PromptTokens)
	assert.Equal(t, 25, comp.Usage.CompletionTokens)
}

// ---------------------------------------------------------------------------
// Part A extension: Completion.Streamed + the plain-JSON fallback
// ---------------------------------------------------------------------------

func TestOpenAIStreamedFlag(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.True(t, comp.Streamed, "an SSE response streamed")
	assert.Equal(t, "hi", comp.Message.Content)
}

func TestOpenAINonStreamingJSONFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json; charset=utf-8")
		_, _ = w.Write([]byte(`{"choices":[{"message":{"role":"assistant","content":"plain answer","tool_calls":[{"id":"c1","type":"function","function":{"name":"Grep","arguments":"{\"pattern\":\"Foo\"}"}}]},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":12,"completion_tokens":7,"total_tokens":19}}`))
	}))
	defer srv.Close()

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.False(t, comp.Streamed, "a plain JSON answer was not streamed")
	assert.Equal(t, "plain answer", comp.Message.Content)
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, ToolCall{ID: "c1", Name: "Grep", Arguments: `{"pattern":"Foo"}`}, comp.Message.ToolCalls[0])
	assert.Equal(t, StopToolUse, comp.StopReason)
	assert.Equal(t, 19, comp.Usage.TotalTokens)
	assert.True(t, comp.UsageReported)
}

func TestOpenAINonStreamingContentTypeInsensitive(t *testing.T) {
	// The content-type check is a substring match, so a parameterized event-stream header still routes through SSE.
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"sse"},"finish_reason":"stop"}]}`}}
	h.contentType = "text/event-stream; charset=utf-8"
	srv := httptest.NewServer(h)
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.True(t, comp.Streamed)
	assert.Equal(t, "sse", comp.Message.Content)
}

func TestAnthropicNonStreamingJSONFallback(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"content":[
			{"type":"thinking","thinking":"hmm","signature":"sig1"},
			{"type":"text","text":"hello "},
			{"type":"text","text":"world"},
			{"type":"tool_use","id":"t1","name":"Glob","input":{"pattern":"*.go"}}
		],"stop_reason":"tool_use","usage":{"input_tokens":100,"output_tokens":25,"cache_read_input_tokens":40}}`))
	}))
	defer srv.Close()

	comp, err := anProvider(t, srv.URL).Complete(
		context.Background(),
		Request{Model: "m", MaxTokens: 256},
		nil,
	)
	require.NoError(t, err)
	assert.False(t, comp.Streamed, "a plain JSON answer was not streamed")
	assert.Equal(t, "hello world", comp.Message.Content)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, ThinkingBlock{Text: "hmm", Signature: "sig1"}, comp.Message.Thinking[0])
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, ToolCall{ID: "t1", Name: "Glob", Arguments: `{"pattern":"*.go"}`}, comp.Message.ToolCalls[0])
	assert.Equal(t, StopToolUse, comp.StopReason)
	assert.True(t, comp.UsageReported)
	assert.Equal(t, 140, comp.Usage.PromptTokens, "input + cache_read, cache fields excluded from input_tokens")
	assert.JSONEq(t, `{"cache_read_input_tokens":40,"input_tokens":100,"output_tokens":25}`, string(comp.RawUsage))
}

func TestAnthropicNonStreamingMalformedBody(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{`))
	}))
	defer srv.Close()
	comp, err := anProvider(t, srv.URL).Complete(
		context.Background(),
		Request{Model: "m", MaxTokens: 256},
		nil,
	)
	assert.Nil(t, comp)
	require.Error(t, err)
	assert.False(t, IsTransient(err), "a malformed body is permanent, not retried")
}

// Part A extension: OpenAIConfig.PromptCache (static system + moving tail, clean transcript).

// countCacheControls counts the literal `"cache_control"` occurrences in a
// serialized request body.
func countCacheControls(t *testing.T, body []byte) int {
	t.Helper()
	return strings.Count(string(body), `"cache_control"`)
}

func TestOpenAIPromptCacheBreakpoints(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{
		ProviderConfig: ProviderConfig{BaseURL: srv.URL, Retry: retryTestPolicy(4)},
		PromptCache:    true,
	})
	messages := []Message{{Role: RoleUser, Content: "q"}}
	_, err := p.Complete(context.Background(), Request{Model: "m", System: "s", Messages: messages}, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	require.Len(t, msgs, 2)
	// Static: the leading system message's string content is a marked one-block array.
	sys := msgs[0].(map[string]any)
	assert.Equal(t, "system", sys["role"])
	assert.Equal(t, []any{map[string]any{
		"type": "text", "text": "s", "cache_control": map[string]any{"type": "ephemeral"},
	}}, sys["content"])
	// Moving: the tail (user prompt) is marked too.
	user := msgs[1].(map[string]any)
	assert.Equal(t, "user", user["role"])
	assert.Equal(t, []any{map[string]any{
		"type": "text", "text": "q", "cache_control": map[string]any{"type": "ephemeral"},
	}}, user["content"])
	// Exactly breakpoints per request.
	assert.Equal(t, 2, countCacheControls(t, h.body))
	// The stored transcript never carries a marker (per-request copies only).
	assert.Equal(t, []Message{{Role: RoleUser, Content: "q"}}, messages)
}

func TestOpenAIPromptCacheMovingTailOnToolMessage(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{
		ProviderConfig: ProviderConfig{BaseURL: srv.URL, Retry: retryTestPolicy(4)},
		PromptCache:    true,
	})
	messages := []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "Grep", Arguments: `{"pattern":"Foo"}`}}},
		{Role: RoleTool, Content: "src/foo.ts:12:class Foo", ToolCallID: "call_1"},
	}
	_, err := p.Complete(context.Background(), Request{Model: "m", System: "s", Messages: messages}, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	require.Len(t, msgs, 4)
	assert.Equal(t, 2, countCacheControls(t, h.body))
	// The moving marker landed on the role:"tool" tail, in openai shape.
	tail := msgs[3].(map[string]any)
	assert.Equal(t, "tool", tail["role"])
	assert.Equal(t, []any{map[string]any{
		"type": "text", "text": "src/foo.ts:12:class Foo", "cache_control": map[string]any{"type": "ephemeral"},
	}}, tail["content"])
	assert.Equal(t, "call_1", tail["tool_call_id"])
	// The stored transcript never carries a marker.
	assert.Equal(t, []Message{
		{Role: RoleUser, Content: "q"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "Grep", Arguments: `{"pattern":"Foo"}`}}},
		{Role: RoleTool, Content: "src/foo.ts:12:class Foo", ToolCallID: "call_1"},
	}, messages)
}

func TestOpenAIPromptCacheDefaultsOff(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := oaProvider(t, srv.URL) // PromptCache unset -> false
	_, err := p.Complete(context.Background(),
		Request{Model: "m", System: "s", Messages: []Message{{Role: RoleUser, Content: "q"}}}, nil)
	require.NoError(t, err)
	assert.Equal(t, 0, countCacheControls(t, h.body),
		"plain OpenAI-compatible servers must never see a cache_control marker by default")
}

// ---------------------------------------------------------------------------
// Part A extension: OpenAIConfig.ReplayReasoning
// (gateway extension: message.reasoning on assistant messages; default off)
// ---------------------------------------------------------------------------

func TestOpenAIReplayReasoning(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{
		ProviderConfig:  ProviderConfig{BaseURL: srv.URL, Retry: retryTestPolicy(4)},
		ReplayReasoning: true,
	})
	messages := []Message{
		{Role: RoleAssistant, Content: "checked", Thinking: []ThinkingBlock{{Text: "let me think"}}},
		{Role: RoleUser, Content: "and?"},
	}
	_, err := p.Complete(context.Background(), Request{Model: "m", Messages: messages}, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	require.Len(t, msgs, 2)
	assistant := msgs[0].(map[string]any)
	assert.Equal(t, "assistant", assistant["role"])
	assert.Equal(t, "let me think", assistant["reasoning"],
		"the accumulated reasoning rides back as message.reasoning")
	assert.Equal(t, "checked", assistant["content"])
	// Only assistant messages with reasoning carry the field.
	user := msgs[1].(map[string]any)
	_, hasReasoning := user["reasoning"]
	assert.False(t, hasReasoning)
}

func TestOpenAIReplayReasoningDefaultsOff(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := oaProvider(t, srv.URL) // ReplayReasoning unset -> false
	_, err := p.Complete(context.Background(), Request{Model: "m", Messages: []Message{
		{Role: RoleAssistant, Content: "checked", Thinking: []ThinkingBlock{{Text: "let me think"}}},
	}}, nil)
	require.NoError(t, err)
	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	assistant := msgs[0].(map[string]any)
	_, hasReasoning := assistant["reasoning"]
	assert.False(t, hasReasoning,
		"strict OpenAI-compatible servers must never see a reasoning field by default")
}

func TestOpenAIReplayReasoningSkipsEmptyOrRedacted(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := mustOpenAI(t, OpenAIConfig{
		ProviderConfig:  ProviderConfig{BaseURL: srv.URL, Retry: retryTestPolicy(4)},
		ReplayReasoning: true,
	})
	messages := []Message{
		{Role: RoleAssistant, Content: "a", Thinking: []ThinkingBlock{{Text: ""}}},
		{Role: RoleAssistant, Content: "b", Thinking: []ThinkingBlock{{Redacted: "opaque"}}},
	}
	_, err := p.Complete(context.Background(), Request{Model: "m", Messages: messages}, nil)
	require.NoError(t, err)
	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	for _, raw := range msgs {
		m := raw.(map[string]any)
		_, hasReasoning := m["reasoning"]
		assert.False(t, hasReasoning, "empty and redacted reasoning never replays as text")
	}
}
