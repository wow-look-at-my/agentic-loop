package agentic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// anSSEHandler serves named Anthropic stream events, capturing the request.
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
	return [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"` + text + `"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

func TestAnthropicRequestBody(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := &Anthropic{
		BaseURL:   srv.URL,
		APIKey:    "sk-test",
		UserAgent: "agentic-test/1.0",
		Headers:   map[string]string{"X-Custom": "yes"},
	}
	messages := []Message{
		{Role: RoleUser, Content: "look this up"},
		{
			Role:    RoleAssistant,
			Content: "on it",
			Thinking: []ThinkingBlock{
				{Text: "pondering", Signature: "sig-1"},
				{Redacted: "opaque-blob"},
			},
			ToolCalls: []ToolCall{
				{ID: "toolu_1", Name: "lookup", Arguments: `{"city":"Oslo"}`},
				{ID: "toolu_2", Name: "lookup", Arguments: ""},
			},
		},
		{Role: RoleTool, Content: "rainy", ToolCallID: "toolu_1"},
		{Role: RoleTool, Content: "lookup failed", ToolCallID: "toolu_2", ToolIsError: true},
	}
	req := Request{
		Model:     "test-model",
		System:    "be brief",
		Messages:  messages,
		Tools:     []Tool{{Name: "lookup", Description: "look things up", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}, {Name: "bare"}},
		MaxTokens: 512,
		Extra:     map[string]any{"thinking": map[string]any{"type": "adaptive"}, "model": "evil", "stream": false},
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	assert.Equal(t, "test-model", body["model"], "reserved Extra keys cannot override the core")
	assert.Equal(t, float64(512), body["max_tokens"])
	assert.Equal(t, true, body["stream"])
	assert.Equal(t, map[string]any{"type": "adaptive"}, body["thinking"], "Extra passthrough, no model gating")

	sys := body["system"].([]any)
	require.Len(t, sys, 1)
	sysBlock := sys[0].(map[string]any)
	assert.Equal(t, "text", sysBlock["type"])
	assert.Equal(t, "be brief", sysBlock["text"])
	assert.Equal(t, map[string]any{"type": "ephemeral"}, sysBlock["cache_control"], "static breakpoint on the system block")

	msgs := body["messages"].([]any)
	require.Len(t, msgs, 3, "two consecutive tool messages fold into one user message")

	user := msgs[0].(map[string]any)
	assert.Equal(t, "user", user["role"])
	assert.Equal(t, "look this up", user["content"])

	asst := msgs[1].(map[string]any)
	assert.Equal(t, "assistant", asst["role"])
	blocks := asst["content"].([]any)
	require.Len(t, blocks, 5)
	think := blocks[0].(map[string]any)
	assert.Equal(t, "thinking", think["type"], "thinking blocks replayed FIRST")
	assert.Equal(t, "pondering", think["thinking"])
	assert.Equal(t, "sig-1", think["signature"], "signature replayed unchanged")
	redacted := blocks[1].(map[string]any)
	assert.Equal(t, "redacted_thinking", redacted["type"])
	assert.Equal(t, "opaque-blob", redacted["data"])
	text := blocks[2].(map[string]any)
	assert.Equal(t, "text", text["type"])
	assert.Equal(t, "on it", text["text"])
	tu := blocks[3].(map[string]any)
	assert.Equal(t, "tool_use", tu["type"])
	assert.Equal(t, "toolu_1", tu["id"])
	assert.Equal(t, "lookup", tu["name"])
	assert.Equal(t, map[string]any{"city": "Oslo"}, tu["input"], "input is the PARSED object, not a JSON string")
	tu2 := blocks[4].(map[string]any)
	assert.Equal(t, map[string]any{}, tu2["input"], "empty arguments parse to {}")

	results := msgs[2].(map[string]any)
	assert.Equal(t, "user", results["role"], "tool results ride as a user message")
	rblocks := results["content"].([]any)
	require.Len(t, rblocks, 2)
	r0 := rblocks[0].(map[string]any)
	assert.Equal(t, "tool_result", r0["type"])
	assert.Equal(t, "toolu_1", r0["tool_use_id"])
	assert.Equal(t, "rainy", r0["content"])
	assert.Equal(t, false, r0["is_error"])
	r1 := rblocks[1].(map[string]any)
	assert.Equal(t, true, r1["is_error"])
	assert.Equal(t, map[string]any{"type": "ephemeral"}, r1["cache_control"],
		"moving breakpoint on the last content block of the last message")
	_, r0Marked := r0["cache_control"]
	assert.False(t, r0Marked, "exactly one moving breakpoint")

	tools := body["tools"].([]any)
	require.Len(t, tools, 2)
	tool0 := tools[0].(map[string]any)
	assert.Equal(t, "lookup", tool0["name"])
	assert.Equal(t, "look things up", tool0["description"])
	assert.Equal(t, "object", tool0["input_schema"].(map[string]any)["type"])
	tool1 := tools[1].(map[string]any)
	assert.Equal(t, map[string]any{"type": "object"}, tool1["input_schema"], "nil schema defaults")

	assert.Equal(t, "sk-test", h.header.Get("x-api-key"))
	assert.Equal(t, "2023-06-01", h.header.Get("anthropic-version"), "default version")
	assert.Equal(t, "agentic-test/1.0", h.header.Get("User-Agent"))
	assert.Equal(t, "yes", h.header.Get("X-Custom"))
	assert.Empty(t, h.header.Get("anthropic-dangerous-direct-browser-access"), "browser-only header never sent")
}

func TestAnthropicCallerTranscriptUnchanged(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL}

	messages := []Message{
		{Role: RoleUser, Content: "hello"},
		{Role: RoleAssistant, Content: "hi", Thinking: []ThinkingBlock{{Text: "t", Signature: "s"}}},
		{Role: RoleUser, Content: "again"},
	}
	want := make([]Message, len(messages))
	copy(want, messages)
	req := Request{Model: "m", Messages: messages, MaxTokens: 64}

	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	_, err = p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	assert.Equal(t, want, messages, "cache markers are applied to a per-request copy; the transcript stays marker-free")

	// And the wire really carried the moving marker both times.
	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	lastMsg := msgs[len(msgs)-1].(map[string]any)
	blocks := lastMsg["content"].([]any)
	tail := blocks[len(blocks)-1].(map[string]any)
	assert.Equal(t, map[string]any{"type": "ephemeral"}, tail["cache_control"])
	assert.Equal(t, "again", tail["text"], "a string tail becomes a one-block array carrying the marker")
}

func TestAnthropicDisableCaching(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL, DisableCaching: true, Version: "2024-01-01"}

	req := Request{
		Model:     "m",
		System:    "sys",
		Messages:  []Message{{Role: RoleUser, Content: "hello"}},
		MaxTokens: 64,
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	assert.NotContains(t, string(h.body), "cache_control", "DisableCaching removes all markers")
	assert.Equal(t, "2024-01-01", h.header.Get("anthropic-version"), "custom version honored")
}

func TestAnthropicMaxTokensRequired(t *testing.T) {
	p := &Anthropic{BaseURL: "http://placeholder.invalid"}
	comp, err := p.Complete(context.Background(), Request{Model: "m"}, nil)
	assert.Nil(t, comp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "max_tokens")
	assert.False(t, IsTransient(err), "a validation failure is never retried")
}

func TestAnthropicStreamDecode(t *testing.T) {
	h := &anSSEHandler{events: [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"cache_read_input_tokens":40,"cache_creation_input_tokens":7,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ponder"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"ing"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"content_block_start", `{"type":"content_block_start","index":1,"content_block":{"type":"redacted_thinking","data":"opaque-blob"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":1}`},
		{"content_block_start", `{"type":"content_block_start","index":2,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Hello, "}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"world"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":2}`},
		{"content_block_start", `{"type":"content_block_start","index":3,"content_block":{"type":"tool_use","id":"toolu_1","name":"lookup"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"{\"city\":"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":3,"delta":{"type":"input_json_delta","partial_json":"\"Oslo\"}"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":3}`},
		{"ping", `{"type":"ping"}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":9}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	var texts, reasonings []string
	var usages []Usage
	ev := &StreamEvents{
		OnText:      func(s string) { texts = append(texts, s) },
		OnReasoning: func(s string) { reasonings = append(reasonings, s) },
		OnUsage:     func(u Usage) { usages = append(usages, u) },
	}
	p := &Anthropic{BaseURL: srv.URL}
	comp, err := p.Complete(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}, MaxTokens: 128}, ev)
	require.NoError(t, err)

	assert.Equal(t, "Hello, world", comp.Message.Content)
	assert.Equal(t, []string{"Hello, ", "world"}, texts)
	assert.Equal(t, []string{"ponder", "ing"}, reasonings)

	require.Len(t, comp.Message.Thinking, 2)
	assert.Equal(t, ThinkingBlock{Text: "pondering", Signature: "sig-1"}, comp.Message.Thinking[0])
	assert.Equal(t, ThinkingBlock{Redacted: "opaque-blob"}, comp.Message.Thinking[1])

	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, ToolCall{ID: "toolu_1", Name: "lookup", Arguments: `{"city":"Oslo"}`}, comp.Message.ToolCalls[0],
		"input_json_delta fragments accumulate as the raw JSON text")
	assert.Equal(t, StopToolUse, comp.StopReason)

	// input_tokens EXCLUDES cached tokens: full prompt = input + read + write.
	assert.Equal(t, 50, comp.Usage.PromptTokens)
	assert.Equal(t, 9, comp.Usage.CompletionTokens)
	assert.Equal(t, 59, comp.Usage.TotalTokens)
	require.NotNil(t, comp.Usage.CacheReadTokens)
	assert.Equal(t, 40, *comp.Usage.CacheReadTokens)
	require.NotNil(t, comp.Usage.CacheWriteTokens)
	assert.Equal(t, 7, *comp.Usage.CacheWriteTokens)
	assert.Equal(t, 40, comp.Usage.CachedTokens())
	require.Len(t, usages, 2, "message_start and message_delta each fire OnUsage")
	assert.Equal(t, 1, usages[0].CompletionTokens)
	assert.Equal(t, 9, usages[1].CompletionTokens, "message_delta output_tokens overwrites (cumulative)")
}

func TestAnthropicTriStateCacheFields(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("hi")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL}
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.NoError(t, err)
	assert.Nil(t, comp.Usage.CacheReadTokens, "absent cache fields stay nil, never zero-filled")
	assert.Nil(t, comp.Usage.CacheWriteTokens)
	assert.Equal(t, 3, comp.Usage.PromptTokens)
	assert.Equal(t, 2, comp.Usage.CompletionTokens)
}

func TestAnthropicErrorEvent(t *testing.T) {
	h := &anSSEHandler{events: [][2]string{
		{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL}
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	assert.Nil(t, comp, "nothing streamed before the error event")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Overloaded")
}

func TestAnthropicNonOKOverflow(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`, http.StatusBadRequest)
	}))
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL}
	_, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.Error(t, err)
	assert.True(t, IsContextOverflow(err))
	assert.False(t, IsTransient(err))
}

func TestAnthropicAssistantOnlyTextNoToolContinuation(t *testing.T) {
	// A text-only assistant message still maps to a block array, and a lone
	// user tail (string content) receives the moving marker as a one-block
	// array — pinned by TestAnthropicCallerTranscriptUnchanged. Here: the
	// assistant tail case.
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := &Anthropic{BaseURL: srv.URL}
	req := Request{
		Model:     "m",
		Messages:  []Message{{Role: RoleUser, Content: "q"}, {Role: RoleAssistant, Content: "a"}},
		MaxTokens: 64,
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	asst := msgs[1].(map[string]any)
	blocks := asst["content"].([]any)
	require.Len(t, blocks, 1)
	tail := blocks[0].(map[string]any)
	assert.Equal(t, "text", tail["type"])
	assert.Equal(t, "a", tail["text"])
	assert.Equal(t, map[string]any{"type": "ephemeral"}, tail["cache_control"])
}
