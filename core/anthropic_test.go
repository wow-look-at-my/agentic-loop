package commonai

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
		{"content_block_delta", jsonMust(jsonObj{"type": "content_block_delta", "index": 0, "delta": jsonObj{"type": "text_delta", "text": text}})},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":2}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}
}

func TestAnthropicRequestBody(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()

	p := mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{
		BaseURL:   srv.URL,
		APIKey:    "sk-test",
		UserAgent: "agentic-test/1.0",
		Headers:   map[string]string{"X-Custom": "yes"},
	}})
	messages := []Message{
		{Role: RoleUser, Content: "look this up"},
		{
			Role:    RoleAssistant,
			Content: "on it",
			Thinking: []ThinkingBlock{
				{Text: "pondering", Signature: "sig-1"},
				// Unreplayable: no signature, so sending it as a thinking block would 400 the whole turn.
				{Text: "text-only reasoning from somewhere else"},
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
		Tools:     []ToolDecl{{Name: "lookup", Description: "look things up", InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`)}, {Name: "bare"}},
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
	for _, b := range blocks {
		m := b.(map[string]any)
		if m["type"] == "thinking" {
			assert.NotEmpty(t, m["signature"],
				"an unsigned thinking block is DROPPED, never sent signature-less: Anthropic rejects it, "+
					"so one unreplayable block would fail the whole turn")
		}
	}
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
	p := anProvider(t, srv.URL)

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
	p := mustAnthropic(t, AnthropicConfig{
		ProviderConfig: ProviderConfig{BaseURL: srv.URL},
		Version:        "2024-01-01",
		DisableCaching: true,
	})

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
	p := anProvider(t, "http://placeholder.invalid")
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
		OnText:      func(s string) error { texts = append(texts, s); return nil },
		OnReasoning: func(s string) error { reasonings = append(reasonings, s); return nil },
		OnUsage:     func(u Usage) error { usages = append(usages, u); return nil },
	}
	p := anProvider(t, srv.URL)
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
	assert.Equal(t, 50, firstUsage(comp).PromptTokens)
	assert.Equal(t, 9, firstUsage(comp).CompletionTokens)
	assert.Equal(t, 59, firstUsage(comp).TotalTokens)
	require.NotNil(t, firstUsage(comp).CacheReadTokens)
	assert.Equal(t, 40, *firstUsage(comp).CacheReadTokens)
	require.NotNil(t, firstUsage(comp).CacheWriteTokens)
	assert.Equal(t, 7, *firstUsage(comp).CacheWriteTokens)
	assert.Equal(t, 40, firstUsage(comp).CachedTokens())
	require.Len(t, usages, 2, "message_start and message_delta each fire OnUsage")
	assert.Equal(t, 1, usages[0].CompletionTokens)
	assert.Equal(t, 9, usages[1].CompletionTokens, "message_delta output_tokens overwrites (cumulative)")
}

func TestAnthropicTriStateCacheFields(t *testing.T) {
	h := &anSSEHandler{events: minimalAnEvents("hi")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.NoError(t, err)
	assert.Nil(t, firstUsage(comp).CacheReadTokens, "absent cache fields stay nil, never zero-filled")
	assert.Nil(t, firstUsage(comp).CacheWriteTokens)
	assert.Equal(t, 3, firstUsage(comp).PromptTokens)
	assert.Equal(t, 2, firstUsage(comp).CompletionTokens)
}

func TestAnthropicErrorEventMapsToAPIError(t *testing.T) {
	cases := []struct {
		errType    string
		wantStatus int
		transient  bool
	}{
		{"overloaded_error", 529, true},
		{"rate_limit_error", 429, true},
		{"api_error", 500, true},
		{"invalid_request_error", 400, false},
		{"authentication_error", 401, false},
		{"permission_error", 403, false},
		{"billing_error", 403, false},
		{"not_found_error", 404, false},
		{"request_too_large", 413, false},
		{"mystery_future_error", 500, true},
	}
	for _, tc := range cases {
		t.Run(tc.errType, func(t *testing.T) {
			payload := jsonMust(jsonObj{"type": "error", "error": jsonObj{"type": tc.errType, "message": "boom"}})
			h := &anSSEHandler{events: [][2]string{{"error", payload}}}
			srv := httptest.NewServer(h)
			defer srv.Close()
			p := anProvider(t, srv.URL)
			comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
			assert.Nil(t, comp, "nothing streamed before the error event, so the call stays retryable")
			require.Error(t, err)
			var ae *APIError
			require.ErrorAs(t, err, &ae, "in-stream error events map to APIError")
			assert.Equal(t, tc.wantStatus, ae.Status, "status from the documented error-type table")
			assert.Equal(t, payload, ae.Body, "the raw event JSON is the error body")
			assert.Equal(t, tc.transient, IsTransient(err))
			assert.Contains(t, err.Error(), "boom", "the server's message stays visible in the error text")
		})
	}
}

func TestAnthropicErrorEventOverflowConsistent(t *testing.T) {
	// A 400-mapped in-stream error is checked against the overflow regex just
	// like a non-2xx body.
	h := &anSSEHandler{events: [][2]string{
		{"error", `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	_, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.Error(t, err)
	assert.True(t, IsContextOverflow(err))
	assert.False(t, IsTransient(err))
}

func TestAnthropicErrorEventAfterDataKeepsPartial(t *testing.T) {
	// An error event after data has streamed still classifies (529 transient),
	// but the partial completion rides alongside the error — and Run's
	// nothing-streamed guard is what keeps such a call from being re-sent.
	h := &anSSEHandler{events: [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`},
		{"error", `{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.Error(t, err)
	assert.True(t, IsTransient(err))
	require.NotNil(t, comp, "partial completion returned alongside the error")
	assert.Equal(t, "par", comp.Message.Content)
}

func TestAnthropicNonOKOverflow(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"type":"error","error":{"type":"invalid_request_error","message":"prompt is too long: 210000 tokens > 200000 maximum"}}`, http.StatusBadRequest)
	}))
	defer srv.Close()
	p := anProvider(t, srv.URL)
	_, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.Error(t, err)
	assert.True(t, IsContextOverflow(err))
	assert.False(t, IsTransient(err))
}

func TestAnthropicEmptyTailUnmarked(t *testing.T) {
	// An empty-string tail must not become a marked empty text block — the API rejects those.
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	req := Request{
		Model:     "m",
		Messages:  []Message{{Role: RoleAssistant, Content: "a"}, {Role: RoleUser, Content: ""}},
		MaxTokens: 64,
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	lastMsg := msgs[len(msgs)-1].(map[string]any)
	assert.Equal(t, "", lastMsg["content"], "an empty tail stays a plain string, never an empty text block")
	assert.NotContains(t, string(h.body), "cache_control",
		"no marker anywhere: the tail is unmarkable and there is no system prompt")
}

func TestAnthropicSaysNothingForATurnThatSaidNothing(t *testing.T) {
	// A content-less assistant turn is dropped: an empty text block would fail the whole request.
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	req := Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "first"},
			{Role: RoleAssistant, Content: ""},
			// Thinking with no signature is not replayable, so this turn is empty too.
			{Role: RoleAssistant, Thinking: []ThinkingBlock{{Text: "hm"}}},
			{Role: RoleUser, Content: "second"},
		},
		MaxTokens: 64,
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	require.Len(t, msgs, 2, "both content-less assistant turns are gone")
	assert.Equal(t, "user", msgs[0].(map[string]any)["role"])
	assert.Equal(t, "first", msgs[0].(map[string]any)["content"])
	assert.Equal(t, "user", msgs[1].(map[string]any)["role"])
	blocks := msgs[1].(map[string]any)["content"].([]any)
	require.Len(t, blocks, 1)
	assert.Equal(t, "second", blocks[0].(map[string]any)["text"])
}

func TestAnthropicKeepsAnAssistantTurnThatOnlyCalledATool(t *testing.T) {
	// A tool-call-only turn has no text but must still be replayed, or its tool_result won't attach.
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
	req := Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "go"},
			{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "c1", Name: "ls", Arguments: `{}`}}},
			{Role: RoleTool, ToolCallID: "c1", Content: "out"},
		},
		MaxTokens: 64,
	}
	_, err := p.Complete(context.Background(), req, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	msgs := body["messages"].([]any)
	require.Len(t, msgs, 3)
	asst := msgs[1].(map[string]any)
	assert.Equal(t, "assistant", asst["role"])
	blocks := asst["content"].([]any)
	require.Len(t, blocks, 1)
	assert.Equal(t, "tool_use", blocks[0].(map[string]any)["type"])
}

func TestAnthropicAssistantOnlyTextNoToolContinuation(t *testing.T) {
	// A text-only assistant tail becomes a block array and receives the moving marker.
	h := &anSSEHandler{events: minimalAnEvents("ok")}
	srv := httptest.NewServer(h)
	defer srv.Close()
	p := anProvider(t, srv.URL)
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
