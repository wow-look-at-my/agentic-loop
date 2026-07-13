package agentic

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"sync/atomic"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errSink is the sentinel a consumer's failing callback returns in these
// tests; every abort path must keep it reachable via errors.Is.
var errSink = errors.New("sink closed")

func TestOpenAICallbackErrorAbortsStream(t *testing.T) {
	h := &sseHandler{payloads: []string{
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"srch","arguments":"{}"}}]}}],"usage":{"prompt_tokens":7,"completion_tokens":1,"total_tokens":8}}`,
		`{"choices":[{"delta":{"content":"first"}}]}`,
		`{"choices":[{"delta":{"content":" second — never delivered"}}]}`,
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	var got []string
	ev := &StreamEvents{OnText: func(s string) error {
		got = append(got, s)
		return errSink
	}}
	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}}, ev)

	require.Error(t, err)
	assert.ErrorIs(t, err, errSink, "the sentinel the callback returned surfaces via errors.Is")
	assert.False(t, IsTransient(err), "a sink failure is never transient")
	var ae *APIError
	assert.False(t, errors.As(err, &ae), "a callback error is never an APIError")

	require.NotNil(t, comp, "partial completion returned alongside the error")
	assert.Equal(t, []string{"first"}, got, "the stream read stopped at the failing delta")
	assert.Equal(t, "first", comp.Message.Content, "content so far — including the failing delta — is kept")
	require.Len(t, comp.Message.ToolCalls, 1, "tool calls so far are kept")
	assert.True(t, comp.UsageReported)
	assert.Equal(t, 8, comp.Usage.TotalTokens, "usage so far is kept")
}

func TestOpenAICallbackErrorPerCallbackType(t *testing.T) {
	payloads := []string{
		`{"prompt_progress":{"total":10,"cache":0,"processed":5,"time_ms":1}}`,
		`{"choices":[{"delta":{"reasoning_content":"hmm"}}]}`,
		`{"choices":[{"delta":{"content":"hi"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}`,
		`{"timings":{"prompt_n":1,"prompt_ms":2,"predicted_n":3,"predicted_ms":4}}`,
	}
	cases := []struct {
		name string
		ev   func() *StreamEvents
	}{
		{"OnText", func() *StreamEvents {
			return &StreamEvents{OnText: func(string) error { return errSink }}
		}},
		{"OnReasoning", func() *StreamEvents {
			return &StreamEvents{OnReasoning: func(string) error { return errSink }}
		}},
		{"OnUsage", func() *StreamEvents {
			return &StreamEvents{OnUsage: func(Usage) error { return errSink }}
		}},
		{"OnProgress", func() *StreamEvents {
			return &StreamEvents{OnProgress: func(PromptProgress) error { return errSink }}
		}},
		{"OnTimings", func() *StreamEvents {
			return &StreamEvents{OnTimings: func(Timings) error { return errSink }}
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			h := &sseHandler{payloads: payloads}
			srv := httptest.NewServer(h)
			defer srv.Close()
			p := oaProvider(t, srv.URL)
			comp, err := p.Complete(context.Background(), Request{Model: "m"}, tc.ev())
			require.Error(t, err)
			assert.ErrorIs(t, err, errSink)
			assert.False(t, IsTransient(err))
			assert.NotNil(t, comp, "every callback abort keeps the partial completion")
		})
	}
}

func TestAnthropicCallbackErrorAbortsStream(t *testing.T) {
	h := &anSSEHandler{events: [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"par"}}`},
		{"content_block_delta", `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"tial"}}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	ev := &StreamEvents{OnText: func(string) error { return errSink }}
	p := anProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, ev)
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	assert.False(t, IsTransient(err))
	require.NotNil(t, comp)
	assert.Equal(t, "par", comp.Message.Content)
	assert.True(t, comp.UsageReported, "the message_start usage was seen before the abort")
	assert.Nil(t, comp.Timings, "the Anthropic dialect never reports timings")
}

func TestRunCallbackErrorSingleRequestAndPartialResult(t *testing.T) {
	// Delivery is marked BEFORE the callback runs, so a callback failing on
	// the very FIRST delta still counts as "streamed something": exactly one
	// upstream request, no re-attempt, and the partial transcript survives.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.Header().Set("Content-Type", "text/event-stream")
		fl := w.(http.Flusher)
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"part","tool_calls":[{"index":0,"id":"c1","function":{"name":"x","arguments":"{}"}}]}}]}` + "\n\n"))
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"never"}}]}` + "\n\ndata: [DONE]\n\n"))
		fl.Flush()
	}))
	defer srv.Close()

	cfg := Config{
		Provider: oaProvider(t, srv.URL),
		Retry:    retryTestPolicy(4),
		Events: Events{StreamEvents: StreamEvents{
			OnText: func(string) error { return errSink },
		}},
	}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	assert.Equal(t, int32(1), hits.Load(), "a failing first callback still blocks the re-attempt")
	require.NotNil(t, res, "partial Result returned alongside the error")
	require.Len(t, res.Messages, 2)
	assert.Equal(t, "part", res.Messages[1].Content)
	assert.Nil(t, res.Messages[1].ToolCalls, "never-executed calls are cleared from the finalized turn")
	assert.Equal(t, 1, res.Turns)
}

func TestRunOnToolCallErrorAbortsBatch(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{
			Role:     RoleAssistant,
			Content:  "checking",
			Thinking: []ThinkingBlock{{Text: "hmm"}},
			ToolCalls: []ToolCall{
				{ID: "c1", Name: "alpha", Arguments: "{}"},
				{ID: "c2", Name: "beta", Arguments: "{}"},
			},
		}, StopReason: StopToolUse}},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}, {Name: "beta"}}}
	fails := 0
	cfg := Config{Provider: provider, Tools: exec, Retry: &noSleep, Events: Events{
		OnToolCall: func(c ToolCall) error {
			if c.Name == "beta" {
				fails++
				return errSink
			}
			return nil
		},
	}}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	assert.False(t, IsTransient(err))
	require.NotNil(t, res)
	assert.Equal(t, 1, fails)

	// alpha ran before the abort, but the whole batch is cleared: the
	// assistant keeps content/thinking, loses its tool calls, and alpha's
	// appended result is dropped — no orphan tool calls remain.
	assert.Len(t, exec.executed, 1)
	require.Len(t, res.Messages, 2)
	final := res.Messages[1]
	assert.Equal(t, "checking", final.Content)
	assert.Equal(t, "hmm", final.Thinking[0].Text)
	assert.Nil(t, final.ToolCalls)
	assert.Equal(t, final, res.Final)
}

func TestRunOnToolResultErrorAbortsBatch(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
	}}
	exec := &fakeExec{tools: []Tool{{Name: "alpha"}}}
	cfg := Config{Provider: provider, Tools: exec, Retry: &noSleep, Events: Events{
		OnToolResult: func(ToolCall, ToolResult) error { return errSink },
	}}
	res, err := Run(context.Background(), cfg, Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "go"}}})
	require.Error(t, err)
	assert.ErrorIs(t, err, errSink)
	require.NotNil(t, res)
	assert.Len(t, exec.executed, 1, "the tool DID run; only its delivery failed")
	require.Len(t, res.Messages, 2, "the executed result is dropped so the transcript stays replayable")
	assert.Nil(t, res.Messages[1].ToolCalls)
	assert.Equal(t, RoleAssistant, res.Messages[1].Role)
}

func TestWrapCallbackErrIdempotentAndTransparent(t *testing.T) {
	assert.NoError(t, wrapCallbackErr(nil))
	w := wrapCallbackErr(errSink)
	assert.ErrorIs(t, w, errSink)
	assert.Equal(t, errSink.Error(), w.Error(), "the marker adds no text of its own")
	assert.Same(t, w, wrapCallbackErr(w), "already-marked errors are not re-wrapped")
	assert.False(t, IsTransient(w))
}
