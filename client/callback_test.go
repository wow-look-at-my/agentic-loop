package client

import (
	"context"
	"errors"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// errSink is the sentinel a failing callback returns; every abort path must keep it reachable via errors.Is.
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
