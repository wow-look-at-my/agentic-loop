package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// jsonServer answers every request with JSON body, which is what an
// upstream that ignored stream:true (or a proxy that buffered the stream) hands
// back: a that is not an event stream.
func jsonServer(t *testing.T, body string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	t.Cleanup(srv.Close)
	return srv
}

func TestOpenAINonStreamResponse(t *testing.T) {
	srv := jsonServer(t, `{
	  "choices": [{
	    "finish_reason": "tool_calls",
	    "message": {
	      "content": "on it",
	      "reasoning": "the user wants a search",
	      "tool_calls": [{"id":"c1","function":{"name":"grep","arguments":"{\"q\":\"x\"}"}}]
	    }
	  }],
	  "usage": {"prompt_tokens":11,"completion_tokens":4,"total_tokens":15,"cost":0.5}
	}`)

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.False(t, comp.Streamed, "a buffered answer did not stream")
	assert.Equal(t, "on it", comp.Message.Content)
	assert.Equal(t, StopToolUse, comp.StopReason)
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "grep", comp.Message.ToolCalls[0].Name)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "the user wants a search", comp.Message.Thinking[0].Text)

	require.Len(t, comp.Usages, 1)
	u := comp.Usages[0]
	assert.Equal(t, 11, u.PromptTokens)
	require.NotNil(t, u.CostUsd)
	assert.InDelta(t, 0.5, *u.CostUsd, 1e-9)
	assert.NotEmpty(t, u.Raw, "the provider's own usage object is kept verbatim")
}

// A response with no finish_reason still has to say how it ended: the shape of
// what came back is the only evidence left.
func TestOpenAINonStreamInfersStopReason(t *testing.T) {
	srv := jsonServer(t, `{"choices":[{"message":{"content":"done"}}]}`)
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, StopEndTurn, comp.StopReason)

	empty := jsonServer(t, `{"choices":[]}`)
	_, err = oaProvider(t, empty.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.Error(t, err, "no choices is not an answer")
}

func TestAnthropicNonStreamResponse(t *testing.T) {
	srv := jsonServer(t, `{
	  "stop_reason": "tool_use",
	  "content": [
	    {"type":"thinking","thinking":"weighing it","signature":"sig-1"},
	    {"type":"redacted_thinking","data":"AAAA"},
	    {"type":"text","text":"here"},
	    {"type":"tool_use","id":"c1","name":"grep","input":{"q":"x"}}
	  ],
	  "usage": {"input_tokens":9,"output_tokens":3,"cache_read_input_tokens":7}
	}`)

	comp, err := anProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.NoError(t, err)
	assert.False(t, comp.Streamed)
	assert.Equal(t, StopToolUse, comp.StopReason)

	// The blocks keep the order they arrived in: a reply reads differently when its thinking sits after its text.
	kinds := make([]PartKind, 0, len(comp.Message.Parts))
	for _, p := range comp.Message.Parts {
		kinds = append(kinds, p.Kind())
	}
	assert.Equal(t, []PartKind{
		PartKindThinking, PartKindRedactedThinking, PartKindText, PartKindToolCall,
	}, kinds)
	assert.Equal(t, "here", comp.Message.Content)
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.JSONEq(t, `{"q":"x"}`, comp.Message.ToolCalls[0].Arguments)

	require.Len(t, comp.Usages, 1)
	require.NotNil(t, comp.Usages[0].CacheReadTokens)
	assert.Equal(t, 7, *comp.Usages[0].CacheReadTokens)
}

// A tool_use block with no input at all still has to produce arguments a caller
// can hand to a decoder.
func TestAnthropicNonStreamEmptyToolInput(t *testing.T) {
	srv := jsonServer(t, `{"content":[{"type":"tool_use","id":"c1","name":"noop"}]}`)
	comp, err := anProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, nil)
	require.NoError(t, err)
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "{}", comp.Message.ToolCalls[0].Arguments)
	assert.Equal(t, StopToolUse, comp.StopReason, "a call is how it ended, whatever the body omitted")
}
