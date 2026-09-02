package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// zeroArgTranscript is one assistant turn that called a tool taking no
// arguments, as a host replays it out of storage: the model sent no argument
// bytes, so Arguments is empty.
func zeroArgTranscript() []Message {
	return []Message{
		{Role: RoleUser, Content: "who am i"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "whoami"}}},
		{Role: RoleTool, ToolCallID: "call_1", Content: "you"},
	}
}

// A zero-argument tool call still carries an arguments field on the openai
// wire. Z.AI answers a function object without one with 400 "Invalid API
// parameter, please check the documentation", which fails the turn -- and the
// call is in the stored transcript, so every later turn fails the same way.
func TestOpenAIReplaysZeroArgumentToolCall(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	_, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{
		Model:    "m",
		Messages: zeroArgTranscript(),
	}, nil)
	require.NoError(t, err)

	msgs := bodyMap(t, h.body)["messages"].([]any)
	fn := msgs[1].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	args, ok := fn["arguments"]
	require.True(t, ok, "the arguments field is never omitted, whatever the model sent")
	assert.Equal(t, "{}", args, "no arguments is the empty object, not a missing field")
}

// malformedArgTranscript is one assistant turn whose tool call carried text
// that is not JSON, as a host replays it out of storage. The tool result after
// it already told the model what was wrong.
func malformedArgTranscript() []Message {
	return []Message{
		{Role: RoleUser, Content: "search"},
		{Role: RoleAssistant, ToolCalls: []ToolCall{{ID: "call_1", Name: "grep", Arguments: `{"pattern": "x`}}},
		{Role: RoleTool, ToolCallID: "call_1", ToolIsError: true, Content: "invalid grep arguments: unexpected end of JSON input"},
	}
}

// A tool call whose arguments are not valid JSON is replayed as {}: the
// backend rejects the raw text with 400 "function.arguments must be valid
// JSON", and since the call is persisted, every later turn of that
// conversation would fail the same way. The transcript itself keeps the
// model's text; only the wire copy is repaired.
func TestOpenAIReplaysMalformedToolArgumentsAsEmptyObject(t *testing.T) {
	h := &sseHandler{payloads: []string{`{"choices":[{"delta":{"content":"ok"},"finish_reason":"stop"}]}`}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	transcript := malformedArgTranscript()
	_, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{
		Model:    "m",
		Messages: transcript,
	}, nil)
	require.NoError(t, err)

	msgs := bodyMap(t, h.body)["messages"].([]any)
	fn := msgs[1].(map[string]any)["tool_calls"].([]any)[0].(map[string]any)["function"].(map[string]any)
	assert.Equal(t, "{}", fn["arguments"])
	assert.Equal(t, `{"pattern": "x`, transcript[1].ToolCalls[0].Arguments, "the transcript keeps what the model said")
}

// The Responses dialect repairs the same call the same way.
func TestResponsesReplaysMalformedToolArgumentsAsEmptyObject(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))

	_, err := respTestProvider(t, base, false).Complete(context.Background(), Request{
		Model:    "m",
		Messages: malformedArgTranscript(),
	}, nil)
	require.NoError(t, err)

	call := bodyMap(t, h.body)["input"].([]any)[1].(map[string]any)
	require.Equal(t, "function_call", call["type"])
	assert.Equal(t, "{}", call["arguments"])
}

// Valid JSON that is not an object (a model answering a bare string) is still
// what the model said, and still valid on this wire: it is sent as it is.
func TestReplayToolArgsKeepsValidJSON(t *testing.T) {
	assert.Equal(t, `{"a":1}`, replayToolArgs(`{"a":1}`))
	assert.Equal(t, `"x"`, replayToolArgs(`"x"`))
	assert.Equal(t, "{}", replayToolArgs(""))
	assert.Equal(t, "{}", replayToolArgs("   "))
	assert.Equal(t, "{}", replayToolArgs(`{"a":`))
	assert.Equal(t, "{}", replayToolArgs("not json"))
}

// The Responses dialect answers for the same call the same way.
func TestResponsesReplaysZeroArgumentToolCall(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))

	_, err := respTestProvider(t, base, false).Complete(context.Background(), Request{
		Model:    "m",
		Messages: zeroArgTranscript(),
	}, nil)
	require.NoError(t, err)

	call := bodyMap(t, h.body)["input"].([]any)[1].(map[string]any)
	require.Equal(t, "function_call", call["type"])
	args, ok := call["arguments"]
	require.True(t, ok, "the arguments field is never omitted, whatever the model sent")
	assert.Equal(t, "{}", args)
}

// Anthropic streams a zero-argument tool_use as a block with no
// input_json_delta at all. The non-streaming path already reads that as {};
// the streamed one must agree, or which transport served the turn decides
// whether the transcript can be replayed later.
func TestAnthropicStreamZeroArgumentToolCall(t *testing.T) {
	h := &anSSEHandler{events: [][2]string{
		{"message_start", `{"type":"message_start","message":{"usage":{"input_tokens":3,"output_tokens":1}}}`},
		{"content_block_start", `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"toolu_1","name":"whoami"}}`},
		{"content_block_stop", `{"type":"content_block_stop","index":0}`},
		{"message_delta", `{"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}`},
		{"message_stop", `{"type":"message_stop"}`},
	}}
	srv := httptest.NewServer(h)
	defer srv.Close()

	comp, err := anProvider(t, srv.URL).Complete(context.Background(),
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}, MaxTokens: 64}, nil)
	require.NoError(t, err)

	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, ToolCall{ID: "toolu_1", Name: "whoami", Arguments: "{}"}, comp.Message.ToolCalls[0])
}

// An OpenAI-compatible upstream reports the same call with no arguments
// fragments, and a decoder downstream is entitled to valid JSON either way.
func TestOpenAIStreamZeroArgumentToolCall(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","type":"function","function":{"name":"whoami"}}]},"finish_reason":"tool_calls"}]}`,
	)

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)

	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, ToolCall{ID: "c1", Name: "whoami", Arguments: "{}"}, comp.Message.ToolCalls[0])
}

// The same call reported by a server that ignored stream:true.
func TestOpenAINonStreamZeroArgumentToolCall(t *testing.T) {
	body := `{"choices":[{"message":{"role":"assistant","tool_calls":[{"id":"c1","function":{"name":"whoami"}}]},"finish_reason":"tool_calls"}]}`
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)

	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "{}", comp.Message.ToolCalls[0].Arguments)
}

// A Responses function_call item arrives whole, and can arrive with no
// arguments field.
func TestResponsesStreamZeroArgumentToolCall(t *testing.T) {
	_, base := respServer(t,
		jsonMust(jsonObj{"type": "response.output_item.done", "item": jsonObj{
			"type": "function_call", "call_id": "c1", "name": "whoami",
		}}),
		respCompletedEvent(jsonObj{"status": "completed"}),
	)

	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)

	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "{}", comp.Message.ToolCalls[0].Arguments)
}
