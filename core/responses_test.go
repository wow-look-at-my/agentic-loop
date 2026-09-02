package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// respServer serves payloads as an SSE stream and captures the request.
func respServer(t *testing.T, payloads ...string) (*sseHandler, string) {
	t.Helper()
	h := &sseHandler{payloads: payloads}
	srv := httptest.NewServer(h)
	t.Cleanup(srv.Close)
	return h, srv.URL
}

// respTestProvider is a Responses provider with retry off, so a test never
// waits out real backoff.
func respTestProvider(t *testing.T, baseURL string, store bool) Provider {
	t.Helper()
	return mustResponses(t, ResponsesConfig{
		ProviderConfig: ProviderConfig{BaseURL: baseURL, APIKey: "k"},
		Store:          store,
	})
}

// respCompletedEvent wraps a Response object in a response.completed event.
func respCompletedEvent(response jsonObj) string {
	return jsonMust(jsonObj{"type": "response.completed", "response": response})
}

// The request is the Responses shape, not chat-completions wearing a new URL:
// the system prompt is `instructions`, the transcript is `input` items, and a
// tool declares its name at the top level rather than inside a "function".
func TestResponsesRequestShape(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))

	_, err := respTestProvider(t, base, false).Complete(context.Background(), Request{
		Model:     "gpt-x",
		System:    "be brief",
		Messages:  []Message{{Role: RoleUser, Content: "hi"}},
		Tools:     []ToolDecl{{Name: "look", Description: "look at it"}},
		MaxTokens: 512,
		CacheKey:  "conv-1",
		Extra:     map[string]any{"reasoning": jsonObj{"effort": "high"}, "model": "ignored"},
	}, nil)
	require.NoError(t, err)

	body := bodyMap(t, h.body)
	assert.Equal(t, "gpt-x", body["model"], "a reserved Extra key never wins")
	assert.Equal(t, "be brief", body["instructions"], "the system prompt is not a message here")
	assert.Equal(t, true, body["stream"])
	assert.Equal(t, float64(512), body["max_output_tokens"])
	assert.Equal(t, "conv-1", body["prompt_cache_key"])
	assert.Equal(t, jsonObj{"effort": "high"}, body["reasoning"], "Extra passes through verbatim")

	assert.Equal(t, false, body["store"],
		"retaining a caller's conversations on a third party's servers is their decision, not a default")
	assert.Equal(t, []any{"reasoning.encrypted_content"}, body["include"],
		"the encrypted payload is what makes reasoning replayable with store off")

	require.Len(t, body["tools"], 1)
	tool := body["tools"].([]any)[0].(map[string]any)
	assert.Equal(t, "function", tool["type"])
	assert.Equal(t, "look", tool["name"], "no nested function object on this dialect")
	assert.Equal(t, "look at it", tool["description"])
	assert.Equal(t, jsonObj{"type": "object"}, tool["parameters"])

	require.Len(t, body["input"], 1)
	item := body["input"].([]any)[0].(map[string]any)
	assert.Equal(t, "message", item["type"])
	assert.Equal(t, "user", item["role"])
	assert.Equal(t, []any{jsonObj{"type": "input_text", "text": "hi"}}, item["content"])

	assert.Equal(t, "Bearer k", h.header.Get("Authorization"))
}

// Store is an opt-in, and opting in must actually reach the wire.
func TestResponsesStoreIsOptIn(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))
	_, err := respTestProvider(t, base, true).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, true, bodyMap(t, h.body)["store"])
}

// A caller that wants different includes is not fighting the default.
func TestResponsesIncludeIsOverridable(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))
	_, err := respTestProvider(t, base, false).Complete(context.Background(), Request{
		Model: "m", Extra: map[string]any{"include": jsonArr{"message.output_text.logprobs"}},
	}, nil)
	require.NoError(t, err)
	assert.Equal(t, []any{"message.output_text.logprobs"}, bodyMap(t, h.body)["include"])
}

// The whole point of the dialect: reasoning survives a tool call. It goes back
// as a reasoning ITEM, before the text and the calls it produced, and only when
// it carries the payload that makes it the model's own thinking rather than a
// paraphrase of it.
func TestResponsesReplaysReasoningAcrossAToolCall(t *testing.T) {
	h, base := respServer(t, respCompletedEvent(jsonObj{"status": "completed"}))

	_, err := respTestProvider(t, base, false).Complete(context.Background(), Request{
		Model: "m",
		Messages: []Message{
			{Role: RoleUser, Content: "count them"},
			{
				Role:    RoleAssistant,
				Content: "let me look",
				Thinking: []ThinkingBlock{
					{ID: "rs_1", Text: "I should list the files", Signature: "enc-1"},
					{Text: "a summary with no payload"},
				},
				ToolCalls: []ToolCall{{ID: "call_1", Name: "look", Arguments: `{"path":"/"}`}},
			},
			{Role: RoleTool, ToolCallID: "call_1", Content: "three files"},
		},
	}, nil)
	require.NoError(t, err)

	input := bodyMap(t, h.body)["input"].([]any)
	types := make([]string, len(input))
	for i, it := range input {
		types[i] = it.(map[string]any)["type"].(string)
	}
	assert.Equal(t, []string{"message", "reasoning", "message", "function_call", "function_call_output"}, types,
		"reasoning precedes the text and the calls it produced, in the order the model emitted them")

	reasoning := input[1].(map[string]any)
	assert.Equal(t, "rs_1", reasoning["id"])
	assert.Equal(t, "enc-1", reasoning["encrypted_content"])
	assert.Equal(t, []any{jsonObj{"type": "summary_text", "text": "I should list the files"}}, reasoning["summary"])

	call := input[3].(map[string]any)
	assert.Equal(t, "call_1", call["call_id"], "call_id, not the item id, is what pairs a call with its output")
	assert.Equal(t, "look", call["name"])
	assert.Equal(t, `{"path":"/"}`, call["arguments"])

	out := input[4].(map[string]any)
	assert.Equal(t, "call_1", out["call_id"])
	assert.Equal(t, "three files", out["output"])
	assert.NotContains(t, out, "role", "a tool result is an item, not a message with a role")
}

// A finished item arrives whole, so tool calls need no fragment accumulator --
// and text must not be taken from both the deltas and the final item.
func TestResponsesStreamsTextReasoningAndCalls(t *testing.T) {
	_, base := respServer(t,
		jsonMust(jsonObj{"type": "response.reasoning_summary_text.delta", "delta": "thinking "}),
		jsonMust(jsonObj{"type": "response.reasoning_summary_text.delta", "delta": "hard"}),
		jsonMust(jsonObj{"type": "response.output_item.done", "item": jsonObj{
			"type": "reasoning", "id": "rs_9", "encrypted_content": "enc-9",
			"summary": jsonArr{jsonObj{"type": "summary_text", "text": "thinking hard"}},
		}}),
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": "one "}),
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": "two"}),
		jsonMust(jsonObj{"type": "response.output_item.done", "item": jsonObj{
			"type": "function_call", "call_id": "call_7", "name": "look", "arguments": `{"a":1}`,
		}}),
		respCompletedEvent(jsonObj{
			"status": "completed",
			"output": jsonArr{jsonObj{"type": "function_call", "call_id": "call_7", "name": "look"}},
			"usage": jsonObj{
				"input_tokens": 100, "output_tokens": 20, "total_tokens": 120,
				"input_tokens_details":  jsonObj{"cached_tokens": 80},
				"output_tokens_details": jsonObj{"reasoning_tokens": 12},
			},
		}),
	)

	var text, reasoning string
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, &StreamEvents{
		OnText:      func(s string) error { text += s; return nil },
		OnReasoning: func(s string) error { reasoning += s; return nil },
	})
	require.NoError(t, err)

	assert.Equal(t, "one two", text)
	assert.Equal(t, "thinking hard", reasoning)
	assert.Equal(t, "one two", comp.Message.Content, "the final item never re-adds the streamed text")
	assert.True(t, comp.Streamed)

	require.Len(t, comp.Message.ToolCalls, 1, "response.output carries the call again; it is not counted twice")
	assert.Equal(t, ToolCall{ID: "call_7", Name: "look", Arguments: `{"a":1}`}, comp.Message.ToolCalls[0])

	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, ThinkingBlock{ID: "rs_9", Text: "thinking hard", Signature: "enc-9"}, comp.Message.Thinking[0],
		"the replayable block wins over the streamed copy of the same words")

	assert.Equal(t, StopToolUse, comp.StopReason, "this API has no finish_reason; the shape of the turn says it")
	require.True(t, (len(comp.Usages) > 0))
	assert.Equal(t, 100, firstUsage(comp).PromptTokens, "input_tokens already includes the cached ones")
	assert.Equal(t, 20, firstUsage(comp).CompletionTokens)
	require.NotNil(t, firstUsage(comp).CacheReadTokens)
	assert.Equal(t, 80, *firstUsage(comp).CacheReadTokens)
	require.NotNil(t, firstUsage(comp).CacheWriteTokens)
	assert.Equal(t, 0, *firstUsage(comp).CacheWriteTokens, "this API bills no separate cache-write class")
	require.NotNil(t, firstUsage(comp).ReasoningTokens)
	assert.Equal(t, 12, *firstUsage(comp).ReasoningTokens)
	assert.NotEmpty(t, firstUsage(comp).Raw, "the verbatim usage object survives for logging")
}

// No cache detail at all is UNKNOWN, never a: the tri-state contract.
func TestResponsesUsageWithoutCacheDetailStaysUnknown(t *testing.T) {
	_, base := respServer(t, respCompletedEvent(jsonObj{
		"status": "completed",
		"usage":  jsonObj{"input_tokens": 5, "output_tokens": 1, "total_tokens": 6},
	}))
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Nil(t, firstUsage(comp).CacheReadTokens)
	assert.Nil(t, firstUsage(comp).CacheWriteTokens)
	assert.Nil(t, firstUsage(comp).ReasoningTokens)
	assert.Equal(t, StopEndTurn, comp.StopReason)
}

// A run that hit the output cap must say so, and this API reports it in only
// place.
func TestResponsesIncompleteReportsWhy(t *testing.T) {
	cases := []struct {
		reason string
		want   string
	}{
		{"max_output_tokens", StopMaxTokens},
		{"content_filter", "content_filter"},
	}
	for _, tc := range cases {
		t.Run(tc.reason, func(t *testing.T) {
			_, base := respServer(t, jsonMust(jsonObj{
				"type": "response.incomplete",
				"response": jsonObj{
					"status":             "incomplete",
					"incomplete_details": jsonObj{"reason": tc.reason},
				},
			}))
			comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
			require.NoError(t, err)
			assert.Equal(t, tc.want, comp.StopReason)
		})
	}
}

// A whose body says the response failed is still a failure, and re-sending
// a request the server accepted and then rejected would just be billed.
func TestResponsesFailureInsideA200(t *testing.T) {
	cases := []struct {
		name    string
		payload string
		want    string
	}{
		{"response.failed", jsonMust(jsonObj{
			"type":     "response.failed",
			"response": jsonObj{"status": "failed", "error": jsonObj{"message": "model overloaded", "code": "server_error"}},
		}), "model overloaded (server_error)"},
		{"stream error event", jsonMust(jsonObj{
			"type": "error", "error": jsonObj{"message": "bad tool schema"},
		}), "bad tool schema"},
		{"failed with no reason", jsonMust(jsonObj{
			"type": "response.failed", "response": jsonObj{"status": "failed"},
		}), "no reason given"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, base := respServer(t, tc.payload)
			comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
			require.Error(t, err)
			assert.Nil(t, comp, "nothing streamed, so there is no partial to keep")
			assert.Contains(t, err.Error(), tc.want)
			assert.False(t, IsTransient(err), "the server accepted the request and then rejected it")
		})
	}
}

// A failure AFTER deltas keeps what the caller already saw.
func TestResponsesFailureAfterDataKeepsThePartial(t *testing.T) {
	_, base := respServer(t,
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": "half an ans"}),
		jsonMust(jsonObj{"type": "error", "error": jsonObj{"message": "upstream died"}}),
	)
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.Error(t, err)
	require.NotNil(t, comp, "a streamed call is not safe to re-send, and the partial is how the layers above know")
	assert.Equal(t, "half an ans", comp.Message.Content)
}

// A server that ignores stream:true is accepted transparently, and there the
// items are the only source there is.
func TestResponsesNonStreamingBody(t *testing.T) {
	body := jsonMust(jsonObj{
		"status": "completed",
		"output": jsonArr{
			jsonObj{"type": "reasoning", "id": "rs_2", "encrypted_content": "enc-2"},
			jsonObj{"type": "message", "role": "assistant", "content": jsonArr{
				jsonObj{"type": "output_text", "text": "done"},
			}},
			jsonObj{"type": "function_call", "call_id": "c1", "name": "look", "arguments": "{}"},
		},
		"usage": jsonObj{"input_tokens": 7, "output_tokens": 2, "total_tokens": 9},
	})
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(body))
	}))
	defer srv.Close()

	comp, err := respTestProvider(t, srv.URL, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.False(t, comp.Streamed, "the transport is recorded truthfully")
	assert.Equal(t, "done", comp.Message.Content)
	require.Len(t, comp.Message.ToolCalls, 1)
	assert.Equal(t, "c1", comp.Message.ToolCalls[0].ID)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "enc-2", comp.Message.Thinking[0].Signature)
	assert.Equal(t, StopToolUse, comp.StopReason)
	assert.Equal(t, 7, firstUsage(comp).PromptTokens)
	assert.NotEmpty(t, firstUsage(comp).Raw)
}

// A non-streaming body that reports a failure is an error, not a blank answer.
func TestResponsesNonStreamingFailure(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(jsonMust(jsonObj{
			"status": "failed", "error": jsonObj{"message": "nope"},
		})))
	}))
	defer srv.Close()

	comp, err := respTestProvider(t, srv.URL, false).Complete(context.Background(), Request{Model: "m"}, nil)
	assert.Nil(t, comp)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nope")

	bad := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte("<html>"))
	}))
	defer bad.Close()
	_, err = respTestProvider(t, bad.URL, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "decode non-streaming response")
}

// A non-2xx is classified exactly as on the other dialects, so retry and the
// overflow detector behave the same behind this seam.
func TestResponsesHTTPErrorsClassifyNormally(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(jsonMust(jsonObj{"error": jsonObj{"message": "slow down"}})))
	}))
	defer srv.Close()

	comp, err := respTestProvider(t, srv.URL, false).Complete(context.Background(), Request{Model: "m"}, nil)
	assert.Nil(t, comp, "nothing streamed")
	require.Error(t, err)
	assert.True(t, IsTransient(err), "a 429 is transient on every dialect")
}

// A stream sink that fails aborts the read and hands back what it already had.
func TestResponsesCallbackErrorAbortsWithAPartial(t *testing.T) {
	_, base := respServer(t,
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": "seen"}),
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": " unseen"}),
	)
	sentinel := assertAnError{}
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, &StreamEvents{
		OnText: func(string) error { return sentinel },
	})
	require.Error(t, err)
	require.NotNil(t, comp)
	assert.Equal(t, "seen", comp.Message.Content, "state is recorded before each emit")
	assert.False(t, IsTransient(err), "a failed sink never makes the call re-sendable")
}

// A stream that produced summary text but no replayable item still keeps the
// reasoning: not being able to replay it is better than losing it.
func TestResponsesKeepsUnreplayableReasoning(t *testing.T) {
	_, base := respServer(t,
		jsonMust(jsonObj{"type": "response.reasoning_summary_text.delta", "delta": "pondering"}),
		respCompletedEvent(jsonObj{"status": "completed"}),
	)
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "pondering", comp.Message.Thinking[0].Text)
	assert.Empty(t, comp.Message.Thinking[0].Signature)
}

// Noise between events must not derail the read.
func TestResponsesToleratesUnknownAndUnparseableEvents(t *testing.T) {
	_, base := respServer(t,
		"not json at all",
		jsonMust(jsonObj{"type": "response.output_item.added", "item": jsonObj{"type": "message"}}),
		jsonMust(jsonObj{"type": "response.function_call_arguments.delta", "delta": `{"a`}),
		jsonMust(jsonObj{"type": "response.output_text.delta", "delta": "ok"}),
		respCompletedEvent(jsonObj{"status": "completed"}),
	)
	comp, err := respTestProvider(t, base, false).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content,
		"argument fragments are ignored: the finished item is the one source of truth")
}

// assertAnError is a sentinel a stream callback returns.
type assertAnError struct{}

func (assertAnError) Error() string { return "sink failed" }
