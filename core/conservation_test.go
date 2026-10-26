package commonai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// conservationTranscript holds a unique sentinel in every field a turn can
// carry. Each dialect must put every sentinel on the wire, except the ones
// its row in TestTranscriptConservation names with a reason.
func conservationTranscript() []Message {
	return []Message{
		{Role: RoleUser, Content: "S-USER"},
		{
			Role:      RoleAssistant,
			Thinking:  []ThinkingBlock{{Text: "S-THINK", Signature: "S-SIG", ID: "S-RSID"}},
			Content:   "S-TEXT",
			ToolCalls: []ToolCall{{ID: "S-CALLID", Name: "S_TOOL", Arguments: `{"q":"S-ARGS"}`}},
		},
		{Role: RoleTool, Content: "S-RESULT", ToolCallID: "S-CALLID"},
	}
}

var conservationSentinels = []string{
	"S-USER", "S-THINK", "S-SIG", "S-RSID", "S-TEXT", "S-CALLID", "S_TOOL", "S-ARGS", "S-RESULT",
}

// captureServer records the request body and refuses the call, so one
// handler serves every dialect.
func captureServer(t *testing.T) (*httptest.Server, *[]byte) {
	t.Helper()
	var body []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusTeapot)
	}))
	t.Cleanup(srv.Close)
	return srv, &body
}

// TestTranscriptConservation fails when a dialect stops sending part of the
// transcript. A dropped field is state the model no longer has on the next
// call. A new drop must be added to `dropped` with its reason, or fixed.
func TestTranscriptConservation(t *testing.T) {
	cases := []struct {
		name    string
		build   func(base ProviderConfig) Provider
		dropped map[string]string
	}{
		{
			name: "openai",
			build: func(base ProviderConfig) Provider {
				return mustOpenAI(t, OpenAIConfig{ProviderConfig: base})
			},
			dropped: map[string]string{
				"S-SIG":  "a signature that is not a reasoning_details array has no chat-completions field",
				"S-RSID": "chat completions has no reasoning item id",
			},
		},
		{
			name: "anthropic",
			build: func(base ProviderConfig) Provider {
				return mustAnthropic(t, AnthropicConfig{ProviderConfig: base})
			},
			dropped: map[string]string{
				"S-RSID": "an Anthropic thinking block has no id",
			},
		},
		{
			name: "responses",
			build: func(base ProviderConfig) Provider {
				return mustResponses(t, ResponsesConfig{ProviderConfig: base})
			},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			srv, body := captureServer(t)
			p := tc.build(ProviderConfig{BaseURL: srv.URL})
			_, err := p.Complete(context.Background(), Request{
				Model: "m", MaxTokens: 16, Messages: conservationTranscript(),
			}, nil)
			require.Error(t, err, "the capture server refuses every call")
			require.NotEmpty(t, *body)
			sent := string(*body)
			for _, s := range conservationSentinels {
				if why, ok := tc.dropped[s]; ok {
					assert.NotContains(t, sent, s, "%s is declared dropped (%s); update the table if it is sent now", s, why)
					continue
				}
				assert.Contains(t, sent, s, "%s dialect lost %s from the transcript", tc.name, s)
			}
		})
	}
}

// TestResponsesConservesReasoningText: a reasoning_text block with no encrypted_content is the
// thinking itself, and the Responses dialect must put it on the wire.
func TestResponsesConservesReasoningText(t *testing.T) {
	srv, body := captureServer(t)
	p := mustResponses(t, ResponsesConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})
	msgs := conservationTranscript()
	msgs[1].Thinking = []ThinkingBlock{{Text: "S-THINK", ID: "S-RSID", Raw: true}}
	_, err := p.Complete(context.Background(), Request{Model: "m", MaxTokens: 16, Messages: msgs}, nil)
	require.Error(t, err, "the capture server refuses every call")
	for _, s := range []string{"S-THINK", "S-RSID"} {
		assert.Contains(t, string(*body), s, "responses dialect lost %s from the transcript", s)
	}
}

// TestOpenAINonStreamKeepsReasoningContent covers a server that ignores
// stream:true and answers with reasoning_content, the DeepSeek field name.
func TestOpenAINonStreamKeepsReasoningContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{"model":"m","choices":[{"message":{"role":"assistant","content":"ok",` +
			`"reasoning_content":"I worked it out"},"finish_reason":"stop"}]}`))
	}))
	defer srv.Close()
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, nil)
	require.NoError(t, err)
	require.Len(t, comp.Message.Thinking, 1)
	assert.Equal(t, "I worked it out", comp.Message.Thinking[0].Text)
}

// TestOpenAIReasoningRefusedFails covers an endpoint that 400s on replayed
// reasoning. The call fails with ErrReasoningRefused. It is never sent again
// without the reasoning.
func TestOpenAIReasoningRefusedFails(t *testing.T) {
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		bodies = append(bodies, string(b))
		if strings.Contains(string(b), "reasoning_content") {
			w.Header().Set("Content-Type", "application/json")
			w.WriteHeader(http.StatusBadRequest)
			_, _ = w.Write([]byte(`{"error":{"message":"Extra inputs are not permitted: messages.1.reasoning_content"}}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	p := oaProvider(t, srv.URL)
	comp, err := p.Complete(context.Background(), Request{Model: "m", Messages: conservationTranscript()}, nil)
	assert.Nil(t, comp)
	require.ErrorIs(t, err, ErrReasoningRefused)
	assert.False(t, IsTransient(err), "the same request is refused every time")
	require.Len(t, bodies, 1, "the call is never sent again without the reasoning")
	assert.Contains(t, bodies[0], "S-THINK")
}

func TestOpenAIUnrelated400KeepsReasoning(t *testing.T) {
	hits := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error":{"message":"unknown field \"temperature\""}}`))
	}))
	defer srv.Close()
	_, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m", Messages: conservationTranscript()}, nil)
	require.Error(t, err)
	assert.NotErrorIs(t, err, ErrReasoningRefused)
	assert.Equal(t, 1, hits)
}
