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

func mustCompletions(t *testing.T, base ProviderConfig) Provider {
	t.Helper()
	p, err := NewCompletionsProvider(CompletionsConfig{ProviderConfig: base})
	require.NoError(t, err)
	return p
}

func TestCompletionsStreamsTextAndUsage(t *testing.T) {
	var got map[string]any
	var path, auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		path, auth = r.URL.Path, r.Header.Get("Authorization")
		b, _ := io.ReadAll(r.Body)
		require.NoError(t, json.Unmarshal(b, &got))
		w.Header().Set("Content-Type", "text/event-stream")
		for _, line := range []string{
			`{"model":"m-served","choices":[{"text":" Hel"}]}`,
			`{"choices":[{"text":"lo","finish_reason":"stop"}]}`,
			`{"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":2,"total_tokens":9}}`,
		} {
			_, _ = io.WriteString(w, "data: "+line+"\n\n")
		}
		_, _ = io.WriteString(w, "data: [DONE]\n\n")
	}))
	defer srv.Close()

	var deltas []string
	comp, err := mustCompletions(t, ProviderConfig{BaseURL: srv.URL + "/v1", APIKey: "k", HTTPClient: srv.Client()}).
		Complete(context.Background(), Request{
			Model: "m", System: "Be brief.",
			Messages: []Message{{Role: RoleUser, Content: "Hi"}},
			Extra:    map[string]any{"temperature": 0.5, "prompt": "ignored"},
		}, &StreamEvents{OnText: func(s string) error { deltas = append(deltas, s); return nil }})
	require.NoError(t, err)
	assert.Equal(t, "/v1/completions", path)
	assert.Equal(t, "Bearer k", auth)
	assert.Equal(t, "System: Be brief.\n\nUser: Hi\n\nAssistant:", got["prompt"], "the core owns the prompt")
	assert.EqualValues(t, DefaultCompletionsMaxTokens, got["max_tokens"], "the API's own default of 16 is never left in place")
	assert.InDelta(t, 0.5, got["temperature"], 1e-9)
	assert.Equal(t, true, got["stream"])
	assert.NotNil(t, got["stop"])
	assert.Equal(t, []string{" Hel", "lo"}, deltas)
	assert.Equal(t, "Hello", comp.Message.Content)
	assert.Equal(t, StopEndTurn, comp.StopReason)
	assert.Equal(t, "m-served", comp.Model)
	assert.True(t, comp.Streamed)
	require.Len(t, comp.Usages, 1)
	assert.Equal(t, 9, comp.Usages[0].TotalTokens)
	assert.NotEmpty(t, comp.Usages[0].Raw)
}

func TestCompletionsAcceptsAPlainJSONAnswer(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"choices":[{"text":" ok","finish_reason":"length"}]}`)
	}))
	defer srv.Close()
	comp, err := mustCompletions(t, ProviderConfig{BaseURL: srv.URL}).
		Complete(context.Background(), Request{Model: "m", MaxTokens: 3, Messages: []Message{{Role: RoleUser, Content: "x"}}}, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	assert.Equal(t, StopMaxTokens, comp.StopReason)
	assert.False(t, comp.Streamed)
}

// A dialect that cannot express something FAILS: no tools field, no image field.
func TestCompletionsFailsOnWhatItCannotExpress(t *testing.T) {
	p := mustCompletions(t, ProviderConfig{BaseURL: "http://127.0.0.1:1"})
	_, err := p.Complete(context.Background(), Request{Model: "m", Tools: []ToolDecl{{Name: "t"}}}, nil)
	require.Error(t, err)
	assert.True(t, IsUnsupported(err))
	assert.Contains(t, err.Error(), "tool")

	_, err = p.Complete(context.Background(), Request{Model: "m", Messages: []Message{
		NewMessage(RoleUser, ImagePart{MediaType: "image/png", Data: "AA=="}),
	}}, nil)
	require.Error(t, err)
	assert.True(t, IsUnsupported(err))
}

func TestCompletionsReportsARefusal(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = io.WriteString(w, `{"error":{"message":"nope"}}`)
	}))
	defer srv.Close()
	_, err := mustCompletions(t, ProviderConfig{BaseURL: srv.URL}).Complete(context.Background(), Request{Model: "m"}, nil)
	var ae *APIError
	require.ErrorAs(t, err, &ae)
	assert.Equal(t, 400, ae.Status)
}

func TestCompletionsNeedsABaseURL(t *testing.T) {
	_, err := NewCompletionsProvider(CompletionsConfig{})
	require.Error(t, err)
}
