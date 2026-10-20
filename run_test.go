package agentic

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Run answers req.Messages from a fresh session in one turn.
func TestRunAnswersOneTurn(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n"))
	}))
	defer srv.Close()

	provider, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL}})
	require.NoError(t, err)

	res, err := Run(context.Background(),
		Config{Provider: provider},
		Request{Model: "m", Messages: []Message{{Role: RoleUser, Content: "q"}}})
	require.NoError(t, err)
	require.NotNil(t, res)

	assert.Equal(t, "ok", res.Final.Content)
	assert.Equal(t, 1, res.Turns, "one call, one turn")
	require.NotEmpty(t, res.Messages, "the request's user message is in the transcript")
	assert.Equal(t, RoleUser, res.Messages[0].Role)
	assert.Equal(t, "q", res.Messages[0].Content)
}

// Run refuses a config with no provider rather than answering from nowhere.
func TestRunRequiresProvider(t *testing.T) {
	_, err := Run(context.Background(), Config{}, Request{Model: "m"})
	require.Error(t, err)
}
