package client

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type callerProvider struct{ inner Provider }

func (c callerProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	return c.inner.Complete(ctx, req, ev)
}

func (c callerProvider) ModelLimits(ctx context.Context, model string) (Limits, error) {
	return ForwardModelLimits(ctx, c.inner, model)
}

func TestEveryWrapperPassesTheLimitsQuestionThrough(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"object":"list","data":[{"id":"m","context_length":32768,"max_completion_tokens":4096}]}`))
	}))
	defer srv.Close()
	base, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: srv.URL, HTTPClient: srv.Client()}})
	require.NoError(t, err)

	for name, p := range map[string]Provider{
		"retrying":            base,
		"param stripper":      NewParamStripper(base),
		"caller's own":        Retrying(callerProvider{base}, nil),
		"caller's, unwrapped": up(Unwrap(callerProvider{base})),
	} {
		l, ok, err := ModelLimitsOf(context.Background(), p, "m")
		require.NoError(t, err, name)
		assert.True(t, ok, name)
		assert.Equal(t, Limits{ContextWindow: 32768, MaxOutput: 4096}, l, name)
	}
}

type silentProvider struct{}

func (silentProvider) Complete(context.Context, Request, *StreamEvents) (*Completion, error) {
	return nil, nil
}

func TestAWrappedProviderWithNoModelListStillCannotSay(t *testing.T) {
	for name, p := range map[string]Provider{
		"retrying":       Retrying(silentProvider{}, nil),
		"param stripper": NewParamStripper(silentProvider{}),
		"forwarding":     callerProvider{silentProvider{}},
	} {
		l, ok, err := ModelLimitsOf(context.Background(), p, "m")
		require.NoError(t, err, name)
		assert.False(t, ok, name)
		assert.Zero(t, l, name)
	}
}
