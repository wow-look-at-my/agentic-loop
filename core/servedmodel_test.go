package commonai

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A gateway that resolves an alias names the model it served. The list knows that name, not the alias.
func TestOpenAINonStreamNamesTheModelServed(t *testing.T) {
	srv := jsonServer(t, `{
	  "model": "deepseek/deepseek-v4.1-flash",
	  "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
	  "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}
	}`)

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "auto-alias"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "deepseek/deepseek-v4.1-flash", comp.Model)
}

func TestLimitsForMatchesAServedModelBySuffix(t *testing.T) {
	list := decoded(t, `{"object": "list", "data": [
	  {"id": "openrouter/deepseek/deepseek-v4.1-flash", "context_length": 1048576},
	  {"id": "IncoAI/deepseek-v4.1-flash", "context_length": 128000},
	  {"id": "a/twin", "context_length": 10},
	  {"id": "b/twin", "context_length": 20}
	]}`)

	exact, ok := list.LimitsFor("IncoAI/deepseek-v4.1-flash")
	require.True(t, ok)
	assert.Equal(t, 128000, exact.ContextWindow)

	served, ok := list.LimitsFor("deepseek/deepseek-v4.1-flash")
	require.True(t, ok, "the one id ending in /deepseek/deepseek-v4.1-flash is the served model")
	assert.Equal(t, 1048576, served.ContextWindow)

	_, ok = list.LimitsFor("twin")
	assert.False(t, ok, "two ids with the suffix name nothing")
	_, ok = list.LimitsFor("nobody")
	assert.False(t, ok)
	_, ok = list.LimitsFor("")
	assert.False(t, ok)
	var none *ModelList
	_, ok = none.LimitsFor("x")
	assert.False(t, ok)
}

// A Bifrost gateway states the list id it routed to in extra_fields: the provider, a slash, and the provider's model.
// The body's own model is the upstream's name for it, which the list does not hold.
func TestOpenAINonStreamNamesTheListIDTheGatewayRoutedTo(t *testing.T) {
	srv := jsonServer(t, `{
	  "model": "deepseek-ai/DeepSeek-V4.1-Flash",
	  "choices": [{"finish_reason": "stop", "message": {"content": "ok"}}],
	  "extra_fields": {"routing_info": {"provider": "Synthetic", "model": "hf:deepseek-ai/DeepSeek-V4.1-Flash", "requested_model": "auto-pr-resolve"}}
	}`)

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "auto-pr-resolve"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "deepseek-ai/DeepSeek-V4.1-Flash", comp.Model)
	assert.Equal(t, "Synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash", comp.Routed)
}

func TestOpenAIStreamNamesTheListIDTheGatewayRoutedTo(t *testing.T) {
	srv := sseServer(t,
		`{"model": "deepseek-ai/DeepSeek-V4.1-Flash", "choices": [{"delta": {"content": "o"}}], "extra_fields": {"routing_info": {"provider": "Synthetic", "model": "hf:deepseek-ai/DeepSeek-V4.1-Flash"}}}`,
		`{"model": "deepseek-ai/DeepSeek-V4.1-Flash", "choices": [{"delta": {"content": "k"}, "finish_reason": "stop"}], "extra_fields": {"routing_info": {"provider": "Synthetic", "model": "hf:deepseek-ai/DeepSeek-V4.1-Flash"}}}`,
	)

	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "auto-pr-resolve"}, nil)
	require.NoError(t, err)
	assert.Equal(t, "Synthetic/hf:deepseek-ai/DeepSeek-V4.1-Flash", comp.Routed)
}

func TestARoutingModelWithNoProviderIsTheListID(t *testing.T) {
	f := &oaExtraFields{}
	assert.Empty(t, f.routed())
	assert.Empty(t, (*oaExtraFields)(nil).routed())
	require.NoError(t, json.Unmarshal([]byte(`{"routing_info": {"model": "solo"}}`), f))
	assert.Equal(t, "solo", f.routed())
}
