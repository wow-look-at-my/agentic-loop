package commonai

import (
	"context"
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
