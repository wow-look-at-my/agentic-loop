package agentic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mustOpenAI builds a Provider via NewOpenAIProvider, failing the test on error.
func mustOpenAI(t *testing.T, cfg OpenAIConfig) Provider {
	t.Helper()
	p, err := NewOpenAIProvider(cfg)
	require.NoError(t, err)
	return p
}

// mustAnthropic builds a Provider via NewAnthropicProvider, failing the test on error.
func mustAnthropic(t *testing.T, cfg AnthropicConfig) Provider {
	t.Helper()
	p, err := NewAnthropicProvider(cfg)
	require.NoError(t, err)
	return p
}

// oaProvider is shorthand for an OpenAI-dialect test provider.
func oaProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL}})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL}})
}

func TestConstructorsSelectDialect(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.openai.com/v1"}})
	require.NoError(t, err)
	_, ok := p.(*openaiProvider)
	assert.True(t, ok, "NewOpenAIProvider builds the OpenAI-compatible provider")

	p, err = NewAnthropicProvider(AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.anthropic.com"}})
	require.NoError(t, err)
	_, ok = p.(*anthropicProvider)
	assert.True(t, ok, "NewAnthropicProvider builds the Messages API provider")
}

func TestConstructorsRequireBaseURL(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIConfig{})
	assert.Nil(t, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "OpenAIConfig.BaseURL")
	assert.False(t, IsTransient(err), "misconfiguration is permanent")

	p, err = NewAnthropicProvider(AnthropicConfig{})
	assert.Nil(t, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "AnthropicConfig.BaseURL")
	assert.False(t, IsTransient(err), "misconfiguration is permanent")
}
