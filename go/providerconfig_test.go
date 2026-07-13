package agentic

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mustProvider builds a Provider via NewProvider, failing the test on error.
func mustProvider(t *testing.T, cfg ProviderConfig) Provider {
	t.Helper()
	p, err := NewProvider(cfg)
	require.NoError(t, err)
	return p
}

// oaProvider is shorthand for an OpenAI-dialect test provider.
func oaProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustProvider(t, ProviderConfig{Dialect: DialectOpenAI, BaseURL: baseURL})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustProvider(t, ProviderConfig{Dialect: DialectAnthropic, BaseURL: baseURL})
}

func TestNewProviderSelectsDialect(t *testing.T) {
	p, err := NewProvider(ProviderConfig{Dialect: DialectOpenAI, BaseURL: "https://api.openai.com/v1"})
	require.NoError(t, err)
	_, ok := p.(*openaiProvider)
	assert.True(t, ok, "openai dialect builds the OpenAI-compatible provider")

	p, err = NewProvider(ProviderConfig{Dialect: DialectAnthropic, BaseURL: "https://api.anthropic.com"})
	require.NoError(t, err)
	_, ok = p.(*anthropicProvider)
	assert.True(t, ok, "anthropic dialect builds the Messages API provider")
}

func TestNewProviderRequiresBaseURL(t *testing.T) {
	p, err := NewProvider(ProviderConfig{Dialect: DialectOpenAI})
	assert.Nil(t, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "BaseURL")
	assert.False(t, IsTransient(err), "misconfiguration is permanent")
}

func TestNewProviderRejectsUnknownDialect(t *testing.T) {
	p, err := NewProvider(ProviderConfig{BaseURL: "https://api.openai.com/v1"})
	assert.Nil(t, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "Dialect is required")

	p, err = NewProvider(ProviderConfig{Dialect: "cohere", BaseURL: "https://example.invalid"})
	assert.Nil(t, p)
	require.Error(t, err)
	assert.Contains(t, err.Error(), `unknown ProviderConfig.Dialect "cohere"`)
	assert.False(t, IsTransient(err))
}
