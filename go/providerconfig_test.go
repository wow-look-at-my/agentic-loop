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

// oaProvider is shorthand for an OpenAI-dialect test provider. Providers retry
// by default, so tests inject the fast no-sleep policy — otherwise every
// retrying test would wait out real exponential backoff.
func oaProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return oaProviderRetry(t, baseURL, retryTestPolicy(4))
}

// oaProviderRetry is oaProvider with an explicit retry policy.
func oaProviderRetry(t *testing.T, baseURL string, retry *RetryPolicy) Provider {
	t.Helper()
	return mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retry}})
}

// anProvider is shorthand for an Anthropic-dialect test provider.
func anProvider(t *testing.T, baseURL string) Provider {
	t.Helper()
	return mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: baseURL, Retry: retryTestPolicy(4)}})
}

// unwrapRetry returns the dialect provider inside the retry wrapper every
// constructor applies.
func unwrapRetry(p Provider) Provider {
	if r, ok := p.(*retryingProvider); ok {
		return r.inner
	}
	return p
}

func TestConstructorsSelectDialect(t *testing.T) {
	p, err := NewOpenAIProvider(OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.openai.com/v1"}})
	require.NoError(t, err)
	_, ok := unwrapRetry(p).(*openaiProvider)
	assert.True(t, ok, "NewOpenAIProvider builds the OpenAI-compatible provider")

	p, err = NewAnthropicProvider(AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.anthropic.com"}})
	require.NoError(t, err)
	_, ok = unwrapRetry(p).(*anthropicProvider)
	assert.True(t, ok, "NewAnthropicProvider builds the Messages API provider")
}

func TestConstructorsRetryByDefault(t *testing.T) {
	// The point of moving retry out of the loop: a caller who never thinks
	// about retry still gets it, on both dialects.
	oa := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.openai.com/v1"}})
	assert.IsType(t, &retryingProvider{}, oa, "no Retry set still retries")

	an := mustAnthropic(t, AnthropicConfig{ProviderConfig: ProviderConfig{BaseURL: "https://api.anthropic.com"}})
	assert.IsType(t, &retryingProvider{}, an, "no Retry set still retries")

	// Opting out is explicit, and leaves the dialect provider unwrapped rather
	// than paying for a probe wrapper that can never fire.
	off := mustOpenAI(t, OpenAIConfig{ProviderConfig: ProviderConfig{
		BaseURL: "https://api.openai.com/v1", Retry: &RetryPolicy{MaxAttempts: 1}}})
	assert.IsType(t, &openaiProvider{}, off, "a single-attempt policy disables retry")
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
