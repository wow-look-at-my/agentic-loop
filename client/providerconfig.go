package client

import (
	"context"
	"net/http"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/extras"
)

// ProviderConfig holds the connection settings shared by every dialect. It is
// not used on its own: embed it in the per-dialect config types (OpenAIConfig,
// AnthropicConfig) accepted by the dialect constructors.
type ProviderConfig struct {
	// BaseURL is the required API root; OpenAI includes the version segment, Anthropic the bare root.
	BaseURL string
	// APIKey, when non-empty, authenticates: Bearer on OpenAI dialects, x-api-key on Anthropic.
	APIKey string
	// HTTPClient performs the requests; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header.
	UserAgent string
	// Headers are applied after the dialect defaults, so a caller-supplied header can override them.
	Headers map[string]string
	// Retry is the transient-failure policy; nil means DefaultRetry (on by default); MaxAttempts: turns it off.
	Retry *RetryPolicy
	// RateLimiter, when non-nil, throttles request starts; shared across providers throttles them together.
	RateLimiter *RateLimiter
}

// core is the connection base without the policies, which decorate the built
// provider rather than travelling into it.
func (c ProviderConfig) core() commonai.ProviderConfig {
	return commonai.ProviderConfig{
		BaseURL:    c.BaseURL,
		APIKey:     c.APIKey,
		HTTPClient: extras.RateLimitedClient(c.HTTPClient, c.RateLimiter),
		UserAgent:  c.UserAgent,
		Headers:    c.Headers,
	}
}

// OpenAIConfig configures NewOpenAIProvider: the shared ProviderConfig
// connection base plus the knobs specific to the OpenAI-compatible dialect.
type OpenAIConfig struct {
	ProviderConfig

	// SelfHosted adds cache_prompt:true to every request; keep false for hosted OpenAI/Azure, which reject it.
	SelfHosted bool
	// PromptCache emits ephemeral cache_control breakpoints in openai shape; default false, plain servers reject them.
	PromptCache bool
	// ReplayReasoning replays accumulated reasoning as message.reasoning; default false, strict servers reject it.
	ReplayReasoning bool
}

// ResponsesConfig configures NewResponsesProvider: the shared ProviderConfig
// connection base plus the knob specific to the OpenAI Responses dialect.
type ResponsesConfig struct {
	ProviderConfig

	// Store opts into the API's server-side retention; defaults false because retaining caller data is a caller's decision.
	Store bool
}

// AnthropicConfig configures NewAnthropicProvider: the shared ProviderConfig
// connection base plus the knobs specific to the Anthropic Messages dialect.
type AnthropicConfig struct {
	ProviderConfig

	// Version sets the anthropic-version header; empty defaults to "--".
	Version string
	// DisableCaching drops the ephemeral cache_control breakpoints the provider otherwise places on every request.
	DisableCaching bool
}

// NewOpenAIProvider builds the Provider for OpenAI-compatible chat-completions
// APIs. It fails fast -- with a permanent (never-retried) error -- on an empty
// BaseURL.
func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error) {
	p, err := commonai.NewOpenAIProvider(commonai.OpenAIConfig{
		ProviderConfig:  cfg.core(),
		SelfHosted:      cfg.SelfHosted,
		PromptCache:     cfg.PromptCache,
		ReplayReasoning: cfg.ReplayReasoning,
	})
	return finish(p, cfg.Retry, err)
}

// NewResponsesProvider builds the Provider for the OpenAI Responses API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error) {
	p, err := commonai.NewResponsesProvider(commonai.ResponsesConfig{
		ProviderConfig: cfg.core(),
		Store:          cfg.Store,
	})
	return finish(p, cfg.Retry, err)
}

// NewAnthropicProvider builds the Provider for the Anthropic Messages API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
// A thinking block whose signature the API rejects is stripped and the call
// retried automatically -- see commonai.NewThinkingSignatureRepair.
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error) {
	p, err := commonai.NewAnthropicProvider(commonai.AnthropicConfig{
		ProviderConfig: cfg.core(),
		Version:        cfg.Version,
		DisableCaching: cfg.DisableCaching,
	})
	if err != nil {
		return nil, err
	}
	return finish(commonai.NewThinkingSignatureRepair(p), cfg.Retry, err)
}

// finish turns a dialect implementation into the Provider a caller holds: retry behavior plus folded usage.
func finish(p commonai.Provider, policy *RetryPolicy, err error) (Provider, error) {
	if err != nil {
		return nil, err
	}
	return up(extras.Retrying(p, policy)), nil
}

// FetchModelList reads an endpoint's model list: the dialect it speaks and what its models charge.
func FetchModelList(ctx context.Context, cfg ProviderConfig) (*ModelList, error) {
	return commonai.FetchModelList(ctx, cfg.core())
}
