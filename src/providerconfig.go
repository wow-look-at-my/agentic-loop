package agentic

import "net/http"

// ProviderConfig holds the connection settings shared by every dialect. It is
// not used on its own: embed it in the per-dialect config types (OpenAIConfig,
// AnthropicConfig) accepted by the dialect constructors.
type ProviderConfig struct {
	// BaseURL is the API root and is required. OpenAI and Responses: include
	// the version segment (e.g. "https://api.openai.com/v1"); requests POST to
	// BaseURL + "/chat/completions" and BaseURL + "/responses" respectively.
	// Anthropic: the bare root (e.g. "https://api.anthropic.com"); requests
	// POST to BaseURL + "/v1/messages". Trailing slashes are trimmed before
	// joining.
	BaseURL string
	// APIKey, when non-empty, authenticates requests: a Bearer token on both
	// OpenAI dialects, the x-api-key header on Anthropic.
	APIKey string
	// HTTPClient performs the requests; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header.
	UserAgent string
	// Headers are applied after the dialect defaults, so a caller-supplied
	// header can override them.
	Headers map[string]string
	// Retry is the transient-failure retry policy for every call this provider
	// makes. Nil means DefaultRetry -- retry is ON by default, because a call
	// that fails before streaming anything is always safe to re-send and an
	// opt-in retry is one a caller forgets to enable. Set
	// &RetryPolicy{MaxAttempts: 1} to turn it off.
	//
	// Retry lives here, not around the loop, because the provider is the layer
	// that knows whether a call streamed anything -- the condition that decides
	// whether re-sending is safe.
	Retry *RetryPolicy
	// RateLimiter, when non-nil, throttles this provider's outgoing request
	// starts (its transient-failure retries included, which ride the same
	// http.Client) to the limiter's fixed rate -- a provider's per-minute
	// request cap, spaced evenly. A single RateLimiter shared across several
	// providers throttles them TOGETHER, so concurrent callers (e.g. the jobs
	// of one benchmark run) stay under one per-endpoint cap. Like Retry, it
	// lives here because HOW a call gets made -- including without tripping the
	// upstream's rate limit -- is the provider's job.
	RateLimiter *RateLimiter
}

// OpenAIConfig configures NewOpenAIProvider: the shared ProviderConfig
// connection base plus the knobs specific to the OpenAI-compatible dialect.
type OpenAIConfig struct {
	ProviderConfig

	// SelfHosted adds cache_prompt:true to every request -- the KV-cache
	// prefix-reuse opt-in llama.cpp-style servers honor. It must stay false
	// for hosted OpenAI/Azure, which reject unknown body fields with a 400.
	SelfHosted bool
	// PromptCache emits two Anthropic-style ephemeral cache_control
	// breakpoints in openai dialect shape -- a static one on the leading
	// system message and a moving one on the tail content block -- for
	// Anthropic-fronting gateways that pass cache_control through. Default
	// false: plain OpenAI-compatible servers reject the unknown marker.
	PromptCache bool
	// ReplayReasoning replays the accumulated reasoning text as
	// message.reasoning on each assistant message, the gateway-extension
	// behavior that keeps a model seeing its own chain-of-thought on the
	// openai dialect. Default false: strict OpenAI-compatible servers reject
	// the unknown field.
	ReplayReasoning bool
}

// ResponsesConfig configures NewResponsesProvider: the shared ProviderConfig
// connection base plus the one knob specific to the OpenAI Responses dialect.
type ResponsesConfig struct {
	ProviderConfig

	// Store opts INTO the API's server-side retention of every prompt and
	// response. It defaults to false, unlike the API itself, because retaining
	// a caller's conversations on a third party's servers is a decision for
	// the caller to make out loud rather than one a library makes for them by
	// inheriting a default. Reasoning still survives across tool calls with
	// Store false: the provider asks for the encrypted reasoning payload and
	// replays it.
	Store bool
}

// AnthropicConfig configures NewAnthropicProvider: the shared ProviderConfig
// connection base plus the knobs specific to the Anthropic Messages dialect.
type AnthropicConfig struct {
	ProviderConfig

	// Version sets the anthropic-version header; empty defaults to
	// "2023-06-01".
	Version string
	// DisableCaching drops the two ephemeral cache_control breakpoints the
	// provider otherwise places on every request.
	DisableCaching bool
}

// NewOpenAIProvider builds the Provider for OpenAI-compatible chat-completions
// APIs. It fails fast -- with a permanent (never-retried) error -- on an empty
// BaseURL. The returned Provider retries transient failures per
// ProviderConfig.Retry. The concrete implementation is unexported; consumers
// hold only the Provider interface.
func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("agentic: OpenAIConfig.BaseURL is required")
	}
	return newProvider(&openaiProvider{
		baseURL:         cfg.BaseURL,
		apiKey:          cfg.APIKey,
		httpClient:      rateLimitedClient(cfg.HTTPClient, cfg.RateLimiter),
		userAgent:       cfg.UserAgent,
		selfHosted:      cfg.SelfHosted,
		promptCache:     cfg.PromptCache,
		replayReasoning: cfg.ReplayReasoning,
		headers:         cfg.Headers,
	}, cfg.Retry), nil
}

// NewResponsesProvider builds the Provider for the OpenAI Responses API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
// The returned Provider retries transient failures per ProviderConfig.Retry.
// The concrete implementation is unexported; consumers hold only the Provider
// interface.
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("agentic: ResponsesConfig.BaseURL is required")
	}
	return newProvider(&responsesProvider{
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
		httpClient: rateLimitedClient(cfg.HTTPClient, cfg.RateLimiter),
		userAgent:  cfg.UserAgent,
		store:      cfg.Store,
		headers:    cfg.Headers,
	}, cfg.Retry), nil
}

// NewAnthropicProvider builds the Provider for the Anthropic Messages API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
// The returned Provider retries transient failures per ProviderConfig.Retry.
// The concrete implementation is unexported; consumers hold only the Provider
// interface.
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("agentic: AnthropicConfig.BaseURL is required")
	}
	return newProvider(&anthropicProvider{
		baseURL:        cfg.BaseURL,
		apiKey:         cfg.APIKey,
		version:        cfg.Version,
		httpClient:     rateLimitedClient(cfg.HTTPClient, cfg.RateLimiter),
		userAgent:      cfg.UserAgent,
		disableCaching: cfg.DisableCaching,
		headers:        cfg.Headers,
	}, cfg.Retry), nil
}
