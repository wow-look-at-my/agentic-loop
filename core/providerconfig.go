package commonai

import "net/http"

// ProviderConfig holds the connection settings shared by every dialect. It is
// not used on its own: embed it in the per-dialect config types (OpenAIConfig,
// AnthropicConfig) accepted by the dialect constructors.
type ProviderConfig struct {
	// BaseURL is the required API root; OpenAI includes the version segment, Anthropic the bare root.
	BaseURL string
	// APIKey, when non-empty, authenticates requests: Bearer on OpenAI, x-api-key on Anthropic.
	APIKey string
	// HTTPClient performs the requests; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header.
	UserAgent string
	// Headers are applied after the dialect defaults, so a caller-supplied header can override them.
	Headers map[string]string
}

// OpenAIConfig configures NewOpenAIProvider: the shared ProviderConfig
// connection base plus the knobs specific to the OpenAI-compatible dialect.
type OpenAIConfig struct {
	ProviderConfig

	// SelfHosted adds cache_prompt:true to every request; must stay false for hosted OpenAI/Azure.
	SelfHosted bool
	// PromptCache emits two ephemeral cache_control breakpoints for Anthropic-fronting gateways; default false.
	PromptCache bool
	// ReplayReasoning replays reasoning text as message.reasoning; default false for strict OpenAI servers.
	ReplayReasoning bool
}

// ResponsesConfig configures NewResponsesProvider: the shared ProviderConfig
// connection base plus the one knob specific to the OpenAI Responses dialect.
type ResponsesConfig struct {
	ProviderConfig

	// Store opts into the API's server-side retention; defaults false, the caller decides.
	Store bool
}

// AnthropicConfig configures NewAnthropicProvider: the shared ProviderConfig
// connection base plus the knobs specific to the Anthropic Messages dialect.
type AnthropicConfig struct {
	ProviderConfig

	// Version sets the anthropic-version header; empty defaults to "2023-06-01".
	Version string
	// DisableCaching drops the two ephemeral cache_control breakpoints the provider otherwise places.
	DisableCaching bool
}

// NewOpenAIProvider builds the Provider for OpenAI-compatible chat-completions
// APIs. It fails fast -- with a permanent (never-retried) error -- on an empty
// BaseURL. The concrete implementation is
// unexported; consumers hold only the Provider interface.
func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("commonai: OpenAIConfig.BaseURL is required")
	}
	return (&openaiProvider{
		baseURL:         cfg.BaseURL,
		apiKey:          cfg.APIKey,
		httpClient:      cfg.HTTPClient,
		userAgent:       cfg.UserAgent,
		selfHosted:      cfg.SelfHosted,
		promptCache:     cfg.PromptCache,
		replayReasoning: cfg.ReplayReasoning,
		headers:         cfg.Headers,
	}), nil
}

// NewResponsesProvider builds the Provider for the OpenAI Responses API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
// The concrete implementation is unexported; consumers hold only the Provider
// interface.
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("commonai: ResponsesConfig.BaseURL is required")
	}
	return (&responsesProvider{
		baseURL:    cfg.BaseURL,
		apiKey:     cfg.APIKey,
		httpClient: cfg.HTTPClient,
		userAgent:  cfg.UserAgent,
		store:      cfg.Store,
		headers:    cfg.Headers,
	}), nil
}

// NewAnthropicProvider builds the Provider for the Anthropic Messages API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
// The concrete implementation is unexported; consumers hold only the Provider
// interface.
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("commonai: AnthropicConfig.BaseURL is required")
	}
	return (&anthropicProvider{
		baseURL:        cfg.BaseURL,
		apiKey:         cfg.APIKey,
		version:        cfg.Version,
		httpClient:     cfg.HTTPClient,
		userAgent:      cfg.UserAgent,
		disableCaching: cfg.DisableCaching,
		headers:        cfg.Headers,
	}), nil
}
