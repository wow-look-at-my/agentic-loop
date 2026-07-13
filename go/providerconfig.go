package agentic

import (
	"fmt"
	"net/http"
)

// Dialect selects the wire dialect a Provider built by NewProvider speaks.
// The string values match the "mode"/"dialect" convention of the source
// applications' configuration ("openai" / "anthropic").
type Dialect string

// The two supported dialects.
const (
	DialectOpenAI    Dialect = "openai"
	DialectAnthropic Dialect = "anthropic"
)

// ProviderConfig configures NewProvider. Dialect and BaseURL are required;
// everything else is optional. The dialect-specific knobs are ignored by the
// other dialect.
type ProviderConfig struct {
	// Dialect selects the wire protocol: DialectOpenAI for OpenAI-compatible
	// chat completions, DialectAnthropic for the Anthropic Messages API.
	Dialect Dialect
	// BaseURL is the API root and is required. OpenAI: include the version
	// segment (e.g. "https://api.openai.com/v1"); requests POST to
	// BaseURL + "/chat/completions". Anthropic: the bare root (e.g.
	// "https://api.anthropic.com"); requests POST to BaseURL + "/v1/messages".
	// Trailing slashes are trimmed before joining.
	BaseURL string
	// APIKey, when non-empty, authenticates requests: a Bearer token on the
	// OpenAI dialect, the x-api-key header on Anthropic.
	APIKey string
	// HTTPClient performs the requests; nil uses http.DefaultClient.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header.
	UserAgent string
	// Headers are applied after the dialect defaults, so a caller-supplied
	// header can override them.
	Headers map[string]string

	// SelfHosted (OpenAI dialect only) adds cache_prompt:true to every request
	// — the KV-cache prefix-reuse opt-in llama.cpp-style servers honor. It must
	// stay false for hosted OpenAI/Azure, which reject unknown body fields with
	// a 400.
	SelfHosted bool
	// AnthropicVersion (Anthropic dialect only) sets the anthropic-version
	// header; empty defaults to "2023-06-01".
	AnthropicVersion string
	// DisableCaching (Anthropic dialect only) drops the two ephemeral
	// cache_control breakpoints the provider otherwise places on every request.
	DisableCaching bool
}

// NewProvider builds a Provider for the configured dialect. It fails fast —
// with a permanent (never-retried) error — on a missing/unknown Dialect or an
// empty BaseURL. The concrete dialect implementations are unexported;
// consumers hold only the Provider interface.
func NewProvider(cfg ProviderConfig) (Provider, error) {
	if cfg.BaseURL == "" {
		return nil, badRequestErr("agentic: ProviderConfig.BaseURL is required")
	}
	switch cfg.Dialect {
	case DialectOpenAI:
		return &openaiProvider{
			baseURL:    cfg.BaseURL,
			apiKey:     cfg.APIKey,
			httpClient: cfg.HTTPClient,
			userAgent:  cfg.UserAgent,
			selfHosted: cfg.SelfHosted,
			headers:    cfg.Headers,
		}, nil
	case DialectAnthropic:
		return &anthropicProvider{
			baseURL:        cfg.BaseURL,
			apiKey:         cfg.APIKey,
			version:        cfg.AnthropicVersion,
			httpClient:     cfg.HTTPClient,
			userAgent:      cfg.UserAgent,
			disableCaching: cfg.DisableCaching,
			headers:        cfg.Headers,
		}, nil
	case "":
		return nil, badRequestErr(`agentic: ProviderConfig.Dialect is required ("openai" or "anthropic")`)
	default:
		return nil, badRequestErr(fmt.Sprintf("agentic: unknown ProviderConfig.Dialect %q (want %q or %q)",
			cfg.Dialect, DialectOpenAI, DialectAnthropic))
	}
}
