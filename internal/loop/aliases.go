// aliases.go is this package's wire half, which lives in client/ over core/.
// Everything here is an ALIAS or a thin call through, never a copy: a value
// built as agentic.Message IS a client.Message, so the loop and the wire half
// hand values to each other without a conversion step and nothing has to be
// kept in sync between declarations of the same type.
package loop

import (
	"context"

	"github.com/wow-look-at-my/agentic-loop/client"
)

// The conversation vocabulary.
type (
	Role          = client.Role
	ThinkingBlock = client.ThinkingBlock
	ToolCall      = client.ToolCall
	Message       = client.Message
	ToolDecl      = client.ToolDecl
	Usage         = client.Usage
)

// The roles: RoleSystem via Request.System, RoleTool carries tool results.
const (
	RoleSystem    = client.RoleSystem
	RoleUser      = client.RoleUser
	RoleAssistant = client.RoleAssistant
	RoleTool      = client.RoleTool
)

// The model call and its answer.
type (
	Request        = client.Request
	Completion     = client.Completion
	Provider       = client.Provider
	StreamEvents   = client.StreamEvents
	PromptProgress = client.PromptProgress
	Timings        = client.Timings
	RetryAttempt   = client.RetryAttempt
	APIError       = client.APIError
)

// Normalized stop reasons. A provider maps its own vocabulary onto these, so a
const (
	StopEndTurn   = client.StopEndTurn
	StopToolUse   = client.StopToolUse
	StopMaxTokens = client.StopMaxTokens
)

// Provider construction and its policies.
type (
	ProviderConfig  = client.ProviderConfig
	OpenAIConfig    = client.OpenAIConfig
	ResponsesConfig = client.ResponsesConfig
	AnthropicConfig = client.AnthropicConfig
	RetryPolicy     = client.RetryPolicy
	RateLimiter     = client.RateLimiter
	Dialect         = client.Dialect
)

// Wire dialects.
const (
	DialectAuto      = client.DialectAuto
	DialectOpenAI    = client.DialectOpenAI
	DialectAnthropic = client.DialectAnthropic
	DialectResponses = client.DialectResponses
)

// DefaultRetry is the policy used when a config names none: attempts, 500ms base delay.
var DefaultRetry = client.DefaultRetry

// Error classification: what kind of failure am I holding, not should I retry.
var (
	IsTransient       = client.IsTransient
	IsContextOverflow = client.IsContextOverflow
)

// NewOpenAIProvider builds the OpenAI chat-completions Provider; fails fast on an empty BaseURL.
func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error) {
	return client.NewOpenAIProvider(cfg)
}

// NewResponsesProvider builds the OpenAI Responses API Provider; fails fast on an empty BaseURL.
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error) {
	return client.NewResponsesProvider(cfg)
}

// NewAnthropicProvider builds the Anthropic Messages API Provider; fails fast on an empty BaseURL.
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error) {
	return client.NewAnthropicProvider(cfg)
}

// NewParamStripper retries when a failed call names a rejected parameter, dropping it.
func NewParamStripper(p Provider) Provider { return client.NewParamStripper(p) }

// NewRateLimiter permits n request starts per minute; n <= returns nil (no limiting).
func NewRateLimiter(n int) *RateLimiter { return client.NewRateLimiter(n) }

// Dialects returns every dialect that can be named, in a stable order.
func Dialects() []Dialect { return client.Dialects() }

// Rates is model's USD-per-token charges; ModelList is the endpoint's rates document.
type (
	Rates     = client.Rates
	ModelList = client.ModelList
)

// FetchModelList reads an endpoint's model list; unpriced models are ABSENT from Prices.
func FetchModelList(ctx context.Context, cfg ProviderConfig) (*ModelList, error) {
	return client.FetchModelList(ctx, cfg)
}

// DecodeModelList reads a model-list document; a document that will not parse is an error.
func DecodeModelList(body []byte) (*ModelList, error) { return client.DecodeModelList(body) }

// Anomalous reports a usage that priced more cached tokens than prompt tokens.
func Anomalous(u Usage) bool { return client.Anomalous(u) }

// Bool addresses a boolean for tri-state fields; nil means "not stated", not false.
func Bool(v bool) *bool { return client.Bool(v) }
