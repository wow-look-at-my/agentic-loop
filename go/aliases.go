// aliases.go is this package's wire half, which now lives in common-ai-api.
// Everything here is an ALIAS or a thin call through, never a copy: a value
// built as agentic.Message IS a client.Message, so the two libraries hand
// values to each other without a conversion step and nothing has to be kept in
// sync between two declarations of the same type.
//
// The split is what moved, not the API. Every name below meant the same thing
// before it, and the test suite that proves it is the one that came with the
// loop -- unchanged.
package agentic

import (
	"context"

	"github.com/wow-look-at-my/common-ai-api/go/client"
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

// The four conversation roles. RoleSystem messages are normally supplied via
// Request.System rather than the transcript; RoleTool messages carry tool
// results back to the model.
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
// caller branches on one set of strings rather than three.
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
	DialectOpenAI    = client.DialectOpenAI
	DialectAnthropic = client.DialectAnthropic
	DialectResponses = client.DialectResponses
)

// DefaultRetry is the policy a provider gets when its config names none: 10
// attempts, 500ms base delay.
var DefaultRetry = client.DefaultRetry

// Error classification. A call the loop sees has already been through whatever
// retrying its provider does, so these answer "what kind of failure am I
// holding", not "should I try again".
var (
	IsTransient       = client.IsTransient
	IsContextOverflow = client.IsContextOverflow
)

// NewOpenAIProvider builds the Provider for OpenAI-compatible chat-completions
// APIs. It fails fast -- with a permanent (never-retried) error -- on an empty
// BaseURL.
func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error) {
	return client.NewOpenAIProvider(cfg)
}

// NewResponsesProvider builds the Provider for the OpenAI Responses API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error) {
	return client.NewResponsesProvider(cfg)
}

// NewAnthropicProvider builds the Provider for the Anthropic Messages API. It
// fails fast -- with a permanent (never-retried) error -- on an empty BaseURL.
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error) {
	return client.NewAnthropicProvider(cfg)
}

// NewParamStripper wraps a Provider with rejected-parameter recovery: when a
// call fails before anything streamed and the error text names a parameter
// present in the request's Extra (matched by normalized form, so
// reasoningEffort matches reasoning_effort), that key is removed and the call
// is retried ONCE. The strip is remembered, so subsequent calls through the
// same stripper drop the key up front, without mutating the caller's Extra
// map. A context cancellation is never treated as a parameter problem, and a
// call that already streamed (a non-nil completion) is never retried.
func NewParamStripper(p Provider) Provider { return client.NewParamStripper(p) }

// NewRateLimiter returns a limiter permitting n request starts per minute,
// spaced evenly. n <= 0 returns nil, meaning no limiting.
func NewRateLimiter(n int) *RateLimiter { return client.NewRateLimiter(n) }

// DetectDialect asks an endpoint which dialect it speaks. It is a package
// function rather than a Provider method because it answers what to BUILD,
// before there is one.
func DetectDialect(ctx context.Context, cfg ProviderConfig) (Dialect, error) {
	return client.DetectDialect(ctx, cfg)
}

// Dialects returns every dialect that can be named, in a stable order.
func Dialects() []Dialect { return client.Dialects() }

// DialectOfModelList guesses a dialect from the body of a model-list response.
func DialectOfModelList(body []byte) Dialect { return client.DialectOfModelList(body) }
