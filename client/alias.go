package client

import (
	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/extras"
)

// The format's own types are the client's types. They are aliases, not
// conversions: a value built here IS the core value, so a caller can hand
// to the encoder, a transport, or another client without a copy step, and
// nothing has to be kept in sync between declarations of the same thing.
//
// [Completion] is the exception, and the reason this package exists -- see
// completion.go.
type (
	Role                 = commonai.Role
	Message              = commonai.Message
	Part                 = commonai.Part
	PartKind             = commonai.PartKind
	TextPart             = commonai.TextPart
	ImagePart            = commonai.ImagePart
	ThinkingPart         = commonai.ThinkingPart
	RedactedThinkingPart = commonai.RedactedThinkingPart
	ToolCallPart         = commonai.ToolCallPart
	ThinkingBlock        = commonai.ThinkingBlock
	ToolCall             = commonai.ToolCall
	ToolDecl             = commonai.ToolDecl
	Request              = commonai.Request
	Usage                = commonai.Usage
	Timings              = commonai.Timings
	PromptProgress       = commonai.PromptProgress
	StreamEvents         = commonai.StreamEvents
	RetryAttempt         = commonai.RetryAttempt
	APIError             = commonai.APIError
	UnsupportedError     = commonai.UnsupportedError
	Dialect              = commonai.Dialect
	Param                = commonai.Param

	// Rates is what model charges per token; ModelList is the document it comes out of.
	Rates     = commonai.Rates
	ModelList = commonai.ModelList

	// RetryPolicy and RateLimiter are the extras' policies, reachable here via ProviderConfig.
	RetryPolicy = extras.RetryPolicy
	RateLimiter = extras.RateLimiter
)

// Roles.
const (
	RoleSystem    = commonai.RoleSystem
	RoleUser      = commonai.RoleUser
	RoleAssistant = commonai.RoleAssistant
	RoleTool      = commonai.RoleTool
)

// Part kinds.
const (
	PartKindText             = commonai.PartKindText
	PartKindImage            = commonai.PartKindImage
	PartKindThinking         = commonai.PartKindThinking
	PartKindRedactedThinking = commonai.PartKindRedactedThinking
	PartKindToolCall         = commonai.PartKindToolCall
)

// Normalized stop reasons.
const (
	StopEndTurn   = commonai.StopEndTurn
	StopToolUse   = commonai.StopToolUse
	StopMaxTokens = commonai.StopMaxTokens
)

// Wire dialects.
const (
	DialectAuto      = commonai.DialectAuto
	DialectOpenAI    = commonai.DialectOpenAI
	DialectAnthropic = commonai.DialectAnthropic
	DialectResponses = commonai.DialectResponses
)

// DefaultRetry is the retry policy a provider gets when its config names none.
var DefaultRetry = extras.DefaultRetry

// Error classification. A call the loop above sees has already been through
// whatever retrying its provider does, so these answer "what kind of failure
// am I holding", not "should I try again".
var (
	IsTransient       = commonai.IsTransient
	IsContextOverflow = commonai.IsContextOverflow
	IsUnsupported     = commonai.IsUnsupported
	Unsupported       = commonai.Unsupported
	ErrorKind         = commonai.ErrorKind
	IsBadRequest      = commonai.IsBadRequest
	// DialectRefused: see docs/dialect-refusal.md
	DialectRefused = commonai.DialectRefused
)

// Error constructors; a caller's own marker for refusal would be classified transient and re-sent.
var (
	BadRequest    = commonai.BadRequest
	CallbackError = commonai.CallbackError
)

// Format entry points, for a caller that wants the document a call would
// produce -- to store a conversation, to send over a transport, or to
// check against the schema.
var (
	NewMessage           = commonai.NewMessage
	Validate             = commonai.Validate
	DecodeRequest        = commonai.DecodeRequest
	DecodeConversation   = commonai.DecodeConversation
	DecodeResponse       = commonai.DecodeResponse
	EncodeRequest        = commonai.EncodeRequest
	EncodeConversation   = commonai.EncodeConversation
	EncodeRequestBytes   = commonai.EncodeRequestBytes
	ParamsFromJSONObject = commonai.ParamsFromJSONObject
	ParamsJSON           = commonai.ParamsJSON
	Dialects             = commonai.Dialects
	DecodeModelList      = commonai.DecodeModelList
	Anomalous            = commonai.Anomalous
	NewRateLimiter       = extras.NewRateLimiter
	// Bool addresses a boolean for ToolDecl's tri-state behaviour fields.
	Bool = commonai.Bool
)
