// Package agentic is the module-root package: the agentic tool loop,
// re-exported from the implementation in internal/loop so the module
// root stays small forwarding file.
//
// Everything the loop package exposes — the Run loop, tools, approval,
// the wire half (aliases of the client types), compaction, sub-agent and
// resource-watch machinery — is available here by the same names, and
// the optional tool families (vfs, repo, subagent, webfetch, todo,
// resources) import this package. The implementation lives in
// internal/loop; this file only forwards it.
package agentic

import (
	"encoding/json"

	loop "github.com/wow-look-at-my/agentic-loop/internal/loop"
)

type APIError = loop.APIError
type AnthropicConfig = loop.AnthropicConfig
type Approval = loop.Approval
type Approver = loop.Approver
type AssistantMessageEvent = loop.AssistantMessageEvent
type CompactResult = loop.CompactResult
type Completion = loop.Completion
type CompactionEvent = loop.CompactionEvent
type Config = loop.Config
type Dialect = loop.Dialect
type ElapsedTime = loop.ElapsedTime
type Events = loop.Events
type FinalizeAssistantEvent = loop.FinalizeAssistantEvent
type Gate = loop.Gate
type Message = loop.Message
type MessageID = loop.MessageID
type MessageQueue = loop.MessageQueue
type ModelList = loop.ModelList
type OpenAIConfig = loop.OpenAIConfig
type OutputDeduper = loop.OutputDeduper
type PromptProgress = loop.PromptProgress
type Provider = loop.Provider
type ProviderConfig = loop.ProviderConfig
type QueuedMessage = loop.QueuedMessage
type RateLimiter = loop.RateLimiter
type Rates = loop.Rates
type Request = loop.Request
type ResourceChange = loop.ResourceChange
type ResourceNoticeEvent = loop.ResourceNoticeEvent
type ResourcePoll = loop.ResourcePoll
type ResourceWatcher = loop.ResourceWatcher
type ResponsesConfig = loop.ResponsesConfig
type Result = loop.Result
type RetryAttempt = loop.RetryAttempt
type RetryPolicy = loop.RetryPolicy
type Role = loop.Role
type StopEvent = loop.StopEvent
type StreamEvents = loop.StreamEvents
type SubagentReport = loop.SubagentReport
type SubagentRuns = loop.SubagentRuns
type SubagentState = loop.SubagentState
type SubagentUpdate = loop.SubagentUpdate
type SystemMessage = loop.SystemMessage
type SystemMessageEvent = loop.SystemMessageEvent
type ThinkingBlock = loop.ThinkingBlock
type Timings = loop.Timings
type Tool = loop.Tool
type ToolCall = loop.ToolCall
type ToolCallEvent = loop.ToolCallEvent
type ToolContentPart = loop.ToolContentPart
type ToolDecl = loop.ToolDecl
type ToolMessageEvent = loop.ToolMessageEvent
type ToolResult = loop.ToolResult
type ToolResultEvent = loop.ToolResultEvent
type Tools = loop.Tools
type TurnBeginEvent = loop.TurnBeginEvent
type TurnEndEvent = loop.TurnEndEvent
type Usage = loop.Usage
type UserMessage = loop.UserMessage

const RoleSystem = loop.RoleSystem
const RoleUser = loop.RoleUser
const RoleAssistant = loop.RoleAssistant
const RoleTool = loop.RoleTool
const StopEndTurn = loop.StopEndTurn
const StopToolUse = loop.StopToolUse
const StopMaxTokens = loop.StopMaxTokens
const DialectAuto = loop.DialectAuto
const DialectOpenAI = loop.DialectOpenAI
const DialectAnthropic = loop.DialectAnthropic
const DialectResponses = loop.DialectResponses

var IsTransient = loop.IsTransient
var IsContextOverflow = loop.IsContextOverflow

const ResourceAdded = loop.ResourceAdded
const ResourceModified = loop.ResourceModified
const ResourceRemoved = loop.ResourceRemoved
const StuckNudgeAt = loop.StuckNudgeAt
const StuckFailAt = loop.StuckFailAt
const SubagentQueued = loop.SubagentQueued
const SubagentRunning = loop.SubagentRunning
const SubagentDone = loop.SubagentDone
const SubagentFailed = loop.SubagentFailed
const SubagentAbandoned = loop.SubagentAbandoned
const CompactRequestText = loop.CompactRequestText
const CompactionHandoffPrefix = loop.CompactionHandoffPrefix
const CompactionKind = loop.CompactionKind
const DefaultAutoCompact = loop.DefaultAutoCompact
const DeniedMessage = loop.DeniedMessage
const ElapsedKind = loop.ElapsedKind
const SubagentDeliveryHeader = loop.SubagentDeliveryHeader
const SubagentReportKind = loop.SubagentReportKind
const UnchangedPrefix = loop.UnchangedPrefix
const Version = loop.Version

var DefaultRetry = loop.DefaultRetry
var ErrStuck = loop.ErrStuck

var Anomalous = loop.Anomalous
var Compact = loop.Compact
var CountLineChanges = loop.CountLineChanges
var CountLines = loop.CountLines

// Bool addresses a boolean, for ToolDecl's tri-state behaviour fields:
var Bool = loop.Bool
var DecodeModelList = loop.DecodeModelList
var Dialects = loop.Dialects
var FetchModelList = loop.FetchModelList
var FormatElapsed = loop.FormatElapsed
var FormatElapsedNotice = loop.FormatElapsedNotice
var FormatResourceNotice = loop.FormatResourceNotice
var FormatSubagentDelivery = loop.FormatSubagentDelivery
var HumanSize = loop.HumanSize
var NewAnthropicProvider = loop.NewAnthropicProvider
var NewGate = loop.NewGate
var NewOpenAIProvider = loop.NewOpenAIProvider
var NewOutputDeduper = loop.NewOutputDeduper
var NewParamStripper = loop.NewParamStripper
var NewRateLimiter = loop.NewRateLimiter
var NewResponsesProvider = loop.NewResponsesProvider
var NewSubagentRuns = loop.NewSubagentRuns
var NewTool = loop.NewTool
var OneShot = loop.OneShot
var Plural = loop.Plural
var ReadCapped = loop.ReadCapped
var Run = loop.Run
var ToolCallID = loop.ToolCallID
var TruncateRunes = loop.TruncateRunes
var UnifiedDiff = loop.UnifiedDiff
var WithToolCallID = loop.WithToolCallID

// EnumSchema is forwarded from internal/loop. A generic function cannot
func EnumSchema[In any](enums map[string][]string) json.RawMessage { return loop.EnumSchema[In](enums) }

// InferSchema is forwarded from internal/loop. A generic function cannot
func InferSchema[In any]() json.RawMessage { return loop.InferSchema[In]() }
