package loop

import (
	"errors"
)

// MaxTurns is an optional host-enforced cap on model calls.

// DefaultAutoCompact is the auto-compact fraction (0.8); zero disables the feature.
const DefaultAutoCompact = 0.8

// SubagentReportKind is the Kind set on subagent delivery messages the loop injects.
const SubagentReportKind = "subagent_report"

// CompactionKind marks the summary that replaces a compacted transcript.
const CompactionKind = "compaction"

// wrapUpInstruction forces a stalled model to write its answer from what it gathered.
const wrapUpInstruction = "Stop researching and write your final answer now, using only the information already gathered above. " +
	"Do not call any tools and do not keep thinking -- output the complete, self-contained report that directly answers the task."

// noOutputPlaceholder is the final content when a turn has no text, tool call, or thinking.
const noOutputPlaceholder = "(no output was produced this turn)"

// StuckNudgeAt and StuckFailAt bound a model repeating identical tool-call batches.
const (
	StuckNudgeAt = 3
	StuckFailAt  = 6
)

// stuckNudgeInstruction tells the model it is repeating itself and repeating ends the run.
const stuckNudgeInstruction = "You have now requested the same tool calls several times in a row and received the same results each time. " +
	"Repeating them again cannot tell you anything new. Do something different: act on the results you already have, " +
	"call a different tool, or write your final answer now. Another identical request ends this run."

// ErrStuck ends a run whose model kept repeating a byte-identical tool-call batch.
var ErrStuck = errors.New("agentic: model is stuck repeating the same tool calls")

// Config wires one Run: the Provider to call, the Tools advertised and
// executed (empty runs tool-less), the Approver consulted for EVERY tool call
// (nil allows a Readonly tool and denies anything else with DeniedMessage),
// and the event callbacks. MaxTurns, when positive, caps model calls.
//
// Output dedup is ON by default: a read-only tool result whose content is
// byte-identical to an earlier call in the same run is fed back as a short
// [unchanged] marker instead of the full text (see OutputDeduper). Set
// DisableOutputDedup to turn that off.
//
// There is deliberately NO retry knob. The loop is a high-level construct:
// it knows nothing about connections, status codes, or backoff, and an error
// that reaches it is treated as REAL and PERMANENT -- the layer whose job was
// to make the call happen has already given up, so Run stops rather than
// second-guessing it. Riding out transient failure belongs to the Provider
// (ProviderConfig.Retry), which is also the only layer that can see whether a
// call streamed anything -- the condition that decides whether re-sending is
// safe. See "Layering" in README.md.
type Config struct {
	Provider Provider
	Tools    Tools
	Approver Approver
	Events   *Events
	// MaxTurns caps model calls when positive; the final permitted call is tool-less.
	MaxTurns int

	// ResourceWatcher, when set, is polled at every turn boundary; nil disables it.
	ResourceWatcher  ResourceWatcher
	ResourceDiffTool string

	// KeepAlive keeps subscribed event callbacks from being garbage-collected.
	KeepAlive any

	// Messages delivers system notices and user messages INTO the run; a
	// queued message reaches the model. One queue, both kinds.
	Messages *MessageQueue

	// Subagents is the registry an asynchronous run_subagent reports into; nil = none.
	Subagents *SubagentRuns

	// DisableOutputDedup opts out of collapsing byte-identical read-only results.
	DisableOutputDedup bool

	// ContextWindow is the model's context window size; zero disables auto-compaction.
	ContextWindow int

	// ElapsedTime, when set, states how long has passed since the previous request on every call.
	ElapsedTime *ElapsedTime

	// turnHook, when non-nil, is invoked with the 1-based turn number as each turn begins.
	TurnHook func(turn int)

	// unknownTool, when non-nil, replaces the text answering a call to an unoffered name.
	UnknownTool func(name string) string
}

// Result is the outcome of a Run: the transcript, usages in order, turn count, undelivered.
type Result struct {
	Messages    []Message
	Final       Message
	Usages      []Usage
	Turns       int
	Undelivered []Message
}
