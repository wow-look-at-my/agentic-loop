package loop

import (
	"errors"
)

// MaxTurns is an optional host-enforced cap on model calls. It is useful for
// interactive hosts that must bound a turn; the cap is checked before each
// call, and a final capped call is made without tools so it can answer rather
// than produce unhandled tool calls.

// SubagentReportKind is the Kind the loop sets on subagent delivery messages
// it injects, so a host can persist them with the right label.
const SubagentReportKind = "subagent_report"

// wrapUpInstruction is appended as a final user turn to force a model that
// stalled at "thinking" (no content) into actually writing its answer from
// the information it already gathered, instead of the loop surfacing raw
// chain-of-thought as the result.
const wrapUpInstruction = "Stop researching and write your final answer now, using only the information already gathered above. " +
	"Do not call any tools and do not keep thinking -- output the complete, self-contained report that directly answers the task."

// noOutputPlaceholder is returned as the final content when the model
// produced neither content nor reasoning, so the caller never gets a
// confusing empty result.
const noOutputPlaceholder = "(subagent produced no output)"

// StuckNudgeAt and StuckFailAt bound a model that stops making progress: a
// turn whose tool calls are byte-identical to the previous turn's cannot
// learn anything new, because the same calls produce the same results, which
// produce the same turn again. The StuckNudgeAt-th identical turn in a row
// gets one nudge appended after its tool results; the StuckFailAt-th ends the
// run with ErrStuck. Any change in what the model asks for clears the count.
//
// With no turn cap, this is the loop's own bound on a model that has stopped
// progressing -- and unlike a cap it fires on evidence of uselessness rather
// than on a budget running out.
//
// They are constants, not knobs: a loop repeating itself verbatim is never
// the model working, so there is nothing to tune.
const (
	StuckNudgeAt = 3
	StuckFailAt  = 6
)

// stuckNudgeInstruction is appended as a user turn after the results of the
// StuckNudgeAt-th identical tool-call batch, telling the model what the
// transcript alone does not: that it is repeating itself, and that repeating
// again ends the run.
const stuckNudgeInstruction = "You have now requested the same tool calls several times in a row and received the same results each time. " +
	"Repeating them again cannot tell you anything new. Do something different: act on the results you already have, " +
	"call a different tool, or write your final answer now. Another identical request ends this run."

// ErrStuck ends a run whose model kept requesting a byte-identical tool-call
// batch after being nudged. Callers match it with errors.Is; the partial
// Result rides alongside it like every other mid-run failure.
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
	// MaxTurns caps model calls when positive. The final permitted call is
	// tool-less, so requested tools are not stranded without results.
	MaxTurns int

	// ResourceWatcher, when set, is polled at every turn boundary. A non-empty
	// poll is delivered to the model as a user-role notice (and mirrored to the
	// host via OnResourceNotice) before the next model call. nil disables
	// resource watching. ResourceDiffTool, when non-empty, is quoted in the
	// notice so the model calls the name it was actually given.
	ResourceWatcher  ResourceWatcher
	ResourceDiffTool string

	// KeepAlive holds a reference that keeps subscribed event callbacks from
	// being garbage-collected for the life of the run. event.Event holds weak
	// pointers, so a caller that subscribes must retain the *func(T) error it
	// passed to Subscribe — store the struct holding those function fields here.
	KeepAlive any

	// SystemMessages is the queue for automated notices -- a CI status
	// change, a stop-hook nudge, a sub-agent report. UserMessages is the same
	// channel for what the user sent while the model was working. Any
	// goroutine calls Queue to deliver a message INTO this run.
	//
	// Both are drained at the top of every turn, system first, and a queued
	// message starts another turn when the model would otherwise finish -- so
	// a message either queue ACCEPTS always reaches the model. Run closes both
	// as it returns, which is how a producer racing the end of a run learns to
	// start a new run instead (Queue reports false); anything queued and never
	// delivered comes back in Result.Undelivered. A nil queue accepts nothing,
	// for the same reason: there is no run to deliver it.
	SystemMessages *MessageQueue
	UserMessages   *MessageQueue

	// Subagents is the registry an asynchronous run_subagent reports into (the
	// same value given to SubagentConfig.Runs). When set, a turn that would
	// otherwise END while sub-agents are still out instead waits for the next
	// report and delivers it as a user message, so the loop keeps the promise
	// the launch receipt made. nil means nothing was launched asynchronously.
	Subagents *SubagentRuns

	// DisableOutputDedup opts out of collapsing byte-identical read-only tool
	// results into [unchanged] markers. On by default; only set when the full
	// output must always reach the model.
	DisableOutputDedup bool

	// turnHook, when non-nil, is invoked with the 1-based turn number as each
	// numbered turn begins (the stall-fallback wrap-up call is not a numbered
	// turn). It is unexported: package-internal machinery -- the subagent
	// tool's live activity telemetry -- not public API.
	TurnHook func(turn int)

	// unknownTool, when non-nil, replaces the text a call to an unoffered name
	// is answered with. Unexported: the sub-agent run uses it to say WHY a name
	// its parent has is not in this run's toolset (read-only only, or outside
	// the granted allowed_tools), which a bare "unknown tool" would not teach.
	UnknownTool func(name string) string
}

// Result is the outcome of a Run. Messages is the input transcript plus
// everything the loop appended (assistant turns, tool results, and the final
// message). Usages holds one entry per model call IN ORDER -- deliberately not
// summed, because successive prompts overlap (each turn re-sends the growing
// transcript) and summing would double-count the shared prefix many times
// over. Turns is the number of model calls made.
//
// Undelivered holds messages that were queued (SystemMessages or
// UserMessages) but never reached the model, system first. It is empty on a
// run that ended normally -- a queued message starts another turn -- and
// non-empty only when the run ended for another reason first: a cancelled
// ctx, a model-call error, an aborted tool batch. Whoever produced those
// messages believes the model saw them, so they come back here to be
// re-delivered rather than disappearing with the run.
type Result struct {
	Messages    []Message
	Final       Message
	Usages      []Usage
	Turns       int
	Undelivered []Message
}
