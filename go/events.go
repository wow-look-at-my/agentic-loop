package agentic

import (
	"github.com/wow-look-at-my/go-containers/event"
)

// MessageID is the loop's strong type for a transcript entry's identifier.
// It is a string under the hood (a host mints it, or the loop attributes it
// from the host's OnAssistantMessage / OnResourceNotice return), but wrapping
// it keeps an id from being confused with a branch name, a tool name, or any
// other string that happens to flow through the same struct.
type MessageID string

// --- Event param structs ---
//
// Every struct embeds event.Args so it satisfies event.EventArgs, and every
// Events field is an event.Event[T]. Adding a field to a struct is a
// non-breaking change; adding a parameter to a function signature is not.
// "First wins" events (OnStop, OnAssistantMessage, OnResourceNotice) carry
// out-fields: the first listener that claims the event sets the field, and the
// loop reads it after Invoke.

// TurnBeginEvent is the param to OnTurnBegin: the 1-based turn number and the
// per-call Request the hook may mutate (mutations apply to that one call only).
type TurnBeginEvent struct {
	event.Args
	Turn int
	Req  *Request
}

// TurnEndEvent is the param to OnTurnEnd: the turn number, the Completion
// (nil when the call failed before producing one), and the call's error.
type TurnEndEvent struct {
	event.Args
	Turn int
	Comp *Completion
	Err  error
}

// StopEvent is the param to OnStop: the turn that produced a non-empty final
// answer and the completion the loop would finish with. The event is purely
// observational — a listener that wants the loop to continue calls
// cfg.SystemMessages.Queue(msg) rather than returning anything. The loop
// checks the queue after Invoke and continues if non-empty. Invoked at most
// once per run (guarded by stopHookFired in the loop).
type StopEvent struct {
	event.Args
	Turn int
	Comp *Completion
}

// ToolCallEvent is the param to OnToolCall: the call about to be handled. The
// hook may mutate the pointed-to call; mutations are what actually runs.
type ToolCallEvent struct {
	event.Args
	Call *ToolCall
}

// ToolResultEvent is the param to OnToolResult: the call as executed (any
// OnToolCall rewrite included), the tool's own result, and the RoleTool message
// the loop appended (which may carry a dedup marker instead of the full result).
type ToolResultEvent struct {
	event.Args
	Call     ToolCall
	Result   ToolResult
	Recorded Message
}

// AssistantMessageEvent is the param to OnAssistantMessage. The first listener
// that mints a durable row sets ID; subsequent listeners should check and skip.
// The loop attributes the returned ID to the completion.
type AssistantMessageEvent struct {
	event.Args
	ParentID MessageID
	ID       *MessageID
}

// FinalizeAssistantEvent is the param to OnFinalizeAssistant: the id returned
// from OnAssistantMessage (or "" when no hook was set), the finalized Message as
// the loop sees it, and the status the loop reached ("complete", "cancelled", or
// "error"). The host serializes Thinking/ToolCalls however its storage format
// requires; the loop does not prescribe a JSON shape.
type FinalizeAssistantEvent struct {
	event.Args
	ID     MessageID
	Msg    Message
	Status string
}

// ToolMessageEvent is the param to OnToolMessage: the model's tool-call id, the
// parent assistant message id, the recorded content (possibly a dedup marker),
// the structured Parts the tool produced, and whether the result was an error.
type ToolMessageEvent struct {
	event.Args
	ToolCallID        string
	ParentAssistantID MessageID
	Content           string
	Parts             []ToolContentPart
	IsError           bool
}

// ResourceNoticeEvent is the param to OnResourceNotice. The first listener that
// persists the notice sets ID; subsequent listeners should check and skip. The
// loop attributes the returned ID to the transcript entry.
type ResourceNoticeEvent struct {
	event.Args
	Content   string
	ChangeIDs []string
	ID        *MessageID
}

// SystemMessageEvent is the param to OnSystemMessage: a message the loop is
// about to append to the transcript from the system or user message queue
// (subagent deliveries, stop-hook nudges, etc.). The host persists it as a
// user-role message and may set ID so the loop attributes the transcript entry.
type SystemMessageEvent struct {
	event.Args
	Msg Message
	ID  *MessageID
}
