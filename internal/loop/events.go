package loop

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

// Events are the loop's callbacks: the embedded StreamEvents fire during each
// model call, OnTurnBegin fires before each numbered model call (with the
// 1-based turn number and a pointer to the per-call Request, which the hook
// may mutate -- the transcript, system, or Extra it is about to send), OnTurnEnd
// fires after each call (with the turn number, the Completion -- nil when the
// call failed before producing one -- and the call's error), OnToolCall fires
// before each requested tool call is handled and may rewrite it, and
// OnToolResult fires with the outcome (executed, refused, or a teaching error)
// and the message recorded for it. All optional.
//
// Turns are numbered from 1; the stall wrap-up call fires as one past the
// turn that stalled. Like the stream callbacks, OnTurnBegin and OnTurnEnd may
// return a non-nil error to abort the run: OnTurnBegin aborts before the call
// (no completion), OnTurnEnd aborts after it with the completed data kept (the
// assistant message is finalized the way a mid-stream break is).
//
// Like the stream callbacks, OnToolCall and OnToolResult may return a non-nil
// error to abort the run: the turn is finalized the way a cancellation is --
// the pending batch is cleared so the transcript stays replayable with no
// orphan tool calls -- and the partial Result is returned together with that
// error (errors.Is against the caller's sentinel holds; the error is never
// classified transient).
type Events struct {
	StreamEvents
	OnTurnBegin         event.Event[TurnBeginEvent]
	OnTurnEnd           event.Event[TurnEndEvent]
	OnStop              event.Event[StopEvent]
	OnToolCall          event.Event[ToolCallEvent]
	OnToolResult        event.Event[ToolResultEvent]
	OnAssistantMessage  event.Event[AssistantMessageEvent]
	OnFinalizeAssistant event.Event[FinalizeAssistantEvent]
	OnToolMessage       event.Event[ToolMessageEvent]
	OnResourceNotice    event.Event[ResourceNoticeEvent]
	OnSystemMessage     event.Event[SystemMessageEvent]
}

// emitTurnBegin forwards a numbered turn's begin, tolerating nil callbacks.
// The hook receives the per-call Request that is about to be sent and may
// mutate it; the mutations apply to that one call only.
func (e *Events) emitTurnBegin(ev TurnBeginEvent) error {
	return wrapCallbackErr(e.OnTurnBegin.Invoke(ev))
}

// emitTurnEnd forwards a numbered turn's end, tolerating nil callbacks.
func (e *Events) emitTurnEnd(ev TurnEndEvent) error {
	return wrapCallbackErr(e.OnTurnEnd.Invoke(ev))
}

// emitStop notifies listeners that the model is about to stop. A listener
// that wants the loop to continue calls cfg.SystemMessages.Queue(msg) during
// the callback; the loop checks the queue after Invoke.
func (e *Events) emitStop(ev StopEvent) {
	_ = e.OnStop.Invoke(ev)
}

// emitToolCall forwards the call about to be handled, tolerating nil
// callbacks. The hook may rewrite what it points at; the caller then resolves
// the rewritten call.
func (e *Events) emitToolCall(ev ToolCallEvent) error {
	return wrapCallbackErr(e.OnToolCall.Invoke(ev))
}

// emitToolResult forwards a tool result together with the message the loop
// recorded for it, tolerating nil callbacks.
func (e *Events) emitToolResult(ev ToolResultEvent) error {
	return wrapCallbackErr(e.OnToolResult.Invoke(ev))
}

// emitAssistantMessage asks the host to mint (or name) the durable row for one
// assistant turn, hanging off parentID. Returns "" when no hook is set, so the
// loop falls back to its own transcript-only id.
func (e *Events) emitAssistantMessage(ev AssistantMessageEvent) (MessageID, error) {
	var id MessageID
	ev.ID = &id
	err := e.OnAssistantMessage.Invoke(ev)
	return id, err
}

// emitFinalizeAssistant tells the host the turn reached a terminal status and
// the loop's view of the finalized message. The host mirrors the append in its
// durable tree and advances the leaf here.
func (e *Events) emitFinalizeAssistant(ev FinalizeAssistantEvent) {
	e.OnFinalizeAssistant.Invoke(ev)
}

// emitToolMessage tells the host to persist one tool result in transcript order,
// carrying the model's tool-call id and the parent assistant message id. The
// host advances its leaf here.
func (e *Events) emitToolMessage(ev ToolMessageEvent) error {
	return e.OnToolMessage.Invoke(ev)
}

// emitResourceNotice tells the host the loop is about to deliver a resource
// notice as a user message; the host may persist it and return its id so the
// loop can attribute the transcript entry to the host's durable row.
func (e *Events) emitResourceNotice(ev ResourceNoticeEvent) MessageID {
	var id MessageID
	ev.ID = &id
	_ = e.OnResourceNotice.Invoke(ev)
	return id
}

func (e *Events) emitSystemMessage(ev SystemMessageEvent) {
	var id MessageID
	ev.ID = &id
	_ = e.OnSystemMessage.Invoke(ev)
}
