package loop

import (
	"github.com/wow-look-at-my/go-containers/event"
)

// MessageID is the loop's strong type for a transcript entry's identifier.
type MessageID string

// --- Event param structs: every struct embeds event.Args and is an event.Event[T]. ---

// TurnBeginEvent is the param to OnTurnBegin: turn number and the mutable per-call Request.
type TurnBeginEvent struct {
	event.Args
	Turn int
	Req  *Request
}

// TurnEndEvent is the param to OnTurnEnd: turn number, Completion, and the call's error.
type TurnEndEvent struct {
	event.Args
	Turn int
	Comp *Completion
	Err  error
}

// StopEvent is the param to OnStop: purely observational; continue by queueing a message.
type StopEvent struct {
	event.Args
	Turn int
	Comp *Completion
}

// ToolCallEvent is the param to OnToolCall; the hook may mutate the call to run.
type ToolCallEvent struct {
	event.Args
	Call *ToolCall
}

// ToolResultEvent is the param to OnToolResult: executed call, result, and recorded message.
type ToolResultEvent struct {
	event.Args
	Call     ToolCall
	Result   ToolResult
	Recorded Message
}

// AssistantMessageEvent is the param to OnAssistantMessage; listener sets ID.
type AssistantMessageEvent struct {
	event.Args
	ParentID MessageID
	ID       *MessageID
}

// FinalizeAssistantEvent is the param to OnFinalizeAssistant: id, finalized Message, and status.
type FinalizeAssistantEvent struct {
	event.Args
	ID     MessageID
	Msg    Message
	Status string
}

// ToolMessageEvent is the param to OnToolMessage: ids, content, Parts, and error flag.
type ToolMessageEvent struct {
	event.Args
	ToolCallID        string
	ParentAssistantID MessageID
	Content           string
	Parts             []ToolContentPart
	IsError           bool
}

// ResourceNoticeEvent is the param to OnResourceNotice; listener sets ID.
type ResourceNoticeEvent struct {
	event.Args
	Content   string
	ChangeIDs []string
	ID        *MessageID
}

// SystemMessageEvent is the param to OnSystemMessage; the host may set ID.
type SystemMessageEvent struct {
	event.Args
	Msg Message
	ID  *MessageID
}

// CompactionEvent is the param to OnCompaction; the host may set ID to the row it stored.
type CompactionEvent struct {
	event.Args
	Summary    string
	Messages   []Message
	Completion *Completion
	ID         *MessageID
}

// Events are the loop's callbacks; all optional, and a returned error aborts the run.
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
	OnCompaction        event.Event[CompactionEvent]
}

// emitTurnBegin forwards a numbered turn's begin, tolerating nil callbacks.
func (e *Events) emitTurnBegin(ev TurnBeginEvent) error {
	return wrapCallbackErr(e.OnTurnBegin.Invoke(ev))
}

// emitTurnEnd forwards a numbered turn's end, tolerating nil callbacks.
func (e *Events) emitTurnEnd(ev TurnEndEvent) error {
	return wrapCallbackErr(e.OnTurnEnd.Invoke(ev))
}

// emitStop notifies listeners the model is about to stop; continue by queueing a message.
func (e *Events) emitStop(ev StopEvent) {
	_ = e.OnStop.Invoke(ev)
}

// emitToolCall forwards the call about to be handled; the hook may rewrite it.
func (e *Events) emitToolCall(ev ToolCallEvent) error {
	return wrapCallbackErr(e.OnToolCall.Invoke(ev))
}

// emitToolResult forwards a tool result with the message the loop recorded for it.
func (e *Events) emitToolResult(ev ToolResultEvent) error {
	return wrapCallbackErr(e.OnToolResult.Invoke(ev))
}

// emitAssistantMessage asks the host to mint the durable row; "" falls back to a local id.
func (e *Events) emitAssistantMessage(ev AssistantMessageEvent) (MessageID, error) {
	var id MessageID
	ev.ID = &id
	err := e.OnAssistantMessage.Invoke(ev)
	return id, err
}

// emitFinalizeAssistant tells the host the turn reached a terminal status.
func (e *Events) emitFinalizeAssistant(ev FinalizeAssistantEvent) {
	e.OnFinalizeAssistant.Invoke(ev)
}

// emitToolMessage tells the host to persist tool result in transcript order.
func (e *Events) emitToolMessage(ev ToolMessageEvent) error {
	return e.OnToolMessage.Invoke(ev)
}

// emitResourceNotice tells the host a notice is coming; it may persist it and return its id.
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

// emitCompaction reports the compaction; the returned row id keeps the next turn attached.
func (e *Events) emitCompaction(ev CompactionEvent) MessageID {
	var id MessageID
	ev.ID = &id
	_ = e.OnCompaction.Invoke(ev)
	return id
}
