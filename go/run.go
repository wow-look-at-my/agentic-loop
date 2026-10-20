package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// There is deliberately NO turn cap. A counted cap cannot tell "looping uselessly" from "deep in a hard task". It fires at the worst moment: after the model has spent every call gathering context and before it
// writes any of it down.

// wrapUpInstruction is appended as a final user turn to force a model. That model stalled at "thinking" (no content) into writing its answer from the information it already gathered, instead of the loop.
const wrapUpInstruction = "Stop researching and write your final answer now, using only the information already gathered above. " +
	"Do not call any tools and do not keep thinking -- output the complete, self-contained report that directly answers the task."

// noOutputPlaceholder is returned as the final content when the model produced neither content nor reasoning.
const noOutputPlaceholder = "(subagent produced no output)"

// StuckNudgeAt and StuckFailAt bound a model that stops making progress: a
// turn whose tool calls are byte-identical to the turn's cannot learn
// anything new.
const (
	StuckNudgeAt = 3
	StuckFailAt  = 6
)

// stuckNudgeInstruction is appended as a user turn after the results of the StuckNudgeAt-th identical tool-call batch. That stuckNudgeInstruction is telling the model what the transcript alone does not: that it is repeating itself, and that
// repeating again ends the run.
const stuckNudgeInstruction = "You have now requested the same tool calls several times in a row and received the same results each time. " +
	"Repeating them again cannot tell you anything new. Do something different: act on the results you already have, " +
	"call a different tool, or write your final answer now. Another identical request ends this run."

// ErrStuck ends a run whose model kept requesting a byte-identical tool-call batch after being nudged.
var ErrStuck = errors.New("agentic: model is stuck repeating the same tool calls")

// batchFingerprint identifies a tool-call batch by what the model asked for
// -- the calls' names and raw arguments, in order.
func batchFingerprint(calls []ToolCall) string {
	var b strings.Builder
	for _, c := range calls {
		fmt.Fprintf(&b, "%d:%s|%d:%s|", len(c.Name), c.Name, len(c.Arguments), c.Arguments)
	}
	return b.String()
}

// Events are the loop's callbacks: the embedded StreamEvents fire during each
// model call, OnTurnBegin fires before each numbered model call (with the
// 1-based turn number and a pointer to the per-call Request, which the hook
// may mutate -- the transcript, system, or Extra it is about to send).
type Events struct {
	StreamEvents
	OnTurnBegin func(turn int, req *Request) error
	OnTurnEnd   func(turn int, comp *Completion, err error) error
	// OnToolCall receives a POINTER to the call about to be handled, and mutations to it -- Arguments, in practice -- are what runs.
	OnToolCall func(call *ToolCall) error
	// OnToolResult receives the call as EXECUTED (any OnToolCall rewrite included), the tool's own result, and the RoleTool message the loop appended -- which is not always the result.
	OnToolResult func(call ToolCall, result ToolResult, recorded Message) error
}

// emitTurnBegin forwards a numbered turn's begin, tolerating nil callbacks.
func (e *Events) emitTurnBegin(turn int, req *Request) error {
	if e == nil || e.OnTurnBegin == nil {
		return nil
	}
	return wrapCallbackErr(e.OnTurnBegin(turn, req))
}

// emitTurnEnd forwards a numbered turn's end, tolerating nil callbacks.
func (e *Events) emitTurnEnd(turn int, comp *Completion, err error) error {
	if e == nil || e.OnTurnEnd == nil {
		return nil
	}
	return wrapCallbackErr(e.OnTurnEnd(turn, comp, err))
}

// emitToolCall forwards the call about to be handled, tolerating nil
// callbacks.
func (e *Events) emitToolCall(c *ToolCall) error {
	if e == nil || e.OnToolCall == nil {
		return nil
	}
	return wrapCallbackErr(e.OnToolCall(c))
}

// emitToolResult forwards a tool result together with the message the loop
// recorded for it, tolerating nil callbacks.
func (e *Events) emitToolResult(c ToolCall, r ToolResult, recorded Message) error {
	if e == nil || e.OnToolResult == nil {
		return nil
	}
	return wrapCallbackErr(e.OnToolResult(c, r, recorded))
}

// Config wires one Run: the Provider to call, the Tools advertised and
// executed (empty runs tool-less), the Approver consulted for EVERY tool call
// (nil allows a Readonly tool and denies anything else with DeniedMessage).
type Config struct {
	Provider Provider
	Tools    Tools
	Approver Approver
	Events   Events

	// Subagents is the registry an asynchronous run_subagent reports into (the same value given to SubagentConfig.Runs).
	Subagents *SubagentRuns

	// DisableOutputDedup opts out of collapsing byte-identical read-only tool results into [unchanged] markers.
	DisableOutputDedup bool

	// Inbox, when non-nil, is drained at the top of every turn.
	Inbox Inbox

	// turnHook, when non-nil, is invoked with the 1-based turn number as each numbered turn begins (the stall-fallback wrap-up call is not a numbered turn).
	turnHook func(turn int)

	// unknownTool, when non-nil, replaces the text a call to an unoffered name is answered with.
	unknownTool func(name string) string
}

// Result is the outcome of a Run.
type Result struct {
	Messages []Message
	Final    Message
	Usages   []Usage
	Turns    int
}

// Run drives the agentic tool loop. It calls cfg.Provider on the growing
// transcript and executes the tool calls each turn requests via cfg.Tools. It
// also feeds the results back, and stops when the model answers with text.
//
// Each turn advertises cfg.Tools.Decls(); req.Tools is ignored and overwritten
// (an empty cfg.Tools advertises no tools). Every requested call goes to
// cfg.Approver first -- read-only ones included, so a host's deny rules apply
// to the whole toolset. A nil Approver allows a Readonly tool and denies the
// rest. Tool failures never abort the loop: an Execute error becomes a
// recoverable "tool execution failed: ..." error result, a call the model
// hallucinated with no executor configured gets an "unknown tool: ..." error
// result, and a refused call records the Approval's Reason, or DeniedMessage
// when it carried none. In every case the loop continues so the model can
// react.
func Run(ctx context.Context, cfg Config, req Request) (*Result, error) {
	if cfg.Provider == nil {
		return nil, badRequestErr("agentic: Config.Provider is required")
	}
	advertised := cfg.Tools.Decls()

	// Output dedup: one deduper for the whole run.
	var deduper *OutputDeduper
	if !cfg.DisableOutputDedup {
		deduper = NewOutputDeduper()
	}

	transcript := make([]Message, len(req.Messages), len(req.Messages)+8)
	copy(transcript, req.Messages)

	res := &Result{}
	// Stuck detection (see StuckNudgeAt): the turn's tool-call fingerprint and how many turns in a row have repeated it.
	lastBatch := ""
	repeats := 0
	// finish returns the run's result. Before it does, it drains the inbox: a
	// message that arrived during the final turn must be answered, not dropped.
	finish := func(final Message) (*Result, bool) {
		transcript = append(transcript, final)
		if cfg.Inbox != nil {
			var drained bool
			transcript, drained = drainInbox(ctx, cfg.Inbox, transcript)
			if drained {
				return nil, true
			}
		}
		r := &Result{}
		r.Messages = transcript
		r.Final = final
		return r, false
	}

	for turn := 0; ; turn++ {
		if cfg.turnHook != nil {
			cfg.turnHook(turn + 1)
		}
		// Drain the inbox before every model call.
		if cfg.Inbox != nil {
			transcript, _ = drainInbox(ctx, cfg.Inbox, transcript)
		}
		comp, err := runModelCall(ctx, &cfg, req, turn+1, transcript, advertised, res)
		if err != nil {
			if comp != nil {
				// Mid-stream break/cancel: keep the partial content, reasoning and usage, but drop any assembled tool calls -- they were never executed.
				partial := comp.Message
				partial.ToolCalls = nil
				transcript = append(transcript, partial)
				res.Messages = transcript
				res.Final = partial
				return res, err
			}
			res.Messages = transcript
			return res, err
		}
		assistant := comp.Message
		calls := assistant.ToolCalls

		// Keep looping while the model is still requesting tools and we are
		// allowed to run them. Replay the assistant's tool-call message, then
		// each tool result, so the next turn sees the full sub-conversation.
		if len(calls) > 0 {
			// A batch identical to the turn's makes no progress: the same calls return
			// the same results, which produce the same batch again.
			if fp := batchFingerprint(calls); fp == lastBatch {
				repeats++
			} else {
				lastBatch, repeats = fp, 1
			}
			if repeats >= StuckFailAt {
				stuck := assistant
				stuck.ToolCalls = nil
				transcript = append(transcript, stuck)
				res.Messages = transcript
				res.Final = stuck
				return res, fmt.Errorf("%w: %d identical turns in a row", ErrStuck, repeats)
			}

			transcript = append(transcript, assistant)
			aIdx := len(transcript) - 1
			// abortBatch ends the run mid-batch -- an approval decision that never
			// arrived, or a tool callback that returned an error.
			abortBatch := func(cause error) (*Result, error) {
				transcript = transcript[:aIdx+1]
				cleared := transcript[aIdx]
				cleared.ToolCalls = nil
				transcript[aIdx] = cleared
				res.Messages = transcript
				res.Final = cleared
				return res, cause
			}
			for _, asked := range calls {
				// The hook sees a copy, not the transcript's own entry.
				call := asked
				if cberr := cfg.Events.emitToolCall(&call); cberr != nil {
					return abortBatch(cberr)
				}
				result, aerr := resolveCall(ctx, &cfg, call)
				if aerr != nil {
					return abortBatch(aerr)
				}
				content := result.Content
				if deduper != nil {
					if tool, known := cfg.Tools.Find(call.Name); known {
						if collapsed, deduped := deduper.Collapse(tool.Decl(), result); deduped {
							content = collapsed
						}
					}
				}
				// The id answered is the MODEL's, never a rewritten one.
				recorded := Message{
					Role:        RoleTool,
					Content:     content,
					ToolCallID:  asked.ID,
					ToolIsError: result.IsError,
				}
				if cberr := cfg.Events.emitToolResult(call, result, recorded); cberr != nil {
					return abortBatch(cberr)
				}
				transcript = append(transcript, recorded)
			}
			if repeats == StuckNudgeAt {
				transcript = append(transcript, Message{Role: RoleUser, Content: stuckNudgeInstruction})
			}
			continue
		}

		// The model asked for no tools -- but sub-agents launched earlier in this
		// run may still be out. Their launch receipt promised the model it would be
		// notified. Deliver what has landed (waiting for the next report if none
		// has) and keep looping, so the model sees them. That promise is the whole
		// reason an asynchronous run_subagent may return before it has an answer.
		if cfg.Subagents.Pending() > 0 {
			reports, cerr := cfg.Subagents.Collect(ctx)
			if cerr != nil {
				res.Messages = transcript
				return res, cerr
			}
			if len(reports) > 0 {
				if strings.TrimSpace(assistant.Content) != "" {
					answered := assistant
					answered.ToolCalls = nil
					transcript = append(transcript, answered)
				}
				transcript = append(transcript, Message{
					Role:    RoleUser,
					Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
				})
				continue
			}
		}

		// The loop is ending: the model asked for no tools.
		if strings.TrimSpace(assistant.Content) != "" {
			final := assistant
			final.ToolCalls = nil
			if r, loop := finish(final); loop {
				continue
			} else {
				return r, nil
			}
		}

		// The model stopped without writing an answer -- it produced only
		// reasoning. When tools were in play (so it may already have gathered
		// useful results), make one final tool-less request that forces it to
		// synthesize an answer from what it has. The stalling turn's assistant
		// message is deliberately NOT in the transcript (it is only appended on the
		// tool-execution branch), so the wrap-up request can't be rejected for an
		// unanswered tool call.
		if len(cfg.Tools) > 0 {
			wrapMsg := Message{Role: RoleUser, Content: wrapUpInstruction}
			wrapMsgs := make([]Message, len(transcript), len(transcript)+1)
			copy(wrapMsgs, transcript)
			wrapMsgs = append(wrapMsgs, wrapMsg)
			comp2, err2 := runModelCall(ctx, &cfg, req, turn+2, wrapMsgs, nil, res)
			if err2 == nil {
				if s := strings.TrimSpace(comp2.Message.Content); s != "" {
					final := comp2.Message
					final.ToolCalls = nil
					transcript = append(transcript, wrapMsg)
					if r, loop := finish(final); loop {
						continue
					} else {
						return r, nil
					}
				}
			}
		}

		// Last resort.
		final := assistant
		final.ToolCalls = nil
		if strings.TrimSpace(final.Content) == "" {
			final.Content = fallbackOutput(assistant)
		}
		if r, loop := finish(final); loop {
			continue
		} else {
			return r, nil
		}
	}
}

// runModelCall executes one model call. It counts it as one turn -- including
// when the provider re-attempted it internally, which Run neither sees nor
// needs to. Every call that produced a completion -- success or partial --
// appends its usage to the result. turn is the 1-based turn number (the stall
// wrap-up call is one past the turn that stalled): OnTurnBegin fires with the
// per-call Request (mutations apply to this call only), OnTurnEnd after it.
func runModelCall(
	ctx context.Context, cfg *Config,
	req Request, turn int, msgs []Message, tools []ToolDecl, res *Result,
) (*Completion, error) {
	r := req
	r.Messages = msgs
	r.Tools = tools
	if cberr := cfg.Events.emitTurnBegin(turn, &r); cberr != nil {
		// The call never happened; nothing to count or record.
		return nil, cberr
	}

	comp, err := cfg.Provider.Complete(ctx, r, &cfg.Events.StreamEvents)
	res.Turns++
	if comp != nil {
		res.Usages = append(res.Usages, comp.Usage)
	}
	if cberr := cfg.Events.emitTurnEnd(turn, comp, err); cberr != nil {
		// The call happened; its data is kept, the run aborts on the sink failure.
		return comp, cberr
	}
	return comp, err
}

// resolveCall produces the recorded ToolResult for one requested call:
// executed, denied, or a teaching error. EVERY call is put to the Approver --
// a tool has no say in whether it is asked about. A host's deny rule reaches
// read-only calls too. The returned error is non-nil ONLY when an approval
// decision never arrived (Approver.Ask failed), which ends the run.
func resolveCall(ctx context.Context, cfg *Config, call ToolCall) (ToolResult, error) {
	tool, known := cfg.Tools.Find(call.Name)
	if !known {
		// A name this run does not offer: the model hallucinated a tool, or it remembers one its parent has.
		text := "unknown tool: " + call.Name
		if cfg.unknownTool != nil {
			text = cfg.unknownTool(call.Name)
		}
		return ToolResult{Content: text, IsError: true}, nil
	}
	if cfg.Approver == nil {
		// Nobody to ask: read, but change nothing.
		if !tool.Decl().Readonly {
			return ToolResult{Content: DeniedMessage, IsError: true}, nil
		}
	} else {
		verdict, aerr := cfg.Approver.Ask(ctx, call)
		if aerr != nil {
			return ToolResult{}, fmt.Errorf("agentic: tool approval interrupted: %w", aerr)
		}
		if !verdict.OK {
			return ToolResult{Content: deniedText(verdict.Reason), IsError: true}, nil
		}
	}
	// The id is threaded on the context, where it is known -- the Tool interface stays id-free (see toolcallid.go).
	result, exErr := tool.Execute(WithToolCallID(ctx, call.ID), json.RawMessage(call.Arguments))
	if exErr != nil {
		// Defensive: internal failures are surfaced as tool text.
		return ToolResult{Content: "tool execution failed: " + exErr.Error(), IsError: true}, nil
	}
	return result, nil
}

// deniedText is what a refused call records: the approver's own reason, and
// DeniedMessage when it gave none.
func deniedText(reason string) string {
	if r := strings.TrimSpace(reason); r != "" {
		return r
	}
	return DeniedMessage
}

// fallbackOutput picks the text to surface when the loop ends without a
// written answer: the content, then the accumulated reasoning (a thinking
// model's only output), then a clear placeholder.
func fallbackOutput(m Message) string {
	if s := strings.TrimSpace(m.Content); s != "" {
		return s
	}
	var b strings.Builder
	for _, t := range m.Thinking {
		b.WriteString(t.Text)
	}
	if s := strings.TrimSpace(b.String()); s != "" {
		return s
	}
	return noOutputPlaceholder
}
