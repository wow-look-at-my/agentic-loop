package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// There is deliberately NO turn cap. A counted cap cannot tell "looping
// uselessly" from "deep in a hard task", so it fires at the worst moment:
// after the model has spent every call gathering context and just before it
// writes any of it down. What bounds a run instead is ErrStuck (repetition is
// the only mechanically detectable form of not-progressing) and the caller's
// ctx, which bounds wall-clock and spend without discarding work in flight.

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

// batchFingerprint identifies a tool-call batch by what the model asked for --
// the calls' names and raw arguments, in order. Call IDs are deliberately
// excluded: providers mint a fresh ID per call, so including them would make
// every batch unique and the detector dead code. Lengths are interleaved so
// no pair of adjacent fields can be re-cut into a different batch with the
// same fingerprint.
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
// may mutate -- the transcript, system, or Extra it is about to send), OnTurnEnd
// fires after each call (with the turn number, the Completion -- nil when the
// call failed before producing one -- and the call's error), OnToolCall fires
// before each requested tool call is handled, and OnToolResult fires with its
// recorded result (executed, denied, or a teaching error). All optional.
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
	OnTurnBegin  func(turn int, req *Request) error
	OnTurnEnd    func(turn int, comp *Completion, err error) error
	OnToolCall   func(ToolCall) error
	OnToolResult func(ToolCall, ToolResult) error
}

// emitTurnBegin forwards a numbered turn's begin, tolerating nil callbacks.
// The hook receives the per-call Request that is about to be sent and may
// mutate it; the mutations apply to that one call only.
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

// emitToolCall forwards a requested tool call, tolerating nil callbacks.
func (e *Events) emitToolCall(c ToolCall) error {
	if e == nil || e.OnToolCall == nil {
		return nil
	}
	return wrapCallbackErr(e.OnToolCall(c))
}

// emitToolResult forwards a recorded tool result, tolerating nil callbacks.
func (e *Events) emitToolResult(c ToolCall, r ToolResult) error {
	if e == nil || e.OnToolResult == nil {
		return nil
	}
	return wrapCallbackErr(e.OnToolResult(c, r))
}

// Config wires one Run: the Provider to call, the Tools advertised and
// executed (empty runs tool-less), the Approver consulted for calls a tool
// flags via NeedsApproval (nil denies gated calls with DeniedMessage), and
// the event callbacks. There is deliberately no turn cap
// -- see the note above wrapUpInstruction for what bounds a run instead.
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
	Events   Events

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
	turnHook func(turn int)

	// unknownTool, when non-nil, replaces the text a call to an unoffered name
	// is answered with. Unexported: the sub-agent run uses it to say WHY a name
	// its parent has is not in this run's toolset (read-only only, or outside
	// the granted allowed_tools), which a bare "unknown tool" would not teach.
	unknownTool func(name string) string
}

// Result is the outcome of a Run. Messages is the input transcript plus
// everything the loop appended (assistant turns, tool results, and the final
// message). Usages holds one entry per model call IN ORDER -- deliberately not
// summed, because successive prompts overlap (each turn re-sends the growing
// transcript) and summing would double-count the shared prefix many times
// over. Turns is the number of model calls made.
type Result struct {
	Messages []Message
	Final    Message
	Usages   []Usage
	Turns    int
}

// Run drives the agentic tool loop: it calls cfg.Provider on the growing
// transcript, executes the tool calls each turn requests via cfg.Tools, feeds
// the results back, and stops when the model answers with text.
//
// Each turn advertises cfg.Tools.Decls(); req.Tools is ignored and
// overwritten (an empty cfg.Tools advertises no tools). On the final permitted
// turn tools are WITHHELD so the model must answer rather than request
// another never-executed call. Tool failures never abort the loop: an
// Execute error becomes a recoverable "tool execution failed: ..." error
// result, a call the model hallucinated with no executor configured gets an
// "unknown tool: ..." error result, and a denied approval records
// DeniedMessage -- in every case the loop continues so the model can react.
//
// An Approver.Ask error (the decision never arrived) ends the run: the
// current assistant message keeps its content and reasoning but its ToolCalls
// are cleared and the batch's already-appended tool results are dropped, so
// the returned transcript stays replayable with no orphan tool calls; the
// partial Result is returned alongside the error. An error returned by
// OnToolCall or OnToolResult ends the run the same way, and a stream
// callback error surfaces through the model call as a partial completion
// plus the callback's error -- in every case the partial Result rides
// alongside the error and the transcript carries no orphan tool calls.
//
// Read-only tool results that are byte-identical to an earlier call in the
// run are fed back as a short [unchanged] marker instead of the full content
// (see OutputDeduper; Config.DisableOutputDedup opts out).
//
// A model-call error ENDS the run -- the loop assumes any failure reaching it
// is permanent (see Config). Transient failures never get this far: the
// Provider rides them out, and a retried call is one turn here because Run
// only ever sees the outcome. When a call fails after data arrived, the
// partial assistant message is finalized into the transcript (tool calls
// cleared) and the partial Result is returned alongside the error. Whenever
// Run returns an error together with a non-nil Result, the Result carries the
// transcript accumulated so far.
//
// If the loop ends with the model having produced no content (a
// thinking-only turn, or the cap hit mid-research), one extra tool-less
// wrap-up turn asks it to synthesize an answer from what it gathered; failing
// that, the final content falls back to the accumulated reasoning, then to a
// clear placeholder.
func Run(ctx context.Context, cfg Config, req Request) (*Result, error) {
	if cfg.Provider == nil {
		return nil, badRequestErr("agentic: Config.Provider is required")
	}
	advertised := cfg.Tools.Decls()

	// Output dedup: one deduper for the whole run, so an unchanged read-only
	// result collapses to a marker instead of re-dumping a huge output. What
	// is eligible is the deduper's own decision -- it reads the declaration.
	var deduper *OutputDeduper
	if !cfg.DisableOutputDedup {
		deduper = NewOutputDeduper()
	}

	transcript := make([]Message, len(req.Messages), len(req.Messages)+8)
	copy(transcript, req.Messages)

	res := &Result{}
	// Stuck detection (see StuckNudgeAt): the previous turn's tool-call
	// fingerprint and how many turns in a row have repeated it.
	lastBatch := ""
	repeats := 0
	finish := func(final Message) (*Result, error) {
		transcript = append(transcript, final)
		res.Messages = transcript
		res.Final = final
		return res, nil
	}

	for turn := 0; ; turn++ {
		if cfg.turnHook != nil {
			cfg.turnHook(turn + 1)
		}
		comp, err := runModelCall(ctx, &cfg, req, turn+1, transcript, advertised, res)
		if err != nil {
			if comp != nil {
				// Mid-stream break/cancel: keep the partial content, reasoning
				// and usage, but drop any assembled tool calls -- they were
				// never executed, and replaying an assistant tool_call with no
				// matching result 400s on most upstreams.
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
		// allowed to run them: replay the assistant's tool-call message, then
		// each tool result, so the next turn sees the full sub-conversation.
		if len(calls) > 0 {
			// A batch identical to the previous turn's makes no progress: the
			// same calls return the same results, which produce the same
			// batch again. Nudge once, then end the run rather than spending
			// the remaining turns on it. The failing batch is never executed
			// -- it would only repeat work already in the transcript -- so the
			// assistant message is finalized with its tool calls cleared,
			// leaving no orphan to replay.
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
			// abortBatch ends the run mid-batch -- an approval decision that
			// never arrived, or a tool callback that returned an error. It
			// clears the pending batch: the assistant message keeps its
			// content/reasoning but loses its tool_calls, and this batch's
			// already-appended results are dropped, so no orphans remain to
			// replay.
			abortBatch := func(cause error) (*Result, error) {
				transcript = transcript[:aIdx+1]
				cleared := transcript[aIdx]
				cleared.ToolCalls = nil
				transcript[aIdx] = cleared
				res.Messages = transcript
				res.Final = cleared
				return res, cause
			}
			for _, call := range calls {
				if cberr := cfg.Events.emitToolCall(call); cberr != nil {
					return abortBatch(cberr)
				}
				result, aerr := resolveCall(ctx, &cfg, call)
				if aerr != nil {
					return abortBatch(aerr)
				}
				if cberr := cfg.Events.emitToolResult(call, result); cberr != nil {
					return abortBatch(cberr)
				}
				content := result.Content
				if deduper != nil {
					if tool, known := cfg.Tools.Find(call.Name); known {
						if collapsed, deduped := deduper.Collapse(tool.Decl(), result); deduped {
							content = collapsed
						}
					}
				}
				transcript = append(transcript, Message{
					Role:        RoleTool,
					Content:     content,
					ToolCallID:  call.ID,
					ToolIsError: result.IsError,
				})
			}
			if repeats == StuckNudgeAt {
				transcript = append(transcript, Message{Role: RoleUser, Content: stuckNudgeInstruction})
			}
			continue
		}

		// The model asked for no tools -- but sub-agents launched earlier in
		// this run may still be out, and their launch receipt promised the
		// model it would be notified. Deliver what has landed (waiting for the
		// next report if none has) and keep looping, so the model actually
		// sees them; that promise is the whole reason an asynchronous
		// run_subagent may return before it has an answer.
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

		// The loop is ending: the model asked for no tools. ToolCalls is
		// cleared defensively so a replayable transcript can never carry an
		// orphan.
		if strings.TrimSpace(assistant.Content) != "" {
			final := assistant
			final.ToolCalls = nil
			return finish(final)
		}

		// The model stopped without writing an answer -- it produced only
		// reasoning, or hit the turn cap mid-research. When tools were in
		// play (so it may already have gathered useful results), make one
		// final tool-less request that forces it to synthesize an answer from
		// what it has. The stalling turn's assistant message is deliberately
		// NOT in the transcript (it is only appended on the tool-execution
		// branch), so the wrap-up request can't be rejected for an unanswered
		// tool call.
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
					return finish(final)
				}
			}
			// The wrap-up failed or still produced nothing: fall through to
			// the last-resort fallback (the error, if any, is swallowed like
			// the source's synthesize step).
		}

		// Last resort: the reasoning (a thinking model's only output), then a
		// clear placeholder, so the caller never gets a confusing empty
		// result.
		final := assistant
		final.ToolCalls = nil
		if strings.TrimSpace(final.Content) == "" {
			final.Content = fallbackOutput(assistant)
		}
		return finish(final)
	}
}

// runModelCall executes one model call and counts it as one turn -- including
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
		// The call happened; its data is kept, the run aborts on the sink
		// failure (Run finalizes the completion like a mid-stream break).
		return comp, cberr
	}
	return comp, err
}

// resolveCall produces the recorded ToolResult for one requested call:
// executed, denied, or a teaching error. The returned error is non-nil ONLY
// when an approval decision never arrived (Approver.Ask failed), which ends
// the run.
func resolveCall(ctx context.Context, cfg *Config, call ToolCall) (ToolResult, error) {
	tool, known := cfg.Tools.Find(call.Name)
	if !known {
		// A name this run does not offer: the model hallucinated a tool, or it
		// remembers one its parent has. Teach it rather than aborting.
		text := "unknown tool: " + call.Name
		if cfg.unknownTool != nil {
			text = cfg.unknownTool(call.Name)
		}
		return ToolResult{Content: text, IsError: true}, nil
	}
	if tool.NeedsApproval() {
		if cfg.Approver == nil {
			// Fail closed: a gated tool with nobody to ask is denied.
			return ToolResult{Content: DeniedMessage, IsError: true}, nil
		}
		allowed, aerr := cfg.Approver.Ask(ctx, call)
		if aerr != nil {
			return ToolResult{}, fmt.Errorf("agentic: tool approval interrupted: %w", aerr)
		}
		if !allowed {
			return ToolResult{Content: DeniedMessage, IsError: true}, nil
		}
	}
	// The id is threaded on the context, where it is known -- the Tool
	// interface stays id-free (see toolcallid.go).
	result, exErr := tool.Execute(WithToolCallID(ctx, call.ID), json.RawMessage(call.Arguments))
	if exErr != nil {
		// Defensive: internal failures are surfaced as tool text so the model
		// can react rather than aborting the conversation.
		return ToolResult{Content: "tool execution failed: " + exErr.Error(), IsError: true}, nil
	}
	return result, nil
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
