package agentic

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// Run drives the agentic tool loop: it calls cfg.Provider on the growing
// transcript, executes the tool calls each turn requests via cfg.Tools, feeds
// the results back, and stops when the model answers with text.
//
// Each turn advertises cfg.Tools.Decls(); req.Tools is ignored and
// overwritten (an empty cfg.Tools advertises no tools). Every requested call
// goes to cfg.Approver first -- read-only ones included, so a host's deny
// rules apply to the whole toolset -- and a nil Approver allows a Readonly
// tool and denies the rest. Tool failures never abort the loop: an Execute
// error becomes a recoverable "tool execution failed: ..." error result, a
// call the model hallucinated with no executor configured gets an "unknown
// tool: ..." error result, and a refused call records the Approval's Reason,
// or DeniedMessage when it carried none -- in every case the loop continues so
// the model can react.
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
// Within a batch, read-only ungated tool calls (ToolDecl.Readonly set, no
// NeedsApproval) execute concurrently via goroutines. Mutating or gated calls
// execute sequentially in call order; each mutating call waits for every
// in-flight read-only call to finish first, so workspace state is consistent
// at the start of each mutation. OnToolCall fires in call order, each call's
// hook immediately before that call is dispatched; OnToolResult and
// transcript append happen in call order after every call has resolved. The
// only observable nondeterminism is the execution order among read-only
// calls; the transcript, event callbacks, and exec count on abort are all
// deterministic.
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
// thinking-only turn, or a run its ctx cut short), one extra tool-less
// wrap-up turn asks it to synthesize an answer from what it gathered; failing
// that, the final content falls back to the accumulated reasoning, then to a
// clear placeholder.
func Run(ctx context.Context, cfg Config, req Request) (*Result, error) {
	if cfg.Provider == nil {
		return nil, badRequestErr("agentic: Config.Provider is required")
	}
	if cfg.Events == nil {
		cfg.Events = &Events{}
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
	// stopHookFired prevents a host hook from trapping the run in an infinite
	// continuation cycle.
	stopHookFired := false
	// Stuck detection (see StuckNudgeAt): the previous turn's tool-call
	// fingerprint and how many turns in a row have repeated it.
	lastBatch := ""
	repeats := 0
	// currentAssistantID tracks the id of the in-flight assistant turn so a
	// panic recovery can finalize it. It is set by emitAssistantMessage and
	// cleared by emitFinalizeAssistant.
	currentAssistantID := MessageID("")
	finalizeAssistant := func(ev FinalizeAssistantEvent) {
		cfg.Events.emitFinalizeAssistant(ev)
		currentAssistantID = ""
	}
	defer func() {
		if r := recover(); r != nil {
			if currentAssistantID != "" {
				finalizeAssistant(FinalizeAssistantEvent{ID: currentAssistantID, Status: "error"})
			}
			panic(r) // re-panic so the caller's recover sees it
		}
	}()
	finish := func(final Message) (*Result, error) {
		transcript = append(transcript, final)
		res.Messages = transcript
		res.Final = final
		return res, nil
	}

	for turn := 0; ; turn++ {
		if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns {
			break
		}
		// Sub-agent delivery: if any reports are ready (without waiting),
		// deliver them at the top of the turn so the model sees them. Skip on
		// turn 0 — nothing has launched yet. This is the "between turns"
		// path: a report that landed while the model was busy with tools.
		if turn > 0 && cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			reports := cfg.Subagents.Take()
			if len(reports) > 0 {
				delivery := Message{
					Role:    RoleUser,
					Kind:    SubagentReportKind,
					Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
				}
				if cfg.SystemMessages != nil {
					cfg.SystemMessages.Queue(delivery)
				} else {
					cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
					transcript = append(transcript, delivery)
				}
			}
		}
		// Drain queued messages: system first, then user. System messages
		// always precede user messages so an automated nudge is seen before
		// anything the user queued mid-run.
		for _, msg := range DrainBoth(cfg.SystemMessages, cfg.UserMessages) {
			cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: msg})
			transcript = append(transcript, msg)
		}
		// Resource watch: poll at the turn boundary, before the model call,
		// exactly where a host loop used to. A non-empty poll is delivered to
		// the model as a user-role notice and mirrored to the host via
		// OnResourceNotice. A poll error becomes a warning in the notice
		// (silence would read as "nothing changed"); only a cancelled ctx
		// aborts.
		if cfg.ResourceWatcher != nil {
			poll, perr := cfg.ResourceWatcher.Poll(ctx)
			if perr != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					res.Messages = transcript
					return res, ctxErr
				}
				poll.Warnings = append(poll.Warnings,
					"the resource watch itself failed this turn ("+perr.Error()+"), so no resource is known to be current")
			}
			if !poll.Empty() {
				notice := FormatResourceNotice(poll, cfg.ResourceDiffTool)
				ids := make([]string, 0, len(poll.Changes))
				for _, c := range poll.Changes {
					ids = append(ids, c.ChangeID)
				}
				hostID := cfg.Events.emitResourceNotice(ResourceNoticeEvent{Content: notice, ChangeIDs: ids})
				noticeMsg := Message{Role: RoleUser, Content: notice}
				if hostID != "" {
					noticeMsg.ID = string(hostID)
				}
				transcript = append(transcript, noticeMsg)
			}
		}
		if cfg.turnHook != nil {
			cfg.turnHook(turn + 1)
		}
		turnTools := advertised
		if cfg.MaxTurns > 0 && turn == cfg.MaxTurns-1 {
			turnTools = nil
		}
		// Ask the host to mint the durable row for this turn, hanging off the
		// last appended transcript entry. The returned id is attributed to the
		// completion; "" means the host is not persisting and the loop's
		// transcript is the only record.
		parentID := ""
		if n := len(transcript); n > 0 {
			parentID = transcript[n-1].ID
		}
		assistantID, aerr := cfg.Events.emitAssistantMessage(AssistantMessageEvent{ParentID: MessageID(parentID)})
		currentAssistantID = assistantID
		if aerr != nil {
			// The host failed to create or announce the row (e.g. the SSE
			// sink died on the meta event). Finalize the turn as an error so
			// the durable row is not stranded, then return the error.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: "error"})
			res.Messages = transcript
			return res, aerr
		}
		comp, err := runModelCall(ctx, &cfg, req, turn+1, transcript, turnTools, res)
		if err != nil {
			if comp != nil {
				// Mid-stream break/cancel: keep the partial content, reasoning
				// and usage, but drop any assembled tool calls -- they were
				// never executed, and replaying an assistant tool_call with no
				// matching result 400s on most upstreams.
				partial := comp.Message
				partial.ToolCalls = nil
				if assistantID != "" {
					partial.ID = string(assistantID)
				}
				status := "cancelled"
				if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
					status = "error"
				}
				transcript = append(transcript, partial)
				res.Messages = transcript
				res.Final = partial
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: partial, Status: status})
				return res, err
			}
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: "error"})
			res.Messages = transcript
			return res, err
		}
		assistant := comp.Message
		if assistantID != "" {
			assistant.ID = string(assistantID)
		}
		calls := assistant.ToolCalls

		// Keep looping while the model is still requesting tools and we are
		// allowed to run them: replay the assistant's tool-call message, then
		// each tool result, so the next turn sees the full sub-conversation.
		if len(calls) > 0 && (cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1) {
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
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: stuck, Status: "complete"})
				return res, fmt.Errorf("%w: %d identical turns in a row", ErrStuck, repeats)
			}

			transcript = append(transcript, assistant)
			aIdx := len(transcript) - 1
			// Finalize the assistant as complete with its tool calls before
			// executing them, matching the host's lifecycle: the row is
			// persisted (with tool calls for display) before tools run. If the
			// batch is aborted, abortBatch re-finalizes as cancelled.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: assistant, Status: "complete"})
			// abortBatch ends the run mid-batch -- an approval decision that
			// never arrived, or a tool callback that returned an error. It
			// clears the pending batch: the assistant message keeps its
			// content/reasoning but loses its tool_calls, and this batch's
			// already-appended results are dropped, so no orphans remain to
			// replay. The host is told to finalize the assistant row as
			// cancelled with no tool calls, so its durable tree matches the
			// loop's transcript.
			abortBatch := func(cause error) (*Result, error) {
				transcript = transcript[:aIdx+1]
				cleared := transcript[aIdx]
				cleared.ToolCalls = nil
				transcript[aIdx] = cleared
				res.Messages = transcript
				res.Final = cleared
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: cleared, Status: "cancelled"})
				return res, cause
			}
			asks := make([]ToolCall, 0, len(calls))
			results := make([]ToolResult, len(calls))
			var mu sync.Mutex
			var firstErr error
			var wg sync.WaitGroup    // all goroutines, for the abort path
			var reads sync.WaitGroup // read-only calls since the last barrier

			// Dispatch: read-only ungated calls run concurrently via
			// goroutines; mutating or gated calls run sequentially in call
			// order. Each mutating call is a barrier -- it waits for every
			// in-flight read-only call to finish first, so workspace state
			// is consistent at the start of each mutation. When no
			// read-only calls are present the entire batch runs sequentially
			// on the calling goroutine.
			//
			// resolveCall returns a non-nil error ONLY when an approval
			// decision never arrived (Approver.Ask failed). Read-only
			// ungated calls skip approval entirely, so they never produce
			// that error -- but the guard is kept for robustness.
			for i, asked := range calls {
				// The hook sees a copy, not the transcript's own entry: what
				// the model asked for is already recorded above and stays that
				// way, while everything downstream from here -- the approval
				// decision and the execution -- uses whatever the hook left.
				call := asked
				if cberr := cfg.Events.emitToolCall(ToolCallEvent{Call: &call}); cberr != nil {
					// Wait for any read-only calls already dispatched so no
					// goroutine outlives the batch, then clear it.
					wg.Wait()
					return abortBatch(cberr)
				}
				asks = append(asks, call)

				tool, known := cfg.Tools.Find(call.Name)
				readonly := known && tool.Decl().Readonly && !tool.NeedsApproval()

				if readonly {
					reads.Add(1)
					wg.Add(1)
					go func(idx int, c ToolCall) {
						defer wg.Done()
						defer reads.Done()
						r, aerr := resolveCall(ctx, &cfg, c)
						results[idx] = r
						if aerr != nil {
							mu.Lock()
							if firstErr == nil {
								firstErr = aerr
							}
							mu.Unlock()
						}
					}(i, call)
					continue
				}
				// Mutating or gated: barrier -- wait for every in-flight
				// read-only call to finish so the sequential execution has
				// a consistent view of workspace state.
				reads.Wait()

				result, aerr := resolveCall(ctx, &cfg, call)
				results[i] = result
				if aerr != nil {
					mu.Lock()
					if firstErr == nil {
						firstErr = aerr
					}
					mu.Unlock()
					break
				}
			}

			// Wait for all goroutines (including any still in flight when a
			// mutating call errored above) before reading results or
			// aborting, so no goroutine outlives the batch.
			wg.Wait()
			if firstErr != nil {
				return abortBatch(firstErr)
			}

			// Record every result in call order: OnToolResult fires
			// sequentially in call order, the transcript messages are
			// appended in call order, and the deduper collapses byte-
			// identical read-only results. This keeps the transcript fully
			// deterministic regardless of goroutine completion order.
			for i, call := range calls {
				content := results[i].Content
				parts := results[i].Parts
				if deduper != nil {
					if tool, known := cfg.Tools.Find(call.Name); known {
						if collapsed, deduped := deduper.Collapse(tool.Decl(), results[i]); deduped {
							content = collapsed
							parts = nil
						}
					}
				}
				// The id answered is the MODEL's, never a rewritten one: it
				// pairs this message with the tool_call already in the
				// transcript, and a mismatch there is an orphan no upstream
				// will replay.
				recorded := Message{
					Role:        RoleTool,
					Content:     content,
					ToolCallID:  call.ID,
					ToolIsError: results[i].IsError,
				}
				if cberr := cfg.Events.emitToolResult(ToolResultEvent{Call: asks[i], Result: results[i], Recorded: recorded}); cberr != nil {
					return abortBatch(cberr)
				}
				transcript = append(transcript, recorded)
				if terr := cfg.Events.emitToolMessage(ToolMessageEvent{
					ToolCallID:        asked.ID,
					ParentAssistantID: assistantID,
					Content:           content,
					Parts:             parts,
					IsError:           result.IsError,
				}); terr != nil {
					return abortBatch(terr)
				}
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
		if cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns-1 {
				// Capped final turn: don't wait for what's still running, but
				// deliver what has arrived and declare remaining subagents lost.
				reports := cfg.Subagents.Take()
				lost := cfg.Subagents.CancelRemaining()
				if len(reports) > 0 || lost > 0 {
					delivery := Message{
						Role:    RoleUser,
						Kind:    SubagentReportKind,
						Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), lost),
					}
					if cfg.SystemMessages != nil {
						cfg.SystemMessages.Queue(delivery)
					} else {
						cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
						transcript = append(transcript, delivery)
					}
					if strings.TrimSpace(assistant.Content) != "" {
						answered := assistant
						answered.ToolCalls = nil
						finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: answered, Status: "complete"})
						transcript = append(transcript, answered)
					}
					for _, msg := range DrainBoth(cfg.SystemMessages, cfg.UserMessages) {
						cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: msg})
						transcript = append(transcript, msg)
					}
					final := assistant
					final.ToolCalls = nil
					if strings.TrimSpace(final.Content) == "" {
						final.Content = fallbackOutput(assistant)
					}
					return finish(final)
				}
			} else {
				reports, cerr := cfg.Subagents.Collect(ctx)
				if cerr != nil {
					// Cancellation while waiting: finalize as cancelled, end gracefully.
					if errors.Is(cerr, context.Canceled) || errors.Is(cerr, context.DeadlineExceeded) {
						final := assistant
						final.ToolCalls = nil
						if strings.TrimSpace(final.Content) == "" {
							final.Content = fallbackOutput(assistant)
						}
						finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "cancelled"})
						return finish(final)
					}
					res.Messages = transcript
					return res, cerr
				}
				if len(reports) > 0 {
					if strings.TrimSpace(assistant.Content) != "" {
						answered := assistant
						answered.ToolCalls = nil
						finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: answered, Status: "complete"})
						transcript = append(transcript, answered)
					}
					if cfg.SystemMessages != nil {
						cfg.SystemMessages.Queue(Message{
							Role:    RoleUser,
							Kind:    SubagentReportKind,
							Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
						})
					} else {
						transcript = append(transcript, Message{
							Role:    RoleUser,
							Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
						})
					}
					continue
				}
			}
		}

		// The loop is ending: the model asked for no tools. ToolCalls is
		// cleared defensively so a replayable transcript can never carry an
		// orphan.
		if strings.TrimSpace(assistant.Content) != "" {
			final := assistant
			final.ToolCalls = nil
			if !stopHookFired {
				stopHookFired = true
				cfg.Events.emitStop(StopEvent{Turn: turn + 1, Comp: comp})
				if Pending(cfg.SystemMessages, cfg.UserMessages) {
					finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
					transcript = append(transcript, final)
					continue
				}
			}
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
			return finish(final)
		}

		// The model stopped without writing an answer -- it produced only
		// reasoning. When tools were in
		// play (so it may already have gathered useful results), make one
		// final tool-less request that forces it to synthesize an answer from
		// what it has. The stalling turn's assistant message is deliberately
		// NOT in the transcript (it is only appended on the tool-execution
		// branch), so the wrap-up request can't be rejected for an unanswered
		// tool call.
		if len(cfg.Tools) > 0 && (cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1) {
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

		if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns-1 {
			final := assistant
			final.ToolCalls = nil
			if strings.TrimSpace(final.Content) == "" {
				final.Content = fallbackOutput(assistant)
			}
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
			return finish(final)
		}

		// Last resort: the reasoning (a thinking model's only output), then a
		// clear placeholder, so the caller never gets a confusing empty
		// result.
		final := assistant
		final.ToolCalls = nil
		if strings.TrimSpace(final.Content) == "" {
			final.Content = fallbackOutput(assistant)
		}
		finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
		return finish(final)
	}

	// A positive cap always permits at least one call; this return is only
	// reachable if the cap broke the loop. Deliver any pending sub-agent
	// reports as a final delivery (whatever is ready, declaring the rest lost)
	// before finishing.
	if cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
		reports := cfg.Subagents.Take()
		lost := cfg.Subagents.CancelRemaining()
		if len(reports) > 0 || lost > 0 {
			delivery := Message{
				Role:    RoleUser,
				Kind:    SubagentReportKind,
				Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), lost),
			}
			cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
			transcript = append(transcript, delivery)
		}
	}
	return finish(Message{Role: RoleAssistant, Content: noOutputPlaceholder})
}
