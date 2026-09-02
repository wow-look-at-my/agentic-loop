package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
)

// Run drives the agentic tool loop: call the provider, execute tools, feed results back, stop on an answer.
func Run(ctx context.Context, cfg Config, req Request) (*Result, error) {
	if cfg.Provider == nil {
		return nil, badRequestErr("agentic: Config.Provider is required")
	}
	if cfg.Events == nil {
		cfg.Events = &Events{}
	}
	advertised := cfg.Tools.Decls()

	// Output dedup: deduper for the whole run collapses unchanged read-only results.
	var deduper *OutputDeduper
	if !cfg.DisableOutputDedup {
		deduper = NewOutputDeduper()
	}

	transcript := make([]Message, len(req.Messages), len(req.Messages)+8)
	copy(transcript, req.Messages)

	// Elapsed-time notices ride each request only; nil config, nil tracker, no notice.
	elapsed := newElapsedTracker(cfg.ElapsedTime)

	res := &Result{}
	// The queue belongs to this run; closing it tells a racing producer its message missed.
	defer func() {
		if left := cfg.Messages.Close(); len(left) > 0 && res != nil {
			res.Undelivered = left
		}
	}()
	// Stuck detection (see StuckNudgeAt): the previous turn's tool-call fingerprint.
	lastBatch := ""
	repeats := 0
	// lastComp is the newest completion, whose PromptTokens decide compaction.
	var lastComp *Completion
	// currentAssistantID tracks the in-flight assistant turn so panic recovery can finalize it.
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
	// moreTurnsAllowed reports whether a host cap still permits a turn after this.
	moreTurnsAllowed := func(turn int) bool {
		return cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1
	}
	finish := func(final Message) (*Result, error) {
		transcript = append(transcript, final)
		res.Messages = transcript
		res.Final = final
		return res, nil
	}
	// answer ends turn on final: the stop hook is asked, the host's row is
	// finalized, and a queued message takes another turn instead (nil result).
	answer := func(turn int, comp *Completion, id MessageID, final Message) *Result {
		cfg.Events.emitStop(StopEvent{Turn: turn + 1, Comp: comp})
		finalizeAssistant(FinalizeAssistantEvent{ID: id, Msg: final, Status: "complete"})
		// Something is queued: either the stop hook above put it there, or
		// it arrived while the model was working and the answer could not
		// have accounted for it. Keep the answer and take another turn,
		// which drains the queue at the top.
		if cfg.Messages.Pending() && moreTurnsAllowed(turn) {
			transcript = append(transcript, final)
			return nil
		}
		r, _ := finish(final)
		return r
	}

	for turn := 0; ; turn++ {
		if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns {
			break
		}
		// Turn boundary ONLY, before the drain: mid-turn compaction dropped the
		// turn's tool calls. Depth: USAGE.md, auto-compaction.
		if next, ok := compactHere(ctx, &cfg, req, transcript, lastComp, res, deduper); ok {
			transcript = next
			lastComp = nil
		}
		// Sub-agent delivery: deliver ready reports at the top of the turn so the model sees them.
		if turn > 0 && cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			reports := cfg.Subagents.Take()
			if len(reports) > 0 {
				delivery := Message{
					Role:    RoleUser,
					Kind:    SubagentReportKind,
					Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
				}
				if cfg.Messages != nil {
					cfg.Messages.Queue(SystemMessage{delivery})
				} else {
					cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
					transcript = append(transcript, delivery)
				}
			}
		}
		// Drain queued messages: system, then user.
		for _, msg := range cfg.Messages.Drain() {
			cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: msg})
			transcript = append(transcript, msg)
		}
		// Resource watch: poll at the turn boundary; a non-empty poll is delivered as a notice.
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
		if cfg.TurnHook != nil {
			cfg.TurnHook(turn + 1)
		}
		turnTools := advertised
		if cfg.MaxTurns > 0 && turn == cfg.MaxTurns-1 {
			turnTools = nil
		}
		// Ask the host to mint the durable row for this turn; "" means not persisting.
		parentID := ""
		if n := len(transcript); n > 0 {
			parentID = transcript[n-1].ID
		}
		assistantID, aerr := cfg.Events.emitAssistantMessage(AssistantMessageEvent{ParentID: MessageID(parentID)})
		currentAssistantID = assistantID
		if aerr != nil {
			// The host failed to announce the row (e.g. the SSE sink died); finalize as error.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: "error"})
			res.Messages = transcript
			return res, aerr
		}
		comp, err := runModelCall(ctx, &cfg, req, turn+1, transcript, turnTools, res, elapsed)
		if err != nil {
			// A cancelled/timed-out call is never an "error"; a stopped stream finalizes cancelled.
			status := "cancelled"
			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				status = "error"
			}
			if comp != nil {
				// Mid-stream break: keep partial content, drop tool calls (never executed; replay 400s).
				partial := comp.Message
				partial.ToolCalls = nil
				if assistantID != "" {
					partial.ID = string(assistantID)
				}
				transcript = append(transcript, partial)
				res.Messages = transcript
				res.Final = partial
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: partial, Status: status})
				return res, err
			}
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: status})
			res.Messages = transcript
			return res, err
		}
		assistant := comp.Message
		if assistantID != "" {
			assistant.ID = string(assistantID)
		}
		calls := assistant.ToolCalls
		lastComp = comp

		// Keep looping while the model requests tools: replay the tool-call message and results.
		if len(calls) > 0 && (cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1) {
			// A batch identical to the previous turn's makes no progress; nudge, then end the run.
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
			// Finalize the assistant as complete with its tool calls before executing them.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: assistant, Status: "complete"})
			// abortBatch ends the run mid-batch (approval/callback error); clears pending batch.
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

			// Dispatch: read-only calls run concurrently; mutating calls run sequentially as barriers.
			for i, asked := range calls {
				// The hook sees a copy; downstream uses whatever the hook left.
				call := asked
				if cberr := cfg.Events.emitToolCall(ToolCallEvent{Call: &call}); cberr != nil {
					// Wait for in-flight read-only calls so none outlive the batch, then clear it.
					wg.Wait()
					return abortBatch(cberr)
				}
				asks = append(asks, call)

				tool, known := cfg.Tools.Find(call.Name)
				readonly := known && tool.Decl().Readonly

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
				// Mutating: barrier -- wait for in-flight read-only calls to finish.
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

			// Wait for all goroutines before reading results or aborting, so none outlive the batch.
			wg.Wait()
			if firstErr != nil {
				return abortBatch(firstErr)
			}

			// Record every result in call order, keeping the transcript deterministic.
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
				// The id answered is the MODEL's, never a rewritten; a mismatch is an orphan.
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
					ToolCallID:        call.ID,
					ParentAssistantID: assistantID,
					Content:           content,
					Parts:             parts,
					IsError:           results[i].IsError,
				}); terr != nil {
					return abortBatch(terr)
				}
			}
			if repeats == StuckNudgeAt {
				transcript = append(transcript, Message{Role: RoleUser, Content: stuckNudgeInstruction})
			}
			continue
		}

		// The model asked for no tools, but sub-agents may still be out; deliver what has landed.
		if cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns-1 {
				// Capped final turn: deliver what arrived, declare remaining subagents lost.
				reports := cfg.Subagents.Take()
				lost := cfg.Subagents.CancelRemaining()
				if len(reports) > 0 || lost > 0 {
					// The answer is recorded, then the delivery trails it: the
					// cap leaves no turn to read it, but the host persists it.
					final := assistant
					final.ToolCalls = nil
					if strings.TrimSpace(final.Content) == "" {
						final.Content = fallbackOutput(assistant)
					}
					finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
					transcript = append(transcript, final)
					delivery := Message{
						Role:    RoleUser,
						Kind:    SubagentReportKind,
						Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), lost),
					}
					if cfg.Messages != nil {
						cfg.Messages.Queue(SystemMessage{delivery})
					} else {
						cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
						transcript = append(transcript, delivery)
					}
					for _, msg := range cfg.Messages.Drain() {
						cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: msg})
						transcript = append(transcript, msg)
					}
					res.Messages = transcript
					res.Final = final
					return res, nil
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
					if cfg.Messages != nil {
						cfg.Messages.Queue(SystemMessage{Message{
							Role:    RoleUser,
							Kind:    SubagentReportKind,
							Content: FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0),
						}})
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
			if r := answer(turn, comp, assistantID, final); r != nil {
				return r, nil
			}
			continue
		}

		// The model stopped without writing an answer, and something is
		// queued. Deliver that instead of spending a wrap-up call on a turn
		// with nothing to wrap up: the queued message is newer than anything
		// the model could synthesize here, and the next turn drains it.
		if cfg.Messages.Pending() && moreTurnsAllowed(turn) {
			stalled := assistant
			stalled.ToolCalls = nil
			stalled.Content = fallbackOutput(assistant)
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: stalled, Status: "complete"})
			transcript = append(transcript, stalled)
			continue
		}

		// The model stopped without writing an answer -- it produced only
		// reasoning. When tools were in
		// play (so it may already have gathered useful results), make
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
			comp2, err2 := runModelCall(ctx, &cfg, req, turn+2, wrapMsgs, nil, res, elapsed)
			if err2 == nil {
				if s := strings.TrimSpace(comp2.Message.Content); s != "" {
					// The wrap-up's answer is this turn's answer: it lands on the
					// row the host minted above, and is a stop boundary like any.
					final := comp2.Message
					final.ToolCalls = nil
					if assistantID != "" {
						final.ID = string(assistantID)
					}
					transcript = append(transcript, wrapMsg)
					if r := answer(turn, comp2, assistantID, final); r != nil {
						return r, nil
					}
					continue
				}
			}
			// The wrap-up failed or still produced nothing: fall through to
			// the last-resort fallback (the error, if any, is swallowed like
			// the source's synthesize step).
		}

		// Last resort: surface the reasoning (a thinking model's only output), else a placeholder.
		final := assistant
		final.ToolCalls = nil
		if strings.TrimSpace(final.Content) == "" {
			final.Content = fallbackOutput(assistant)
		}
		if r := answer(turn, comp, assistantID, final); r != nil {
			return r, nil
		}
		continue
	}

	// Cap broke the loop; deliver any pending sub-agent reports before finishing.
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
