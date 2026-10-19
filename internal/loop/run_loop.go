package loop

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// runOn is a session's turn: call the provider, execute tools, feed results back, stop on an answer.
// It runs from s's leaf. req is the request template; its Messages are ignored.
func runOn(ctx context.Context, cfg Config, req Request, s *session) (*Result, error) {
	if cfg.Provider == nil {
		return nil, badRequestErr("agentic: Config.Provider is required")
	}
	if cfg.Events == nil {
		cfg.Events = &Events{}
	}
	advertised := cfg.Tools.Decls()
	if err := checkWindowConfig(cfg); err != nil {
		return nil, err
	}
	if err := resolveContextWindow(ctx, &cfg, req); err != nil {
		return nil, err
	}

	// Output dedup: deduper for the whole run collapses unchanged read-only results.
	var deduper *OutputDeduper
	if !cfg.DisableOutputDedup {
		deduper = NewOutputDeduper()
	}
	s.checkMonotonic(0)

	// Elapsed-time notices ride each request only; nil config, nil tracker, no notice.
	elapsed := newElapsedTracker(cfg.ElapsedTime)

	res := &Result{session: s}
	// The queue belongs to this run; closing it tells a racing producer its message missed.
	defer func() {
		if left := cfg.Messages.Close(); len(left) > 0 && res != nil {
			res.Undelivered = left
		}
	}()
	// Stuck detection (see StuckNudgeAt): the turn's tool-call fingerprint.
	lastBatch := ""
	repeats := 0
	// lastComp is the newest completion, whose PromptTokens decide compaction.
	var lastComp *Completion
	// states checks the provider side: that each turn's output reached the next call.
	var states stateCheck
	losses := lossTracker{}
	// lastSent is how many window messages the newest call carried.
	lastSent := 0
	// servedAsked is the served model name the model list was last asked about, so an alias is resolved once.
	var servedAsked string
	// overflowTurn is the turn whose call overflowed the window and was compacted once to retry it.
	overflowTurn := -1
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
	// done ends the run on final, the flat answer the host finalized.
	done := func(final Message, err error) (*Result, error) {
		res.Messages = s.messages()
		res.Final = final
		return res, err
	}
	// answer ends turn on final: the stop hook is asked, the host's row is
	// finalized, and a queued message takes another turn instead (nil result).
	answer := func(turn int, comp *Completion, id MessageID, final Message) *Result {
		cfg.Events.emitStop(StopEvent{Turn: turn + 1, Comp: comp})
		finalizeAssistant(FinalizeAssistantEvent{ID: id, Msg: final, Status: "complete"})
		if cfg.Messages.Pending() && moreTurnsAllowed(turn) {
			return nil
		}
		r, _ := done(final, nil)
		return r
	}
	// deliver adds a sub-agent delivery: through the queue when there is one, else at once.
	deliver := func(content string, announce bool) {
		delivery := Message{Role: RoleUser, Kind: SubagentReportKind, Content: content}
		if cfg.Messages != nil {
			cfg.Messages.Queue(SystemMessage{delivery})
			return
		}
		if announce {
			cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: delivery})
		}
		s.appendFlat(delivery)
	}
	// drain adds every queued message, announcing each.
	drain := func() {
		for _, msg := range cfg.Messages.Drain() {
			cfg.Events.emitSystemMessage(SystemMessageEvent{Msg: msg})
			s.appendFlat(msg)
		}
	}

	for turn := 0; ; turn++ {
		if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns {
			break
		}
		// Turn boundary ONLY, before the drain: mid-turn compaction dropped the
		// turn's tool calls. Depth: USAGE.md, auto-compaction.
		if compactHere(ctx, &cfg, req, s, lastComp, res, deduper) {
			lastComp = nil
			states = stateCheck{}
			lastSent = 0
		}
		// Sub-agent delivery: deliver ready reports at the top of the turn so the model sees them.
		if turn > 0 && cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			if reports := cfg.Subagents.Take(); len(reports) > 0 {
				deliver(FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0), true)
			}
		}
		drain()
		// Resource watch: poll at the turn boundary; a non-empty poll is delivered as a notice.
		if cfg.ResourceWatcher != nil {
			poll, perr := cfg.ResourceWatcher.Poll(ctx)
			if perr != nil {
				if ctxErr := ctx.Err(); ctxErr != nil {
					res.Messages = s.messages()
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
				s.appendFlat(Message{Role: RoleUser, ID: string(hostID), Kind: ResourceNoticeKind, Content: notice})
			}
		}
		if compacted, err := capToWindow(ctx, &cfg, req, s, lastComp, lastSent, res, deduper); err != nil {
			res.Messages = s.messages()
			return res, err
		} else if compacted {
			lastComp = nil
			states = stateCheck{}
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
		if msgs := s.messages(); len(msgs) > 0 {
			parentID = msgs[len(msgs)-1].ID
		}
		assistantID, aerr := cfg.Events.emitAssistantMessage(AssistantMessageEvent{ParentID: MessageID(parentID)})
		currentAssistantID = assistantID
		if aerr != nil {
			// The host failed to announce the row (e.g. the SSE sink died); finalize as error.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: "error"})
			res.Messages = s.messages()
			return res, aerr
		}
		msgs := s.messages()
		lastSent = len(msgs)
		mc, err := runModelCall(ctx, &cfg, s, req, turn+1, msgs, turnTools, res, elapsed, string(assistantID))
		comp := mc.comp
		if err != nil {
			// A cancelled/timed-out call is never an "error"; a stopped stream finalizes cancelled.
			status := "cancelled"
			if !errors.Is(err, context.Canceled) && !errors.Is(err, context.DeadlineExceeded) {
				status = "error"
			}
			// A request larger than the window compacts and asks again, once per turn. Without this
			// a stored transcript past the window fails every later run at its first call.
			if IsContextOverflow(err) && req.AutoCompact > 0 && comp == nil && len(mc.nodes) == 0 && overflowTurn != turn {
				overflowTurn = turn
				if compactNow(ctx, &cfg, req, s, res, deduper) {
					finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: "error"})
					lastComp = nil
					states = stateCheck{}
					lastSent = 0
					turn--
					continue
				}
			}
			// What streamed stays. A tool call that will never run gets an omitted result.
			s.omitCalls(mc.nodes, "the call ended before this tool ran: "+err.Error())
			if comp != nil || len(mc.nodes) > 0 {
				partial := withoutCalls(mc.message(), assistantID)
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: partial, Status: status})
				return done(partial, err)
			}
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Status: status})
			res.Messages = s.messages()
			return res, err
		}
		assistant := mc.message()
		if assistantID != "" {
			assistant.ID = string(assistantID)
		}
		// Both checks report and never end the run: a run that continues can still recover.
		if loss := states.observe(turn+1, msgs, comp); loss != nil {
			losses.record(&cfg, res, *loss, cfg.Provider)
		}
		if loss := s.checkMonotonic(turn + 1); loss != nil {
			losses.record(&cfg, res, *loss, cfg.Provider)
		}
		calls := assistant.ToolCalls
		lastComp = comp
		if werr := resolveServedWindow(ctx, &cfg, req, comp, &servedAsked); werr != nil {
			s.omitCalls(mc.nodes, "the run stopped before this tool ran: "+werr.Error())
			stopped := withoutCalls(assistant, assistantID)
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: stopped, Status: "error"})
			return done(stopped, werr)
		}

		// Keep looping while the model requests tools: replay the tool-call message and results.
		if len(calls) > 0 && (cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1) {
			// A batch identical to the turn's makes no progress; nudge, then end the run.
			if fp := batchFingerprint(calls); fp == lastBatch {
				repeats++
			} else {
				lastBatch, repeats = fp, 1
			}
			if repeats >= StuckFailAt {
				s.omitCalls(mc.nodes, "the run stopped: the same tool calls repeated "+fmt.Sprint(repeats)+" times")
				stuck := withoutCalls(assistant, assistantID)
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: stuck, Status: "complete"})
				return done(stuck, fmt.Errorf("%w: %d identical turns in a row", ErrStuck, repeats))
			}

			// The host may replace the batch. The turn records the replacement, so the model's next request asks about what ran.
			expanded, berr := cfg.Events.emitToolBatch(calls)
			if berr != nil {
				s.omitCalls(mc.nodes, "the batch was cancelled before it ran: "+berr.Error())
				cleared := withoutCalls(assistant, assistantID)
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: cleared, Status: "cancelled"})
				return done(cleared, berr)
			}
			callNodes := toolCalls(mc.nodes)
			if !sameCalls(calls, expanded) {
				callNodes = s.replaceCalls(mc.nodes, expanded)
			}
			calls = expanded
			assistant.ToolCalls = calls

			// Finalize the assistant as complete with its tool calls before executing them.
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: assistant, Status: "complete"})
			// abortBatch ends the run mid-batch (approval/callback error). Every call no result
			// answered gets an omitted one, so the tree keeps the calls and stays sendable.
			abortBatch := func(cause error) (*Result, error) {
				pending := make([]IMessage, len(callNodes))
				for i, tc := range callNodes {
					pending[i] = tc
				}
				s.omitCalls(pending, "the batch was cancelled before this tool ran: "+cause.Error())
				cleared := withoutCalls(assistant, assistantID)
				finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: cleared, Status: "cancelled"})
				return done(cleared, cause)
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
				s.toolResult(callNodes[i], content, results[i].IsError, "")
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
				s.appendFlat(Message{Role: RoleUser, Kind: StopNudgeKind, Content: stuckNudgeInstruction})
			}
			continue
		}
		// The loop is ending: no call in this turn will run.
		s.omitCalls(mc.nodes, "the run is ending, so this tool call was not run")

		// The model asked for no tools, but sub-agents may still be out; deliver what has landed.
		if cfg.Subagents != nil && cfg.Subagents.Pending() > 0 {
			if cfg.MaxTurns > 0 && turn >= cfg.MaxTurns-1 {
				// Capped final turn: deliver what arrived, declare remaining subagents lost.
				reports := cfg.Subagents.Take()
				lost := cfg.Subagents.CancelRemaining()
				if len(reports) > 0 || lost > 0 {
					// The answer is recorded, then the delivery trails it for the host.
					final := withoutCalls(assistant, assistantID)
					if strings.TrimSpace(final.Content) == "" {
						final.Content = fallbackOutput(assistant)
						s.addText(mc.rec, final.Content)
					}
					finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "complete"})
					deliver(FormatSubagentDelivery(reports, cfg.Subagents.Running(), lost), true)
					drain()
					return done(final, nil)
				}
			} else {
				reports, cerr := cfg.Subagents.Collect(ctx)
				if cerr != nil {
					// Cancellation while waiting: finalize as cancelled, end gracefully.
					if errors.Is(cerr, context.Canceled) || errors.Is(cerr, context.DeadlineExceeded) {
						final := withoutCalls(assistant, assistantID)
						if strings.TrimSpace(final.Content) == "" {
							final.Content = fallbackOutput(assistant)
							s.addText(mc.rec, final.Content)
						}
						finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: final, Status: "cancelled"})
						return done(final, nil)
					}
					res.Messages = s.messages()
					return res, cerr
				}
				if len(reports) > 0 {
					if strings.TrimSpace(assistant.Content) != "" {
						finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: withoutCalls(assistant, assistantID), Status: "complete"})
					}
					deliver(FormatSubagentDelivery(reports, cfg.Subagents.Running(), 0), true)
					continue
				}
			}
		}

		if strings.TrimSpace(assistant.Content) != "" {
			if r := answer(turn, comp, assistantID, withoutCalls(assistant, assistantID)); r != nil {
				return r, nil
			}
			continue
		}

		// The model stopped without writing an answer, and something is queued. Deliver that
		// instead of spending a wrap-up call. The queued message is newer than anything the model
		// could synthesize here, and the next turn drains it.
		if cfg.Messages.Pending() && moreTurnsAllowed(turn) {
			stalled := withoutCalls(assistant, assistantID)
			stalled.Content = fallbackOutput(assistant)
			s.addText(mc.rec, stalled.Content)
			finalizeAssistant(FinalizeAssistantEvent{ID: assistantID, Msg: stalled, Status: "complete"})
			continue
		}

		// The model stopped without writing an answer -- it produced only reasoning. When tools
		// were in play, a final tool-less request asks it to write its answer from what it has.
		// The reasoning stays in the tree and in the request.
		if len(cfg.Tools) > 0 && (cfg.MaxTurns <= 0 || turn < cfg.MaxTurns-1) {
			wrap := Message{Role: RoleUser, Kind: WrapUpKind, Content: wrapUpInstruction}
			s.appendFlat(wrap)
			mc2, err2 := runModelCall(ctx, &cfg, s, req, turn+2, s.messages(), nil, res, elapsed, "")
			if err2 == nil {
				if strings.TrimSpace(mc2.message().Content) != "" {
					// The wrap-up's answer is this turn's answer, on the row minted above.
					s.omitCalls(mc2.nodes, "the run is ending, so this tool call was not run")
					final := withoutCalls(mc2.message(), assistantID)
					if r := answer(turn, mc2.comp, assistantID, final); r != nil {
						return r, nil
					}
					continue
				}
			}
			s.omitCalls(mc2.nodes, "the run is ending, so this tool call was not run")
		}

		// Last resort: surface the reasoning (a thinking model's only output), else a placeholder.
		final := withoutCalls(assistant, assistantID)
		if strings.TrimSpace(final.Content) == "" {
			final.Content = fallbackOutput(assistant)
			s.addText(mc.rec, final.Content)
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
			s.appendFlat(delivery)
		}
	}
	final := Message{Role: RoleAssistant, Content: noOutputPlaceholder}
	s.appendFlat(final)
	return done(final, nil)
}

// withoutCalls is the turn's flat message with its tool calls cleared, on the host's row id.
func withoutCalls(m Message, id MessageID) Message {
	out := m
	out.ToolCalls = nil
	var parts []commonai.Part
	for _, p := range out.Parts {
		if p.Kind() != commonai.PartKindToolCall {
			parts = append(parts, p)
		}
	}
	out.Parts = parts
	if id != "" {
		out.ID = string(id)
	}
	return out
}

// sameCalls reports whether batches are the same calls.
func sameCalls(a, b []ToolCall) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
