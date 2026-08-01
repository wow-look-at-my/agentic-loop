package agentic

import (
	"context"
	"fmt"
	"strings"
)

// DefaultMaxTurns is the model-call cap applied when Config.MaxTurns is not
// positive.
const DefaultMaxTurns = 10

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

// Events are the loop's callbacks: the embedded StreamEvents fire during each
// model call, OnToolCall fires before each requested tool call is handled,
// and OnToolResult fires with its recorded result (executed, denied, or a
// teaching error). All optional.
//
// Like the stream callbacks, OnToolCall and OnToolResult may return a non-nil
// error to abort the run: the turn is finalized the way a cancellation is —
// the pending batch is cleared so the transcript stays replayable with no
// orphan tool calls — and the partial Result is returned together with that
// error (errors.Is against the caller's sentinel holds; the error is never
// classified transient).
type Events struct {
	StreamEvents
	OnToolCall   func(ToolCall) error
	OnToolResult func(ToolCall, ToolResult) error
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

// Config wires one Run: the Provider to call, the ToolExecutor whose tools
// are advertised and executed (nil runs tool-less), the Approver consulted
// for calls the executor flags via NeedsApproval (nil denies gated calls with
// DeniedMessage), the turn cap (<= 0 means DefaultMaxTurns), the retry policy
// for model calls (nil means DefaultRetry), and the event callbacks.
type Config struct {
	Provider Provider
	Tools    ToolExecutor
	Approver Approver
	MaxTurns int
	Retry    *RetryPolicy
	Events   Events

	// turnHook, when non-nil, is invoked with the 1-based turn number as each
	// numbered turn begins (the stall-fallback wrap-up call is not a numbered
	// turn). It is unexported: package-internal machinery — the subagent
	// executor's live activity telemetry — not public API.
	turnHook func(turn int)
}

// Result is the outcome of a Run. Messages is the input transcript plus
// everything the loop appended (assistant turns, tool results, and the final
// message). Usages holds one entry per model call IN ORDER — deliberately not
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
// Each turn advertises cfg.Tools.Tools(); req.Tools is ignored and
// overwritten (nil cfg.Tools advertises no tools). On the final permitted
// turn tools are WITHHELD so the model must answer rather than request
// another never-executed call. Tool failures never abort the loop: an
// Execute error becomes a recoverable "tool execution failed: ..." error
// result, a call the model hallucinated with no executor configured gets an
// "unknown tool: ..." error result, and a denied approval records
// DeniedMessage — in every case the loop continues so the model can react.
//
// An Approver.Ask error (the decision never arrived) ends the run: the
// current assistant message keeps its content and reasoning but its ToolCalls
// are cleared and the batch's already-appended tool results are dropped, so
// the returned transcript stays replayable with no orphan tool calls; the
// partial Result is returned alongside the error. An error returned by
// OnToolCall or OnToolResult ends the run the same way, and a stream
// callback error surfaces through the model call as a partial completion
// plus the callback's error — in every case the partial Result rides
// alongside the error and the transcript carries no orphan tool calls.
//
// Model calls are retried per cfg.Retry, but only when the failed attempt
// streamed nothing; once data arrived the partial assistant message is
// finalized into the transcript (tool calls cleared) and the partial Result
// is returned alongside the error. Whenever Run returns an error together
// with a non-nil Result, the Result carries the transcript accumulated so
// far.
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
	maxTurns := cfg.MaxTurns
	if maxTurns <= 0 {
		maxTurns = DefaultMaxTurns
	}
	retry := DefaultRetry
	if cfg.Retry != nil {
		retry = *cfg.Retry
	}

	var advertised []Tool
	if cfg.Tools != nil {
		advertised = cfg.Tools.Tools()
	}

	transcript := make([]Message, len(req.Messages), len(req.Messages)+8)
	copy(transcript, req.Messages)

	res := &Result{}
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
		// On the last permitted turn, withhold tools so the model must answer
		// rather than request another (never-executed) tool call.
		lastTurn := turn == maxTurns-1
		var turnTools []Tool
		if !lastTurn {
			turnTools = advertised
		}

		comp, err := runModelCall(ctx, &cfg, retry, req, transcript, turnTools, res)
		if err != nil {
			if comp != nil {
				// Mid-stream break/cancel: keep the partial content, reasoning
				// and usage, but drop any assembled tool calls — they were
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
		if len(calls) > 0 && !lastTurn {
			transcript = append(transcript, assistant)
			aIdx := len(transcript) - 1
			// abortBatch ends the run mid-batch — an approval decision that
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
				transcript = append(transcript, Message{
					Role:        RoleTool,
					Content:     result.Content,
					ToolCallID:  call.ID,
					ToolIsError: result.IsError,
				})
			}
			continue
		}

		// The loop is ending. A real textual answer is the result. Dangling
		// tool calls (a capped last turn) are cleared — they will never
		// execute, and a replayable transcript must not carry orphans.
		if strings.TrimSpace(assistant.Content) != "" {
			final := assistant
			final.ToolCalls = nil
			return finish(final)
		}

		// The model stopped without writing an answer — it produced only
		// reasoning, or hit the turn cap mid-research. When tools were in
		// play (so it may already have gathered useful results), make one
		// final tool-less request that forces it to synthesize an answer from
		// what it has. The stalling turn's assistant message is deliberately
		// NOT in the transcript (it is only appended on the tool-execution
		// branch), so the wrap-up request can't be rejected for an unanswered
		// tool call.
		if cfg.Tools != nil {
			wrapMsg := Message{Role: RoleUser, Content: wrapUpInstruction}
			wrapMsgs := make([]Message, len(transcript), len(transcript)+1)
			copy(wrapMsgs, transcript)
			wrapMsgs = append(wrapMsgs, wrapMsg)
			comp2, err2 := runModelCall(ctx, &cfg, retry, req, wrapMsgs, nil, res)
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

// runModelCall executes one model call with retry (see retryComplete for the
// nothing-streamed guard); the retry loop counts as ONE turn. Every call that
// produced a completion — success or partial — appends its usage to the
// result.
func runModelCall(
	ctx context.Context, cfg *Config, retry RetryPolicy,
	req Request, msgs []Message, tools []Tool, res *Result,
) (*Completion, error) {
	r := req
	r.Messages = msgs
	r.Tools = tools

	comp, err := retryComplete(ctx, cfg.Provider, retry, r, &cfg.Events.StreamEvents)
	res.Turns++
	if comp != nil {
		res.Usages = append(res.Usages, comp.Usage)
	}
	return comp, err
}

// resolveCall produces the recorded ToolResult for one requested call:
// executed, denied, or a teaching error. The returned error is non-nil ONLY
// when an approval decision never arrived (Approver.Ask failed), which ends
// the run.
func resolveCall(ctx context.Context, cfg *Config, call ToolCall) (ToolResult, error) {
	if cfg.Tools == nil {
		// No executor is configured; the model hallucinated a tool. Teach it
		// rather than aborting.
		return ToolResult{Content: "unknown tool: " + call.Name, IsError: true}, nil
	}
	if cfg.Tools.NeedsApproval(call.Name) {
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
	result, exErr := cfg.Tools.Execute(ctx, call)
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
