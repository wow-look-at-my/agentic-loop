package subagent

import (
	"context"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
	"strconv"
	"strings"
)

// Running one sub-agent: the launch (asynchronous when a registry is
// configured), the nested agentic.Run, and the toolset it is given.

// Execute runs one sub-agent. Every misuse — a bad share_context selection,
// an allowed_tools name that resolves to nothing — is a recoverable error
// tool result that teaches the valid shape, never a Go error.
func (e *subagentTool) Execute(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	// A body nobody can parse is the one failure still answered synchronously
	// even in async mode: there is nothing to launch, and the model should
	// learn that from the call it just made.
	var in subagentArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return agentic.ToolResult{Content: "invalid run_subagent arguments: " + err.Error(), IsError: true}, nil
	}
	if e.cfg.Runs == nil {
		return e.run(ctx, in), nil
	}

	// Asynchronous: register the run, hand back a receipt, and let the
	// goroutine report through the registry. Every other failure -- an
	// unconfigured model, a misused argument -- reaches the model as that
	// report, one delivery later, rather than as this call's result.
	callID := agentic.ToolCallID(ctx)
	if callID == "" {
		callID = e.cfg.Runs.nextID()
	}
	e.cfg.Runs.Launch(callID, in.Description, in.Prompt)
	go e.launched(ctx, callID, in)
	return agentic.ToolResult{Content: SubagentLaunchReceipt(in.Description)}, nil
}

// launched runs one asynchronously started sub-agent to completion and records
// its outcome. It never returns anything to its caller -- the registry is the
// only path back -- so every exit has to record something: a lost report would
// leave the loop waiting on a promise nothing will keep. That includes a
// panic, which on this goroutine would otherwise take the whole process down
// rather than one turn.
func (e *subagentTool) launched(ctx context.Context, callID string, in subagentArgs) {
	defer func() {
		if rec := recover(); rec != nil {
			e.cfg.Runs.Complete(callID, fmt.Sprintf("the sub-agent crashed: %v", rec), true)
		}
	}()
	release, err := e.cfg.Gate.Acquire(ctx)
	if err != nil {
		e.cfg.Runs.Complete(callID, "the sub-agent was cancelled while waiting for a free slot: "+err.Error(), true)
		return
	}
	defer release()
	e.cfg.Runs.MarkRunning(callID)

	res := e.runGated(agentic.WithToolCallID(ctx, callID), in)
	e.cfg.Runs.Complete(callID, res.Content, res.IsError)
}

// run executes one sub-agent to completion. Every misuse — a bad
// share_context selection, an allowed_tools name that resolves to nothing — is
// a recoverable error tool result that teaches the valid shape.
func (e *subagentTool) run(ctx context.Context, in subagentArgs) agentic.ToolResult {
	// Serialize per the shared agentic.Gate. Acquisition is cancellable so a caller
	// disconnect (or a stopped turn) while waiting returns promptly. The
	// asynchronous path takes the slot itself, BEFORE marking the run running,
	// so a queued launch reads as queued rather than as an unexplained delay.
	release, err := e.cfg.Gate.Acquire(ctx)
	if err != nil {
		return agentic.ToolResult{Content: "run_subagent was cancelled before it could start: " + err.Error(), IsError: true}
	}
	defer release()
	return e.runGated(ctx, in)
}

// runGated executes one sub-agent with its concurrency slot already held.
func (e *subagentTool) runGated(ctx context.Context, in subagentArgs) agentic.ToolResult {
	if e.cfg.Provider == nil || e.cfg.Model == "" {
		return agentic.ToolResult{Content: "run_subagent is unavailable: no model is configured for the sub-agent", IsError: true}
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return agentic.ToolResult{Content: "run_subagent requires a non-empty prompt describing the task", IsError: true}
	}

	// Pick the sub-agent's toolset. Default: the read-only subset only. When
	// the orchestrator pins it with allowed_tools, select that subset from the
	// FULL toolset instead — so an explicitly-named non-read-only tool IS
	// granted. An unresolved name is a recoverable tool error (it lists the
	// valid tools) so the model can correct the call.
	subTools := e.readonly
	granted := false
	if len(in.AllowedTools) > 0 {
		keep, terr := resolveAllowedTools(grantableToolNames(e.grantableTools()), in.AllowedTools)
		if terr != "" {
			return agentic.ToolResult{Content: terr, IsError: true}
		}
		subTools, granted = e.cfg.Tools.Subset(keep), true
	}

	// Build the optional parent-context block the orchestrator asked to share
	// and fold it into the task. A bad selection (e.g. last_n without a count)
	// is a recoverable tool error so the model can correct the call.
	block, errMsg := e.buildContextBlock(ctx, in)
	if errMsg != "" {
		return agentic.ToolResult{Content: errMsg, IsError: true}
	}
	task := composeSubagentTask(block, in.Prompt)

	res, runErr := agentic.Run(ctx, e.runConfig(agentic.ToolCallID(ctx), subTools, granted), agentic.Request{
		Model:     e.cfg.Model,
		System:    e.system,
		Messages:  []agentic.Message{{Role: agentic.RoleUser, Content: task}},
		MaxTokens: e.cfg.MaxTokens,
		Extra:     e.cfg.Extra,
	})
	if runErr != nil {
		return agentic.ToolResult{Content: "sub-agent failed: " + runErr.Error(), IsError: true}
	}
	// Not just the final text: a run that ended by emitting a tool-call
	// envelope as TEXT never answered, and passing that up as findings is the
	// one failure the orchestrator cannot detect for itself.
	return subagentReport(res.Final.Content)
}

// unavailableTool is what a sub-agent is told when it names a tool this run
// does not offer. A bare "unknown tool" would be misleading: the name usually
// IS a real tool of the parent, withheld either because it modifies state or
// because the orchestrator pinned the run to a smaller set -- and the
// sub-agent can only stop asking for it if it is told which.
func (e *subagentTool) unavailableTool(granted bool) func(string) string {
	if granted {
		return func(name string) string { return "tool not in the sub-agent's allowed set: " + name }
	}
	return func(name string) string {
		return "tool not available to subagent (read-only tools only): " + name
	}
}

// SubagentLaunchReceipt is what the model gets back in place of an answer
// when run_subagent is asynchronous. It has to carry its own weight: the model
// is about to choose what to do with a turn it did not expect to have, and a
// bare "started" invites it to either idle-poll or invent the findings.
func SubagentLaunchReceipt(description string) string {
	what := strings.TrimSpace(description)
	if what == "" {
		what = "(no description given)"
	}
	return "Sub-agent launched: " + what + ". It is running in the background and this call is already finished. " +
		"Its report will be delivered to you automatically as a new message in this conversation when it lands -- " +
		"there is nothing to poll and no way to wait for it here. Launch more sub-agents now if other areas need " +
		"covering, get on with work that does not depend on this one, or, if nothing is left until the report " +
		"arrives, say so in one short line and end your turn."
}

// runConfig assembles the nested agentic.Run's Config: the sub-agent's toolset, the
// approve-everything agentic.Approver (the explicit allowed_tools grant is itself the
// authorization — the source loop never consulted NeedsApproval), and, when
// the host listens, the activity telemetry hooks stamped with the parent
// call's ID.
func (e *subagentTool) runConfig(callID string, subTools agentic.Tools, granted bool) agentic.Config {
	cfg := agentic.Config{
		Provider:    e.cfg.Provider,
		Tools:       subTools,
		Approver:    approveAll{},
		UnknownTool: e.unavailableTool(granted),
	}
	act := e.cfg.OnActivity
	if act == nil {
		return cfg
	}
	cfg.TurnHook = func(turn int) {
		act(SubagentActivity{CallID: callID, Kind: SubagentActivityTurn, Turn: turn})
	}
	cfg.Events = agentic.Events{
		OnToolCall: func(c agentic.ToolCall) error {
			act(SubagentActivity{
				CallID:  callID,
				Kind:    SubagentActivityToolCall,
				Tool:    c.Name,
				Detail:  subagentPreview(c.Arguments),
				Content: c.Arguments,
			})
			return nil
		},
		OnToolResult: func(c agentic.ToolCall, r agentic.ToolResult) error {
			act(SubagentActivity{
				CallID:  callID,
				Kind:    SubagentActivityToolResult,
				Tool:    c.Name,
				Detail:  subagentPreview(r.Content),
				Content: r.Content,
				IsError: r.IsError,
			})
			return nil
		},
		// What the sub-agent itself said each turn. Without this a host can
		// show every tool a sub-agent touched and still not show a word of its
		// own reasoning or working notes — the run reads as a list of file
		// accesses with a report appearing from nowhere at the end.
		OnTurnEnd: func(turn int, comp *agentic.Completion, _ error) error {
			if comp == nil {
				return nil
			}
			if think := thinkingText(comp.Message.Thinking); think != "" {
				act(SubagentActivity{
					CallID:  callID,
					Kind:    SubagentActivityThinking,
					Turn:    turn,
					Detail:  subagentPreview(think),
					Content: think,
				})
			}
			if text := comp.Message.Content; strings.TrimSpace(text) != "" {
				act(SubagentActivity{
					CallID:  callID,
					Kind:    SubagentActivityText,
					Turn:    turn,
					Detail:  subagentPreview(text),
					Content: text,
				})
			}
			return nil
		},
	}
	return cfg
}

// thinkingText joins a completion's reasoning blocks into one string.
func thinkingText(blocks []agentic.ThinkingBlock) string {
	if len(blocks) == 0 {
		return ""
	}
	parts := make([]string, 0, len(blocks))
	for _, b := range blocks {
		if b.Text != "" {
			parts = append(parts, b.Text)
		}
	}
	return strings.Join(parts, "\n")
}

// approveAll is the nested run's Approver: a tool the sub-agent holds is
// authorized by construction (read-only by default, or explicitly granted via
// allowed_tools), so nothing is gated — matching the source loop, which
// executed sub-agent tool calls without consulting the approval flow.
type approveAll struct{}

// Ask always allows.
func (approveAll) Ask(context.Context, agentic.ToolCall) (bool, error) { return true, nil }

// parentContext is the parent conversation's full input context: the system
// prompt (when non-empty) followed by the messages — the same list shape the
// source application selected over, so last_n/messages indices count
// identically.
func (e *subagentTool) parentContext() []agentic.Message {
	if e.cfg.ParentSystem == "" {
		return e.cfg.ParentMessages
	}
	out := make([]agentic.Message, 0, len(e.cfg.ParentMessages)+1)
	out = append(out, agentic.Message{Role: agentic.RoleSystem, Content: e.cfg.ParentSystem})
	out = append(out, e.cfg.ParentMessages...)
	return out
}

// buildContextBlock renders the parent-conversation context the orchestrator
// chose to share (share_context). It returns the rendered block (possibly
// empty when there is nothing to share) or a non-empty errMsg describing a
// misuse the model should fix. The summary mode makes one bounded model call.
func (e *subagentTool) buildContextBlock(ctx context.Context, in subagentArgs) (block, errMsg string) {
	switch strings.ToLower(strings.TrimSpace(in.ShareContext)) {
	case "", "none":
		return "", ""
	case "custom":
		c := strings.TrimSpace(in.CustomContext)
		if c == "" {
			return "", "share_context=custom requires custom_context text"
		}
		return c, ""
	case "full":
		return RenderTranscript(e.parentContext()), ""
	case "last_n":
		if in.ContextMessageCount <= 0 {
			return "", "share_context=last_n requires context_message_count (a positive integer)"
		}
		return RenderTranscript(SelectLastN(e.parentContext(), in.ContextMessageCount)), ""
	case "messages":
		if len(in.ContextMessageIndices) == 0 {
			return "", "share_context=messages requires context_message_indices (1 = the most recent message)"
		}
		return RenderTranscript(SelectByEndIndices(e.parentContext(), in.ContextMessageIndices)), ""
	case "summary":
		parent := e.parentContext()
		if len(parent) == 0 {
			return "", ""
		}
		summary, err := generateContextSummary(ctx, e.cfg.Provider, e.cfg.Model, parent, e.cfg.MaxTokens, e.cfg.Extra)
		if err != nil {
			return "", "failed to summarize the parent conversation: " + err.Error()
		}
		return summary, ""
	default:
		return "", "unknown share_context mode " + strconv.Quote(in.ShareContext) + " (want none, full, last_n, messages, summary, or custom)"
	}
}

// resolveAllowedTools maps the orchestrator's requested tool names onto the
// exact advertised names of the sub-agent's grantable tools (read-only ones
// plus any non-read-only tool the orchestrator may explicitly name). A request
// matches by exact advertised name, or — leniently — by its bare name (the
// part after "__") when that is unambiguous, so dropping a server prefix
// still works. It returns the advertised names to keep (in advertised order),
// or a non-empty errMsg (which lists the available tools) when a requested
// name resolves to nothing or the sub-agent has no tools at all — a
// recoverable tool error the model can correct.
func resolveAllowedTools(available, requested []string) (keep []string, errMsg string) {
	if len(available) == 0 {
		return nil, "run_subagent: the sub-agent has no tools available, so allowed_tools cannot be applied -- omit it."
	}
	set := make(map[string]bool, len(available))
	for _, n := range available {
		set[n] = true
	}

	chosen := make(map[string]bool)
	var unknown []string
	for _, raw := range requested {
		req := strings.TrimSpace(raw)
		if req == "" {
			continue
		}
		if set[req] {
			chosen[req] = true
			continue
		}
		// Bare-name fallback: match "<server>__<req>" when exactly one tool does.
		var hits []string
		for _, adv := range available {
			if strings.HasSuffix(adv, "__"+req) {
				hits = append(hits, adv)
			}
		}
		if len(hits) == 1 {
			chosen[hits[0]] = true
			continue
		}
		unknown = append(unknown, req)
	}

	if len(unknown) > 0 {
		return nil, "run_subagent: allowed_tools names no available tool: " + strings.Join(unknown, ", ") +
			". Available tools: " + strings.Join(available, ", ") +
			". Use these exact names, or omit allowed_tools to allow every read-only tool."
	}
	if len(chosen) == 0 {
		return nil, "run_subagent: allowed_tools contained no usable tool names. Available tools: " +
			strings.Join(available, ", ") + "."
	}
	for _, n := range available {
		if chosen[n] {
			keep = append(keep, n)
		}
	}
	return keep, ""
}
