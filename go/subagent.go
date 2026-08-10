package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// SubagentToolName is the advertised name of the built-in sub-agent tool.
const SubagentToolName = "run_subagent"

// DefaultSubagentSystemPrompt instructs a sub-agent how to behave: work
// autonomously with read-only tools and return a single self-contained report.
// It is used when SubagentConfig.SystemPrompt is empty. Ported verbatim from
// the source application.
const DefaultSubagentSystemPrompt = "You are a sub-agent launched by another assistant to carry out one focused, read-only task. " +
	"You cannot modify anything; you have only read-only tools (web and repository access) to gather information. " +
	"Work autonomously — you cannot ask follow-up questions, and you do not see the parent conversation, only the task below. " +
	"Use the available tools as needed, then return a single, self-contained final report that directly answers the task: " +
	"give the concrete findings the calling assistant needs, not a narration of your process. Be concise and factual."

// subagentToolDescription is the model-facing tool description, ported from
// the source application. One deliberate adaptation: the source enumerated
// its own application's read-only tools ("fetch a web page (web_fetch), read
// GitHub repositories (repo_read: ...), and any read-only MCP tools that are
// enabled") inside the CAPABILITIES sentence; the library cannot know the
// host's toolset, so that enumeration is dropped. Everything else is
// verbatim.
const subagentToolDescription = "Launch a sub-agent: an autonomous helper that runs its own agentic loop in a separate, " +
	"throwaway context and reports back only its final answer. " +
	"WHAT IT'S FOR: offload a focused, self-contained, read-only task so all the intermediate work -- many tool " +
	"calls, large search results, long file dumps -- stays out of your context, and only the distilled result " +
	"comes back to you. This keeps your own context clean. " +
	"CAPABILITIES: the sub-agent runs on the same model. By DEFAULT it may use only read-only tools " +
	"and causes no side effect. It can use a NON-read-only (state-changing) tool only " +
	"if you explicitly grant that tool by name via 'allowed_tools' (see below); otherwise it cannot write, edit, or " +
	"change anything. " +
	"FOCUS IT WITH 'allowed_tools' (STRONGLY RECOMMENDED): by default the sub-agent may use EVERY read-only tool, " +
	"and an unfocused sub-agent tends to squander its limited turns wandering through irrelevant tools -- most " +
	"often defaulting to web searches even when the answer is in a repo or page you already pointed it at. Pass " +
	"'allowed_tools' with the exact tool names (as they appear in your own tool list) the task actually needs, and " +
	"the sub-agent is offered ONLY those. Pin it to the smallest set that can do the job; omit 'allowed_tools' only " +
	"when the task genuinely needs the full read-only toolset. Naming a tool here is ALSO the only way to grant a " +
	"non-read-only tool. Restricting tools keeps the sub-agent focused, faster, and cheaper. " +
	"WHEN TO USE IT: multi-step research or exploration where you mainly need the conclusion -- e.g. 'find where X " +
	"is implemented across this repo and summarize how it works', 'read these pages and compare what they say " +
	"about Y', or any investigation that would otherwise flood your context with raw tool output. " +
	"WHEN NOT TO USE IT: a single quick lookup you can just do yourself in one tool call; a state-changing task " +
	"unless you deliberately grant the needed tool via 'allowed_tools' (the sub-agent is read-only by default); or " +
	"a task that depends on details from this conversation (the sub-agent does NOT see it). " +
	"HOW TO PROMPT IT: the sub-agent starts fresh and cannot ask follow-up questions, so make 'prompt' fully " +
	"self-contained -- state exactly what to investigate and precisely what to report back, in what form. " +
	"SHARING CONTEXT: by default the sub-agent sees ONLY your prompt, not this conversation. If it needs history, " +
	"set 'share_context' to share the full input context, the last N messages, specific recent messages, an " +
	"auto-generated summary, or your own custom text (see that parameter). Prefer a self-contained prompt; widen " +
	"the shared context only when the task truly depends on this conversation. " +
	"CONSTRAINTS: sub-agents run one at a time and this call blocks until the sub-agent finishes; a sub-agent " +
	"cannot launch further sub-agents."

// subagentSchema is the static parameter schema, ported verbatim.
// advertisedSchema specializes its allowed_tools field per turn when
// grantable tools exist.
var subagentSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "description": {
      "type": "string",
      "description": "A short (3-5 word) label for the task, shown in the UI."
    },
    "prompt": {
      "type": "string",
      "description": "The full task for the sub-agent. Be specific and self-contained: by default the sub-agent does not see this conversation, only this prompt (and whatever you choose to share via share_context). State exactly what to investigate and what to report back."
    },
    "share_context": {
      "type": "string",
      "enum": ["none", "full", "last_n", "messages", "summary", "custom"],
      "description": "What to share from THIS conversation with the sub-agent, in addition to the prompt. 'none' (default): only the prompt. 'full': the entire input context (system prompt + every message). 'last_n': the last N messages (set context_message_count). 'messages': specific recent messages by position (set context_message_indices). 'summary': an auto-generated briefing summarizing this conversation. 'custom': the exact text you supply in custom_context. Default 'none' -- only widen it when the sub-agent genuinely needs this conversation's history."
    },
    "context_message_count": {
      "type": "integer",
      "minimum": 1,
      "description": "Required when share_context='last_n': how many of the most recent messages to include."
    },
    "context_message_indices": {
      "type": "array",
      "items": { "type": "integer", "minimum": 1 },
      "description": "Required when share_context='messages': positions of specific recent messages to include, counted from the end (1 = the most recent message, 2 = the one before it, ...)."
    },
    "custom_context": {
      "type": "string",
      "description": "Required when share_context='custom': the exact background text to give the sub-agent."
    },
    "allowed_tools": {
      "type": "array",
      "items": { "type": "string" },
      "description": "Restrict the sub-agent to ONLY these tools, by their exact names (as shown in your own tool list). STRONGLY RECOMMENDED: pinning the sub-agent to just the tools its task needs keeps it focused and stops it wasting turns on irrelevant tools like web search. By default the sub-agent gets only read-only tools; naming a tool here is also how you grant a NON-read-only (state-changing) tool when the task needs it. Omit to allow every read-only tool."
    }
  },
  "required": ["prompt"]
}`)

// Subagent activity kinds delivered to SubagentConfig.OnActivity while a
// run_subagent call executes, so a host can show what the otherwise silent
// sub-agent is doing instead of an opaque, indefinite "running" state. They
// are transient telemetry only: never fed back into any model's context.
const (
	SubagentActivityTurn       = "turn"        // a new sub-agent turn began
	SubagentActivityToolCall   = "tool_call"   // the sub-agent invoked a tool
	SubagentActivityToolResult = "tool_result" // a sub-agent tool returned
	SubagentActivityText       = "text"        // the sub-agent's own answer for a turn
	SubagentActivityThinking   = "thinking"    // its reasoning for a turn
)

// SubagentActivity is one progress step from a running sub-agent. CallID is
// the parent run_subagent tool call's ID, so a host can attach each step to
// the right tool block. Detail is a whitespace-flattened preview capped at
// 160 runes (an argument preview for tool_call, a result preview for
// tool_result).
type SubagentActivity struct {
	CallID string
	Kind   string // one of the SubagentActivity* constants
	Turn   int    // 1-based turn number (every kind but tool_call/tool_result)
	Tool   string // tool name (tool_call / tool_result)
	Detail string // arguments preview, result preview, or other short context
	// Content is the SAME text as Detail but WHOLE: the full arguments, the
	// full tool output, the full answer or the full reasoning, never capped or
	// whitespace-flattened. Detail alone left a host with no way to show what a
	// sub-agent actually read or said — a 160-rune preview of a file listing
	// answers nothing — so hosts that can render a scrollable block use this
	// and keep Detail for the one-line summary.
	Content string
	IsError bool // tool_result only: the tool reported an error
}

// SubagentConfig configures NewSubagentTool.
type SubagentConfig struct {
	// Provider and Model run the sub-agent (typically the same as the parent
	// turn's). MaxTokens and Extra are forwarded to every sub-agent model call
	// (MaxTokens is required when Provider speaks the Anthropic dialect).
	// There is no Retry here for the same reason Config has none: the nested
	// run is a loop, and loops do not retry. The sub-agent's calls inherit
	// whatever Provider does, like every other call in the library.
	Provider  Provider
	Model     string
	MaxTokens int
	Extra     map[string]any
	// Tools is the parent's FULL toolset (every tool the parent turn has). The
	// read-only subset of it is the default sub-agent toolset, but a
	// non-read-only tool the orchestrator names in allowed_tools is granted
	// explicitly. Empty runs the sub-agent tool-less.
	Tools Tools
	// ParentSystem and ParentMessages are the parent conversation's input
	// context — the source for the share_context modes. The system prompt is
	// prepended as a system message before selection, so last_n/messages
	// indices count over the same list the model context held. Empty means
	// the sub-agent can only ever run with the prompt alone.
	ParentSystem   string
	ParentMessages []Message
	// Gate bounds concurrent sub-agent execution (share one Gate across the
	// tools that should share the limit). nil = no limit.
	Gate *Gate
	// Runs makes run_subagent ASYNCHRONOUS: the call returns a receipt as soon
	// as the sub-agent is launched, so the orchestrator can fan several out in
	// one turn and keep working, and each report is delivered between turns
	// through this registry (Run drains it; a host driving its own loop calls
	// Pending/Collect itself).
	//
	// nil keeps the call synchronous -- it blocks until the sub-agent answers,
	// and one call can only ever produce one running sub-agent.
	Runs *SubagentRuns
	// SystemPrompt overrides DefaultSubagentSystemPrompt when non-empty.
	SystemPrompt string
	// OnActivity, when non-nil, receives live telemetry: a step per sub-agent
	// turn and around each of the sub-agent's own tool calls. It is called
	// synchronously from the sub-agent's loop.
	OnActivity func(SubagentActivity)
}

// subagentTool implements run_subagent. It is deliberately NOT marked
// read-only, so Tools.Readonly excludes it from a sub-agent's default toolset,
// and grantableTools omits it from the set allowed_tools can name — so a
// sub-agent can never spawn another (no recursion).
type subagentTool struct {
	cfg      SubagentConfig
	system   string
	readonly Tools
}

// NewSubagentTool builds the run_subagent tool: one tool that runs a nested,
// in-memory agentic loop (this package's Run) on cfg.Provider and reports back
// only the sub-agent's final answer. Append it to the rest of the toolset like
// any other tool. NeedsApproval always reports false — wrap it if launching
// sub-agents should be approval-gated.
func NewSubagentTool(cfg SubagentConfig) Tool {
	system := strings.TrimSpace(cfg.SystemPrompt)
	if system == "" {
		system = DefaultSubagentSystemPrompt
	}
	return &subagentTool{cfg: cfg, system: system, readonly: cfg.Tools.Readonly()}
}

// Decl advertises run_subagent. Readonly is intentionally left false — see the
// type doc.
func (e *subagentTool) Decl() ToolDecl {
	return ToolDecl{
		Name:        SubagentToolName,
		Description: subagentToolDescription,
		InputSchema: e.advertisedSchema(e.grantableTools()),
	}
}

// NeedsApproval always reports false: approval wiring stays the caller's
// concern (the source application keyed it to a user setting).
func (e *subagentTool) NeedsApproval() bool { return false }

// grantableTools returns the tools allowed_tools may name, in deterministic
// order: every tool in the full toolset EXCEPT run_subagent itself (excluding
// it here is what stops a sub-agent being granted the power to spawn another).
// The set includes non-read-only tools — naming one in allowed_tools is how
// the orchestrator explicitly grants it; without that, only the read-only
// subset is used by default.
func (e *subagentTool) grantableTools() []ToolDecl {
	var out []ToolDecl
	for _, d := range e.cfg.Tools.Decls() {
		if d.Name != SubagentToolName {
			out = append(out, d)
		}
	}
	return out
}

// grantableToolNames is the advertised names of grantableTools.
func grantableToolNames(tools []ToolDecl) []string {
	names := make([]string, 0, len(tools))
	for _, t := range tools {
		names = append(names, t.Name)
	}
	return names
}

// advertisedSchema returns the run_subagent parameter schema with
// allowed_tools specialised to this turn's grantable toolset: its description
// lists the available tools (flagging the ones that modify state) and its
// items carry an enum of their exact names, so the model is both told and
// constrained to the valid names. With no grantable tools it returns the
// static schema unchanged (allowed_tools is then inert). The map round-trip
// keeps the static schema literal the single source of truth; any defensive
// fall-through returns it intact.
func (e *subagentTool) advertisedSchema(tools []ToolDecl) json.RawMessage {
	if len(tools) == 0 {
		return subagentSchema
	}
	var m map[string]any
	if json.Unmarshal(subagentSchema, &m) != nil {
		return subagentSchema
	}
	props, _ := m["properties"].(map[string]any)
	at, _ := props["allowed_tools"].(map[string]any)
	if at == nil {
		return subagentSchema
	}
	at["description"] = allowedToolsDescription(tools)
	enum := make([]any, len(tools))
	for i, t := range tools {
		enum[i] = t.Name
	}
	at["items"] = map[string]any{"type": "string", "enum": enum}
	if b, err := json.Marshal(m); err == nil {
		return b
	}
	return subagentSchema
}

// allowedToolsDescription is the allowed_tools field description, naming the
// concrete tools available this turn so the model knows exactly what it can
// pin the sub-agent to. Tools that are NOT read-only are flagged "(modifies
// state)" so the model sees that listing one grants a side-effecting tool —
// by default the sub-agent only gets read-only tools, and naming a tool here
// is what makes a non-read-only one available.
func allowedToolsDescription(tools []ToolDecl) string {
	labels := make([]string, len(tools))
	for i, t := range tools {
		labels[i] = t.Name
		if !t.Readonly {
			labels[i] += " (modifies state)"
		}
	}
	return "Restrict the sub-agent to ONLY these tools, by their exact names. " +
		"STRONGLY RECOMMENDED: pin the sub-agent to just the tools its task needs so it stays focused and does not " +
		"waste its limited turns on irrelevant tools (e.g. running web searches when the answer is in a repo or page " +
		"you already gave it). By default the sub-agent gets only read-only tools; naming a tool here is also how you " +
		"grant a NON-read-only tool (flagged below) when the task truly needs it. Omit to allow every read-only tool. " +
		"Available tools: " + strings.Join(labels, ", ") + "."
}

// subagentArgs is the run_subagent argument payload.
type subagentArgs struct {
	Description           string   `json:"description"`
	Prompt                string   `json:"prompt"`
	ShareContext          string   `json:"share_context"`
	ContextMessageCount   int      `json:"context_message_count"`
	ContextMessageIndices []int    `json:"context_message_indices"`
	CustomContext         string   `json:"custom_context"`
	AllowedTools          []string `json:"allowed_tools"`
}

// Execute runs one sub-agent. Every misuse — a bad share_context selection,
// an allowed_tools name that resolves to nothing — is a recoverable error
// tool result that teaches the valid shape, never a Go error.
func (e *subagentTool) Execute(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	// A body nobody can parse is the one failure still answered synchronously
	// even in async mode: there is nothing to launch, and the model should
	// learn that from the call it just made.
	var in subagentArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return ToolResult{Content: "invalid run_subagent arguments: " + err.Error(), IsError: true}, nil
	}
	if e.cfg.Runs == nil {
		return e.run(ctx, in), nil
	}

	// Asynchronous: register the run, hand back a receipt, and let the
	// goroutine report through the registry. Every other failure -- an
	// unconfigured model, a misused argument -- reaches the model as that
	// report, one delivery later, rather than as this call's result.
	callID := ToolCallID(ctx)
	if callID == "" {
		callID = e.cfg.Runs.nextID()
	}
	e.cfg.Runs.Launch(callID, in.Description, in.Prompt)
	go e.launched(ctx, callID, in)
	return ToolResult{Content: SubagentLaunchReceipt(in.Description)}, nil
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

	res := e.runGated(WithToolCallID(ctx, callID), in)
	e.cfg.Runs.Complete(callID, res.Content, res.IsError)
}

// run executes one sub-agent to completion. Every misuse — a bad
// share_context selection, an allowed_tools name that resolves to nothing — is
// a recoverable error tool result that teaches the valid shape.
func (e *subagentTool) run(ctx context.Context, in subagentArgs) ToolResult {
	// Serialize per the shared Gate. Acquisition is cancellable so a caller
	// disconnect (or a stopped turn) while waiting returns promptly. The
	// asynchronous path takes the slot itself, BEFORE marking the run running,
	// so a queued launch reads as queued rather than as an unexplained delay.
	release, err := e.cfg.Gate.Acquire(ctx)
	if err != nil {
		return ToolResult{Content: "run_subagent was cancelled before it could start: " + err.Error(), IsError: true}
	}
	defer release()
	return e.runGated(ctx, in)
}

// runGated executes one sub-agent with its concurrency slot already held.
func (e *subagentTool) runGated(ctx context.Context, in subagentArgs) ToolResult {
	if e.cfg.Provider == nil || e.cfg.Model == "" {
		return ToolResult{Content: "run_subagent is unavailable: no model is configured for the sub-agent", IsError: true}
	}
	if strings.TrimSpace(in.Prompt) == "" {
		return ToolResult{Content: "run_subagent requires a non-empty prompt describing the task", IsError: true}
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
			return ToolResult{Content: terr, IsError: true}
		}
		subTools, granted = e.cfg.Tools.Subset(keep), true
	}

	// Build the optional parent-context block the orchestrator asked to share
	// and fold it into the task. A bad selection (e.g. last_n without a count)
	// is a recoverable tool error so the model can correct the call.
	block, errMsg := e.buildContextBlock(ctx, in)
	if errMsg != "" {
		return ToolResult{Content: errMsg, IsError: true}
	}
	task := composeSubagentTask(block, in.Prompt)

	res, runErr := Run(ctx, e.runConfig(ToolCallID(ctx), subTools, granted), Request{
		Model:     e.cfg.Model,
		System:    e.system,
		Messages:  []Message{{Role: RoleUser, Content: task}},
		MaxTokens: e.cfg.MaxTokens,
		Extra:     e.cfg.Extra,
	})
	if runErr != nil {
		return ToolResult{Content: "sub-agent failed: " + runErr.Error(), IsError: true}
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

// runConfig assembles the nested Run's Config: the sub-agent's toolset, the
// approve-everything Approver (the explicit allowed_tools grant is itself the
// authorization — the source loop never consulted NeedsApproval), and, when
// the host listens, the activity telemetry hooks stamped with the parent
// call's ID.
func (e *subagentTool) runConfig(callID string, subTools Tools, granted bool) Config {
	cfg := Config{
		Provider:    e.cfg.Provider,
		Tools:       subTools,
		Approver:    approveAll{},
		unknownTool: e.unavailableTool(granted),
	}
	act := e.cfg.OnActivity
	if act == nil {
		return cfg
	}
	cfg.turnHook = func(turn int) {
		act(SubagentActivity{CallID: callID, Kind: SubagentActivityTurn, Turn: turn})
	}
	cfg.Events = Events{
		OnToolCall: func(c ToolCall) error {
			act(SubagentActivity{
				CallID:  callID,
				Kind:    SubagentActivityToolCall,
				Tool:    c.Name,
				Detail:  subagentPreview(c.Arguments),
				Content: c.Arguments,
			})
			return nil
		},
		OnToolResult: func(c ToolCall, r ToolResult) error {
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
		OnTurnEnd: func(turn int, comp *Completion, _ error) error {
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
func thinkingText(blocks []ThinkingBlock) string {
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
func (approveAll) Ask(context.Context, ToolCall) (bool, error) { return true, nil }

// parentContext is the parent conversation's full input context: the system
// prompt (when non-empty) followed by the messages — the same list shape the
// source application selected over, so last_n/messages indices count
// identically.
func (e *subagentTool) parentContext() []Message {
	if e.cfg.ParentSystem == "" {
		return e.cfg.ParentMessages
	}
	out := make([]Message, 0, len(e.cfg.ParentMessages)+1)
	out = append(out, Message{Role: RoleSystem, Content: e.cfg.ParentSystem})
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
