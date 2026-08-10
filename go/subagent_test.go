package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// subParentExec is the canonical parent toolset for subagent tests: two
// namespaced tools (one read-only, one not) and a bare read-only one.
func subParentExec() *fakeExec {
	return &fakeExec{tools: []ToolDecl{
		{Name: "Repo__read", Readonly: true},
		{Name: "Repo__write"},
		{Name: "web_read", Readonly: true},
	}}
}

// subCall builds a run_subagent ToolCall with the given JSON arguments.
func subCall(args string) json.RawMessage { return json.RawMessage(args) }

// toolNames extracts the advertised names of a request's tools.
func toolNames(tools []ToolDecl) []string {
	names := make([]string, 0, len(tools))
	for _, tl := range tools {
		names = append(names, tl.Name)
	}
	return names
}

func TestSubagentAdvertisement(t *testing.T) {
	exec := NewSubagentTool(SubagentConfig{Tools: subParentExec().registry()})
	tools := []ToolDecl{exec.Decl()}
	require.Len(t, tools, 1)
	tool := tools[0]
	assert.Equal(t, SubagentToolName, tool.Name)
	assert.False(t, tool.Readonly, "run_subagent is NOT read-only, so ReadonlyView drops it (no recursion)")
	assert.Contains(t, tool.Description, "Launch a sub-agent")
	assert.False(t, exec.NeedsApproval(), "approval gating stays the caller's concern")

	var schema map[string]any
	require.NoError(t, json.Unmarshal(tool.InputSchema, &schema))
	props := schema["properties"].(map[string]any)
	at := props["allowed_tools"].(map[string]any)
	items := at["items"].(map[string]any)
	assert.Equal(t, []any{"Repo__read", "Repo__write", "web_read"}, items["enum"],
		"the advertised schema constrains allowed_tools to the grantable names")
	desc := at["description"].(string)
	assert.Contains(t, desc, "Available tools: Repo__read, Repo__write (modifies state), web_read.",
		"non-read-only tools are flagged")
	assert.NotContains(t, desc, "Repo__read (modifies state)")
	assert.Contains(t, schema["required"], "prompt")
}

func TestSubagentAdvertisementStaticSchemaWithoutTools(t *testing.T) {
	exec := NewSubagentTool(SubagentConfig{})
	tool := exec.Decl()
	assert.Equal(t, string(subagentSchema), string(tool.InputSchema),
		"no grantable tools falls back to the static schema (allowed_tools inert)")
}

func TestSubagentExecuteDefaultReadonlyToolset(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "s1", Name: "Repo__read", Arguments: `{"path":"a"}`})},
		{comp: assistantComp("  the report  ")},
	}}
	parent := subParentExec()
	exec := NewSubagentTool(SubagentConfig{
		Provider:  provider,
		Model:     "sub-model",
		MaxTokens: 512,
		Extra:     map[string]any{"temperature": 0.2},

		Tools: parent.registry(),
	})

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"investigate"}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, "the report", res.Content, "the trimmed final answer is the tool result")

	require.Len(t, provider.reqs, 2)
	first := provider.reqs[0]
	assert.Equal(t, "sub-model", first.Model)
	assert.Equal(t, DefaultSubagentSystemPrompt, first.System)
	assert.Equal(t, 512, first.MaxTokens)
	assert.Equal(t, map[string]any{"temperature": 0.2}, first.Extra)
	require.Len(t, first.Messages, 1)
	assert.Equal(t, Message{Role: RoleUser, Content: "investigate"}, first.Messages[0],
		"share_context defaults to none: the task is the prompt alone")
	assert.Equal(t, []string{"Repo__read", "web_read"}, toolNames(first.Tools),
		"the default sub-agent toolset is the read-only subset")

	require.Len(t, parent.executed, 1)
	assert.Equal(t, `{"path":"a"}`, parent.executed[0].Arguments)
}

func TestSubagentAllowedToolsGrantsNonReadonly(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "s1", Name: "Repo__write", Arguments: "{}"})},
		{comp: assistantComp("wrote it")},
	}}
	parent := subParentExec()
	parent.ask = map[string]bool{"Repo__write": true} // an "always ask" parent flag
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Tools: parent.registry()})

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"write","allowed_tools":["Repo__write"]}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, "wrote it", res.Content)
	assert.Equal(t, []string{"Repo__write"}, toolNames(provider.reqs[0].Tools),
		"allowed_tools pins the sub-agent to exactly the named set")
	require.Len(t, parent.executed, 1, "the explicitly granted non-read-only tool executes")
	// The grant is the authorization: no approver runs inside the sub-loop,
	// and the tool was executed rather than denied.
	tool := provider.reqs[1].Messages[len(provider.reqs[1].Messages)-1]
	assert.Equal(t, "ran Repo__write", tool.Content)
	assert.NotEqual(t, DeniedMessage, tool.Content)
}

func TestSubagentAllowedToolsBareNameFallback(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Tools: subParentExec().registry()})

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["write"]}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, []string{"Repo__write"}, toolNames(provider.reqs[0].Tools),
		"an unambiguous bare name resolves through the __ prefix")
}

func TestSubagentAllowedToolsAmbiguousBareName(t *testing.T) {
	parent := &fakeExec{tools: []ToolDecl{
		{Name: "A__read", Readonly: true},
		{Name: "B__read", Readonly: true},
	}}
	provider := &scriptProvider{}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Tools: parent.registry()})

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["read"]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "run_subagent: allowed_tools names no available tool: read. "+
		"Available tools: A__read, B__read. "+
		"Use these exact names, or omit allowed_tools to allow every read-only tool.", res.Content)
	assert.Empty(t, provider.reqs, "a bad grant never reaches the model")
}

func TestSubagentAllowedToolsUnknownNameTeaches(t *testing.T) {
	provider := &scriptProvider{}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Tools: subParentExec().registry()})

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["nope"]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "run_subagent: allowed_tools names no available tool: nope. "+
		"Available tools: Repo__read, Repo__write, web_read. "+
		"Use these exact names, or omit allowed_tools to allow every read-only tool.", res.Content)
}

func TestSubagentAllowedToolsEdgeCases(t *testing.T) {
	t.Run("no tools at all", func(t *testing.T) {
		exec := NewSubagentTool(SubagentConfig{Provider: &scriptProvider{}, Model: "m"})
		res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["x"]}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "run_subagent: the sub-agent has no tools available, so allowed_tools cannot be applied -- omit it.", res.Content)
	})
	t.Run("only blank names", func(t *testing.T) {
		exec := NewSubagentTool(SubagentConfig{Provider: &scriptProvider{}, Model: "m", Tools: subParentExec().registry()})
		res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["", "  "]}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "run_subagent: allowed_tools contained no usable tool names. "+
			"Available tools: Repo__read, Repo__write, web_read.", res.Content)
	})
}

func TestSubagentNoRecursion(t *testing.T) {
	// A parent toolset that (like the source composite) carries the subagent
	// tool itself: it is excluded from the grantable set, from the schema
	// enum, and — not being read-only — from the default sub toolset.
	parent := &fakeExec{tools: []ToolDecl{
		{Name: SubagentToolName},
		{Name: "Repo__read", Readonly: true},
	}}
	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Tools: parent.registry()})

	var schema map[string]any
	require.NoError(t, json.Unmarshal(exec.Decl().InputSchema, &schema))
	items := schema["properties"].(map[string]any)["allowed_tools"].(map[string]any)["items"].(map[string]any)
	assert.Equal(t, []any{"Repo__read"}, items["enum"], "run_subagent is never grantable")

	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p","allowed_tools":["run_subagent"]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError, "naming run_subagent in allowed_tools is a teaching error")
	assert.Contains(t, res.Content, "allowed_tools names no available tool: run_subagent")

	res, err = exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, []string{"Repo__read"}, toolNames(provider.reqs[0].Tools),
		"the default read-only subset never contains run_subagent")
}

func TestSubagentShareContextModes(t *testing.T) {
	parentMsgs := []Message{
		{Role: RoleUser, Content: "first"},
		{Role: RoleAssistant, Content: "second"},
		{Role: RoleUser, Content: "third"},
	}
	base := func(provider *scriptProvider) SubagentConfig {
		return SubagentConfig{
			Provider: provider, Model: "m",
			ParentSystem:   "sys prompt",
			ParentMessages: parentMsgs,
		}
	}
	taskOf := func(t *testing.T, provider *scriptProvider) string {
		t.Helper()
		require.NotEmpty(t, provider.reqs)
		last := provider.reqs[len(provider.reqs)-1]
		require.Len(t, last.Messages, 1)
		return last.Messages[0].Content
	}

	t.Run("none is the default", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(), subCall(`{"prompt":"task"}`))
		require.NoError(t, err)
		assert.Equal(t, "task", taskOf(t, provider))
	})

	t.Run("custom", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"custom","custom_context":" my notes "}`))
		require.NoError(t, err)
		assert.Equal(t, "Context shared from the parent conversation (background only):\n\n"+
			"my notes"+
			"\n\n----------------------------------------\n\nYour task:\n\ntask", taskOf(t, provider))
	})

	t.Run("full includes the system prompt", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"full"}`))
		require.NoError(t, err)
		task := taskOf(t, provider)
		assert.Contains(t, task, "System:\nsys prompt\n")
		assert.Contains(t, task, "User:\nfirst\n")
		assert.Contains(t, task, "User:\nthird")
		assert.Contains(t, task, "Your task:\n\ntask")
	})

	t.Run("last_n", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"last_n","context_message_count":1}`))
		require.NoError(t, err)
		task := taskOf(t, provider)
		assert.Contains(t, task, "User:\nthird")
		assert.NotContains(t, task, "first")
		assert.NotContains(t, task, "sys prompt")
	})

	t.Run("messages by end index", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"messages","context_message_indices":[2]}`))
		require.NoError(t, err)
		task := taskOf(t, provider)
		assert.Contains(t, task, "Assistant:\nsecond")
		assert.NotContains(t, task, "third")
	})

	t.Run("summary runs one briefing call", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{
			{comp: assistantComp("  the briefing  ")},
			{comp: assistantComp("ok")},
		}}
		_, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"summary"}`))
		require.NoError(t, err)
		require.Len(t, provider.reqs, 2)
		sum := provider.reqs[0]
		assert.Equal(t, contextSummarySystemPrompt, sum.System)
		assert.Empty(t, sum.Tools, "the summary call is tool-less")
		require.Len(t, sum.Messages, 1)
		assert.True(t, strings.HasPrefix(sum.Messages[0].Content, "Conversation to brief the sub-agent on:\n\n"))
		assert.Contains(t, sum.Messages[0].Content, "User:\nfirst")
		assert.Contains(t, taskOf(t, provider), "the briefing"+
			"\n\n----------------------------------------\n\nYour task:\n\ntask")
	})

	t.Run("summary with no parent context shares nothing", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("ok")}}}
		cfg := base(provider)
		cfg.ParentSystem, cfg.ParentMessages = "", nil
		_, err := NewSubagentTool(cfg).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"summary"}`))
		require.NoError(t, err)
		require.Len(t, provider.reqs, 1, "no summary call without parent context")
		assert.Equal(t, "task", taskOf(t, provider))
	})

	t.Run("summary failure is a recoverable error", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 400, Body: "nope"}}}}
		res, err := NewSubagentTool(base(provider)).Execute(context.Background(),
			subCall(`{"prompt":"task","share_context":"summary"}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "failed to summarize the parent conversation: "))
	})

	misuses := []struct {
		name, args, want string
	}{
		{"custom without text", `{"prompt":"p","share_context":"custom"}`,
			"share_context=custom requires custom_context text"},
		{"last_n without count", `{"prompt":"p","share_context":"last_n"}`,
			"share_context=last_n requires context_message_count (a positive integer)"},
		{"messages without indices", `{"prompt":"p","share_context":"messages"}`,
			"share_context=messages requires context_message_indices (1 = the most recent message)"},
		{"unknown mode", `{"prompt":"p","share_context":"weird"}`,
			`unknown share_context mode "weird" (want none, full, last_n, messages, summary, or custom)`},
	}
	for _, tc := range misuses {
		t.Run(tc.name, func(t *testing.T) {
			provider := &scriptProvider{}
			res, err := NewSubagentTool(base(provider)).Execute(context.Background(), subCall(tc.args))
			require.NoError(t, err)
			assert.True(t, res.IsError)
			assert.Equal(t, tc.want, res.Content)
			assert.Empty(t, provider.reqs)
		})
	}
}

func TestSubagentActivityTelemetry(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "s1", Name: "Repo__read", Arguments: "  {\"q\":\n \"x\"} "})},
		{comp: assistantComp("report")},
	}}
	parent := subParentExec()
	longOut := strings.Repeat("r", 300)
	parent.execute = func(context.Context, ToolCall) (ToolResult, error) {
		return ToolResult{Content: longOut}, nil
	}
	var acts []SubagentActivity
	exec := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Tools: parent.registry(),
		OnActivity: func(a SubagentActivity) { acts = append(acts, a) },
	})
	// The parent call's id rides the context, exactly as Run puts it there --
	// it is what lets a host attach the play-by-play to the right tool block.
	res, err := exec.Execute(WithToolCallID(context.Background(), "call-7"), subCall(`{"prompt":"go"}`))
	require.NoError(t, err)
	assert.Equal(t, "report", res.Content)

	require.Len(t, acts, 5, "turn, tool_call, tool_result, turn, text")
	assert.Equal(t, SubagentActivity{CallID: "call-7", Kind: SubagentActivityTurn, Turn: 1}, acts[0])
	assert.Equal(t, SubagentActivity{
		CallID: "call-7", Kind: SubagentActivityToolCall, Tool: "Repo__read", Detail: `{"q": "x"}`,
		Content: "  {\"q\":\n \"x\"} ",
	}, acts[1], "the preview is whitespace-flattened; Content is the arguments verbatim")
	assert.Equal(t, SubagentActivityToolResult, acts[2].Kind)
	assert.Equal(t, "call-7", acts[2].CallID)
	assert.False(t, acts[2].IsError)
	assert.Len(t, []rune(acts[2].Detail), subagentPreviewMaxRunes, "result previews are rune-capped")
	assert.True(t, strings.HasSuffix(acts[2].Detail, "..."))
	assert.Equal(t, longOut, acts[2].Content, "Content carries the WHOLE tool output, uncapped")
	assert.Equal(t, SubagentActivity{CallID: "call-7", Kind: SubagentActivityTurn, Turn: 2}, acts[3])
	// The sub-agent's own words for the turn, so a host can show what it said
	// and not just which files it touched.
	assert.Equal(t, SubagentActivity{
		CallID: "call-7", Kind: SubagentActivityText, Turn: 2, Detail: "report", Content: "report",
	}, acts[4])
}

// TestSubagentActivityReportsThinking: a turn that reasons without answering
// still shows its reasoning, which used to be invisible to the host entirely.
func TestSubagentActivityReportsThinking(t *testing.T) {
	thinking := assistantComp("")
	thinking.Message.Thinking = []ThinkingBlock{{Text: "first I check the layout"}, {Text: "then the callers"}}
	provider := &scriptProvider{steps: []scriptStep{
		{comp: thinking},
		{comp: assistantComp("report")},
	}}
	var acts []SubagentActivity
	exec := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Tools: subParentExec().registry(),
		OnActivity: func(a SubagentActivity) { acts = append(acts, a) },
	})
	_, err := exec.Execute(context.Background(), subCall(`{"prompt":"go"}`))
	require.NoError(t, err)

	var think *SubagentActivity
	for i := range acts {
		if acts[i].Kind == SubagentActivityThinking {
			think = &acts[i]
		}
	}
	require.NotNil(t, think, "reasoning must reach the host")
	assert.Equal(t, 1, think.Turn)
	assert.Equal(t, "first I check the layout\nthen the callers", think.Content)
	assert.Equal(t, "first I check the layout then the callers", think.Detail)
}

func TestSubagentActivityToolError(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "s1", Name: "Repo__read", Arguments: "{}"})},
		{comp: assistantComp("report")},
	}}
	parent := subParentExec()
	parent.execute = func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, errors.New("boom") }
	var acts []SubagentActivity
	exec := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Tools: parent.registry(),
		OnActivity: func(a SubagentActivity) { acts = append(acts, a) },
	})
	_, err := exec.Execute(context.Background(), subCall(`{"prompt":"go"}`))
	require.NoError(t, err)
	require.Len(t, acts, 5)
	assert.True(t, acts[2].IsError, "an executor failure marks the tool_result step")
	assert.Equal(t, "tool execution failed: boom", acts[2].Detail)
	assert.Equal(t, "tool execution failed: boom", acts[2].Content)
}

// countingProvider counts Complete calls race-safely around an inner provider.
type countingProvider struct {
	inner Provider
	calls atomic.Int32
}

func (c *countingProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	c.calls.Add(1)
	return c.inner.Complete(ctx, req, ev)
}

func TestSubagentGateSerializes(t *testing.T) {
	gate := NewGate(1)
	hold, err := gate.Acquire(context.Background())
	require.NoError(t, err)

	provider := &countingProvider{inner: &scriptProvider{steps: []scriptStep{{comp: assistantComp("done")}}}}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Gate: gate})

	done := make(chan ToolResult, 1)
	go func() {
		res, _ := exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
		done <- res
	}()

	time.Sleep(30 * time.Millisecond)
	assert.Equal(t, int32(0), provider.calls.Load(), "the sub-agent waits for the gate")
	hold()

	res := <-done
	assert.False(t, res.IsError)
	assert.Equal(t, "done", res.Content)
	assert.Equal(t, int32(1), provider.calls.Load())
}

func TestSubagentGateCancelledWhileWaiting(t *testing.T) {
	gate := NewGate(1)
	hold, err := gate.Acquire(context.Background())
	require.NoError(t, err)
	defer hold()

	provider := &scriptProvider{}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m", Gate: gate})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	res, err := exec.Execute(ctx, subCall(`{"prompt":"p"}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "run_subagent was cancelled before it could start: context canceled", res.Content)
	assert.Empty(t, provider.reqs)
}

func TestSubagentExecuteErrors(t *testing.T) {
	t.Run("no model configured", func(t *testing.T) {
		exec := NewSubagentTool(SubagentConfig{})
		res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "run_subagent is unavailable: no model is configured for the sub-agent", res.Content)
	})
	t.Run("invalid arguments", func(t *testing.T) {
		exec := NewSubagentTool(SubagentConfig{Provider: &scriptProvider{}, Model: "m"})
		res, err := exec.Execute(context.Background(), subCall(`{`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "invalid run_subagent arguments: "))
	})
	t.Run("empty prompt", func(t *testing.T) {
		exec := NewSubagentTool(SubagentConfig{Provider: &scriptProvider{}, Model: "m"})
		res, err := exec.Execute(context.Background(), subCall(`{"prompt":"  "}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "run_subagent requires a non-empty prompt describing the task", res.Content)
	})
	t.Run("sub-run failure is a recoverable result", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 400, Body: "bad"}}}}
		exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m"})
		res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
		require.NoError(t, err, "the parent loop never aborts on tool failure")
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "sub-agent failed: "))
	})
}

func TestSubagentCustomSystemPromptAndUncappedTurns(t *testing.T) {
	// A custom system prompt replaces the default, and the nested loop runs as
	// long as the sub-agent keeps working -- no cap cuts its research short.
	provider := &scriptProvider{steps: []scriptStep{
		{comp: assistantComp("", ToolCall{ID: "s1", Name: "Repo__read", Arguments: `{"i":1}`})},
		{comp: assistantComp("", ToolCall{ID: "s2", Name: "Repo__read", Arguments: `{"i":2}`})},
		{comp: assistantComp("the report")},
	}}
	exec := NewSubagentTool(SubagentConfig{
		Provider: provider, Model: "m", Tools: subParentExec().registry(),
		SystemPrompt: "  custom brief  ",
	})
	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
	require.NoError(t, err)
	assert.Equal(t, "the report", res.Content)
	require.Len(t, provider.reqs, 3)
	assert.Equal(t, "custom brief", provider.reqs[0].System)
	for i, r := range provider.reqs {
		assert.NotEmpty(t, r.Tools, "turn %d keeps its tools", i+1)
	}
}

func TestSubagentNoOutputPlaceholder(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: &Completion{Message: Message{Role: RoleAssistant}, StopReason: StopEndTurn}},
	}}
	exec := NewSubagentTool(SubagentConfig{Provider: provider, Model: "m"})
	res, err := exec.Execute(context.Background(), subCall(`{"prompt":"p"}`))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, noOutputPlaceholder, res.Content,
		"the nested Run's fallback placeholder surfaces as the report")
}
