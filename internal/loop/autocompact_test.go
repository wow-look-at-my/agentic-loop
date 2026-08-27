package loop

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A provider whose every completion reports prompt tokens at the configured
// count, so the threshold can be exercised without a real model.
func usageComp(content string, promptTokens int, calls ...ToolCall) *Completion {
	stop := StopEndTurn
	if len(calls) > 0 {
		stop = StopToolUse
	}
	return &Completion{
		Message:       Message{Role: RoleAssistant, Content: content, ToolCalls: calls},
		Usage:         Usage{PromptTokens: promptTokens, CompletionTokens: 5, TotalTokens: promptTokens + 5},
		UsageReported: true,
		StopReason:    stop,
	}
}

func TestShouldCompact(t *testing.T) {
	tests := []struct {
		name        string
		autoCompact float64
		window      int
		prompt      int
		want        bool
	}{
		{"at threshold", 0.8, 10000, 8000, true},
		{"above threshold", 0.8, 10000, 9000, true},
		{"below threshold", 0.8, 10000, 7999, false},
		{"autoCompact zero disables", 0, 10000, 99999, false},
		{"window zero disables", 0.8, 0, 99999, false},
		{"autoCompact negative disables", -0.1, 10000, 99999, false},
		{"exactly 80 percent of 200k", 0.8, 200000, 160000, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := Request{AutoCompact: tt.autoCompact}
			cfg := Config{ContextWindow: tt.window}
			comp := usageComp("ok", tt.prompt)
			assert.Equal(t, tt.want, shouldCompact(req, cfg, comp))
		})
	}
}

func TestShouldCompactNoUsageReported(t *testing.T) {
	comp := &Completion{
		Message:       Message{Role: RoleAssistant, Content: "ok"},
		Usage:         Usage{PromptTokens: 99999},
		UsageReported: false,
	}
	assert.False(t, shouldCompact(Request{AutoCompact: 0.8}, Config{ContextWindow: 10000}, comp),
		"a server that reported no usage never triggers compaction")
}

func TestShouldCompactNilCompletion(t *testing.T) {
	assert.False(t, shouldCompact(Request{AutoCompact: 0.8}, Config{ContextWindow: 10000}, nil))
}

func TestRunAutoCompactTriggersAndReplacesTranscript(t *testing.T) {
	// Turn 1: the model asks for a tool, reporting prompt tokens at the
	// threshold. The tool RUNS, and the loop compacts at the turn boundary.
	// The summarize call: returns a summary.
	// Turn 2 (after compaction): the model answers cleanly.
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("here is my answer", 8000,
			ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("this is the summary")},
		{comp: assistantComp("final answer after compaction")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	var compacted bool
	var compactionMessages []Message
	events := Events{}
	compactionCb := func(ev CompactionEvent) error {
		compacted = true
		compactionMessages = ev.Messages
		return nil
	}
	events.OnCompaction.Subscribe(&compactionCb)
	cfg := Config{
		Provider:      provider,
		Tools:         exec.registry(),
		Approver:      allowAll,
		Events:        &events,
		ContextWindow: 10000,
	}
	req := Request{
		Model:       "m",
		AutoCompact: 0.8,
		Messages:    []Message{{Role: RoleUser, Content: "go"}},
	}

	res, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	assert.True(t, compacted, "compaction fired")
	require.Len(t, compactionMessages, 1, "the replacement is ONE handoff message")
	assert.Equal(t, RoleUser, compactionMessages[0].Role)
	assert.Equal(t, CompactionKind, compactionMessages[0].Kind)
	assert.Equal(t, CompactionHandoffPrefix+"this is the summary", compactionMessages[0].Content)

	// The transcript is the summary plus the post-compaction answer.
	require.Len(t, res.Messages, 2)
	assert.Contains(t, res.Messages[0].Content, "this is the summary")
	assert.Equal(t, "final answer after compaction", res.Messages[1].Content)

	// Two numbered turns (the summarize call does not increment res.Turns).
	assert.Equal(t, 2, res.Turns)
	// Compacting before the batch left the host a row nothing ever answered.
	require.Len(t, exec.executed, 1, "the turn that crossed the threshold still ran its tools")
	assert.Equal(t, "alpha", exec.executed[0].Name)
}

// A compacted turn keeps the tool result the model asked for: the summary is
// taken after the batch, so nothing is summarized around an unanswered call.
func TestRunAutoCompactSummarizesAfterTheToolResults(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("working", 8000, ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("the summary")},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll,
		Events: &Events{}, ContextWindow: 10000}
	req := Request{Model: "m", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)

	require.Len(t, provider.reqs, 3)
	summarized := provider.reqs[1].Messages
	require.NotEmpty(t, summarized)
	assert.Equal(t, CompactRequestText, summarized[len(summarized)-1].Content)

	var sawCall, sawResult bool
	for _, m := range summarized {
		if len(m.ToolCalls) > 0 && m.ToolCalls[0].ID == "c1" {
			sawCall = true
		}
		if m.Role == RoleTool && m.ToolCallID == "c1" {
			sawResult = true
		}
	}
	assert.True(t, sawCall, "the summarized transcript includes the turn's tool call")
	assert.True(t, sawResult, "and the result answering it")
}

// The host's stored row id lands on the summary, so the assistant row minted
// after compaction hangs off it instead of detaching from the message tree.
func TestRunAutoCompactAttachesTheNextTurnToTheStoredSummary(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("working", 8000, ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("the summary")},
		{comp: assistantComp("done")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	events := Events{}
	compactionCb := func(ev CompactionEvent) error {
		*ev.ID = "stored-summary-row"
		return nil
	}
	events.OnCompaction.Subscribe(&compactionCb)
	var parents []MessageID
	mintCb := func(ev AssistantMessageEvent) error {
		parents = append(parents, ev.ParentID)
		return nil
	}
	events.OnAssistantMessage.Subscribe(&mintCb)
	cfg := Config{Provider: provider, Tools: exec.registry(), Approver: allowAll,
		Events: &events, ContextWindow: 10000}
	req := Request{Model: "m", AutoCompact: 0.8, Messages: []Message{{Role: RoleUser, Content: "go"}}}

	res, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)
	require.NotEmpty(t, res.Messages)
	assert.Equal(t, "stored-summary-row", res.Messages[0].ID,
		"the summary carries the id the host stored it under")
	require.Len(t, parents, 2)
	assert.Equal(t, MessageID("stored-summary-row"), parents[1],
		"the turn after compaction hangs off the stored summary, not off nothing")
}

func TestRunAutoCompactBelowThresholdDoesNotCompact(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("answer", 1000)}, // well below 8000
	}}
	var compacted bool
	events := Events{}
	cb := func(ev CompactionEvent) error { compacted = true; return nil }
	events.OnCompaction.Subscribe(&cb)
	cfg := Config{
		Provider:      provider,
		Events:        &events,
		ContextWindow: 10000,
	}
	req := Request{Model: "m", AutoCompact: 0.8,
		Messages: []Message{{Role: RoleUser, Content: "go"}}}

	res, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)
	assert.False(t, compacted, "compaction did not fire below threshold")
	assert.Equal(t, 1, res.Turns)
}

func TestRunAutoCompactZeroDisables(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("answer", 99999)},
	}}
	var compacted bool
	events := Events{}
	cb := func(ev CompactionEvent) error { compacted = true; return nil }
	events.OnCompaction.Subscribe(&cb)
	cfg := Config{
		Provider:      provider,
		Events:        &events,
		ContextWindow: 10000,
	}
	req := Request{Model: "m", AutoCompact: 0, // disabled
		Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)
	assert.False(t, compacted, "AutoCompact=0 disables compaction")
}

func TestRunAutoCompactFailureIsNonFatal(t *testing.T) {
	// The compaction call fails (empty summary). The loop must fall through
	// and proceed with the un-compacted transcript rather than erroring.
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("answer with tools", 8000,
			ToolCall{ID: "c1", Name: "alpha", Arguments: "{}"})},
		{comp: assistantComp("   ")}, // empty summary → Compact returns error
		{comp: assistantComp("answer after failed compaction")},
	}}
	exec := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}}}
	var compacted bool
	events := Events{}
	cb := func(ev CompactionEvent) error { compacted = true; return nil }
	events.OnCompaction.Subscribe(&cb)
	cfg := Config{
		Provider:      provider,
		Tools:         exec.registry(),
		Approver:      allowAll,
		Events:        &events,
		ContextWindow: 10000,
	}
	req := Request{Model: "m", AutoCompact: 0.8,
		Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err, "compaction failure is non-fatal")
	assert.False(t, compacted, "the compaction event did not fire")
	// The original tool calls ran (compaction failed, so the loop fell through to normal execution).
	assert.NotEmpty(t, exec.executed, "tools ran after compaction failed")
}

func TestRunAutoCompactWithoutContextWindowDoesNothing(t *testing.T) {
	provider := &scriptProvider{steps: []scriptStep{
		{comp: usageComp("answer", 99999)},
	}}
	var compacted bool
	events := Events{}
	cb := func(ev CompactionEvent) error { compacted = true; return nil }
	events.OnCompaction.Subscribe(&cb)
	cfg := Config{
		Provider: provider,
		Events:   &events,
		// ContextWindow is 0 (unset)
	}
	req := Request{Model: "m", AutoCompact: 0.8,
		Messages: []Message{{Role: RoleUser, Content: "go"}}}

	_, err := Run(context.Background(), cfg, req)
	require.NoError(t, err)
	assert.False(t, compacted, "no ContextWindow means no compaction")
}
