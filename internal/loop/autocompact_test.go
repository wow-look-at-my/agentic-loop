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
	// Turn 1: the model answers with tool calls, reporting prompt tokens at
	// the threshold. The loop compacts before executing the tools.
	// Turn 2 (the compaction call): returns a summary.
	// Turn 3 (after compaction): the model answers cleanly.
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
	require.Len(t, compactionMessages, 2, "the replacement round is two messages")
	assert.Equal(t, RoleUser, compactionMessages[0].Role)
	assert.Equal(t, CompactRequestText, compactionMessages[0].Content)
	assert.Equal(t, RoleAssistant, compactionMessages[1].Role)
	assert.Equal(t, "this is the summary", compactionMessages[1].Content)

	// The transcript is the compacted round plus the post-compaction answer.
	require.Len(t, res.Messages, 3)
	assert.Equal(t, CompactRequestText, res.Messages[0].Content)
	assert.Equal(t, "this is the summary", res.Messages[1].Content)
	assert.Equal(t, "final answer after compaction", res.Messages[2].Content)

	// Two numbered turns (the summarize call does not increment res.Turns).
	assert.Equal(t, 2, res.Turns)
	// The first turn's tool calls were NOT executed (compaction replaced the transcript first).
	assert.Empty(t, exec.executed, "the stale tool calls were not executed")
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
