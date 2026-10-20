package loop

import (
	"context"
	"fmt"
	"strings"
)

// ReplayProbe is the answer of ProbeReasoningReplay. Completions holds both
// probe calls, so a host can bill them.
type ReplayProbe struct {
	// Kept is true when the endpoint counted the replayed reasoning.
	Kept bool
	// WithReasoning and WithoutReasoning are the prompt tokens the endpoint reported for each call.
	WithReasoning    int
	WithoutReasoning int
	// MinGrowth is the smallest difference a faithful endpoint reports.
	MinGrowth   int
	Completions []*Completion
}

// probeReasoning is long enough that every tokenizer counts it as hundreds of tokens.
var probeReasoning = strings.Repeat("The probe reasons about a tool result before it answers. ", 40)

// ProbeReasoningReplay asks the endpoint behind p whether it reads replayed
// reasoning. It sends one transcript twice, with and without an assistant
// reasoning block, and compares the prompt tokens the endpoint reports. Each
// call asks for one output token. It is for chat-completions endpoints: the
// Anthropic and Responses dialects replay only a block that carries the
// provider's own signature or payload. A synthetic block reads as dropped.
func ProbeReasoningReplay(ctx context.Context, p Provider, model string) (*ReplayProbe, error) {
	if p == nil {
		return nil, badRequestErr("agentic: ProbeReasoningReplay requires a Provider")
	}
	transcript := func(thinking []ThinkingBlock) []Message {
		return []Message{
			{Role: RoleUser, Content: "Look up the value, then answer with one word."},
			{
				Role:      RoleAssistant,
				Thinking:  thinking,
				ToolCalls: []ToolCall{{ID: "probe_1", Name: "lookup", Arguments: "{}"}},
			},
			{Role: RoleTool, Content: "42", ToolCallID: "probe_1"},
		}
	}
	tools := []ToolDecl{{Name: "lookup", Description: "Look up the value.", Readonly: true}}
	call := func(thinking []ThinkingBlock) (*Completion, error) {
		return p.Complete(ctx, Request{Model: model, MaxTokens: 1, Messages: transcript(thinking), Tools: tools}, nil)
	}
	probe := &ReplayProbe{MinGrowth: len(probeReasoning) / stateCharsPerToken}
	with, err := call([]ThinkingBlock{{Text: probeReasoning}})
	if err != nil {
		return nil, fmt.Errorf("agentic: reasoning probe, call with reasoning: %w", err)
	}
	without, err := call(nil)
	if err != nil {
		return nil, fmt.Errorf("agentic: reasoning probe, call without reasoning: %w", err)
	}
	probe.Completions = []*Completion{with, without}
	if !with.UsageReported || !without.UsageReported {
		return probe, fmt.Errorf("agentic: reasoning probe: the endpoint reported no usage, so the probe has no evidence")
	}
	probe.WithReasoning = with.Usage.PromptTokens
	probe.WithoutReasoning = without.Usage.PromptTokens
	probe.Kept = probe.WithReasoning-probe.WithoutReasoning >= probe.MinGrowth
	return probe, nil
}
