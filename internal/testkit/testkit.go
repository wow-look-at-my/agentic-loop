// Package testkit holds fixtures shared by the library's optional-package tests.
// It is not part of the public API.
package testkit

import (
	"context"
	"encoding/json"
	"errors"

	agentic "github.com/wow-look-at-my/agentic-loop"
	"github.com/wow-look-at-my/go-containers/set"
)

// ScriptStep is scripted provider response.
type ScriptStep struct {
	Comp *agentic.Completion
	Err  error
	Emit func(ev *agentic.StreamEvents)
}

// ScriptProvider replays scripted responses and records every request.
type ScriptProvider struct {
	Steps []ScriptStep
	Reqs  []agentic.Request
}

// Complete implements agentic.Provider.
func (p *ScriptProvider) Complete(_ context.Context, req agentic.Request, ev *agentic.StreamEvents) (*agentic.Completion, error) {
	p.Reqs = append(p.Reqs, req)
	if len(p.Steps) == 0 {
		return nil, errors.New("script exhausted")
	}
	step := p.Steps[0]
	p.Steps = p.Steps[1:]
	if step.Emit != nil {
		step.Emit(ev)
	}
	return step.Comp, step.Err
}

// AssistantComp is a one-shot assistant completion for tests.
func AssistantComp(content string, calls ...agentic.ToolCall) *agentic.Completion {
	stop := agentic.StopEndTurn
	if len(calls) > 0 {
		stop = agentic.StopToolUse
	}
	return &agentic.Completion{
		Message:    agentic.Message{Role: agentic.RoleAssistant, Content: content, ToolCalls: calls},
		Usage:      agentic.Usage{PromptTokens: 10, CompletionTokens: 5, TotalTokens: 15},
		StopReason: stop,
	}
}

// FakeExec scripts a set of tools for tests.
type FakeExec struct {
	Tools    []agentic.ToolDecl
	Ask      set.Set[string]
	Execute  func(ctx context.Context, call agentic.ToolCall) (agentic.ToolResult, error)
	Executed []agentic.ToolCall
	Results  map[string]agentic.ToolResult
}

// Registry is the flat toolset, Tool per declaration.
func (f *FakeExec) Registry() agentic.Tools {
	var out agentic.Tools
	for _, d := range f.Tools {
		out = append(out, &fakeTool{owner: f, decl: d})
	}
	return out
}

type fakeTool struct {
	owner *FakeExec
	decl  agentic.ToolDecl
}

func (t *fakeTool) Decl() agentic.ToolDecl { return t.decl }
func (t *fakeTool) NeedsApproval() bool    { return t.owner.Ask.Contains(t.decl.Name) }

func (t *fakeTool) Execute(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	call := agentic.ToolCall{ID: agentic.ToolCallID(ctx), Name: t.decl.Name, Arguments: string(args)}
	t.owner.Executed = append(t.owner.Executed, call)
	if t.owner.Execute != nil {
		return t.owner.Execute(ctx, call)
	}
	if r, ok := t.owner.Results[t.decl.Name]; ok {
		return r, nil
	}
	return agentic.ToolResult{Content: "ran " + t.decl.Name}, nil
}
