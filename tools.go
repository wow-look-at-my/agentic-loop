package agentic

import (
	"context"
	"encoding/json"
	"github.com/wow-look-at-my/go-containers/set"
)

// A tool is an individual thing, and nothing groups them.
//
// Every tool the model is offered -- a built-in of this library, or one a host
// discovered on an MCP server -- is one Tool value in a flat Tools slice. The
// loop resolves a requested name against that slice and can no more tell the
// kinds apart than the model can. There is no executor, no routing table, and
// no wrapper whose only job is to hide part of another wrapper: restricting a
// toolset is filtering a slice.

// Tool is ONE callable tool: what the model is told about it, whether each
// call must be approved first, and how to run it.
//
// Execute should return a ToolResult with IsError set for recoverable,
// model-facing failures and reserve the Go error for internal faults; Run
// converts an Execute error into an error tool result rather than aborting the
// loop.
type Tool interface {
	// Decl is what the model is told: name, description, schema, and whether
	// the tool only reads.
	Decl() ToolDecl
	// Execute runs the tool with raw JSON arguments.
	Execute(ctx context.Context, args json.RawMessage) (ToolResult, error)
	// NeedsApproval reports whether each call must be approved by the user
	// first. The tool is still advertised; only its execution is gated.
	NeedsApproval() bool
}

// Tools is the flat set one run offers the model. Order is the advertised
// order and must be deterministic -- it is part of the prompt-cache prefix.
type Tools []Tool

// Decls is the advertised declarations, in order. Tools with an empty name are
// skipped: a provider rejects them, and one malformed entry must not fail
// every turn.
func (ts Tools) Decls() []ToolDecl {
	if len(ts) == 0 {
		return nil
	}
	out := make([]ToolDecl, 0, len(ts))
	for _, t := range ts {
		if t == nil {
			continue
		}
		if d := t.Decl(); d.Name != "" {
			out = append(out, d)
		}
	}
	return out
}

// Find resolves an advertised name. The first tool to claim a name wins, so a
// host that concatenates two sources gets a deterministic answer rather than a
// silent overwrite.
func (ts Tools) Find(name string) (Tool, bool) {
	for _, t := range ts {
		if t != nil && t.Decl().Name == name {
			return t, true
		}
	}
	return nil, false
}

// Readonly is the subset that only reads state -- the default toolset a
// sub-agent is offered.
func (ts Tools) Readonly() Tools {
	var out Tools
	for _, t := range ts {
		if t == nil {
			continue
		}
		if d := t.Decl(); d.Readonly && d.Name != "" {
			out = append(out, t)
		}
	}
	return out
}

// Subset is the tools whose advertised name appears in names, in THIS slice's
// order (not the caller's), so the advertised list stays deterministic.
func (ts Tools) Subset(names []string) Tools {
	if len(names) == 0 {
		return nil
	}
	keep := set.Of[string](names...)
	var out Tools
	for _, t := range ts {
		if t == nil {
			continue
		}
		if d := t.Decl(); d.Name != "" && keep.Contains(d.Name) {
			out = append(out, t)
		}
	}
	return out
}

// Names is the advertised names, in order.
func (ts Tools) Names() []string {
	decls := ts.Decls()
	if len(decls) == 0 {
		return nil
	}
	out := make([]string, len(decls))
	for i, d := range decls {
		out[i] = d.Name
	}
	return out
}

// funcTool is a Tool built from a declaration and a function.
type funcTool struct {
	decl ToolDecl
	run  func(context.Context, json.RawMessage) (ToolResult, error)
}

// NewTool builds a Tool from a declaration and the function that runs it. The
// result never needs approval: a host that gates a tool decides that from its
// own settings, so it implements Tool itself rather than asking a built-in to
// carry a policy it cannot know.
func NewTool(decl ToolDecl, run func(ctx context.Context, args json.RawMessage) (ToolResult, error)) Tool {
	return funcTool{decl: decl, run: run}
}

func (t funcTool) Decl() ToolDecl      { return t.decl }
func (t funcTool) NeedsApproval() bool { return false }

func (t funcTool) Execute(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	return t.run(ctx, args)
}
