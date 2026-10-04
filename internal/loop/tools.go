package loop

import (
	"context"
	"encoding/json"

	"github.com/wow-look-at-my/go-containers/set"
)

// A tool is an individual thing; every tool is Tool value in a flat slice.

// Tool is callable tool; whether a call may run is the Approver's, not the tool's.
type Tool interface {
	// Decl is what the model is told: name, description, schema, and read-only.
	Decl() ToolDecl
	// Execute runs the tool with raw JSON arguments.
	Execute(ctx context.Context, args json.RawMessage) (ToolResult, error)
}

// Tools is the flat set run offers the model; order is deterministic.
type Tools []Tool

// Decls is the advertised declarations, in order. Tools with an empty name are
// skipped: a provider rejects them, and malformed entry must not fail
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

// Find resolves an advertised name; the tool to claim a name wins.
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
	keep := set.Of(names...)
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

// NewTool builds a Tool from a declaration and the function that runs it.
func NewTool(decl ToolDecl, run func(ctx context.Context, args json.RawMessage) (ToolResult, error)) Tool {
	return funcTool{decl: decl, run: run}
}

func (t funcTool) Decl() ToolDecl { return t.decl }

func (t funcTool) Execute(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	return t.run(ctx, args)
}
