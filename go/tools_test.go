package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeExec scripts a set of tools for tests: it declares them, records what
// ran, and hands out the individual Tool values a Config takes.
type fakeExec struct {
	tools    []ToolDecl
	execute  func(ctx context.Context, call ToolCall) (ToolResult, error)
	executed []ToolCall
	results  map[string]ToolResult
}

// registry is the flat toolset, one Tool per declaration.
func (f *fakeExec) registry() Tools {
	var out Tools
	for _, d := range f.tools {
		out = append(out, &fakeTool{owner: f, decl: d})
	}
	return out
}

// fakeTool is one of a fakeExec's tools.
type fakeTool struct {
	owner *fakeExec
	decl  ToolDecl
}

func (t *fakeTool) Decl() ToolDecl { return t.decl }

func (t *fakeTool) Execute(ctx context.Context, args json.RawMessage) (ToolResult, error) {
	call := ToolCall{ID: ToolCallID(ctx), Name: t.decl.Name, Arguments: string(args)}
	t.owner.executed = append(t.owner.executed, call)
	if t.owner.execute != nil {
		return t.owner.execute(ctx, call)
	}
	if r, ok := t.owner.results[t.decl.Name]; ok {
		return r, nil
	}
	return ToolResult{Content: "ran " + t.decl.Name}, nil
}

func TestToolsDeclsAndFind(t *testing.T) {
	f := &fakeExec{tools: []ToolDecl{{Name: "alpha", Readonly: true}, {Name: ""}, {Name: "beta"}}}
	reg := f.registry()

	assert.Equal(t, []string{"alpha", "beta"}, reg.Names(), "an unnamed tool is skipped, never advertised")
	require.Len(t, reg.Decls(), 2)

	tool, ok := reg.Find("alpha")
	require.True(t, ok)
	res, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Equal(t, "ran alpha", res.Content)

	_, ok = reg.Find("nope")
	assert.False(t, ok, "an unoffered name simply is not there; the loop teaches the model")
}

// Two sources concatenate, and the first to claim a name answers it -- so a
// host appending its own tools to the library's gets a deterministic toolset
// rather than one that depends on map iteration.
func TestFindResolvesToTheFirstClaimant(t *testing.T) {
	a := &fakeExec{tools: []ToolDecl{{Name: "alpha"}}}
	b := &fakeExec{tools: []ToolDecl{{Name: "alpha"}, {Name: "beta"}}}
	reg := append(a.registry(), b.registry()...)

	tool, ok := reg.Find("alpha")
	require.True(t, ok)
	_, err := tool.Execute(context.Background(), nil)
	require.NoError(t, err)
	assert.Len(t, a.executed, 1)
	assert.Empty(t, b.executed, "the later duplicate never runs")
}

func TestToolsEmpty(t *testing.T) {
	var none Tools
	assert.Nil(t, none.Decls())
	assert.Nil(t, none.Names())
	assert.Empty(t, none.Readonly())
	assert.Empty(t, none.Subset([]string{"anything"}))
	_, ok := none.Find("anything")
	assert.False(t, ok)
}

func TestToolsReadonly(t *testing.T) {
	f := &fakeExec{tools: []ToolDecl{{Name: "read", Readonly: true}, {Name: "write"}, {Name: "", Readonly: true}}}
	ro := f.registry().Readonly()

	assert.Equal(t, []string{"read"}, ro.Names())
	_, ok := ro.Find("write")
	assert.False(t, ok, "a mutating tool is absent from the set, not merely refused by it")

	assert.Empty(t, (&fakeExec{tools: []ToolDecl{{Name: "write"}}}).registry().Readonly())
}

func TestToolsSubset(t *testing.T) {
	f := &fakeExec{tools: []ToolDecl{{Name: "one"}, {Name: "two"}, {Name: "three"}}}
	reg := f.registry()

	// The order is the registry's, not the caller's: the advertised list is
	// part of the prompt-cache prefix, so it must not depend on argument order.
	assert.Equal(t, []string{"one", "two"}, reg.Subset([]string{"two", "one", "missing"}).Names())
	assert.Empty(t, reg.Subset(nil))
	assert.Empty(t, reg.Subset([]string{"missing"}))
}

// A tool's own Go error is the caller's to see: nothing between the loop and
// the tool swallows or reshapes it.
func TestToolExecuteErrorPassthrough(t *testing.T) {
	boom := errors.New("boom")
	f := &fakeExec{
		tools:   []ToolDecl{{Name: "explode"}},
		execute: func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, boom },
	}
	tool, ok := f.registry().Find("explode")
	require.True(t, ok)
	_, err := tool.Execute(context.Background(), nil)
	assert.ErrorIs(t, err, boom)
}

// NewTool is the shorthand a host uses for a plain function tool.
func TestNewTool(t *testing.T) {
	tool := NewTool(ToolDecl{Name: "echo", Readonly: true},
		func(_ context.Context, args json.RawMessage) (ToolResult, error) {
			return ToolResult{Content: string(args)}, nil
		})
	assert.Equal(t, "echo", tool.Decl().Name)
	assert.True(t, tool.Decl().Readonly, "gating is Config.Approver's, and Readonly is all a tool declares about it")

	res, err := tool.Execute(context.Background(), json.RawMessage(`{"a":1}`))
	require.NoError(t, err)
	assert.Equal(t, `{"a":1}`, res.Content)
}

// The id of the call being answered reaches a tool that needs it (the
// sub-agent tool stamps its telemetry with it) without the interface carrying
// an argument almost no tool wants.
func TestToolCallIDRidesTheContext(t *testing.T) {
	assert.Empty(t, ToolCallID(context.Background()), "no id outside a call")
	assert.Equal(t, "call_7", ToolCallID(WithToolCallID(context.Background(), "call_7")))
}
