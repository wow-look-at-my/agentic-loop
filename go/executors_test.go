package agentic

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeExec is a scriptable ToolExecutor for tests.
type fakeExec struct {
	tools    []Tool
	ask      map[string]bool
	execute  func(ctx context.Context, call ToolCall) (ToolResult, error)
	executed []ToolCall
}

func (f *fakeExec) Tools() []Tool { return f.tools }

func (f *fakeExec) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	f.executed = append(f.executed, call)
	if f.execute != nil {
		return f.execute(ctx, call)
	}
	return ToolResult{Content: "ran " + call.Name}, nil
}

func (f *fakeExec) NeedsApproval(name string) bool { return f.ask[name] }

func TestNewComposite(t *testing.T) {
	a := &fakeExec{tools: []Tool{{Name: "alpha", Readonly: true}, {Name: ""}}}
	b := &fakeExec{tools: []Tool{{Name: "alpha"}, {Name: "beta"}}, ask: map[string]bool{"beta": true}}

	c := NewComposite(nil, a, b)
	require.NotNil(t, c)

	names := make([]string, 0)
	for _, tool := range c.Tools() {
		names = append(names, tool.Name)
	}
	assert.Equal(t, []string{"alpha", "beta"}, names, "first registration wins; empty names skipped")

	res, err := c.Execute(context.Background(), ToolCall{ID: "1", Name: "alpha"})
	require.NoError(t, err)
	assert.Equal(t, "ran alpha", res.Content)
	assert.Len(t, a.executed, 1, "alpha routed to its first registrant")
	assert.Empty(t, b.executed)

	res, err = c.Execute(context.Background(), ToolCall{ID: "2", Name: "nope"})
	require.NoError(t, err, "unknown tool is a recoverable result, not a Go error")
	assert.True(t, res.IsError)
	assert.Equal(t, "unknown tool: nope", res.Content)

	assert.True(t, c.NeedsApproval("beta"))
	assert.False(t, c.NeedsApproval("alpha"))
	assert.False(t, c.NeedsApproval("nope"), "unknown names need no approval")
}

func TestNewCompositeEmpty(t *testing.T) {
	assert.Nil(t, NewComposite())
	assert.Nil(t, NewComposite(nil, &fakeExec{}))
	assert.Nil(t, NewComposite(&fakeExec{tools: []Tool{{Name: ""}}}))
}

func TestReadonlyView(t *testing.T) {
	inner := &fakeExec{
		tools: []Tool{{Name: "read", Readonly: true}, {Name: "write"}, {Name: "", Readonly: true}},
		ask:   map[string]bool{"read": true, "write": true},
	}
	v := ReadonlyView(inner)
	require.NotNil(t, v)
	require.Len(t, v.Tools(), 1)
	assert.Equal(t, "read", v.Tools()[0].Name)

	res, err := v.Execute(context.Background(), ToolCall{Name: "write"})
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "tool not available to subagent (read-only tools only): write", res.Content)
	assert.Empty(t, inner.executed, "refused before reaching the inner executor")

	res, err = v.Execute(context.Background(), ToolCall{Name: "read"})
	require.NoError(t, err)
	assert.Equal(t, "ran read", res.Content)

	assert.True(t, v.NeedsApproval("read"), "mirrors the inner flag for allowed tools")
	assert.False(t, v.NeedsApproval("write"), "never gates a tool it does not expose")

	assert.Nil(t, ReadonlyView(nil))
	assert.Nil(t, ReadonlyView(&fakeExec{tools: []Tool{{Name: "write"}}}), "no read-only tools collapses to nil")
}

func TestSubsetView(t *testing.T) {
	inner := &fakeExec{
		tools: []Tool{{Name: "one"}, {Name: "two"}, {Name: "three"}},
		ask:   map[string]bool{"one": true, "three": true},
	}
	v := SubsetView(inner, []string{"one", "two", "missing"})
	require.NotNil(t, v)
	names := []string{v.Tools()[0].Name, v.Tools()[1].Name}
	assert.Equal(t, []string{"one", "two"}, names)

	res, err := v.Execute(context.Background(), ToolCall{Name: "three"})
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "tool not in the sub-agent's allowed set: three", res.Content)

	res, err = v.Execute(context.Background(), ToolCall{Name: "one"})
	require.NoError(t, err)
	assert.Equal(t, "ran one", res.Content)

	assert.True(t, v.NeedsApproval("one"))
	assert.False(t, v.NeedsApproval("three"), "outside the subset is never gated")

	assert.Nil(t, SubsetView(nil, []string{"one"}))
	assert.Nil(t, SubsetView(inner, nil))
	assert.Nil(t, SubsetView(inner, []string{"missing"}))
}

func TestCompositeExecuteErrorPassthrough(t *testing.T) {
	boom := errors.New("boom")
	inner := &fakeExec{
		tools:   []Tool{{Name: "explode"}},
		execute: func(context.Context, ToolCall) (ToolResult, error) { return ToolResult{}, boom },
	}
	c := NewComposite(inner)
	require.NotNil(t, c)
	_, err := c.Execute(context.Background(), ToolCall{Name: "explode"})
	assert.ErrorIs(t, err, boom)
}
