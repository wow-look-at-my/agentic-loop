package agentic

import "context"

// composite routes tool calls across multiple executors while presenting one
// deterministic tool list to the model.
type composite struct {
	advertised []Tool
	byName     map[string]ToolExecutor
}

// NewComposite combines non-empty executors into one ToolExecutor. Nil
// executors are skipped, tools with empty names are skipped, and the first
// executor to advertise a name wins (later duplicates are ignored). The
// advertised order is the iteration order across executors. If no tools
// remain, NewComposite returns nil so callers can treat "no tools" as the
// single-turn case.
func NewComposite(execs ...ToolExecutor) ToolExecutor {
	c := &composite{byName: map[string]ToolExecutor{}}
	for _, ex := range execs {
		if ex == nil {
			continue
		}
		for _, t := range ex.Tools() {
			if t.Name == "" {
				continue
			}
			if _, exists := c.byName[t.Name]; exists {
				continue
			}
			c.byName[t.Name] = ex
			c.advertised = append(c.advertised, t)
		}
	}
	if len(c.advertised) == 0 {
		return nil
	}
	return c
}

// Tools returns the combined advertised tool list.
func (c *composite) Tools() []Tool { return c.advertised }

// Execute routes the call to the executor that advertised its name. An
// unknown name is a recoverable error tool result, never a Go error.
func (c *composite) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	ex, ok := c.byName[call.Name]
	if !ok {
		return ToolResult{Content: "unknown tool: " + call.Name, IsError: true}, nil
	}
	return ex.Execute(ctx, call)
}

// NeedsApproval routes to the executor that advertised the named tool, so its
// own "always ask" set decides. Unknown names need no approval.
func (c *composite) NeedsApproval(name string) bool {
	ex, ok := c.byName[name]
	if !ok {
		return false
	}
	return ex.NeedsApproval(name)
}

// readonlyView exposes only the read-only tools of an underlying executor.
type readonlyView struct {
	inner      ToolExecutor
	advertised []Tool
	allowed    map[string]bool
}

// ReadonlyView returns an executor that advertises and routes only the
// read-only tools of e (those with Tool.Readonly set). It returns nil when e
// is nil or has no read-only tools, so callers can treat "no read-only tools"
// the same as "no tools".
func ReadonlyView(e ToolExecutor) ToolExecutor {
	if e == nil {
		return nil
	}
	v := &readonlyView{inner: e, allowed: map[string]bool{}}
	for _, t := range e.Tools() {
		if !t.Readonly || t.Name == "" {
			continue
		}
		v.advertised = append(v.advertised, t)
		v.allowed[t.Name] = true
	}
	if len(v.advertised) == 0 {
		return nil
	}
	return v
}

// Tools returns only the read-only tools.
func (v *readonlyView) Tools() []Tool { return v.advertised }

// NeedsApproval mirrors the underlying executor's flag for an allowed tool.
func (v *readonlyView) NeedsApproval(name string) bool {
	return v.allowed[name] && v.inner.NeedsApproval(name)
}

// Execute routes to the underlying executor, refusing any tool outside the
// read-only allow-list — defense in depth so a restricted agent can never
// reach a mutating tool even if it names one that was not advertised to it.
func (v *readonlyView) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if !v.allowed[call.Name] {
		return ToolResult{Content: "tool not available to subagent (read-only tools only): " + call.Name, IsError: true}, nil
	}
	return v.inner.Execute(ctx, call)
}

// subsetView exposes only an explicitly named subset of an executor's tools.
type subsetView struct {
	inner      ToolExecutor
	advertised []Tool
	allowed    map[string]bool
}

// SubsetView returns an executor that advertises and routes only the tools of
// e whose advertised name appears in names. It returns nil when e is nil,
// names is empty, or none of e's tools match — so callers can treat "nothing
// selected" the same as "no tools".
func SubsetView(e ToolExecutor, names []string) ToolExecutor {
	if e == nil || len(names) == 0 {
		return nil
	}
	keep := make(map[string]bool, len(names))
	for _, n := range names {
		keep[n] = true
	}
	v := &subsetView{inner: e, allowed: map[string]bool{}}
	for _, t := range e.Tools() {
		if t.Name == "" || !keep[t.Name] {
			continue
		}
		v.advertised = append(v.advertised, t)
		v.allowed[t.Name] = true
	}
	if len(v.advertised) == 0 {
		return nil
	}
	return v
}

// Tools returns only the kept subset.
func (v *subsetView) Tools() []Tool { return v.advertised }

// NeedsApproval mirrors the underlying executor's flag for a kept tool.
func (v *subsetView) NeedsApproval(name string) bool {
	return v.allowed[name] && v.inner.NeedsApproval(name)
}

// Execute routes to the underlying executor, refusing any tool outside the
// allowed subset.
func (v *subsetView) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if !v.allowed[call.Name] {
		return ToolResult{Content: "tool not in the sub-agent's allowed set: " + call.Name, IsError: true}, nil
	}
	return v.inner.Execute(ctx, call)
}
