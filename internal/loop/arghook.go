// arghook.go lets a tool act on an argument the moment it has streamed whole.

package loop

import (
	"context"
	"encoding/json"
	"errors"
)

// ArgHook watches one top-level argument member of a call.
type ArgHook struct {
	Key string
	// On gets the member's raw JSON value.
	On func(ctx context.Context, value json.RawMessage) *ToolResult
}

// ArgWatcher is a Tool with argument hooks.
type ArgWatcher interface {
	Tool
	ArgHooks() []ArgHook
}

// WithArgHooks gives t hooks on its arguments. The tool keeps its declaration and Execute.
func WithArgHooks(t Tool, hooks ...ArgHook) Tool {
	return argHookTool{Tool: t, hooks: hooks}
}

type argHookTool struct {
	Tool
	hooks []ArgHook
}

func (t argHookTool) ArgHooks() []ArgHook { return t.hooks }

// OnArg is a hook that decodes the member as T. A value that does not decode is left to Execute,
// which reports it with the rest of the arguments.
func OnArg[T any](key string, fn func(ctx context.Context, v T) *ToolResult) ArgHook {
	return ArgHook{Key: key, On: func(ctx context.Context, raw json.RawMessage) *ToolResult {
		var v T
		if json.Unmarshal(raw, &v) != nil {
			return nil
		}
		return fn(ctx, v)
	}}
}

// errArgRefused is the sink error that ends a model call on a hook's refusal.
var errArgRefused = errors.New("agentic: a tool refused the call from a streamed argument")

// argHooksFn names the hooks of the tool a call names; nil watches nothing.
type argHooksFn func(name string) []ArgHook

// argHooksFor is the sink's hook lookup for a turn. It is nil when no advertised tool has a hook,
// so a turn without one scans nothing.
func argHooksFor(tools Tools, advertised []ToolDecl) argHooksFn {
	watched := false
	for _, d := range advertised {
		if t, ok := tools.Find(d.Name); ok {
			if w, ok := t.(ArgWatcher); ok && len(w.ArgHooks()) > 0 {
				watched = true
				break
			}
		}
	}
	if !watched {
		return nil
	}
	return func(name string) []ArgHook {
		t, ok := tools.Find(name)
		if !ok {
			return nil
		}
		if w, ok := t.(ArgWatcher); ok {
			return w.ArgHooks()
		}
		return nil
	}
}
