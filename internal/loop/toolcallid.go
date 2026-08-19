package loop

import "context"

// A Tool executes with its arguments and nothing else -- the interface stays
// id-free, because the id of the call being answered is meaningless to almost
// every tool. The few that need it (the sub-agent tool stamps its live
// activity with the parent call's id, so a host can attach the play-by-play to
// the right tool block) read it off the context, which Run puts it on.

// toolCallIDKey is the context key carrying the id of the tool call currently
// being executed.
type toolCallIDKey struct{}

// WithToolCallID carries the id of the tool call being executed. Run sets it
// around every Tool.Execute; a host running a tool outside the loop sets it
// only if that tool needs it.
func WithToolCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, toolCallIDKey{}, id)
}

// ToolCallID is the id of the tool call being executed, or "" outside one.
func ToolCallID(ctx context.Context) string {
	id, _ := ctx.Value(toolCallIDKey{}).(string)
	return id
}
