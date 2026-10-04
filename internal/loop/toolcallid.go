package loop

import "context"

// A Tool executes with its arguments and nothing else; the interface stays id-free.

// toolCallIDKey is the context key carrying the id of the tool call being executed.
type toolCallIDKey struct{}

// WithToolCallID carries the id of the tool call being executed.
func WithToolCallID(ctx context.Context, id string) context.Context {
	return context.WithValue(ctx, toolCallIDKey{}, id)
}

// ToolCallID is the id of the tool call being executed, or "" outside.
func ToolCallID(ctx context.Context) string {
	id, _ := ctx.Value(toolCallIDKey{}).(string)
	return id
}
