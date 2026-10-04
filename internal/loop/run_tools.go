package loop

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
)

// batchFingerprint identifies a batch by call names and raw arguments, in order.
func batchFingerprint(calls []ToolCall) string {
	var b strings.Builder
	for _, c := range calls {
		fmt.Fprintf(&b, "%d:%s|%d:%s|", len(c.Name), c.Name, len(c.Arguments), c.Arguments)
	}
	return b.String()
}

// resolveCall produces the recorded ToolResult for requested call:
// executed, denied, or a teaching error. EVERY call is put to the Approver --
// a tool has no say in whether it is asked about, so a host's deny rule
// reaches read-only calls too. The returned error is non-nil ONLY when an
// approval decision never arrived (Approver.Ask failed), which ends the run.
func resolveCall(ctx context.Context, cfg *Config, call ToolCall) (ToolResult, error) {
	tool, known := cfg.Tools.Find(call.Name)
	if !known {
		// A name this run does not offer: the model hallucinated a tool; teach it.
		text := "unknown tool: " + call.Name
		if cfg.UnknownTool != nil {
			text = cfg.UnknownTool(call.Name)
		}
		return ToolResult{Content: text, IsError: true}, nil
	}
	if cfg.Approver == nil {
		// Nobody to ask: read, but change nothing.
		if !tool.Decl().Readonly {
			return ToolResult{Content: DeniedMessage, IsError: true}, nil
		}
	} else {
		verdict, aerr := cfg.Approver.Ask(ctx, call)
		if aerr != nil {
			return ToolResult{}, fmt.Errorf("agentic: tool approval interrupted: %w", aerr)
		}
		if !verdict.OK {
			return ToolResult{Content: deniedText(verdict.Reason), IsError: true}, nil
		}
	}
	// The id is threaded on the context; the Tool interface stays id-free.
	result, exErr := tool.Execute(WithToolCallID(ctx, call.ID), json.RawMessage(call.Arguments))
	if exErr != nil {
		// Defensive: internal failures surface as tool text so the model can react.
		return ToolResult{Content: "tool execution failed: " + exErr.Error(), IsError: true}, nil
	}
	return result, nil
}

// deniedText is what a refused call records: the approver's own reason, and
// DeniedMessage when it gave none.
func deniedText(reason string) string {
	if r := strings.TrimSpace(reason); r != "" {
		return r
	}
	return DeniedMessage
}
