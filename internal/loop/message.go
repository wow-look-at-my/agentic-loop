package loop

import "context"

// ToolContentPart is one structured piece of a tool result: an MCP content
// block, or a block a built-in tool produces for its host. The field names are
// MCP's, and the json tags are the wire shape a host persists and ships to its
// own front end.
//
// It exists so a result can carry an image, a file, or a rendered artifact
// WITHOUT that content re-entering the model's context: the model is fed
// ToolResult.Content and nothing else.
type ToolContentPart struct {
	// Type is the MCP block type -- "text", "image", "audio", "resource_link", "resource".
	Type string `json:"type"`
	// Text is the text of a text block or an embedded text resource.
	Text string `json:"text,omitempty"`
	// Data is base64-encoded bytes for an image, audio, or blob resource.
	Data string `json:"data,omitempty"`
	// MimeType is the media type of Data (or of a resource), when known.
	MimeType string `json:"mime_type,omitempty"`
	// URI identifies a resource_link or embedded resource.
	URI string `json:"uri,omitempty"`
	// Name and Description label a resource_link.
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
}

// ToolResult is a call's outcome; Content is model-facing text; IsError marks recoverable failure.
type ToolResult struct {
	Content string
	// Parts is structured content for the HOST to render; NEVER sent to the model.
	Parts   []ToolContentPart
	IsError bool
}

// Approval is the verdict on one tool call, with the reason for a refusal.
type Approval struct {
	// OK allows the call.
	OK bool
	// Reason, when !OK, is recorded instead of DeniedMessage; empty keeps DeniedMessage.
	Reason string
}

// Approver decides whether a tool call may run; nil Approver is fail-closed.
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (Approval, error)
}

// DeniedMessage is the tool-result text recorded when the user denies a tool.
const DeniedMessage = "The user denied permission to run this tool."
