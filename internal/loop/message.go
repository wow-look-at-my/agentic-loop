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
	// Type is the MCP block type -- "text", "image", "audio", "resource_link",
	// "resource" -- or a name a host and its tools agree on.
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

// ToolResult is the outcome of executing one tool call. Content is the
// model-facing text fed back as the tool message; IsError marks a recoverable
// failure the model can react to.
type ToolResult struct {
	Content string
	// Parts is structured content for the HOST to render -- images, audio,
	// embedded files, a tool's own rich block. It is nil for a plain-text
	// result and is NEVER sent to the model: a tool that returns a megabyte of
	// image here still costs the context only what Content says.
	Parts   []ToolContentPart
	IsError bool
}

// Approval is the verdict on one tool call. A bare bool cannot say WHY a call
// was refused, so every denial reached the model as the same sentence about a
// user -- including the ones no user was ever asked about. A model told "the
// write was outside the workspace" retries inside it; a model told "the user
// denied permission" asks a person who never saw the question.
type Approval struct {
	// OK allows the call.
	OK bool
	// Reason, when !OK, is recorded as the tool result INSTEAD of
	// DeniedMessage. Empty keeps DeniedMessage, which is still the right
	// sentence for the case it was written for: a user pressing deny.
	Reason string
}

// Approver decides whether a tool call may run. Ask is consulted for EVERY
// call -- a tool does not get to declare itself unremarkable, because a deny
// rule that cannot fire on read-only tools is a lie about what it protects --
// and blocks until a decision is available. An error means the decision never
// arrived (e.g. the caller went away); Run then finalizes the turn with the
// pending batch cleared and returns.
//
// A nil Approver on Config is the fail-closed default expressed once: a call
// whose ToolDecl.Readonly is true runs, and anything that can change state is
// denied with DeniedMessage.
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (Approval, error)
}

// DeniedMessage is the exact tool-result text recorded when the user denies
// permission to run a tool. The denial is an error tool result and the loop
// continues so the model can react.
const DeniedMessage = "The user denied permission to run this tool."
