package agentic

import (
	"context"
	"encoding/json"
)

// Role identifies the author of a Message.
type Role string

// The four conversation roles. RoleSystem messages are normally supplied via
// Request.System rather than the transcript; RoleTool messages carry tool
// results back to the model.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// ThinkingBlock is one reasoning block produced by the model. Text is always
// the human-readable reasoning; Signature is always the opaque token that must
// be replayed VERBATIM for the block to still count as the model's own
// thinking. What fills them differs by dialect:
//
//   - Anthropic: a native thinking block (Text + Signature) or an opaque
//     redacted block (Redacted set, Text/Signature empty). Replaying blocks
//     verbatim — signatures intact, redacted payloads untouched — is required
//     for tool-use continuations.
//   - Responses: the reasoning item's summary in Text, its encrypted_content in
//     Signature, and its item id in ID.
//   - OpenAI chat-completions: the accumulated reasoning text in a single block
//     with only Text set. There is no replay token on that dialect, which is
//     the whole reason the Responses dialect exists.
type ThinkingBlock struct {
	Text      string
	Signature string
	// ID is the provider's identifier for this reasoning item, replayed with
	// it (a Responses `rs_...`). Empty on the dialects that have none.
	ID string
	// Redacted holds an opaque redacted_thinking payload. When set, Text and
	// Signature are empty and the block is replayed to Anthropic as
	// {type:"redacted_thinking", data:Redacted}.
	Redacted string
}

// ToolCall is one tool invocation requested by the model. Arguments is the raw
// JSON object text: for OpenAI-compatible upstreams it is the concatenation of
// the streamed argument fragments; for Anthropic it is the accumulated
// input_json_delta partial_json.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// Message is one entry in a conversation transcript. Thinking and ToolCalls
// are meaningful only on assistant messages; ToolCallID and ToolIsError only
// on tool messages.
type Message struct {
	Role      Role
	Content   string
	Thinking  []ThinkingBlock
	ToolCalls []ToolCall
	// ToolCallID links a RoleTool message to the assistant ToolCall it answers.
	ToolCallID string
	// ToolIsError marks a tool result as an error. It is surfaced to Anthropic
	// as the tool_result block's is_error flag; OpenAI-compatible upstreams
	// have no equivalent field, so there it only informs the caller.
	ToolIsError bool
}

// ToolDecl is what the model is told about one tool. InputSchema is the JSON
// schema of the tool's arguments; nil marshals as {"type":"object"}. Readonly
// marks a tool that only reads state; it is never sent to the upstream and
// exists to drive Tools.Readonly.
type ToolDecl struct {
	Name        string
	Description string
	InputSchema json.RawMessage
	Readonly    bool
}

// defaultToolSchema is the schema sent for a ToolDecl with a nil InputSchema.
var defaultToolSchema = json.RawMessage(`{"type":"object"}`)

// schema returns the tool's input schema, defaulting nil to {"type":"object"}.
func (t ToolDecl) schema() json.RawMessage {
	if len(t.InputSchema) > 0 {
		return t.InputSchema
	}
	return defaultToolSchema
}

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

// Approver decides whether an approval-gated tool call may run. Ask blocks
// until a decision is available: true allows the call, false denies it (the
// loop records DeniedMessage as the tool result and continues). An error means
// the decision never arrived (e.g. the caller went away); Run then finalizes
// the turn with the pending batch cleared and returns.
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (bool, error)
}

// DeniedMessage is the exact tool-result text recorded when the user denies
// permission to run a tool. The denial is an error tool result and the loop
// continues so the model can react.
const DeniedMessage = "The user denied permission to run this tool."
