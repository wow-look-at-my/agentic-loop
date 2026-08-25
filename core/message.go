package commonai

import (
	"encoding/json"
	"strings"
)

// Role identifies the author of a Message.
type Role string

// Four roles: RoleSystem via Request.System, RoleTool carries tool results back.
const (
	RoleSystem    Role = "system"
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
	RoleTool      Role = "tool"
)

// One reasoning block: Text is the reasoning, Signature must replay VERBATIM; contents differ by dialect.
type ThinkingBlock struct {
	Text      string
	Signature string
	// ID is the provider's identifier for this reasoning item, replayed with it (a Responses rs_...).
	ID string
	// Redacted holds an opaque redacted_thinking payload; Text/Signature empty when set.
	Redacted string
}

// ToolCall is one tool invocation; Arguments is the raw JSON text, "{}" for a zero-argument tool.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// toolArgs maps an empty argument string to "{}" since empty is not valid JSON and would be rejected.
func toolArgs(args string) string {
	if strings.TrimSpace(args) == "" {
		return "{}"
	}
	return args
}

// Message is one entry in a conversation transcript. Thinking and ToolCalls
// are meaningful only on assistant messages; ToolCallID and ToolIsError only
// on tool messages.
type Message struct {
	Role Role
	// ID is the host-assigned transcript identifier; never sent upstream, for the host's durable tree.
	ID string
	// Kind is an optional host-facing classification (e.g. "stop_nudge"); never sent upstream.
	Kind      string
	Parts     []Part
	Content   string
	Thinking  []ThinkingBlock
	ToolCalls []ToolCall
	// ToolCallID links a RoleTool message to the assistant ToolCall it answers.
	ToolCallID string
	// ToolIsError marks a tool result as an error; surfaced to Anthropic's is_error flag.
	ToolIsError bool
}

// ToolDecl is what the model is told about one tool. InputSchema is the JSON
// schema of the tool's arguments; nil marshals as {"type":"object"}.
//
// The four behaviour fields state what the tool DOES to state. They are facts,
// not policy: nothing here says whether a call is allowed, only what running it
// would mean, so an Approver has something to decide from. They are MCP's tool
// annotations (readOnlyHint, destructiveHint, idempotentHint, openWorldHint),
// because an MCP server is a first-class source of tools and throwing away what
// it declares would leave every host to re-derive it.
//
// Two of them are pointers because their MCP defaults are TRUE, so an absent
// fact must not read as false. Never read those fields directly — IsDestructive,
// IsIdempotent and IsOpenWorld apply the defaults and the read-only precedence
// in one place.
type ToolDecl struct {
	Name        string
	Description string
	InputSchema json.RawMessage

	// Readonly marks a tool that only reads state; never sent upstream.
	Readonly bool

	// Destructive reports whether a non-readonly tool may destroy state; nil resolves to destructive.
	Destructive *bool

	// Idempotent reports that repeating the call with the same arguments has no further effect.
	Idempotent bool

	// OpenWorld reports whether the tool reaches outside a closed domain; nil resolves to open.
	OpenWorld *bool

	// Unvouched marks facts the HOST does not stand behind; a destructive tool could be auto-approved.
	Unvouched bool
}

// IsDestructive reports whether running this tool may destroy state; unknown resolves to destructive.
func (t ToolDecl) IsDestructive() bool {
	if t.Readonly {
		return false
	}
	if t.Destructive == nil {
		return true
	}
	return *t.Destructive
}

// IsIdempotent reports whether repeating this call with the same arguments adds nothing.
func (t ToolDecl) IsIdempotent() bool { return t.Readonly || t.Idempotent }

// IsOpenWorld reports whether the tool reaches outside a closed domain; unknown resolves to open.
func (t ToolDecl) IsOpenWorld() bool {
	if t.OpenWorld == nil {
		return true
	}
	return *t.OpenWorld
}

// Vouched reports whether the host stands behind this tool's stated behaviour; inverse of Unvouched.
func (t ToolDecl) Vouched() bool { return !t.Unvouched }

// defaultToolSchema is the schema sent for a ToolDecl with a nil InputSchema.
var defaultToolSchema = json.RawMessage(`{"type":"object"}`)

// schema returns the tool's input schema, defaulting nil to {"type":"object"}.
func (t ToolDecl) schema() json.RawMessage {
	if len(t.InputSchema) > 0 {
		return t.InputSchema
	}
	return defaultToolSchema
}

// A message's content is an ordered list of parts; order is the contract and is never flattened.

// PartKind names one kind of content part; the discriminator the XML codec writes and reads.
type PartKind string

// The kinds of content a message can carry.
const (
	PartKindText             PartKind = "text"
	PartKindImage            PartKind = "image"
	PartKindThinking         PartKind = "thinking"
	PartKindRedactedThinking PartKind = "redacted-thinking"
	PartKindToolCall         PartKind = "tool-call"
)

// Part is one piece of a message's content; concrete types are TextPart, ImagePart, etc.
type Part interface {
	// Kind identifies the part without a type switch.
	Kind() PartKind
}

// TextPart is model-facing text, carrying bytes exactly as they arrived, control characters included.
type TextPart struct {
	Text string
}

// Kind implements Part.
func (TextPart) Kind() PartKind { return PartKindText }

// ImagePart is an image held as supplied (inline MediaType+Data or by Src), never re-encoded.
type ImagePart struct {
	MediaType string
	Data      string
	Src       string
}

// Kind implements Part.
func (ImagePart) Kind() PartKind { return PartKindImage }

// ThinkingPart is one reasoning block: Text the reasoning, Signature replayed verbatim, ID the id.
type ThinkingPart struct {
	Text      string
	Signature string
	ID        string
}

// Kind implements Part.
func (ThinkingPart) Kind() PartKind { return PartKindThinking }

// RedactedThinkingPart is an opaque redacted_thinking payload, replayed to
// Anthropic untouched.
type RedactedThinkingPart struct {
	Data string
}

// Kind implements Part.
func (RedactedThinkingPart) Kind() PartKind { return PartKindRedactedThinking }

// ToolCallPart is one tool invocation; Arguments kept as raw text because a model can emit invalid JSON.
type ToolCallPart struct {
	ID        string
	Name      string
	Arguments string
}

// Kind implements Part.
func (ToolCallPart) Kind() PartKind { return PartKindToolCall }

// EffectiveParts is the message's content as ordered parts. Parts wins when it
// is set; otherwise the parts are derived from Thinking, Content and ToolCalls,
// in the order a turn produces them. That fallback is what lets a caller that
// only ever set Content keep working unchanged.
func (m Message) EffectiveParts() []Part {
	if len(m.Parts) > 0 {
		return m.Parts
	}
	var out []Part
	for _, tb := range m.Thinking {
		if tb.Redacted != "" {
			out = append(out, RedactedThinkingPart{Data: tb.Redacted})
			continue
		}
		out = append(out, ThinkingPart{Text: tb.Text, Signature: tb.Signature, ID: tb.ID})
	}
	if m.Content != "" {
		out = append(out, TextPart{Text: m.Content})
	}
	for _, tc := range m.ToolCalls {
		out = append(out, ToolCallPart{ID: tc.ID, Name: tc.Name, Arguments: tc.Arguments})
	}
	return out
}

// SyncViews fills Content, Thinking and ToolCalls from Parts, so a caller
// reading only the flattened fields sees the same turn the parts describe.
// Text parts are concatenated in order; everything else keeps its own list.
// Decoders call this after building Parts.
func (m *Message) SyncViews() {
	if len(m.Parts) == 0 {
		return
	}
	var text strings.Builder
	m.Thinking = nil
	m.ToolCalls = nil
	for _, p := range m.Parts {
		switch v := p.(type) {
		case TextPart:
			text.WriteString(v.Text)
		case ThinkingPart:
			m.Thinking = append(m.Thinking, ThinkingBlock{Text: v.Text, Signature: v.Signature, ID: v.ID})
		case RedactedThinkingPart:
			m.Thinking = append(m.Thinking, ThinkingBlock{Redacted: v.Data})
		case ToolCallPart:
			m.ToolCalls = append(m.ToolCalls, ToolCall{ID: v.ID, Name: v.Name, Arguments: v.Arguments})
		}
	}
	m.Content = text.String()
}

// NewMessage builds a message from ordered parts, with the flattened views
// filled in.
func NewMessage(role Role, parts ...Part) Message {
	m := Message{Role: role, Parts: parts}
	m.SyncViews()
	return m
}

// Bool is the address of a boolean, for the tri-state ToolDecl fields.
func Bool(v bool) *bool { return &v }
