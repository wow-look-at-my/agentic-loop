package commonai

import (
	"encoding/json"
	"strings"
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
// input_json_delta partial_json. A call this package produces always holds a
// JSON object there, "{}" for a tool that takes no arguments.
type ToolCall struct {
	ID        string
	Name      string
	Arguments string
}

// toolArgs is a tool call's arguments as the object every dialect needs. A
// model that calls a zero-argument tool sends no argument bytes at all, which
// arrives as an empty string, and an empty string is not JSON: a decoder
// rejects it, and Z.AI answers a function object without an arguments field
// with 400 "Invalid API parameter". So empty reads as the empty object, which
// is what the call meant. Every place that reads a tool call off the wire and
// every place that writes one back applies this, because a transcript comes
// from the host's storage as often as from the call that just happened.
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
	// ID is the host-assigned identifier for this transcript entry. The loop
	// attributes it from the OnAssistantMessage / OnResourceNotice return; a
	// host that does not persist leaves it empty. It is never sent to the
	// upstream — it is for the host's durable tree and the loop's own
	// transcript bookkeeping.
	ID string
	// Kind is an optional host-facing classification of the message (e.g.
	// "stop_nudge", "subagent_report"). The loop sets it on injected messages
	// so a host can persist it with the right label. It is never sent to the
	// upstream.
	Kind      string
	Parts     []Part
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

// A message's content is an ordered list of parts. Order is the contract: an
// Anthropic reply whose text blocks bracket a thinking block, or whose image
// sits between two paragraphs, means what it means because of where each piece
// sits. Flattening that into one string is a decision no translator gets to
// make on the caller's behalf.

// PartKind names one kind of content part. It is the discriminator the XML
// codec writes and reads, and the answer to "what is this" without a type
// switch.
type PartKind string

// The kinds of content a message can carry.
const (
	PartKindText             PartKind = "text"
	PartKindImage            PartKind = "image"
	PartKindThinking         PartKind = "thinking"
	PartKindRedactedThinking PartKind = "redacted-thinking"
	PartKindToolCall         PartKind = "tool-call"
)

// Part is one piece of a message's content. The concrete types are TextPart,
// ImagePart, ThinkingPart, RedactedThinkingPart and ToolCallPart.
type Part interface {
	// Kind identifies the part without a type switch.
	Kind() PartKind
}

// TextPart is model-facing text. It carries the bytes exactly as they arrived,
// control characters included -- what a provider sent is what the caller gets.
type TextPart struct {
	Text string
}

// Kind implements Part.
func (TextPart) Kind() PartKind { return PartKindText }

// ImagePart is an image, held the way it was supplied: inline (MediaType plus
// base64 Data) or by reference (Src, any URI including a data: one). It is
// never converted between the two on the way in -- a dialect that cannot
// express the form it was given says so rather than fetching or re-encoding
// behind the caller's back.
type ImagePart struct {
	MediaType string
	Data      string
	Src       string
}

// Kind implements Part.
func (ImagePart) Kind() PartKind { return PartKindImage }

// ThinkingPart is one reasoning block: Text is the human-readable reasoning,
// Signature the opaque token that must be replayed verbatim for the block to
// still count as the model's own thinking, and ID the provider's identifier for
// the item (a Responses `rs_...`). See ThinkingBlock for what fills them per
// dialect.
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

// ToolCallPart is one tool invocation the model requested. Arguments is the raw
// JSON text it produced, kept as text because a model can and does emit invalid
// JSON -- re-encoding it would hide that from the caller who has to handle it.
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
