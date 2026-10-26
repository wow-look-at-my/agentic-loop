// IMessage is one node of a conversation tree.

package commonai

import (
	"encoding/json"
	"time"
)

// IMessage is one node of a conversation tree. Every implementation is a pointer to a struct that
// embeds MessageBase.
type IMessage interface {
	Base() *MessageBase
	NodeKind() NodeKind
	// Final reports whether every stream field on the node is done.
	Final() bool
}

// Producer is who made a model-output node. Opaque payloads replay only to the same producer.
type Producer struct {
	Dialect  Dialect
	Endpoint string
	Model    string
	// Account is a hash of the credential identity, never the credential.
	Account string
}

// MessageBase is what every node shares. A Tree owns the links; Session methods change them.
type MessageBase struct {
	id       string
	kind     string
	parent   IMessage
	children []IMessage
	at       time.Time
	call     *CallRecord
	producer Producer
	meta     Meta
	enc      encCache
	// superseded marks a node a later one replaced, such as a tool call a host rewrote.
	superseded bool
	// generatedID marks an id the tree made because the node had none. A flat message shows it as "".
	generatedID bool
}

// HostID is the id a host or a provider gave the node, or "" when the tree generated it.
func (b *MessageBase) HostID() string {
	if b.generatedID {
		return ""
	}
	return b.id
}

// Superseded reports whether a later node replaced this on the wire.
func (b *MessageBase) Superseded() bool { return b.superseded }

// MarkSuperseded leaves the node out of later requests. It stays in the tree.
func (b *MessageBase) MarkSuperseded() { b.superseded = true }

func (b *MessageBase) Base() *MessageBase { return b }

// ID is the node's identifier. It never changes once the node is in a tree.
func (b *MessageBase) ID() string { return b.id }

// Kind is the host-facing classification: stop_nudge, subagent_report, compaction, and so on.
func (b *MessageBase) Kind() string     { return b.kind }
func (b *MessageBase) Parent() IMessage { return b.parent }

// Children is a copy of the node's children, oldest first.
func (b *MessageBase) Children() []IMessage {
	out := make([]IMessage, len(b.children))
	copy(out, b.children)
	return out
}
func (b *MessageBase) At() time.Time { return b.at }

// Call is the model call that produced the node, or nil for a node this side wrote.
func (b *MessageBase) Call() *CallRecord  { return b.call }
func (b *MessageBase) Producer() Producer { return b.producer }
func (b *MessageBase) Meta() *Meta        { return &b.meta }

// SetID names a node before it joins a tree.
func (b *MessageBase) SetID(id string) { b.id, b.generatedID = id, false }

// SetKind classifies a node before it joins a tree.
func (b *MessageBase) SetKind(kind string) { b.kind = kind }

// UserMessage is one user turn: text and images in order.
type UserMessage struct {
	MessageBase
	Content []IContent
}

// NewUser is a user turn.
func NewUser(content ...IContent) *UserMessage { return &UserMessage{Content: content} }

// NewUserText is a user turn of one text block.
func NewUserText(s string) *UserMessage { return NewUser(Text(s)) }

func (*UserMessage) NodeKind() NodeKind { return NodeUser }
func (*UserMessage) Final() bool        { return true }

// SystemMessage is a note this side injects mid-conversation.
type SystemMessage struct {
	MessageBase
	Content []IContent
}

// NewSystem is a note of kind holding content.
func NewSystem(kind string, content ...IContent) *SystemMessage {
	m := &SystemMessage{Content: content}
	m.kind = kind
	return m
}

func (*SystemMessage) NodeKind() NodeKind { return NodeSystem }
func (*SystemMessage) Final() bool        { return true }

// AssistantMessage is model text.
type AssistantMessage struct {
	MessageBase
	Text      IStreamString
	Citations []json.RawMessage
}

// NewAssistant is an empty model-text node that is still streaming.
func NewAssistant() *AssistantMessage { return &AssistantMessage{Text: NewStreamString()} }

// NewAssistantText is a finished model-text node.
func NewAssistantText(s string) *AssistantMessage {
	return &AssistantMessage{Text: FinishedString(s)}
}

func (*AssistantMessage) NodeKind() NodeKind { return NodeAssistant }
func (m *AssistantMessage) Final() bool      { return m.Text.Done() }

// ResponsesItemID is a Responses reasoning item id, replayed verbatim.
type ResponsesItemID string

// ReasoningDetails is OpenRouter's reasoning_details array, kept verbatim.
type ReasoningDetails json.RawMessage

// ThinkingMessage is model reasoning. Each dialect's opaque payload has its own field.
type ThinkingMessage struct {
	MessageBase
	// Text is the reasoning; on Responses it is the summary.
	Text IStreamString
	// Signature is the Anthropic thinking signature. signature_delta assigns it whole.
	Signature *EncryptedReasoning
	// Encrypted is the Responses encrypted_content, from output_item.done.
	Encrypted *EncryptedReasoning
	// Redacted is Anthropic redacted_thinking data. Text and Signature are empty when it is set.
	Redacted *EncryptedReasoning
	ItemID   ResponsesItemID
	Details  ReasoningDetails
	// Raw marks Text as a Responses reasoning_text rather than a summary.
	Raw bool
	// Dialect produced the node. Any other dialect replays Text only.
	Dialect Dialect
}

// NewThinking is an empty reasoning node that is still streaming.
func NewThinking(d Dialect) *ThinkingMessage {
	return &ThinkingMessage{Text: NewStreamString(), Dialect: d}
}

func (*ThinkingMessage) NodeKind() NodeKind { return NodeThinking }
func (m *ThinkingMessage) Final() bool      { return m.Text.Done() }

// CallID is a tool call's id, replayed verbatim.
type CallID string

// ToolCallMessage is one tool call the model made.
type ToolCallMessage struct {
	MessageBase
	CallID CallID
	Name   string
	// Arguments is the raw text the model wrote. It may be invalid JSON.
	Arguments IStreamString
	// ProviderFields are provider extras on the call, such as Gemini's thought_signature.
	ProviderFields json.RawMessage
	// GeneratedID marks an id this side assigned because the provider sent none.
	GeneratedID bool
	// Subagent is the child session this call ran, as the host's own type.
	Subagent any
}

// NewToolCall is a tool call whose arguments are still streaming.
func NewToolCall(id CallID, name string) *ToolCallMessage {
	return &ToolCallMessage{CallID: id, Name: name, Arguments: NewStreamString()}
}

// NewToolCallArgs is a finished tool call.
func NewToolCallArgs(id CallID, name, args string) *ToolCallMessage {
	return &ToolCallMessage{CallID: id, Name: name, Arguments: FinishedString(args)}
}

func (*ToolCallMessage) NodeKind() NodeKind { return NodeToolCall }
func (m *ToolCallMessage) Final() bool      { return m.Arguments.Done() }

// ToolResultMessage answers one tool call.
type ToolResultMessage struct {
	MessageBase
	Call    *ToolCallMessage
	Content []IContent
	IsError bool
	// Omitted says why the result was withheld, such as an aborted batch. The content says it too.
	Omitted string
}

// NewToolResult answers call.
func NewToolResult(call *ToolCallMessage, isError bool, content ...IContent) *ToolResultMessage {
	return &ToolResultMessage{Call: call, Content: content, IsError: isError}
}

func (*ToolResultMessage) NodeKind() NodeKind { return NodeToolResult }
func (*ToolResultMessage) Final() bool        { return true }

// CompactionMessage is the summary a compaction wrote. The window starts here.
type CompactionMessage struct {
	MessageBase
	Summary IStreamString
	// Replaces is where the window started before this compaction.
	Replaces IMessage
}

// NewCompaction is a summary that is still streaming.
func NewCompaction(replaces IMessage) *CompactionMessage {
	m := &CompactionMessage{Summary: NewStreamString(), Replaces: replaces}
	m.kind = CompactionKind
	return m
}

func (*CompactionMessage) NodeKind() NodeKind { return NodeCompaction }
func (m *CompactionMessage) Final() bool      { return m.Summary.Done() }

// RawBlockMessage is a content block of a type this library does not model, kept verbatim and
// replayed to the dialect that produced it.
type RawBlockMessage struct {
	MessageBase
	Type    string
	JSON    json.RawMessage
	Dialect Dialect
}

func (*RawBlockMessage) NodeKind() NodeKind { return NodeRawBlock }
func (*RawBlockMessage) Final() bool        { return true }

// CompactionKind marks a compaction summary.
const CompactionKind = "compaction"
