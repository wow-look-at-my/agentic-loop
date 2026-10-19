package commonai

import "fmt"

// The closed sets of the conversation tree.

// NodeKind names a node type without a type switch.
type NodeKind uint8

const (
	NodeUser NodeKind = iota
	NodeSystem
	NodeAssistant
	NodeThinking
	NodeToolCall
	NodeToolResult
	NodeCompaction
	NodeRawBlock
	NodeExtension
)

func (k NodeKind) String() string {
	switch k {
	case NodeUser:
		return "user"
	case NodeSystem:
		return "system"
	case NodeAssistant:
		return "assistant"
	case NodeThinking:
		return "thinking"
	case NodeToolCall:
		return "tool-call"
	case NodeToolResult:
		return "tool-result"
	case NodeCompaction:
		return "compaction"
	case NodeRawBlock:
		return "raw-block"
	case NodeExtension:
		return "extension"
	default:
		return fmt.Sprintf("NodeKind(%d)", uint8(k))
	}
}

// CallStatus is where one model call is.
type CallStatus uint8

const (
	CallStreaming CallStatus = iota
	CallDone
	CallFailed
	CallCancelled
	// CallSuperseded marks a try a retry replaced. Its nodes keep what they received.
	CallSuperseded
)

func (s CallStatus) String() string {
	switch s {
	case CallStreaming:
		return "streaming"
	case CallDone:
		return "done"
	case CallFailed:
		return "failed"
	case CallCancelled:
		return "cancelled"
	case CallSuperseded:
		return "superseded"
	default:
		return fmt.Sprintf("CallStatus(%d)", uint8(s))
	}
}

// RevisionSource says where one value of a stream field came from.
type RevisionSource uint8

const (
	RevisionStreamed RevisionSource = iota
	RevisionFinal
	RevisionRetry
	RevisionHostEdit
)

func (r RevisionSource) String() string {
	switch r {
	case RevisionStreamed:
		return "streamed"
	case RevisionFinal:
		return "final"
	case RevisionRetry:
		return "retry"
	case RevisionHostEdit:
		return "host-edit"
	default:
		return fmt.Sprintf("RevisionSource(%d)", uint8(r))
	}
}

// Severity is how loud a state loss is.
type Severity uint8

const (
	// SeverityNotice is a one-off loss, such as a switch of model or provider.
	SeverityNotice Severity = iota
	// SeverityChronic is a loss that repeats on the same provider, model and dialect.
	SeverityChronic
)

func (s Severity) String() string {
	switch s {
	case SeverityNotice:
		return "notice"
	case SeverityChronic:
		return "chronic"
	default:
		return fmt.Sprintf("Severity(%d)", uint8(s))
	}
}

// FieldID names the stream field of a node that a delta writes.
type FieldID uint8

const (
	FieldText FieldID = iota
	FieldArguments
)

func (f FieldID) String() string {
	switch f {
	case FieldText:
		return "text"
	case FieldArguments:
		return "arguments"
	default:
		return fmt.Sprintf("FieldID(%d)", uint8(f))
	}
}

// MisstepKind names one way a model used the protocol wrongly.
type MisstepKind uint8

const (
	MisstepUnknownTool MisstepKind = iota
	MisstepInvalidArguments
	MisstepMalformedToolCall
	MisstepToolCallAsText
	MisstepToolUseWithoutCall
	MisstepEmptyCompletion
)

func (m MisstepKind) String() string {
	switch m {
	case MisstepUnknownTool:
		return "unknown_tool"
	case MisstepInvalidArguments:
		return "invalid_arguments"
	case MisstepMalformedToolCall:
		return "malformed_tool_call"
	case MisstepToolCallAsText:
		return "tool_call_as_text"
	case MisstepToolUseWithoutCall:
		return "tool_use_without_call"
	case MisstepEmptyCompletion:
		return "empty_completion"
	default:
		return fmt.Sprintf("MisstepKind(%d)", uint8(m))
	}
}
