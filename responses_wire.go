package agentic

import "encoding/json"

// The Responses wire vocabulary: the items that go in and come out, the tool
// shape, and the usage object. Its one structural fact -- input and output are
// the SAME item vocabulary -- is what makes reasoning replay possible at all,
// so the mapping lives here and the transport lives in responses.go.

// respIncludeEncryptedReasoning asks the API to return each reasoning item's
// encrypted payload, which is the only way to replay reasoning when store is
// false -- and store is false by default here.
const respIncludeEncryptedReasoning = "reasoning.encrypted_content"

// respTool is the Responses wire shape of an advertised tool. Unlike
// chat-completions there is no nested "function" object: the name, description
// and parameters sit on the tool itself.
type respTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// respItem is one item on the Responses wire, in either direction. The API's
// input and output are the SAME item vocabulary, which is what makes replay
// possible: an item that came out can go straight back in.
type respItem struct {
	Type string `json:"type"`
	ID   string `json:"id,omitempty"`
	Role string `json:"role,omitempty"`
	// Content is a message's parts: input_text on the way in, output_text on
	// the way out.
	Content []respContent `json:"content,omitempty"`
	// CallID, Name and Arguments describe a function_call; CallID and Output a
	// function_call_output. The call_id (not the item id) is what pairs them.
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
	// Summary and EncryptedContent belong to a reasoning item. The summary is
	// the human-readable trace; the encrypted content is the opaque payload
	// that must be replayed verbatim for the model to keep its own reasoning.
	Summary          []respContent `json:"summary,omitempty"`
	EncryptedContent string        `json:"encrypted_content,omitempty"`
}

// respContent is one part of a message or reasoning summary.
type respContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

// The item and content type names this dialect reads and writes.
const (
	respItemMessage    = "message"
	respItemFuncCall   = "function_call"
	respItemFuncOutput = "function_call_output"
	respItemReasoning  = "reasoning"

	respTextInput   = "input_text"
	respTextOutput  = "output_text"
	respTextSummary = "summary_text"
)

// respInputItems maps the neutral transcript onto Responses input items.
//
// An assistant turn can become SEVERAL items, and their ORDER is the contract:
// reasoning first, then the text it produced, then the tool calls it made --
// the order the model itself emitted them. A tool result is a top-level
// function_call_output item keyed by call_id, not a message with a role.
//
// A reasoning item is replayed only when it carries the encrypted payload: the
// summary alone is prose about the reasoning, and sending it as though it were
// the reasoning would hand the model a paraphrase of its own thinking. A block
// with no payload is dropped rather than half-replayed.
func respInputItems(msgs []Message) []respItem {
	out := make([]respItem, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case RoleTool:
			out = append(out, respItem{
				Type: respItemFuncOutput, CallID: m.ToolCallID, Output: m.Content,
			})
		case RoleAssistant:
			for _, tb := range m.Thinking {
				if tb.Signature == "" {
					continue
				}
				item := respItem{Type: respItemReasoning, ID: tb.ID, EncryptedContent: tb.Signature}
				if tb.Text != "" {
					item.Summary = []respContent{{Type: respTextSummary, Text: tb.Text}}
				}
				out = append(out, item)
			}
			if m.Content != "" {
				out = append(out, respItem{
					Type: respItemMessage, Role: string(RoleAssistant),
					Content: []respContent{{Type: respTextOutput, Text: m.Content}},
				})
			}
			for _, tc := range m.ToolCalls {
				out = append(out, respItem{
					Type: respItemFuncCall, CallID: tc.ID, Name: tc.Name, Arguments: tc.Arguments,
				})
			}
		default:
			out = append(out, respItem{
				Type: respItemMessage, Role: string(m.Role),
				Content: []respContent{{Type: respTextInput, Text: m.Content}},
			})
		}
	}
	return out
}

// respUsage is the Responses usage shape. The field names are NOT the
// chat-completions ones -- input_tokens/output_tokens rather than
// prompt_tokens/completion_tokens -- so this dialect decodes its own and does
// not share oaUsage. The detail fields are pointers for the tri-state contract:
// absent is not zero.
type respUsage struct {
	InputTokens         int                `json:"input_tokens"`
	OutputTokens        int                `json:"output_tokens"`
	TotalTokens         int                `json:"total_tokens"`
	InputTokensDetails  *respInputDetails  `json:"input_tokens_details"`
	OutputTokensDetails *respOutputDetails `json:"output_tokens_details"`
}

// respInputDetails carries the cached-prompt figure.
type respInputDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

// respOutputDetails carries the reasoning-token figure.
type respOutputDetails struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// toUsage normalizes a Responses usage snapshot onto the library's shape.
// input_tokens already INCLUDES the cached ones on this dialect, so it passes
// through untouched. When any cache figure was reported, CacheWriteTokens is an
// explicit 0 -- this API neither reports nor bills a separate cache-write class
// -- and a snapshot with no cache detail at all leaves both nil (unknown).
func (u *respUsage) toUsage() Usage {
	out := Usage{
		PromptTokens:     u.InputTokens,
		CompletionTokens: u.OutputTokens,
		TotalTokens:      u.TotalTokens,
	}
	if u.InputTokensDetails != nil && u.InputTokensDetails.CachedTokens != nil {
		out.CacheReadTokens = clonePtr(u.InputTokensDetails.CachedTokens)
		out.CacheWriteTokens = intPtr(0)
	}
	return out
}

// reasoningTokens returns the reported reasoning-token count, or nil.
func (u *respUsage) reasoningTokens() *int {
	if u.OutputTokensDetails == nil {
		return nil
	}
	return clonePtr(u.OutputTokensDetails.ReasoningTokens)
}
