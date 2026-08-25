package commonai

import "encoding/json"

// The Responses wire vocabulary, where input and output share the same item vocabulary.

// respIncludeEncryptedReasoning asks for the encrypted payload, the only way to replay reasoning.
const respIncludeEncryptedReasoning = "reasoning.encrypted_content"

// respTool is the Responses wire shape of a tool; no nested "function" object here.
type respTool struct {
	Type        string          `json:"type"`
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// respItem is one item on the Responses wire, in either direction.
type respItem struct {
	Type string `json:"type"`
	ID   string `json:"id,omitempty"`
	Role string `json:"role,omitempty"`
	// Content is a message's parts: input_text in, output_text out.
	Content []respContent `json:"content,omitempty"`
	// CallID, Name and Arguments describe a function_call; CallID and Output a function_call_output.
	CallID    string `json:"call_id,omitempty"`
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
	Output    string `json:"output,omitempty"`
	// Summary and EncryptedContent belong to a reasoning item; the payload must be replayed verbatim.
	Summary          []respContent `json:"summary,omitempty"`
	EncryptedContent string        `json:"encrypted_content,omitempty"`
}

// respContent is one part of a message or reasoning summary.
type respContent struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	// ImageURL carries an input image, as a URI: the supplied one, or a data:
	// URI built from supplied bytes.
	ImageURL string `json:"image_url,omitempty"`
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
	respImageInput  = "input_image"
)

// respInputItems maps the transcript onto Responses input items, keeping their emitted order.
func respInputItems(msgs []Message) ([]respItem, error) {
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
					Type: respItemFuncCall, CallID: tc.ID, Name: tc.Name, Arguments: toolArgs(tc.Arguments),
				})
			}
		default:
			content, err := respInputContent(m)
			if err != nil {
				return nil, err
			}
			out = append(out, respItem{
				Type: respItemMessage, Role: string(m.Role), Content: content,
			})
		}
	}
	return out, nil
}

// respInputContent builds a user message's content list: text alone is one input_text.
func respInputContent(m Message) ([]respContent, error) {
	if !hasImage(m) {
		return []respContent{{Type: respTextInput, Text: m.Content}}, nil
	}
	var out []respContent
	for _, p := range m.EffectiveParts() {
		switch v := p.(type) {
		case TextPart:
			if v.Text != "" {
				out = append(out, respContent{Type: respTextInput, Text: v.Text})
			}
		case ImagePart:
			ref, err := imageRef(DialectResponses, v)
			if err != nil {
				return nil, err
			}
			out = append(out, respContent{Type: respImageInput, ImageURL: ref})
		}
	}
	return out, nil
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
