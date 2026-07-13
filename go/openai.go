package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// openaiProvider is the Provider for OpenAI-compatible chat-completions APIs,
// built by NewProvider with DialectOpenAI. baseURL is the API root including
// the version segment (e.g. "https://api.openai.com/v1"); requests POST to
// baseURL + "/chat/completions". apiKey, when non-empty, is sent as a Bearer
// token, and headers are applied after the defaults so a caller-supplied
// header can override them. selfHosted adds cache_prompt:true to every
// request — the KV-cache prefix-reuse opt-in llama.cpp-style servers honor —
// and must stay false for hosted OpenAI/Azure, which reject unknown body
// fields with a 400. A nil httpClient uses http.DefaultClient.
//
// The fields are read-only during Complete, so a value is safe for concurrent
// use.
type openaiProvider struct {
	baseURL    string
	apiKey     string
	httpClient *http.Client
	userAgent  string
	selfHosted bool
	headers    map[string]string
}

// oaReserved are the Extra keys the typed core always overrides.
var oaReserved = map[string]bool{"messages": true, "model": true, "stream": true, "tools": true}

// Complete implements Provider over a streaming chat completion.
func (o *openaiProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	body, err := o.buildBody(req)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(o.baseURL, "/") + "/chat/completions"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("openai: build request: " + err.Error())
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if o.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
	if o.userAgent != "" {
		hreq.Header.Set("User-Agent", o.userAgent)
	}
	for k, v := range o.headers {
		hreq.Header.Set(k, v)
	}
	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	st := &oaStream{ev: ev, acc: newToolCallAccumulator()}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("openai: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	return st.completion(), nil
}

// buildBody assembles the JSON request body. Extra passthrough params are
// merged FIRST so the typed core fields always win; reserved keys in Extra
// (messages, model, stream, tools) are silently ignored so they cannot break
// routing. stream is always forced true, tools are sent only when non-empty
// (no tool_choice is ever sent), and stream_options defaults to
// {"include_usage":true} only when the caller has not supplied a
// stream_options key via Extra — without it OpenAI and most compatibles omit
// usage from streamed responses entirely. MaxTokens > 0 sets max_tokens
// (overriding an Extra value); 0 leaves the field to Extra or the provider
// default. CacheKey, when set, rides as prompt_cache_key, and selfHosted adds
// cache_prompt:true.
func (o *openaiProvider) buildBody(req Request) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.Extra {
		if oaReserved[k] {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	body["messages"] = oaWireMessages(req.System, req.Messages)
	body["stream"] = true
	if len(req.Tools) > 0 {
		tools := make([]oaTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, oaTool{Type: "function", Function: oaToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.schema(),
			}})
		}
		body["tools"] = tools
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if _, ok := body["stream_options"]; !ok {
		body["stream_options"] = map[string]bool{"include_usage": true}
	}
	if req.CacheKey != "" {
		body["prompt_cache_key"] = req.CacheKey
	}
	if o.selfHosted {
		body["cache_prompt"] = true
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, badRequestErr("openai: marshal request: " + err.Error())
	}
	return b, nil
}

// oaTool is the OpenAI wire shape of an advertised tool.
type oaTool struct {
	Type     string         `json:"type"`
	Function oaToolFunction `json:"function"`
}

// oaToolFunction is the function descriptor inside an advertised tool.
type oaToolFunction struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters"`
}

// oaToolCall is the OpenAI wire shape of a tool call, used both for replaying
// assistant tool_calls and for decoding streamed deltas (where Index keys the
// fragment accumulation).
type oaToolCall struct {
	Index    int            `json:"index,omitempty"`
	ID       string         `json:"id,omitempty"`
	Type     string         `json:"type,omitempty"`
	Function oaFunctionCall `json:"function"`
}

// oaFunctionCall is the function name and JSON-encoded arguments of a call.
type oaFunctionCall struct {
	Name      string `json:"name,omitempty"`
	Arguments string `json:"arguments,omitempty"`
}

// oaMessage is one chat message on the OpenAI wire. Its encoding is owned by
// MarshalJSON, because the content field has a role-dependent presence rule
// the standard omitempty cannot express.
type oaMessage struct {
	Role       string
	Content    string
	ToolCalls  []oaToolCall
	ToolCallID string
}

// MarshalJSON serializes a message for an OpenAI-compatible request. The
// OpenAI spec requires a content field on tool, user, and system messages
// even when empty: a plain `content,omitempty` drops an empty tool result and
// produces {"role":"tool","tool_call_id":...}, which upstreams reject with
// "invalid message content type: <nil>" / a 400 — failing the whole turn. So
// content is always emitted, except for an assistant message that carries
// tool_calls, where the spec makes content optional and the model originally
// produced none; there an empty content is omitted to match what was
// generated.
func (m oaMessage) MarshalJSON() ([]byte, error) {
	type wire struct {
		Role       string       `json:"role"`
		Content    *string      `json:"content,omitempty"`
		ToolCalls  []oaToolCall `json:"tool_calls,omitempty"`
		ToolCallID string       `json:"tool_call_id,omitempty"`
	}
	w := wire{Role: m.Role, ToolCalls: m.ToolCalls, ToolCallID: m.ToolCallID}
	if !(m.Role == "assistant" && m.Content == "" && len(m.ToolCalls) > 0) {
		// A non-nil pointer (even to "") is always emitted by omitempty, so this
		// forces a content field to appear; nil omits it for the assistant-with-
		// tool-calls case above.
		w.Content = &m.Content
	}
	return json.Marshal(w)
}

// oaWireMessages maps the neutral transcript onto the OpenAI wire: the system
// prompt (when non-empty) is prepended as a system message, assistant
// tool calls are replayed as tool_calls, and tool results ride as role:"tool"
// messages keyed by tool_call_id. Message.Thinking is not replayed on this
// dialect (OpenAI-compatible APIs have no reasoning-replay field), and
// Message.ToolIsError has no wire equivalent.
func oaWireMessages(system string, msgs []Message) []oaMessage {
	out := make([]oaMessage, 0, len(msgs)+1)
	if system != "" {
		out = append(out, oaMessage{Role: string(RoleSystem), Content: system})
	}
	for _, m := range msgs {
		wm := oaMessage{Role: string(m.Role), Content: m.Content, ToolCallID: m.ToolCallID}
		for _, tc := range m.ToolCalls {
			wm.ToolCalls = append(wm.ToolCalls, oaToolCall{
				ID:       tc.ID,
				Type:     "function",
				Function: oaFunctionCall{Name: tc.Name, Arguments: tc.Arguments},
			})
		}
		out = append(out, wm)
	}
	return out
}

// oaChunk is one SSE delta from a streaming chat completion.
type oaChunk struct {
	Choices []oaChoice `json:"choices"`
	Usage   *oaUsage   `json:"usage,omitempty"`
	// Timings is the llama.cpp-style timing snapshot llama.cpp/ollama attach
	// to streamed chunks; each occurrence replaces the previous (last wins).
	Timings *Timings `json:"timings,omitempty"`
	// PromptProgress is a non-standard prefill-progress update some upstreams
	// emit before the first token while a long prompt is ingested. It rides a
	// choices-less chunk.
	PromptProgress *PromptProgress `json:"prompt_progress,omitempty"`
}

// oaChoice is one choice within a chunk.
type oaChoice struct {
	Delta        oaDelta `json:"delta"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// oaDelta is the incremental content of a streaming choice. Reasoning arrives
// under two field names in the wild: reasoning_content (OpenAI/DeepSeek
// style) and reasoning (Ollama style).
type oaDelta struct {
	Content          string       `json:"content,omitempty"`
	ReasoningContent string       `json:"reasoning_content,omitempty"`
	Reasoning        string       `json:"reasoning,omitempty"`
	ToolCalls        []oaToolCall `json:"tool_calls,omitempty"`
}

// reasoning returns the delta's reasoning text from whichever field the
// upstream used; reasoning_content wins when both are present.
func (d oaDelta) reasoning() string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
}

// oaUsage is the wire shape of a usage snapshot, capturing the cache
// accounting of three dialects: OpenAI/vLLM/OpenRouter report cached tokens
// under prompt_tokens_details.cached_tokens, DeepSeek reports
// prompt_cache_hit_tokens, and Anthropic-compatible layers pass through
// cache_read_input_tokens. The cache fields are pointers so an absent field
// is distinguishable from an explicit zero (the tri-state contract).
type oaUsage struct {
	PromptTokens         int                    `json:"prompt_tokens"`
	CompletionTokens     int                    `json:"completion_tokens"`
	TotalTokens          int                    `json:"total_tokens"`
	PromptTokensDetails  *oaPromptTokensDetails `json:"prompt_tokens_details"`
	PromptCacheHitTokens *int                   `json:"prompt_cache_hit_tokens"`
	CacheReadInputTokens *int                   `json:"cache_read_input_tokens"`
}

// oaPromptTokensDetails is the OpenAI breakdown of prompt tokens.
type oaPromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

// toUsage normalizes a wire snapshot: the largest cache signal present wins
// (the dialects are mutually exclusive in practice) and lands in
// CacheReadTokens; when any cache info was reported, CacheWriteTokens is an
// explicit 0 — OpenAI-compatible servers neither report nor bill a separate
// cache-write class — while a snapshot with no cache fields at all leaves
// both nil (unknown). prompt_tokens already includes cached tokens on this
// dialect, so PromptTokens passes through untouched.
func (u *oaUsage) toUsage() Usage {
	out := Usage{
		PromptTokens:     u.PromptTokens,
		CompletionTokens: u.CompletionTokens,
		TotalTokens:      u.TotalTokens,
	}
	var read *int
	consider := func(p *int) {
		if p == nil {
			return
		}
		if read == nil || *p > *read {
			read = clonePtr(p)
		}
	}
	consider(u.PromptCacheHitTokens)
	if u.PromptTokensDetails != nil {
		consider(u.PromptTokensDetails.CachedTokens)
	}
	consider(u.CacheReadInputTokens)
	out.CacheReadTokens = read
	if read != nil {
		out.CacheWriteTokens = intPtr(0)
	}
	return out
}

// toolCallAccumulator reassembles tool calls that arrive in fragments across
// streaming deltas. OpenAI streams tool calls by index: the first delta for
// an index carries id/name, and subsequent deltas append argument fragments.
type toolCallAccumulator struct {
	byIndex map[int]*oaToolCall
	order   []int
}

func newToolCallAccumulator() *toolCallAccumulator {
	return &toolCallAccumulator{byIndex: map[int]*oaToolCall{}}
}

// add merges streamed tool-call deltas into the accumulator: id/type/name
// overwrite when non-empty, argument fragments concatenate.
func (a *toolCallAccumulator) add(deltas []oaToolCall) {
	for _, d := range deltas {
		tc, ok := a.byIndex[d.Index]
		if !ok {
			tc = &oaToolCall{Index: d.Index, Type: "function"}
			a.byIndex[d.Index] = tc
			a.order = append(a.order, d.Index)
		}
		if d.ID != "" {
			tc.ID = d.ID
		}
		if d.Type != "" {
			tc.Type = d.Type
		}
		if d.Function.Name != "" {
			tc.Function.Name = d.Function.Name
		}
		tc.Function.Arguments += d.Function.Arguments
	}
}

// finish returns the assembled calls in the order their indices first
// appeared.
func (a *toolCallAccumulator) finish() []ToolCall {
	out := make([]ToolCall, 0, len(a.order))
	for _, idx := range a.order {
		tc := a.byIndex[idx]
		out = append(out, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: tc.Function.Arguments})
	}
	return out
}

// empty reports whether any tool calls were accumulated.
func (a *toolCallAccumulator) empty() bool { return len(a.order) == 0 }

// oaStream accumulates one streamed completion.
type oaStream struct {
	ev      *StreamEvents
	content strings.Builder
	reason  strings.Builder
	acc     *toolCallAccumulator
	finish  string
	usage   *Usage
	timings *Timings
	sawData bool
}

// onData decodes one SSE payload. Unparseable chunks are tolerated silently
// (keep-alive noise). A prompt_progress chunk is forwarded and carries
// nothing else; usage snapshots are merged newest-wins (never summed) so both
// the OpenAI single-final-chunk and the xAI usage-on-every-chunk conventions
// yield the same result; timings snapshots replace each other (last wins). A
// callback error aborts the stream: state is accumulated BEFORE each emit, so
// the partial completion includes everything through the failing delta.
func (st *oaStream) onData(data []byte) error {
	var chunk oaChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil
	}
	if chunk.PromptProgress != nil {
		st.sawData = true
		return st.ev.emitProgress(*chunk.PromptProgress)
	}
	if chunk.Usage != nil {
		st.sawData = true
		u := chunk.Usage.toUsage()
		st.usage = mergeUsage(st.usage, &u)
		if err := st.ev.emitUsage(*st.usage); err != nil {
			return err
		}
	}
	if chunk.Timings != nil {
		st.sawData = true
		t := *chunk.Timings
		st.timings = &t
		if err := st.ev.emitTimings(t); err != nil {
			return err
		}
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != "" {
			st.sawData = true
			st.content.WriteString(ch.Delta.Content)
			if err := st.ev.emitText(ch.Delta.Content); err != nil {
				return err
			}
		}
		if rc := ch.Delta.reasoning(); rc != "" {
			st.sawData = true
			st.reason.WriteString(rc)
			if err := st.ev.emitReasoning(rc); err != nil {
				return err
			}
		}
		if len(ch.Delta.ToolCalls) > 0 {
			st.sawData = true
			st.acc.add(ch.Delta.ToolCalls)
		}
		if ch.FinishReason != "" {
			st.sawData = true
			st.finish = ch.FinishReason
		}
	}
	return nil
}

// completion assembles the final (or partial) result: accumulated content,
// reasoning as a single ThinkingBlock, assembled tool calls, the merged
// usage with the total floored at prompt+completion (a genuine surplus —
// reasoning tokens — is preserved), the last timings snapshot (nil when the
// upstream reported none), and the normalized stop reason.
func (st *oaStream) completion() *Completion {
	var calls []ToolCall
	if !st.acc.empty() {
		calls = st.acc.finish()
	}
	finish := st.finish
	if finish == "" && len(calls) > 0 {
		finish = "tool_calls"
	}
	if finish == "" {
		finish = "stop"
	}
	msg := Message{Role: RoleAssistant, Content: st.content.String(), ToolCalls: calls}
	if r := st.reason.String(); r != "" {
		msg.Thinking = []ThinkingBlock{{Text: r}}
	}
	var u Usage
	if st.usage != nil {
		u = floorTotal(*st.usage)
	}
	return &Completion{
		Message:       msg,
		Usage:         u,
		UsageReported: st.usage != nil,
		Timings:       st.timings,
		StopReason:    normalizeStopReason(finish),
	}
}

// normalizeStopReason maps OpenAI finish reasons onto the normalized
// constants; anything unrecognized passes through raw.
func normalizeStopReason(reason string) string {
	switch reason {
	case "stop":
		return StopEndTurn
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopMaxTokens
	}
	return reason
}
