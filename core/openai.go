package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// openaiProvider is the Provider for OpenAI-compatible chat-completions APIs,
// built by NewOpenAIProvider. baseURL is the API root including
// the version segment (e.g. "https://api.openai.com/v1"); requests POST to
// baseURL + "/chat/completions". apiKey, when non-empty, is sent as a Bearer
// token, and headers are applied after the defaults so a caller-supplied
// header can override them. selfHosted adds cache_prompt:true to every
// request -- the KV-cache prefix-reuse opt-in llama.cpp-style servers honor --
// and must stay false for hosted OpenAI/Azure, which reject unknown body
// fields with a 400. promptCache adds the two Anthropic-style ephemeral
// cache_control breakpoints in openai shape for Anthropic-fronting gateways
// that pass them through; replayReasoning echoes each assistant message's
// accumulated reasoning back as message.reasoning (the gateway extension a
// model needs to keep seeing its chain-of-thought). A nil httpClient uses
// http.DefaultClient.
//
// The fields are read-only during Complete, so a value is safe for concurrent
// use.
type openaiProvider struct {
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	userAgent       string
	selfHosted      bool
	promptCache     bool
	replayReasoning bool
	headers         map[string]string
}

// oaReserved are the Extra keys the typed core always overrides.
var oaReserved = set.Of("messages", "model", "stream", "tools")

// Complete implements Provider over a streaming chat completion. When the
// first attempt fails before anything streamed with a 400 that names NO
// recoverable parameter at all (Z.AI answers "Invalid API parameter, please
// check the documentation" -- no name whatsoever, so NewParamStripper's
// regexes have nothing to match and a caller wrapping this Provider in it
// gets no help), and the request carried the AUTO-added default
// stream_options (never a caller-requested field, only a usage-in-stream
// convenience), Complete retries once with that default left off. A 400 that
// DOES name a parameter is left untouched here -- that is NewParamStripper's
// job, and guessing "it must be stream_options" over a name that points
// somewhere else would just burn an extra round trip while the real culprit
// (e.g. a caller-supplied reasoning_effort) survives untouched into the
// retry. Dropping stream_options is always safe when it does fire: the
// caller never asked for it, and its absence only means no usage figures on
// this call, the same degradation an upstream with no stream_options support
// already produces. A context-overflow 400 is excluded too: it is permanent
// regardless of stream_options, and IsContextOverflow's callers expect it
// unretried.
func (o *openaiProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	comp, err := o.complete(ctx, req, ev, true)
	if comp != nil || err == nil || !o.shouldRetryWithoutStreamOptions(req, err) {
		return comp, err
	}
	return o.complete(ctx, req, ev, false)
}

// shouldRetryWithoutStreamOptions reports whether a failed first attempt is
// worth retrying with the default stream_options left off: the failure must
// be a pre-stream 400 whose text names no recoverable parameter (a named one
// is NewParamStripper's job, not a guess made here), and the request must
// actually have carried the AUTO-added default -- a caller-supplied
// stream_options (via Extra) is never touched.
func (o *openaiProvider) shouldRetryWithoutStreamOptions(req Request, err error) bool {
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 400 || ae.ContextOverflow {
		return false
	}
	if _, named := rejectedParamName(ae.Body); named {
		return false
	}
	_, hasOwn := req.ParamsFor(DialectOpenAI)["stream_options"]
	return !hasOwn
}

// complete runs one attempt of the streaming chat completion.
// includeDefaultStreamOptions gates the {"include_usage":true} default (see
// buildBody); Complete calls this twice only when a first attempt with it set
// is rejected outright.
func (o *openaiProvider) complete(ctx context.Context, req Request, ev *StreamEvents, includeDefaultStreamOptions bool) (*Completion, error) {
	body, err := o.buildBody(req, includeDefaultStreamOptions)
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

	// A 200 that is NOT an SSE stream is a plain JSON response -- the server
	// ignored stream:true (or a proxy buffered it) and answered with the
	// non-streaming shape. It is accepted transparently and reassembled into a
	// Completion with Streamed false, so the caller keeps a truthful record of
	// how the call was actually transported.
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		body, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, badRequestErr("openai: read non-streaming response: " + rerr.Error())
		}
		return o.parseNonStream(body)
	}

	st := &oaStream{ev: ev, acc: newToolCallAccumulator()}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("openai: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	comp := st.completion()
	if err := st.emitRemaining(comp); err != nil {
		return comp, err
	}
	return comp, nil
}

// buildBody assembles the JSON request body. Extra passthrough params are
// merged FIRST so the typed core fields always win; reserved keys in Extra
// (messages, model, stream, tools) are silently ignored so they cannot break
// routing. stream is always forced true, tools are sent only when non-empty
// (no tool_choice is ever sent), and stream_options defaults to
// {"include_usage":true} only when the caller has not supplied a
// stream_options key via Extra AND includeDefaultStreamOptions is true --
// without it OpenAI and most compatibles omit usage from streamed responses
// entirely, but a few (Z.AI) reject the field outright, which is what
// includeDefaultStreamOptions=false is for (see Complete's retry). MaxTokens
// > 0 sets max_tokens (overriding an Extra value); 0 leaves the field to
// Extra or the provider default. CacheKey, when set, rides as
// prompt_cache_key, and selfHosted adds cache_prompt:true. promptCache marks
// the per-request wire copy with the two ephemeral cache breakpoints;
// replayReasoning echoes assistant reasoning back as message.reasoning.
func (o *openaiProvider) buildBody(req Request, includeDefaultStreamOptions bool) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.ParamsFor(DialectOpenAI) {
		if oaReserved.Contains(k) {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	msgs, err := oaWireMessages(req.System, req.Messages, o.replayReasoning)
	if err != nil {
		return nil, err
	}
	if o.promptCache {
		oaMarkPromptCache(msgs)
	}
	body["messages"] = msgs
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
	if _, ok := body["stream_options"]; !ok && includeDefaultStreamOptions {
		body["stream_options"] = map[string]any{"include_usage": true}
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
// is distinguishable from an explicit zero (the tri-state contract). The
// provider-reported dollar figure rides under cost (OpenRouter/Requesty) or
// estimated_cost (DeepInfra), and reasoning tokens under
// completion_tokens_details.reasoning_tokens -- all pointers for the same
// tri-state reason.
type oaUsage struct {
	PromptTokens           int                      `json:"prompt_tokens"`
	CompletionTokens       int                      `json:"completion_tokens"`
	TotalTokens            int                      `json:"total_tokens"`
	PromptTokensDetails    *oaPromptTokensDetails   `json:"prompt_tokens_details"`
	CompletionTokensDetail *oaCompletionTokenDetail `json:"completion_tokens_details"`
	PromptCacheHitTokens   *int                     `json:"prompt_cache_hit_tokens"`
	CacheReadInputTokens   *int                     `json:"cache_read_input_tokens"`
	Cost                   *float64                 `json:"cost"`
	EstimatedCost          *float64                 `json:"estimated_cost"`
}

// oaPromptTokensDetails is the OpenAI breakdown of prompt tokens.
type oaPromptTokensDetails struct {
	CachedTokens *int `json:"cached_tokens"`
}

// oaCompletionTokenDetail is the OpenAI breakdown of completion tokens; the
// reasoning_tokens figure is the only field the library reads.
type oaCompletionTokenDetail struct {
	ReasoningTokens *int `json:"reasoning_tokens"`
}

// reasoningTokens returns the reported reasoning-token count, or nil when the
// snapshot carries none.
func (u *oaUsage) reasoningTokens() *int {
	if u.CompletionTokensDetail == nil {
		return nil
	}
	return clonePtr(u.CompletionTokensDetail.ReasoningTokens)
}

// costUsd returns the provider-reported dollar cost -- usage.cost wins over
// usage.estimated_cost -- or nil when neither field is present.
func (u *oaUsage) costUsd() *float64 {
	if u.Cost != nil {
		v := *u.Cost
		return &v
	}
	if u.EstimatedCost != nil {
		v := *u.EstimatedCost
		return &v
	}
	return nil
}

// toUsage normalizes a wire snapshot: the largest cache signal present wins
// (the dialects are mutually exclusive in practice) and lands in
// CacheReadTokens; when any cache info was reported, CacheWriteTokens is an
// explicit 0 -- OpenAI-compatible servers neither report nor bill a separate
// cache-write class -- while a snapshot with no cache fields at all leaves
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
		out = append(out, ToolCall{ID: tc.ID, Name: tc.Function.Name, Arguments: toolArgs(tc.Function.Arguments)})
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
	usages  []Usage
	timings []Timings
	sawData bool
	// sentParts counts the parts already delivered through OnPart, so the
	// remainder can go out at the end without any part going twice.
	sentParts int
}

// closeReasoning delivers the reasoning block once it can no longer grow.
func (st *oaStream) closeReasoning() error {
	if st.sentParts > 0 || st.reason.Len() == 0 {
		return nil
	}
	st.sentParts++
	return st.ev.EmitPart(ThinkingPart{Text: st.reason.String()})
}

// emitRemaining delivers the parts that only exist once the stream is over:
// the accumulated text, and the tool calls, which arrive as fragments keyed by
// index and are not a call until the last one lands.
func (st *oaStream) emitRemaining(comp *Completion) error {
	parts := comp.Message.EffectiveParts()
	if st.sentParts > len(parts) {
		return nil
	}
	if err := st.ev.EmitParts(parts[st.sentParts:]); err != nil {
		return err
	}
	st.sentParts = len(parts)
	return nil
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
		return st.ev.EmitProgress(*chunk.PromptProgress)
	}
	if chunk.Usage != nil {
		st.sawData = true
		// Every report is kept, in order, exactly as sent: which of them
		// counts -- and whether they are snapshots of one running total or
		// separate charges -- is the reader's call, not a translator's.
		u := chunk.Usage.toUsage()
		var raw struct {
			Usage json.RawMessage `json:"usage"`
		}
		if err := json.Unmarshal(data, &raw); err == nil {
			u.Raw = raw.Usage
		}
		u.ReasoningTokens = chunk.Usage.reasoningTokens()
		u.CostUsd = chunk.Usage.costUsd()
		st.usages = append(st.usages, u)
		if err := st.ev.EmitUsage(u); err != nil {
			return err
		}
	}
	if chunk.Timings != nil {
		st.sawData = true
		t := *chunk.Timings
		st.timings = append(st.timings, t)
		if err := st.ev.EmitTimings(t); err != nil {
			return err
		}
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != "" {
			st.sawData = true
			// Reasoning and content are two separate streams on this dialect,
			// reasoning first, so content starting IS the reasoning block
			// ending -- the last moment its part can be delivered in the order
			// it actually occupies.
			if err := st.closeReasoning(); err != nil {
				return err
			}
			st.content.WriteString(ch.Delta.Content)
			if err := st.ev.EmitText(ch.Delta.Content); err != nil {
				return err
			}
		}
		if rc := ch.Delta.reasoning(); rc != "" {
			st.sawData = true
			st.reason.WriteString(rc)
			if err := st.ev.EmitReasoning(rc); err != nil {
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
// usage with the total floored at prompt+completion (a genuine surplus --
// reasoning tokens -- is preserved), the last timings snapshot (nil when the
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
	return &Completion{
		Message:    oaAssistantMessage(st.reason.String(), st.content.String(), calls),
		Usages:     st.usages,
		Timings:    st.timings,
		Streamed:   true,
		StopReason: normalizeStopReason(finish),
	}
}

// oaAssistantMessage assembles one assistant turn's parts in the order this
// dialect produces them: reasoning first, then the content it produced, then
// the calls it asked for. Chat-completions accumulates reasoning and content as
// two separate streams rather than as interleaved blocks, so that order is all
// the wire actually says.
func oaAssistantMessage(reasoning, content string, calls []ToolCall) Message {
	var parts []Part
	if reasoning != "" {
		parts = append(parts, ThinkingPart{Text: reasoning})
	}
	if content != "" {
		parts = append(parts, TextPart{Text: content})
	}
	for _, c := range calls {
		parts = append(parts, ToolCallPart{ID: c.ID, Name: c.Name, Arguments: c.Arguments})
	}
	m := Message{Role: RoleAssistant, Parts: parts}
	m.SyncViews()
	return m
}

// oaNonStream is the non-streaming chat-completions response shape a server
// that ignores stream:true answers with.
type oaNonStream struct {
	Choices []struct {
		Message      oaNonStreamMessage `json:"message"`
		FinishReason string             `json:"finish_reason"`
	} `json:"choices"`
	Usage json.RawMessage `json:"usage"`
}

// oaNonStreamMessage is the assistant message inside a non-streaming choice.
// Content is a pointer because a tool-call-only response carries null.
type oaNonStreamMessage struct {
	Role      string       `json:"role"`
	Content   *string      `json:"content"`
	Reasoning string       `json:"reasoning"`
	ToolCalls []oaToolCall `json:"tool_calls"`
}

// parseNonStream reassembles a plain-JSON chat-completions response into a
// Completion with Streamed false -- the transparent acceptance of a server that
// ignored stream:true (or a proxy that buffered it). The stop reason is
// normalized and post-inferred exactly like the streamed path; usage extras
// (reasoning tokens, provider cost, the verbatim usage object) are captured
// when present.
func (o *openaiProvider) parseNonStream(data []byte) (*Completion, error) {
	var resp oaNonStream
	if err := json.Unmarshal(data, &resp); err != nil {
		return nil, badRequestErr("openai: decode non-streaming response: " + err.Error())
	}
	if len(resp.Choices) == 0 {
		return nil, badRequestErr("openai: non-streaming response carries no choices")
	}
	first := resp.Choices[0]
	content := ""
	if first.Message.Content != nil {
		content = *first.Message.Content
	}
	var calls []ToolCall
	for _, tc := range first.Message.ToolCalls {
		calls = append(calls, ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Arguments: toolArgs(tc.Function.Arguments),
		})
	}
	comp := &Completion{
		Message:    oaAssistantMessage(first.Message.Reasoning, content, calls),
		StopReason: normalizeStopReason(first.FinishReason),
		Streamed:   false,
	}
	if len(resp.Usage) > 0 {
		var u oaUsage
		if err := json.Unmarshal(resp.Usage, &u); err == nil {
			usage := u.toUsage()
			usage.Raw = resp.Usage
			usage.ReasoningTokens = u.reasoningTokens()
			usage.CostUsd = u.costUsd()
			comp.Usages = append(comp.Usages, usage)
		}
	}
	if comp.StopReason == "" {
		if len(comp.Message.ToolCalls) > 0 {
			comp.StopReason = StopToolUse
		} else {
			comp.StopReason = StopEndTurn
		}
	}
	return comp, nil
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
