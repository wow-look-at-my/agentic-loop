package commonai

import (
	"encoding/json"
	"strings"
)

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
// layer, so PromptTokens passes through untouched.
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

// onData decodes one SSE payload. Unparseable chunks are tolerated silently;
// a prompt_progress chunk is forwarded and carries nothing else; usage
// snapshots are merged newest-wins (never summed) so both the OpenAI
// single-final-chunk and the xAI usage-on-every-chunk conventions yield the
// same result; timings snapshots replace each other. A callback error aborts
// the stream.
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
			// Reasoning and content are two separate streams on this layer,
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
