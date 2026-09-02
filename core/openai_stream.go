package commonai

import (
	"encoding/json"
	"strings"
)

// oaChunk is SSE delta from a streaming chat completion.
type oaChunk struct {
	Choices []oaChoice `json:"choices"`
	Usage   *oaUsage   `json:"usage,omitempty"`
	// Timings is a llama.cpp/ollama timing snapshot; each replaces the previous (last wins).
	Timings *Timings `json:"timings,omitempty"`
	// PromptProgress is a non-standard prefill-progress update emitted on a choices-less chunk.
	PromptProgress *PromptProgress `json:"prompt_progress,omitempty"`
}

// oaChoice is choice within a chunk.
type oaChoice struct {
	Delta        oaDelta `json:"delta"`
	FinishReason string  `json:"finish_reason,omitempty"`
}

// oaDelta is the incremental content of a streaming choice. Reasoning arrives
// under field names in the wild: reasoning_content (OpenAI/DeepSeek
// style) and reasoning (Ollama style). ReasoningDetails is OpenRouter's
// structured form, streamed as fragments keyed by index the same way
// ToolCalls is.
type oaDelta struct {
	Content          string              `json:"content,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	Reasoning        string              `json:"reasoning,omitempty"`
	ReasoningDetails []oaReasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []oaToolCall        `json:"tool_calls,omitempty"`
}

// reasoning returns the delta's reasoning text from whichever field the
// upstream used; reasoning_content wins when both are present.
func (d oaDelta) reasoning() string {
	if d.ReasoningContent != "" {
		return d.ReasoningContent
	}
	return d.Reasoning
}

// oaUsage is the wire usage shape; cache fields are pointers so absent != explicit.
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

// oaCompletionTokenDetail is the completion-token breakdown; reasoning_tokens is the only field read.
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
// explicit -- OpenAI-compatible servers neither report nor bill a separate
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

// toolCallAccumulator reassembles tool calls by index; delta has id/name, rest append args.
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

// finish returns the assembled calls in the order their indices
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

// reasoningDetailAccumulator merges streamed reasoning_details fragments by index.
type reasoningDetailAccumulator struct {
	byIndex map[int]*oaReasoningDetail
	order   []int
}

func newReasoningDetailAccumulator() *reasoningDetailAccumulator {
	return &reasoningDetailAccumulator{byIndex: map[int]*oaReasoningDetail{}}
}

// add merges streamed reasoning_details deltas into the accumulator:
// type/id/format/signature overwrite when non-empty, text/summary/data fragments concatenate.
func (a *reasoningDetailAccumulator) add(deltas []oaReasoningDetail) {
	for _, d := range deltas {
		rd, ok := a.byIndex[d.Index]
		if !ok {
			rd = &oaReasoningDetail{Index: d.Index}
			a.byIndex[d.Index] = rd
			a.order = append(a.order, d.Index)
		}
		if d.Type != "" {
			rd.Type = d.Type
		}
		if d.ID != "" {
			rd.ID = d.ID
		}
		if d.Format != "" {
			rd.Format = d.Format
		}
		if d.Signature != nil {
			rd.Signature = d.Signature
		}
		rd.Text += d.Text
		rd.Summary += d.Summary
		rd.Data += d.Data
	}
}

// finish returns the assembled details in the order their indices appeared.
func (a *reasoningDetailAccumulator) finish() []oaReasoningDetail {
	if len(a.order) == 0 {
		return nil
	}
	out := make([]oaReasoningDetail, 0, len(a.order))
	for _, idx := range a.order {
		out = append(out, *a.byIndex[idx])
	}
	return out
}

// oaStream accumulates streamed completion.
type oaStream struct {
	ev               *StreamEvents
	content          strings.Builder
	reason           strings.Builder
	acc              *toolCallAccumulator
	reasoningDetails *reasoningDetailAccumulator
	finish           string
	usages           []Usage
	timings          []Timings
	sawData          bool
	// sentParts counts parts already delivered via OnPart; the rest goes out at the end.
	sentParts int
}

// closeReasoning delivers the reasoning block it can no longer grow.
func (st *oaStream) closeReasoning() error {
	if st.sentParts > 0 || st.reason.Len() == 0 {
		return nil
	}
	st.sentParts++
	return st.ev.EmitPart(ThinkingPart{Text: st.reason.String()})
}

// emitRemaining delivers the parts that only exist the stream is over:
// the accumulated text, and the tool calls, which arrive as fragments keyed by
// index and are not a call until the last lands.
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

// onData decodes payload; unparseable chunks ignored, usage newest-wins, timings replace.
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
		// Every usage report is kept in order as sent; which counts is the reader's call, not ours.
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
			// Reasoning and content are separate streams; content starting IS when the reasoning block ends.
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
		if len(ch.Delta.ReasoningDetails) > 0 {
			st.sawData = true
			st.reasoningDetails.add(ch.Delta.ReasoningDetails)
		}
		if ch.FinishReason != "" {
			st.sawData = true
			st.finish = ch.FinishReason
		}
	}
	return nil
}

// completion assembles the final (or partial) result: accumulated content,
// reasoning as a single ThinkingBlock (its Signature carrying the verbatim
// reasoning_details array when the upstream sent), assembled tool calls,
// the merged usage with the total floored at prompt+completion (a genuine
// surplus -- reasoning tokens -- is preserved), the last timings snapshot
// (nil when the upstream reported none), and the normalized stop reason.
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
	details := oaReasoningDetailsJSON(st.reasoningDetails.finish())
	return &Completion{
		Message:    oaAssistantMessage(st.reason.String(), details, st.content.String(), calls),
		Usages:     st.usages,
		Timings:    st.timings,
		Streamed:   true,
		StopReason: normalizeStopReason(finish),
	}
}
