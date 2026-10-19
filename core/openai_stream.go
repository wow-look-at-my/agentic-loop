package commonai

import (
	"encoding/json"
)

// oaChunk is SSE delta from a streaming chat completion.
type oaChunk struct {
	Choices     []oaChoice     `json:"choices"`
	Usage       *oaUsage       `json:"usage,omitempty"`
	Model       string         `json:"model,omitempty"`
	ExtraFields *oaExtraFields `json:"extra_fields,omitempty"`
	// Timings is a llama.cpp/ollama timing snapshot; each replaces the (last wins).
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
// under field names in the wild: reasoning_content (OpenAI/DeepSeek style)
// and reasoning (Ollama style). ReasoningDetails is OpenRouter's structured
// form, streamed as fragments keyed by index the same way ToolCalls is.
type oaDelta struct {
	Content          string              `json:"content,omitempty"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	Reasoning        string              `json:"reasoning,omitempty"`
	ReasoningDetails []oaReasoningDetail `json:"reasoning_details,omitempty"`
	ToolCalls        []oaToolCall        `json:"tool_calls,omitempty"`
}

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
// CacheReadTokens. CacheWriteTokens is an explicit -- OpenAI-compatible
// servers neither report nor bill a separate cache-write class. While a
// snapshot with no cache fields at all leaves both nil (unknown). This
// holds when any cache info was reported. prompt_tokens already includes
// cached tokens on this layer, so PromptTokens passes through untouched.
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

// oaStream writes a streamed chat completion into a CallSink, one node per part as it begins.
type oaStream struct {
	out CallSink
	// think and text are the open reasoning and text nodes. A reasoning node ends when text starts.
	think            *ThinkingMessage
	text             *AssistantMessage
	calls            map[int]*ToolCallMessage
	nodes            []IMessage
	announced        map[IMessage]bool
	reasoningDetails *reasoningDetailAccumulator
	finish           string
	usages           []Usage
	timings          []Timings
	model            string
	routed           string
	sawData          bool
}

func newOAStream(out CallSink) *oaStream {
	return &oaStream{out: out, calls: map[int]*ToolCallMessage{}, announced: map[IMessage]bool{},
		reasoningDetails: newReasoningDetailAccumulator()}
}

func (st *oaStream) start(n IMessage) {
	st.nodes = append(st.nodes, n)
	st.out.Start(n)
}

// done announces a node as finished, once.
func (st *oaStream) done(n IMessage) {
	if st.announced[n] {
		return
	}
	st.announced[n] = true
	st.out.Done(n)
}

// closeReasoning ends the reasoning node, which can no longer grow.
func (st *oaStream) closeReasoning() {
	if st.think != nil && !st.think.Final() {
		st.done(st.think)
	}
}

// announceRemaining announces the nodes that only end with the stream.
func (st *oaStream) announceRemaining() {
	for _, n := range st.nodes {
		st.done(n)
	}
}

// finishNodes ends every node still open, without announcing it.
func (st *oaStream) finishNodes() {
	if details := oaReasoningDetailsJSON(st.reasoningDetails.finish()); details != "" {
		if st.think == nil {
			st.think = NewThinking(DialectOpenAI)
			st.nodes = append([]IMessage{st.think}, st.nodes...)
			st.out.Start(st.think)
		}
		st.think.Details = ReasoningDetails(details)
	}
	for _, n := range st.nodes {
		FinishNode(n)
	}
}

// onData decodes payload; unparseable chunks ignored, usage newest-wins, timings replace.
func (st *oaStream) onData(data []byte) error {
	var chunk oaChunk
	if err := json.Unmarshal(data, &chunk); err != nil {
		return nil
	}
	if chunk.Model != "" {
		st.model = chunk.Model
	}
	if routed := chunk.ExtraFields.routed(); routed != "" {
		st.routed = routed
	}
	if chunk.PromptProgress != nil {
		st.sawData = true
		st.out.Progress(*chunk.PromptProgress)
		return st.out.Err()
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
		st.out.Usage(u)
		if err := st.out.Err(); err != nil {
			return err
		}
	}
	if chunk.Timings != nil {
		st.sawData = true
		t := *chunk.Timings
		st.timings = append(st.timings, t)
		st.out.Timings(t)
		if err := st.out.Err(); err != nil {
			return err
		}
	}
	for _, ch := range chunk.Choices {
		if ch.Delta.Content != "" {
			st.sawData = true
			// Content starting is when the reasoning block ends.
			st.closeReasoning()
			if st.text == nil {
				st.text = NewAssistant()
				st.start(st.text)
			}
			st.out.Delta(st.text, FieldText, ch.Delta.Content)
			if err := st.out.Err(); err != nil {
				return err
			}
		}
		if rc := ch.Delta.reasoning(); rc != "" {
			st.sawData = true
			if st.think == nil || st.think.Final() {
				st.think = NewThinking(DialectOpenAI)
				st.start(st.think)
			}
			st.out.Delta(st.think, FieldText, rc)
			if err := st.out.Err(); err != nil {
				return err
			}
		}
		if len(ch.Delta.ToolCalls) > 0 {
			st.sawData = true
			st.addToolCalls(ch.Delta.ToolCalls)
			if err := st.out.Err(); err != nil {
				return err
			}
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

// addToolCalls merges tool-call fragments. The first fragment of an index makes its node; later
// ones append arguments and fill an id or name that arrived late.
func (st *oaStream) addToolCalls(deltas []oaToolCall) {
	for _, d := range deltas {
		tc, ok := st.calls[d.Index]
		if !ok {
			tc = NewToolCall(CallID(d.ID), d.Function.Name)
			st.calls[d.Index] = tc
			st.start(tc)
		}
		if d.ID != "" {
			tc.CallID = CallID(d.ID)
		}
		if d.Function.Name != "" {
			tc.Name = d.Function.Name
		}
		if d.Function.Arguments != "" {
			st.out.Delta(tc, FieldArguments, d.Function.Arguments)
		}
	}
}

// completion ends every open node and gives the call as a Completion, partial on a broken stream.
func (st *oaStream) completion() *Completion {
	st.finishNodes()
	finish := st.finish
	if finish == "" && len(st.calls) > 0 {
		finish = "tool_calls"
	}
	if finish == "" {
		finish = "stop"
	}
	return &Completion{
		Message:    MessageFromNodes(st.nodes),
		Usages:     st.usages,
		Timings:    st.timings,
		Streamed:   true,
		StopReason: normalizeStopReason(finish),
		Model:      st.model,
		Routed:     st.routed,
	}
}
