package agentic

import "context"

// PromptProgress reports prompt-processing (prefill) progress, when the
// upstream emits it (ollama with the llama.cpp return_progress patch):
// Processed/Total is the fraction of the prompt ingested, Cache the portion
// served from the prompt cache, and TimeMS the elapsed prefill time in
// milliseconds. The JSON tags match the upstream prompt_progress object.
type PromptProgress struct {
	Processed int   `json:"processed"`
	Total     int   `json:"total"`
	Cache     int   `json:"cache"`
	TimeMS    int64 `json:"time_ms"`
}

// StreamEvents are optional streaming callbacks for one model call. All
// fields are optional and a nil *StreamEvents is valid: providers guard every
// emit. OnText receives content deltas, OnReasoning thinking/reasoning
// deltas, OnUsage each merged usage snapshot as it arrives, and OnProgress
// prefill progress updates.
type StreamEvents struct {
	OnText      func(string)
	OnReasoning func(string)
	OnUsage     func(Usage)
	OnProgress  func(PromptProgress)
}

// emitText forwards a non-empty content delta, tolerating nil receivers.
func (ev *StreamEvents) emitText(s string) {
	if ev == nil || ev.OnText == nil || s == "" {
		return
	}
	ev.OnText(s)
}

// emitReasoning forwards a non-empty reasoning delta, tolerating nil receivers.
func (ev *StreamEvents) emitReasoning(s string) {
	if ev == nil || ev.OnReasoning == nil || s == "" {
		return
	}
	ev.OnReasoning(s)
}

// emitUsage forwards a usage snapshot, tolerating nil receivers.
func (ev *StreamEvents) emitUsage(u Usage) {
	if ev == nil || ev.OnUsage == nil {
		return
	}
	ev.OnUsage(u)
}

// emitProgress forwards a prefill-progress update, tolerating nil receivers.
func (ev *StreamEvents) emitProgress(p PromptProgress) {
	if ev == nil || ev.OnProgress == nil {
		return
	}
	ev.OnProgress(p)
}

// probeEvents wraps ev so the caller can observe whether the provider
// delivered any stream event. Retry layers use it to detect a "clean" failure
// (nothing streamed) that is safe to re-attempt.
func probeEvents(ev *StreamEvents, delivered *bool) *StreamEvents {
	return &StreamEvents{
		OnText:      func(s string) { *delivered = true; ev.emitText(s) },
		OnReasoning: func(s string) { *delivered = true; ev.emitReasoning(s) },
		OnUsage:     func(u Usage) { *delivered = true; ev.emitUsage(u) },
		OnProgress:  func(p PromptProgress) { *delivered = true; ev.emitProgress(p) },
	}
}

// Request is one model call. Messages is the transcript; System is the system
// prompt, delivered in the dialect's native position (an OpenAI system
// message, the Anthropic top-level system field).
//
// Extra is a verbatim top-level passthrough for provider-specific parameters
// (reasoning_effort, temperature, num_ctx, thinking, ...). It is merged into
// the request body FIRST, so the typed core fields always win: for OpenAI the
// reserved keys are model/messages/stream/tools, and stream_options defaults
// to {"include_usage":true} only when Extra does not already carry a
// stream_options key; for Anthropic the reserved keys are
// model/max_tokens/stream/system/messages/tools. The library never interprets
// or gates Extra values (no model-specific tables) — what to send is the
// caller's decision.
//
// MaxTokens 0 omits the max_tokens field for OpenAI (the provider default
// governs); Anthropic REQUIRES it and its provider fails fast when it is not
// positive. CacheKey, when non-empty, is sent to OpenAI as prompt_cache_key (a
// cache-routing hint); Anthropic ignores it.
type Request struct {
	Model     string
	System    string
	Messages  []Message
	Tools     []Tool
	MaxTokens int
	Extra     map[string]any
	CacheKey  string
}

// Normalized stop reasons. OpenAI finish reasons are mapped (stop → end_turn,
// tool_calls → tool_use, length → max_tokens); Anthropic already uses these
// names. Anything else passes through raw.
const (
	StopEndTurn   = "end_turn"
	StopToolUse   = "tool_use"
	StopMaxTokens = "max_tokens"
)

// Completion is the outcome of one model call: the assembled assistant
// message, the call's final (merged, total-floored) usage, and the normalized
// stop reason.
type Completion struct {
	Message    Message
	Usage      Usage
	StopReason string
}

// Provider executes one streaming model call. Implementations stream under
// the hood and deliver deltas through ev (which may be nil).
//
// On a mid-stream failure or cancellation AFTER data has arrived, Complete
// returns the PARTIAL *Completion alongside the error — both non-nil — so the
// caller can keep the partial content, reasoning, and the last usage
// snapshot. Before any data (connection errors, non-2xx responses), the
// completion is nil.
//
// Providers must be safe for concurrent use by multiple goroutines.
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}
