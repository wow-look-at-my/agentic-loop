package agentic

import (
	"context"
	"encoding/json"
	"time"
)

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

// Timings is the llama.cpp-style per-call timing snapshot some
// OpenAI-compatible upstreams (llama.cpp, ollama) attach to streamed chunks:
// PromptN prompt tokens processed in PromptMS milliseconds (prefill), and
// PredictedN tokens generated in PredictedMS milliseconds. The field names and
// types are wire-faithful to the upstream timings object. The library only
// surfaces what the provider reported -- it never synthesizes timings from
// wall-clock time; that remains the caller's choice. The Anthropic dialect
// has no equivalent and never reports timings.
type Timings struct {
	PromptN     int     `json:"prompt_n,omitempty"`
	PromptMS    float64 `json:"prompt_ms,omitempty"`
	PredictedN  int     `json:"predicted_n,omitempty"`
	PredictedMS float64 `json:"predicted_ms,omitempty"`
}

// RetryAttempt describes a failed model call that is about to be re-sent:
// which attempt just failed (1-based) out of how many are allowed, how long
// the backoff before the next one is, and why it failed. It exists so a
// caller can SHOW the failure and the wait -- retrying is otherwise a silent
// gap that, at the default ten attempts of uncapped backoff, can run for
// minutes with no sign the call is still alive.
type RetryAttempt struct {
	Attempt int
	Of      int
	Delay   time.Duration
	Err     error
}

// StreamEvents are optional streaming callbacks for one model call. All
// fields are optional and a nil *StreamEvents is valid: providers guard every
// emit. OnText receives content deltas, OnReasoning thinking/reasoning
// deltas, OnUsage each merged usage snapshot as it arrives, OnProgress
// prefill progress updates, OnTimings each provider-reported timings
// snapshot (OpenAI dialect only -- Anthropic never fires it), and OnRetry a
// transient failure about to be re-attempted.
//
// A non-nil error returned by any callback ABORTS the stream read
// immediately: Complete stops consuming the upstream response and returns the
// partial *Completion (content, reasoning, tool calls, and usage accumulated
// so far) together with that error. The error is returned unwrapped or
// %w-wrapped, so errors.Is against a sentinel the callback returned holds; it
// is never converted into an *APIError and never classified transient, so
// neither the retry policy nor the param-strip middleware will re-send a call
// whose sink failed. A provider marks data as seen BEFORE each emit, so even a
// callback that fails on the very first delta yields a partial completion --
// which is what tells the layers above the call is no longer safe to re-send.
type StreamEvents struct {
	OnText      func(string) error
	OnReasoning func(string) error
	OnUsage     func(Usage) error
	OnProgress  func(PromptProgress) error
	OnTimings   func(Timings) error
	// OnRetry fires before each backoff, from the retry layer rather than
	// from a dialect provider. Returning an error stops the retrying and
	// surfaces that error in place of the upstream's. It does NOT count as
	// "streamed something": a notification about a failed attempt cannot make
	// the next one unsafe.
	OnRetry func(RetryAttempt) error
}

// emitText forwards a non-empty content delta, tolerating nil receivers.
func (ev *StreamEvents) emitText(s string) error {
	if ev == nil || ev.OnText == nil || s == "" {
		return nil
	}
	return wrapCallbackErr(ev.OnText(s))
}

// emitReasoning forwards a non-empty reasoning delta, tolerating nil receivers.
func (ev *StreamEvents) emitReasoning(s string) error {
	if ev == nil || ev.OnReasoning == nil || s == "" {
		return nil
	}
	return wrapCallbackErr(ev.OnReasoning(s))
}

// emitUsage forwards a usage snapshot, tolerating nil receivers.
func (ev *StreamEvents) emitUsage(u Usage) error {
	if ev == nil || ev.OnUsage == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnUsage(u))
}

// emitProgress forwards a prefill-progress update, tolerating nil receivers.
func (ev *StreamEvents) emitProgress(p PromptProgress) error {
	if ev == nil || ev.OnProgress == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnProgress(p))
}

// emitTimings forwards a timings snapshot, tolerating nil receivers.
func (ev *StreamEvents) emitTimings(t Timings) error {
	if ev == nil || ev.OnTimings == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnTimings(t))
}

// emitRetry forwards an imminent retry, tolerating nil receivers.
func (ev *StreamEvents) emitRetry(a RetryAttempt) error {
	if ev == nil || ev.OnRetry == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnRetry(a))
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
// or gates Extra values (no model-specific tables) -- what to send is the
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
//
// UsageReported is true iff the provider reported at least one usage snapshot
// during the call. Usage is a value type, so a caller reading only the
// returned Completion could not otherwise distinguish an upstream that
// reported all-zero usage from one that reported none at all (common on local
// OpenAI-compatible servers) -- check UsageReported before persisting or
// displaying Usage.
//
// Timings is the last provider-reported timings snapshot (llama.cpp-style
// upstreams attach one per chunk; the last one wins), or nil when the
// provider never reported timings -- a tri-state like the Usage cache fields.
// The Anthropic dialect never sets it.
//
// RawUsage is the provider's usage object verbatim (the raw JSON the upstream
// sent, or, for the Anthropic dialect where usage arrives as fragments, the
// merged wire-shaped object), for logging and for extracting provider extras --
// reasoning-token and dollar-cost figures -- the normalized Usage drops.
// ReasoningTokens is the openai usage.completion_tokens_details.reasoning_tokens
// figure when present (tri-state; Anthropic never reports one). CostUsd is the
// provider-reported dollar cost from usage.cost or usage.estimated_cost when
// present (OpenRouter/Requesty/DeepInfra style; tri-state). Each stays nil
// unless the upstream reported it -- never zero-fill or estimate.
//
// Streamed records whether the response actually arrived as an SSE stream. A
// 2xx that is NOT text/event-stream is read as a plain JSON body and
// reassembled into a Completion with Streamed false -- a server that ignores
// stream:true is accepted transparently, and the flag preserves the truth of
// how the call was transported.
type Completion struct {
	Message         Message
	Usage           Usage
	UsageReported   bool
	Timings         *Timings
	RawUsage        json.RawMessage
	ReasoningTokens *int
	CostUsd         *float64
	Streamed        bool
	StopReason      string
}

// Provider executes one streaming model call. Implementations stream under
// the hood and deliver deltas through ev (which may be nil). Build one with
// the wire dialect's constructor -- NewOpenAIProvider or NewAnthropicProvider;
// everything else in the library (Run, OneShot, Compact, NewParamStripper,
// the built-in tool executors) works against this interface.
//
// On a mid-stream failure or cancellation AFTER data has arrived -- including
// a stream callback returning an error -- Complete MUST return the PARTIAL
// *Completion alongside the error, both non-nil, so the caller can keep the
// partial content, reasoning, and the last usage snapshot. Before any data
// (connection errors, non-2xx responses), the completion MUST be nil.
//
// That rule is load-bearing, not just convenient: a non-nil completion is how
// the layers above tell that a failed call already streamed, and therefore
// that re-sending it would duplicate output the caller has seen. Retry and
// NewParamStripper both read it, and neither watches the callbacks to
// second-guess it. An implementation that emits deltas and then returns a nil
// completion will be re-sent.
//
// Providers must be safe for concurrent use by multiple goroutines.
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}
