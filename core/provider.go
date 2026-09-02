package commonai

import (
	"context"
	"time"
)

// PromptProgress reports prefill progress (Processed/Total, Cache, TimeMS); JSON tags match upstream.
type PromptProgress struct {
	Processed int   `json:"processed"`
	Total     int   `json:"total"`
	Cache     int   `json:"cache"`
	TimeMS    int64 `json:"time_ms"`
}

// Timings is the llama.cpp-style timing snapshot some OpenAI-compatible upstreams attach.
type Timings struct {
	PromptN     int     `json:"prompt_n,omitempty"`
	PromptMS    float64 `json:"prompt_ms,omitempty"`
	PredictedN  int     `json:"predicted_n,omitempty"`
	PredictedMS float64 `json:"predicted_ms,omitempty"`
}

// RetryAttempt describes a failed call about to be re-sent (attempt, backoff, why) so a caller can show them.
type RetryAttempt struct {
	Attempt int
	Of      int
	Delay   time.Duration
	Err     error
}

// StreamEvents are optional callbacks; a non-nil error ABORTS the stream, returning partial result.
type StreamEvents struct {
	OnText      func(string) error
	OnReasoning func(string) error
	// OnPart fires when a content part is COMPLETE, delivered exactly, in finished-message order.
	OnPart     func(Part) error
	OnUsage    func(Usage) error
	OnProgress func(PromptProgress) error
	OnTimings  func(Timings) error
	// OnRetry fires before each backoff; returning an error stops the retrying and surfaces it.
	OnRetry func(RetryAttempt) error
}

// The Emit helpers deliver events, tolerating nil receivers, so a failed sink is never re-sent.

// EmitText forwards a non-empty content delta, tolerating nil receivers.
func (ev *StreamEvents) EmitText(s string) error {
	if ev == nil || ev.OnText == nil || s == "" {
		return nil
	}
	return wrapCallbackErr(ev.OnText(s))
}

// EmitReasoning forwards a non-empty reasoning delta, tolerating nil receivers.
func (ev *StreamEvents) EmitReasoning(s string) error {
	if ev == nil || ev.OnReasoning == nil || s == "" {
		return nil
	}
	return wrapCallbackErr(ev.OnReasoning(s))
}

// EmitPart forwards a finished content part, tolerating nil receivers.
func (ev *StreamEvents) EmitPart(p Part) error {
	if ev == nil || ev.OnPart == nil || p == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnPart(p))
}

// EmitParts forwards several finished parts in order.
func (ev *StreamEvents) EmitParts(parts []Part) error {
	if ev == nil || ev.OnPart == nil {
		return nil
	}
	for _, p := range parts {
		if err := ev.EmitPart(p); err != nil {
			return err
		}
	}
	return nil
}

// EmitUsage forwards a usage snapshot, tolerating nil receivers.
func (ev *StreamEvents) EmitUsage(u Usage) error {
	if ev == nil || ev.OnUsage == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnUsage(u))
}

// EmitProgress forwards a prefill-progress update, tolerating nil receivers.
func (ev *StreamEvents) EmitProgress(p PromptProgress) error {
	if ev == nil || ev.OnProgress == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnProgress(p))
}

// EmitTimings forwards a timings snapshot, tolerating nil receivers.
func (ev *StreamEvents) EmitTimings(t Timings) error {
	if ev == nil || ev.OnTimings == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnTimings(t))
}

// EmitRetry forwards an imminent retry, tolerating nil receivers.
func (ev *StreamEvents) EmitRetry(a RetryAttempt) error {
	if ev == nil || ev.OnRetry == nil {
		return nil
	}
	return wrapCallbackErr(ev.OnRetry(a))
}

// Extra merges so core fields win; MaxTokens omits for OpenAI, REQUIRES Anthropic
type Request struct {
	Model    string
	System   string
	Messages []Message
	Tools    []ToolDecl
	// SystemParts is the system prompt as ordered parts; System is its flattened text view.
	SystemParts []Part
	MaxTokens   int
	Extra       map[string]any
	// DialectExtra is provider-specific parameters for dialect, so a request can carry both.
	DialectExtra map[Dialect]map[string]any
	CacheKey     string

	// AutoCompact is the fraction (..) of the context window at which the loop compacts; disables it.
	AutoCompact float64
}

// EffectiveSystemParts is the system prompt as ordered parts: SystemParts when
// set, otherwise the System string as a single text part.
func (r Request) EffectiveSystemParts() []Part {
	if len(r.SystemParts) > 0 {
		return r.SystemParts
	}
	if r.System == "" {
		return nil
	}
	return []Part{TextPart{Text: r.System}}
}

// ParamsFor is the parameters a dialect should actually send: the
// dialect-agnostic Extra, overlaid with anything addressed to that dialect
// specifically. The caller's maps are never modified.
func (r Request) ParamsFor(d Dialect) map[string]any {
	own := r.DialectExtra[d]
	if len(own) == 0 {
		return r.Extra
	}
	out := make(map[string]any, len(r.Extra)+len(own))
	for k, v := range r.Extra {
		out[k] = v
	}
	for k, v := range own {
		out[k] = v
	}
	return out
}

// Normalized stop reasons; OpenAI finish reasons are mapped to these, anything else passes raw.
const (
	StopEndTurn   = "end_turn"
	StopToolUse   = "tool_use"
	StopMaxTokens = "max_tokens"
)

// Completion is the outcome of model call: the message, usage reports, and the stop reason.
type Completion struct {
	Message    Message
	Usages     []Usage
	Timings    []Timings
	Streamed   bool
	StopReason string
}

// Provider runs streaming model call; after data arrives, returns partial Completion + error.
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}
