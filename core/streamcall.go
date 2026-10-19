package commonai

import "context"

// Streamer is a provider that writes its call into a CallSink as it streams.
type Streamer interface {
	Stream(ctx context.Context, req Request, out CallSink) (*Completion, error)
}

// StreamCall runs one call into out. A provider that is not a Streamer runs through Complete, and
// its result becomes nodes when the call returns.
func StreamCall(ctx context.Context, p Provider, req Request, out CallSink) (*Completion, error) {
	if s, ok := p.(Streamer); ok {
		return s.Stream(ctx, req, out)
	}
	comp, err := p.Complete(ctx, req, SinkEvents(out))
	if comp != nil {
		AddTurn(out, comp.Message)
	}
	return comp, err
}

// AddTurn adds the nodes of a turn that arrived whole, from a provider that cannot stream nodes.
func AddTurn(out CallSink, m Message) {
	for _, n := range NodesFromMessage(m, DialectAuto) {
		out.Start(n)
		FinishNode(n)
	}
}

// SinkEvents are stream events that report into out, for a provider that only speaks StreamEvents.
func SinkEvents(out CallSink) *StreamEvents { return sinkEvents(out) }

// HostEventer is a sink that carries a host's stream events, which a plain
// provider's text and reasoning deltas reach directly.
type HostEventer interface {
	HostEvents() *StreamEvents
}

// sinkEvents forwards the stream events a plain provider can report into out.
func sinkEvents(out CallSink) *StreamEvents {
	var host *StreamEvents
	if h, ok := out.(HostEventer); ok {
		host = h.HostEvents()
	}
	return &StreamEvents{
		OnText:      func(s string) error { return host.EmitText(s) },
		OnReasoning: func(s string) error { return host.EmitReasoning(s) },
		OnPart:      func(p Part) error { return host.EmitPart(p) },
		OnUsage:     func(u Usage) error { out.Usage(u); return out.Err() },
		OnProgress:  func(p PromptProgress) error { out.Progress(p); return out.Err() },
		OnTimings:   func(t Timings) error { out.Timings(t); return out.Err() },
		OnRetry:     func(a RetryAttempt) error { out.Retry(a); return out.Err() },
	}
}

// completionIntoSink adds the nodes of a call that arrived whole, such as a plain JSON response.
func completionIntoSink(comp *Completion, err error) func(out CallSink, d Dialect) (*Completion, error) {
	return func(out CallSink, d Dialect) (*Completion, error) {
		if comp == nil {
			return nil, err
		}
		for _, n := range NodesFromMessage(comp.Message, d) {
			out.Start(n)
			FinishNode(n)
		}
		for _, u := range comp.Usages {
			out.Usage(u)
		}
		for _, t := range comp.Timings {
			out.Timings(t)
		}
		if err == nil {
			err = out.Err()
		}
		return comp, err
	}
}
