package goal

import (
	"context"

	agentic "github.com/wow-look-at-my/agentic-loop"
)

// DirectiveKind is the Message.Kind a blocked stop's directive carries.
const DirectiveKind = "goal_directive"

// StopListener is goal mode wired to a run: asked at every stop boundary, and a
// blocked stop queues the directive, which takes the loop round again IN PLACE.
// Depth: docs/goal.md.
type StopListener struct {
	// Evaluator decides; a nil State permits the stop.
	Evaluator *Evaluator
	// Report hears every verdict, on the run's goroutine, before the loop moves on.
	Report func(Verdicts)
	// cb is held because Subscribe takes a pointer and does not own the callback.
	cb func(agentic.StopEvent) error
}

// Attach subscribes the listener to a run's stop boundary. ctx is the RUN's
// context, which is what makes cancellation win, and msgs must be the queue the
func (l *StopListener) Attach(ctx context.Context, events *agentic.Events, msgs *agentic.MessageQueue) {
	l.cb = func(agentic.StopEvent) error {
		l.evaluate(ctx, msgs)
		return nil
	}
	events.OnStop.Subscribe(&l.cb)
}

// evaluate runs the policy and queues the directive when the stop is refused. It
// never fails the run: an error out of an OnStop listener loses the answer the
// model just wrote, and goal mode's failure direction is open.
func (l *StopListener) evaluate(ctx context.Context, msgs *agentic.MessageQueue) {
	if l.Evaluator == nil {
		return
	}
	v := l.Evaluator.Evaluate(ctx)
	if l.Report != nil {
		l.Report(v)
	}
	if v.Outcome != Blocked {
		return
	}
	msgs.Queue(agentic.SystemMessage{Message: agentic.Message{
		Role:    agentic.RoleUser,
		Kind:    DirectiveKind,
		Content: v.Directive,
	}})
}
