package goal

import (
	"context"

	agentic "github.com/wow-look-at-my/agentic-loop"
)

// DirectiveKind is the Message.Kind a blocked stop's directive carries, so a
// host can store and render it as what it is rather than as a user turn.
const DirectiveKind = "goal_directive"

// StopListener is goal mode wired to a run: it is asked at every stop boundary,
// and a blocked stop queues the directive, which takes the loop round again.
//
// Re-arming in place is the point. A fresh run would replay the transcript from
// the top and pay for it, and the model would read the directive as the opening
// of a new conversation rather than as the answer to the stop it just tried.
type StopListener struct {
	// Evaluator decides. Its State is the goal, and a nil State permits the stop.
	Evaluator *Evaluator
	// Report receives every verdict, on the run's own goroutine, before the loop
	// moves on. It is how the host records the notice, persists the counters and
	// clears a goal that is met -- a queued directive says only what the MODEL is
	// told, and the session has to record the rest.
	Report func(Verdicts)
	// cb is held so the subscription outlives Attach: Subscribe takes a pointer
	// to the callback and does not own it.
	cb func(agentic.StopEvent) error
}

// Attach subscribes the listener to a run's stop boundary.
//
// ctx is the RUN's context, which is what makes cancellation win: a user who
// stopped the run gets the stop permitted without a model call. msgs must be the
// queue the run was configured with, or a blocked stop queues into nothing and
// the run ends anyway.
func (l *StopListener) Attach(ctx context.Context, events *agentic.Events, msgs *agentic.MessageQueue) {
	l.cb = func(agentic.StopEvent) error {
		l.evaluate(ctx, msgs)
		return nil
	}
	events.OnStop.Subscribe(&l.cb)
}

// evaluate runs the policy and queues the directive when the stop is refused.
//
// It never returns an error. An error out of an OnStop listener aborts the run
// and loses the answer the model just wrote; goal mode's failure direction is
// open, and a policy that cannot evaluate says so through Report.
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
