package loop

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Sub-agent runs are ASYNCHRONOUS: run_subagent returns as soon as the sub-agent is launched.

// SubagentState is the lifecycle state of one launched sub-agent.
type SubagentState string

const (
	// SubagentQueued means launched but waiting for a concurrency slot.
	SubagentQueued SubagentState = "queued"
	// SubagentRunning means the nested loop is executing.
	SubagentRunning SubagentState = "running"
	// SubagentDone means the sub-agent returned its report.
	SubagentDone SubagentState = "done"
	// SubagentFailed means the run ended in error (failed model call or recoverable misuse text).
	SubagentFailed SubagentState = "error"
	// SubagentAbandoned means the turn ended while the sub-agent was still running.
	SubagentAbandoned SubagentState = "abandoned"
)

// SubagentReport is one finished sub-agent's result, waiting to be delivered.
type SubagentReport struct {
	// CallID is the parent run_subagent tool call id; also the live-activity telemetry id.
	CallID string
	// Label is the orchestrator's own one-line description of the task.
	Label string
	// Text is the sub-agent's final report (or the error text).
	Text    string
	IsError bool
	// Usages is what this sub-agent spent: one entry per model call, in order.
	Usages []Usage
	// Duration is wall-clock from launch to report; browser-only, text stays deterministic.
	Duration time.Duration
}

// subagentRun is the registry's per-run bookkeeping.
type subagentRun struct {
	callID    string
	label     string
	prompt    string
	state     SubagentState
	startedAt time.Time
}

// SubagentUpdate is one lifecycle change; transient telemetry, never in a model's context.
type SubagentUpdate struct {
	CallID string
	Label  string
	// Prompt is the task text, carried on the first (queued) update only.
	Prompt string
	State  SubagentState
	// Report/IsError set on the terminal update; Usages is the run's total; Duration is wall-clock.
	Report   string
	IsError  bool
	Usages   []Usage
	Duration time.Duration
}

// SubagentRuns tracks launched sub-agents: still out vs reported but undelivered; nil-safe.
type SubagentRuns struct {
	onUpdate func(SubagentUpdate)
	now      func() time.Time

	mu sync.Mutex
	// active holds runs that have not reported yet, keyed by call id.
	active map[string]*subagentRun
	// ready holds arrived reports not yet delivered into the conversation, oldest first.
	ready []SubagentReport
	// signal is a capacity-1 notification channel: Complete pokes it, Collect waits.
	signal chan struct{}
	// seq numbers ids for backends that assign none, so two launches never collide.
	seq int
}

// nextID mints an id for a launch whose tool call carried none.
func (r *SubagentRuns) nextID() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	return "subagent_" + strconv.Itoa(r.seq)
}

// NewCallID mints a fresh, unique sub-agent call id; exported counterpart of nextID.
func (r *SubagentRuns) NewCallID() string { return r.nextID() }

// NewSubagentRuns returns an empty registry; onUpdate gets one update per change.
func NewSubagentRuns(onUpdate func(SubagentUpdate)) *SubagentRuns {
	return &SubagentRuns{
		onUpdate: onUpdate,
		now:      time.Now,
		active:   make(map[string]*subagentRun),
		signal:   make(chan struct{}, 1),
	}
}

// Launch registers a sub-agent as outstanding, in the queued state. callID is
// the parent tool call's id; label is the orchestrator's description and
// prompt its task text (both are shown in the UI, never re-sent to a model).
// Launching the same call id twice is ignored — a tool call is executed once.
func (r *SubagentRuns) Launch(callID, label, prompt string) {
	r.mu.Lock()
	if _, dup := r.active[callID]; dup {
		r.mu.Unlock()
		return
	}
	run := &subagentRun{callID: callID, label: label, prompt: prompt, state: SubagentQueued, startedAt: r.now()}
	r.active[callID] = run
	r.mu.Unlock()
	r.emit(SubagentUpdate{CallID: callID, Label: label, Prompt: prompt, State: SubagentQueued})
}

// MarkRunning reports that a queued sub-agent acquired its concurrency slot
// and is now executing. Unknown call ids are ignored.
func (r *SubagentRuns) MarkRunning(callID string) {
	r.mu.Lock()
	run, ok := r.active[callID]
	if ok {
		run.state = SubagentRunning
	}
	r.mu.Unlock()
	if !ok {
		return
	}
	r.emit(SubagentUpdate{CallID: callID, Label: run.label, State: SubagentRunning})
}

// Complete records a finished report and wakes Collect; unknown call ids are ignored.
func (r *SubagentRuns) Complete(callID, text string, isErr bool, usages []Usage) {
	r.mu.Lock()
	run, ok := r.active[callID]
	if !ok {
		r.mu.Unlock()
		return
	}
	delete(r.active, callID)
	rep := SubagentReport{
		CallID: callID, Label: run.label, Text: text, IsError: isErr,
		Usages:   usages,
		Duration: r.now().Sub(run.startedAt),
	}
	r.ready = append(r.ready, rep)
	r.mu.Unlock()

	state := SubagentDone
	if isErr {
		state = SubagentFailed
	}
	r.emit(SubagentUpdate{
		CallID: callID, Label: rep.Label, State: state,
		Report: text, IsError: isErr, Usages: usages, Duration: rep.Duration,
	})
	select {
	case r.signal <- struct{}{}:
	default:
	}
}

// Pending reports how many sub-agents are unfinished or undelivered.
func (r *SubagentRuns) Pending() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.active) + len(r.ready)
}

// Running reports how many sub-agents are still executing.
func (r *SubagentRuns) Running() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.active)
}

// Collect blocks until a sub-agent reports, then returns every report ready then.
func (r *SubagentRuns) Collect(ctx context.Context) ([]SubagentReport, error) {
	if r == nil {
		return nil, nil
	}
	for {
		r.mu.Lock()
		if len(r.ready) > 0 {
			out := r.ready
			r.ready = nil
			r.mu.Unlock()
			return out, nil
		}
		outstanding := len(r.active)
		r.mu.Unlock()
		if outstanding == 0 {
			return nil, nil
		}
		select {
		case <-r.signal:
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
}

// Take drains arrived reports without waiting; the capped-final-turn path.
func (r *SubagentRuns) Take() []SubagentReport {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	out := r.ready
	r.ready = nil
	return out
}

// CancelRemaining marks every still-running sub-agent abandoned and returns the count.
func (r *SubagentRuns) CancelRemaining() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	stranded := make([]*subagentRun, 0, len(r.active))
	for _, run := range r.active {
		stranded = append(stranded, run)
	}
	r.active = make(map[string]*subagentRun)
	r.mu.Unlock()
	for _, run := range stranded {
		r.emit(SubagentUpdate{CallID: run.callID, Label: run.label, State: SubagentAbandoned})
	}
	return len(stranded)
}

// emit hands one lifecycle update to the host; telemetry is best-effort.
func (r *SubagentRuns) emit(u SubagentUpdate) {
	if r.onUpdate == nil {
		return
	}
	r.onUpdate(u)
}

// Delivery is a user-role message (accepted mid-thread); the header marks it unmistakable.
const SubagentDeliveryHeader = "[automated delivery -- this is the sub-agent notification you were promised, not a message from the user]"

// FormatSubagentDelivery renders finished reports; still/abandoned are stated for the model.
func FormatSubagentDelivery(reports []SubagentReport, still, abandoned int) string {
	var b strings.Builder
	b.WriteString(SubagentDeliveryHeader)
	for _, rep := range reports {
		b.WriteString("\n\n")
		verb := "finished"
		if rep.IsError {
			verb = "FAILED"
		}
		b.WriteString("Sub-agent " + verb + " -- " + describeSubagent(rep.Label) + ":\n\n")
		text := strings.TrimSpace(rep.Text)
		if text == "" {
			text = "(the sub-agent returned no report)"
		}
		b.WriteString(text)
	}
	b.WriteString("\n\n")
	switch {
	case abandoned == 1:
		b.WriteString("1 sub-agent was still running when this turn hit its limit and was abandoned; " +
			"its work is lost. Answer with what you have, or relaunch it in your next turn.")
	case abandoned > 1:
		b.WriteString(strconv.Itoa(abandoned) + " sub-agents were still running when this turn hit its limit " +
			"and were abandoned; their work is lost. Answer with what you have, or relaunch them in your next turn.")
	case still == 1:
		b.WriteString("1 sub-agent is still running; you will be notified when it reports. " +
			"Continue with other work, or if you have nothing to do until it lands, say so briefly and stop.")
	case still > 1:
		b.WriteString(strconv.Itoa(still) + " sub-agents are still running; you will be notified as each reports. " +
			"Continue with other work, or if you have nothing to do until they land, say so briefly and stop.")
	default:
		b.WriteString("No sub-agents are still running.")
	}
	return b.String()
}

// describeSubagent renders a run's label for the delivery text, falling back
// to a neutral phrase when the orchestrator gave none.
func describeSubagent(label string) string {
	label = strings.TrimSpace(label)
	if label == "" {
		return "(no description given)"
	}
	return strconv.Quote(label)
}
