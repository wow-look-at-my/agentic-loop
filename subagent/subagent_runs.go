package subagent

import (
	"context"
	"strconv"
	"strings"
	"sync"
	"time"
)

// Sub-agent runs are ASYNCHRONOUS: run_subagent returns as soon as the
// sub-agent is launched, so the orchestrator can launch several at once and
// keep working while they run. SubagentRuns is the registry that makes the
// promise in that launch receipt true — it is the one place that knows a
// sub-agent is outstanding, and agentic.Run consults it before finishing a turn.
//
// The two halves:
//
//   - The run_subagent tool calls Launch, MarkRunning and Complete from the
//     goroutine it spawns per sub-agent.
//   - The loop calls Pending/Collect: a turn that would otherwise end while
//     sub-agents are still out instead waits for the next report and delivers
//     it into the conversation, which is what "you will be notified" means.
//
// One registry per turn (SubagentConfig.Runs; a nil one makes run_subagent
// synchronous -- it blocks until the sub-agent answers).

// SubagentState is the lifecycle state of one launched sub-agent, as reported
// to the browser on the "subagent_update" event.
type SubagentState string

const (
	// SubagentQueued means launched but waiting for a concurrency slot.
	SubagentQueued SubagentState = "queued"
	// SubagentRunning means the nested loop is executing.
	SubagentRunning SubagentState = "running"
	// SubagentDone means the sub-agent returned its report.
	SubagentDone SubagentState = "done"
	// SubagentFailed means the run ended in an error (a failed model call, or
	// the library's recoverable misuse text — both reach the orchestrator).
	SubagentFailed SubagentState = "error"
	// SubagentAbandoned means the turn ended (its cap was reached) while the
	// sub-agent was still running, so its report will never be delivered. It
	// is reported rather than dropped quietly — an abandoned run cost real
	// tokens and answered nothing.
	SubagentAbandoned SubagentState = "abandoned"
)

// SubagentReport is one finished sub-agent's result, waiting to be delivered
// into the parent conversation.
type SubagentReport struct {
	// CallID is the parent run_subagent tool call that launched it, which is
	// also the id the live-activity telemetry is stamped with.
	CallID string
	// Label is the orchestrator's own one-line description of the task.
	Label string
	// Text is the sub-agent's final report (or the error text).
	Text    string
	IsError bool
	// Duration is wall-clock from launch to report. Reported to the browser
	// only — the delivered text stays deterministic.
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

// SubagentUpdate is one lifecycle change, delivered to the callback
// NewSubagentRuns was given. It is transient telemetry: a host renders it, and
// it never re-enters any model's context.
type SubagentUpdate struct {
	CallID string
	Label  string
	// Prompt is the task text, carried on the first (queued) update only.
	Prompt string
	State  SubagentState
	// Report and IsError are set on the terminal update; Duration is
	// wall-clock from launch to report.
	Report   string
	IsError  bool
	Duration time.Duration
}

// SubagentRuns tracks the sub-agents launched during one turn: which are still
// out, and which have reported but not yet been delivered to the model. It is
// safe for concurrent use — every launch runs on its own goroutine — and a nil
// *SubagentRuns answers every question as "nothing outstanding", so a loop can
// consult one it was never given.
type SubagentRuns struct {
	onUpdate func(SubagentUpdate)
	now      func() time.Time

	mu sync.Mutex
	// active holds runs that have not reported yet, keyed by call id.
	active map[string]*subagentRun
	// ready holds reports that have arrived but not yet been delivered into
	// the conversation, oldest first.
	ready []SubagentReport
	// signal is a capacity-1 notification channel: Complete pokes it, Collect
	// waits on it. A non-blocking send is enough because Collect always
	// re-reads the whole ready slice under the lock.
	signal chan struct{}
	// seq numbers the ids minted for backends that assign none, so two
	// launches in one turn can never collide.
	seq int
}

// nextID mints an id for a launch whose tool call carried none. A backend that
// assigns no tool_call id is rare but real, and two sub-agents sharing an id
// would report over each other.
func (r *SubagentRuns) nextID() string {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.seq++
	return "subagent_" + strconv.Itoa(r.seq)
}

// NewSubagentRuns returns an empty registry. onUpdate receives one
// SubagentUpdate per lifecycle change (nil = no telemetry); it must be safe
// for concurrent use, since launches report from their own goroutines while
// the turn's own events are still being written.
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

// Complete records a finished sub-agent's report and wakes any Collect waiting
// on it. The report stays pending until Collect hands it to the loop, so a
// sub-agent that finishes while the model is mid-turn is delivered at the end
// of that turn rather than lost. Unknown call ids are ignored.
func (r *SubagentRuns) Complete(callID, text string, isErr bool) {
	r.mu.Lock()
	run, ok := r.active[callID]
	if !ok {
		r.mu.Unlock()
		return
	}
	delete(r.active, callID)
	rep := SubagentReport{
		CallID: callID, Label: run.label, Text: text, IsError: isErr,
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
		Report: text, IsError: isErr, Duration: rep.Duration,
	})
	select {
	case r.signal <- struct{}{}:
	default:
	}
}

// Delivery waits for the next ready report(s) and returns the model-facing
// user-message text the loop should append. Empty string means nothing to
// deliver (idle, or Collect returned no reports).
func (r *SubagentRuns) Delivery(ctx context.Context) (string, error) {
	reports, err := r.Collect(ctx)
	if err != nil || len(reports) == 0 {
		return "", err
	}
	return FormatSubagentDelivery(reports, r.Running(), 0), nil
}

// Pending reports how many sub-agents are unfinished or have reported without
// being delivered yet. agentic.Run uses it to decide whether a turn that produced no
// tool calls may actually end.
func (r *SubagentRuns) Pending() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.active) + len(r.ready)
}

// Running reports how many sub-agents are still executing (nothing delivered
// can bring them back). It is what the delivery message tells the model is
// still outstanding.
func (r *SubagentRuns) Running() int {
	if r == nil {
		return 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.active)
}

// Collect blocks until at least one sub-agent has reported, then returns EVERY
// report ready at that moment (so several finishing together cost one delivery,
// not one each). It returns ctx.Err() if the turn is cancelled first, and nil,
// nil when nothing is outstanding at all.
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

// Take drains the reports that have already arrived without waiting for the
// ones still running. It is the capped-final-turn path: there is no turn left
// to consume a delivery, so what is in hand is delivered and nothing is waited
// for.
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

// CancelRemaining marks every still-running sub-agent abandoned and returns
// how many there were. The runs themselves stop when the turn's context is
// cancelled; this is what makes their loss visible — in the columns, and in
// the count the delivery message states.
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

// emit hands one lifecycle update to the host. Telemetry is best-effort, like
// the activity steps: a host that does not listen never fails a sub-agent.
func (r *SubagentRuns) emit(u SubagentUpdate) {
	if r.onUpdate == nil {
		return
	}
	r.onUpdate(u)
}

// The delivery message. Reports come back into the conversation as a
// user-role message — the one role every OpenAI-compatible upstream accepts
// mid-thread — so it is labeled unmistakably as an automated delivery. How a
// host stores and shows it is the host's business; the TEXT is contract,
// because it is what the model reads.

// SubagentDeliveryHeader opens every delivery so a model reading the
// transcript can never mistake it for the user speaking.
const SubagentDeliveryHeader = "[automated delivery -- this is the sub-agent notification you were promised, not a message from the user]"

// FormatSubagentDelivery renders finished reports as the delivered message
// text. still is how many sub-agents remain outstanding after this delivery
// (stated explicitly so the model can choose between answering now and waiting
// for the rest — it will be notified either way); abandoned is how many were
// dropped because the turn cap left no turn to deliver them into.
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
