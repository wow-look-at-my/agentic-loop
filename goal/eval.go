package goal

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"time"

	agentic "github.com/wow-look-at-my/agentic-loop"
)

// EvalTimeout bounds evaluator call.
const EvalTimeout = 30 * time.Second

// EvalMaxTokens is the evaluator's budget; Anthropic requires.
const EvalMaxTokens = 512

// A tool result in the window keeps its head and tail: the evidence of a test
// run lives at both ends and nowhere in the middle.
const (
	evalToolResultCap = 2000
	evalHead          = 1200
	evalTail          = 800
)

// EvalSystem is the evaluator's system prompt. Its bytes are contract.
const EvalSystem = `You are evaluating a stopping condition for a coding agent.

Below is a transcript of what the agent did. Judge, from the transcript alone,
whether the user's condition is satisfied. Do not assume work happened that the
transcript does not show. The agent's own claim that something is done is evidence,
not proof — look for the tool result that demonstrates it.

Reply with a single JSON object and nothing else:

  {"met": true,  "impossible": false, "reason": "<quote the transcript evidence>"}
  {"met": false, "impossible": false, "reason": "<name exactly what is missing>"}
  {"met": false, "impossible": true,  "reason": "<why this can never be satisfied>"}

"reason" is always required and is shown to the user verbatim, so write it for them:
one or two sentences, concrete, naming files and symbols where you can.

Set "impossible" only when the condition genuinely cannot be satisfied in this
session — it contradicts itself, it needs a resource that does not exist, or the
agent has tried every reasonable approach and demonstrated that it cannot be done.
Slow progress is not impossible. No progress yet is not impossible. When in doubt,
set "met": false and leave "impossible": false.`

// EvalUser is the evaluator's user message.
func EvalUser(condition, window string) string {
	return "Condition: " + condition + "\n\nTranscript:\n" + window
}

// Verdict is the evaluator's answer.
type Verdict struct {
	Met        bool   `json:"met"`
	Impossible bool   `json:"impossible"`
	Reason     string `json:"reason"`
}

// ParseVerdict reads verdict out of a model's reply. An empty reason is a
// parse failure rather than a verdict with nothing to say: the reason is the
// whole of what the user and the model are told, and a block carrying none is a
// refusal nobody can act on.
func ParseVerdict(text string) (Verdict, error) {
	var v Verdict
	body := strings.TrimSpace(unfence(text))
	if body == "" {
		return v, errors.New("goal: the evaluator returned nothing")
	}
	if err := json.Unmarshal([]byte(body), &v); err != nil {
		return v, fmt.Errorf("goal: the evaluator's reply is not a JSON object: %w", err)
	}
	if strings.TrimSpace(v.Reason) == "" {
		return v, errors.New("goal: the evaluator's verdict carries no reason")
	}
	return v, nil
}

// unfence strips a ```json fence when the model wrapped its object in one.
func unfence(text string) string {
	t := strings.TrimSpace(text)
	if !strings.HasPrefix(t, "```") {
		return t
	}
	if i := strings.IndexByte(t, '\n'); i >= 0 {
		t = t[i+1:]
	}
	if i := strings.LastIndex(t, "```"); i >= 0 {
		t = t[:i]
	}
	return t
}

// EvalTokens caps the rendered window. Oldest turns are dropped.
const EvalTokens = 40000

// OmittedMarker heads a window that dropped something. Why: docs/goal.md.
const OmittedMarker = "[earlier turns omitted]"

// EntryKind is what transcript entry is; a host maps its own rows onto these.
type EntryKind uint8

const (
	// EntryUser is a message from the user.
	EntryUser EntryKind = iota
	// EntryAssistant is the model's own text.
	EntryAssistant
	// EntryToolCall is a tool the model asked for, arguments included.
	EntryToolCall
	// EntryToolResult is what a tool answered.
	EntryToolResult
	// EntrySubagent is a sub-agent's activity or report.
	EntrySubagent
	// EntryError is a failure the run recorded.
	EntryError
)

// Entry is line of the window; things never map onto -- private
// reasoning, and goal mode's own notices (docs/goal.md).
type Entry struct {
	Kind EntryKind
	Text string
}

// EntriesFromMessages maps a library transcript onto window entries, for a host
// whose store IS []agentic.Message. A tool call becomes entry per call so a
// truncated window drops calls rather than halves of.
func EntriesFromMessages(msgs []agentic.Message) []Entry {
	out := make([]Entry, 0, len(msgs))
	for _, m := range msgs {
		switch m.Role {
		case agentic.RoleUser:
			out = append(out, Entry{Kind: EntryUser, Text: m.Content})
		case agentic.RoleTool:
			kind := EntryToolResult
			if m.ToolIsError {
				kind = EntryError
			}
			out = append(out, Entry{Kind: kind, Text: m.Content})
		case agentic.RoleAssistant:
			if strings.TrimSpace(m.Content) != "" {
				out = append(out, Entry{Kind: EntryAssistant, Text: m.Content})
			}
			for _, c := range m.ToolCalls {
				out = append(out, Entry{Kind: EntryToolCall, Text: c.Name + " " + c.Arguments})
			}
		}
	}
	return out
}

// RenderWindow turns the goal-scoped entries into the transcript the evaluator
// reads, newest kept, within a budget of maxTokens.
func RenderWindow(entries []Entry, maxTokens int) string {
	rendered := make([]string, 0, len(entries))
	for i := range entries {
		if line, ok := renderEntry(&entries[i]); ok {
			rendered = append(rendered, line)
		}
	}
	if len(rendered) == 0 {
		// Saying so beats handing the evaluator a blank string to interpret.
		return "(no transcript yet: no work has been recorded since the goal was set)"
	}

	// Newest while filling: a small budget keeps the most recent evidence.
	keep, budget := 0, maxTokens
	for i := len(rendered) - 1; i >= 0; i-- {
		budget -= estimateTokens(rendered[i])
		if budget < 0 && keep > 0 {
			break
		}
		keep++
	}
	body := strings.Join(rendered[len(rendered)-keep:], "\n\n")
	if keep < len(rendered) {
		return OmittedMarker + "\n" + body
	}
	return body
}

// estimateTokens is the chars/4 estimator; nothing here pretends to tokenize.
func estimateTokens(s string) int { return len(s)/4 + 1 }

// renderEntry renders entry, or reports that it carries nothing.
func renderEntry(e *Entry) (string, bool) {
	if strings.TrimSpace(e.Text) == "" {
		return "", false
	}
	switch e.Kind {
	case EntryUser:
		return "user: " + e.Text, true
	case EntryAssistant:
		return "assistant: " + e.Text, true
	case EntryToolCall:
		return "tool call: " + e.Text, true
	case EntryToolResult:
		return "tool result: " + truncateMiddle(e.Text), true
	case EntrySubagent:
		return "subagent: " + truncateMiddle(e.Text), true
	case EntryError:
		return "error: " + e.Text, true
	default:
		return "", false
	}
}

// truncateMiddle keeps both ends of a long result and says how much went.
func truncateMiddle(s string) string {
	r := []rune(s)
	if len(r) <= evalToolResultCap {
		return s
	}
	elided := len(r) - evalHead - evalTail
	return string(r[:evalHead]) +
		fmt.Sprintf("\n…[%d runes elided]…\n", elided) +
		string(r[len(r)-evalTail:])
}

// Judge makes bounded, tool-less call; the *Completion is what it spent.
type Judge func(ctx context.Context, system, user string) (*agentic.Completion, error)

// OneShotJudge is the ordinary Judge: bounded, tool-less call on the host's
// own provider and model, with the evaluator's own answer budget.
func OneShotJudge(p agentic.Provider, req agentic.Request) Judge {
	return func(ctx context.Context, system, user string) (*agentic.Completion, error) {
		r := req
		r.System = system
		// SystemParts outranks System, so the host's prompt would replace this.
		r.SystemParts = nil
		r.MaxTokens = EvalMaxTokens
		r.Messages = []agentic.Message{{Role: agentic.RoleUser, Content: user}}
		return agentic.OneShot(ctx, p, r, EvalTimeout)
	}
}

// Outcome is what evaluation decided; the host turns it into notices.
type Outcome uint8

const (
	// Permitted means the run may end and the goal stays as it is.
	Permitted Outcome = iota
	// Blocked means the stop was refused; Directive is what the model is told.
	Blocked
	// Met means the condition holds, so the goal clears.
	Met
	// Failed means the condition can never hold, so the goal clears.
	Failed
	// Suspended means the evaluator could not answer; the stop is PERMITTED.
	Suspended
)

// String names the outcome for a log line.
func (o Outcome) String() string {
	switch o {
	case Permitted:
		return "permitted"
	case Blocked:
		return "blocked"
	case Met:
		return "met"
	case Failed:
		return "failed"
	case Suspended:
		return "suspended"
	default:
		return "outcome(?)"
	}
}

// repetitionLimit identical reasons is a VERDICT, not a budget: docs/goal.md.
const repetitionLimit = 3

// Verdicts is what evaluation produced, for the host to record.
type Verdicts struct {
	Outcome   Outcome
	Reason    string
	Directive string
	// Completion is the evaluator call's own numbers, nil when no call was made.
	Completion *agentic.Completion
}

// Evaluator decides whether a run may stop. Evaluate mutates the state's
// counters, so the caller serializes: goroutine per state at a time.
type Evaluator struct {
	// State is the goal being evaluated; nil permits the stop.
	State *State
	// Window returns the goal-scoped entries, oldest.
	Window func() ([]Entry, error)
	// Judge makes the evaluator call.
	Judge Judge
}

// Evaluate makes the decision. It never returns an error: a provider that would
// not answer is a failure of this policy, whose stated failure direction is open
// -- so it suspends itself loudly and lets the run end.
func (e *Evaluator) Evaluate(ctx context.Context) Verdicts {
	// Cancellation always wins, and it wins WITHOUT a model call. A goal a user
	// cannot interrupt is a wedged session.
	if err := ctx.Err(); err != nil {
		return Verdicts{Outcome: Permitted, Reason: "the run was cancelled"}
	}
	if e.State == nil {
		return Verdicts{Outcome: Permitted, Reason: "no goal is set"}
	}
	if e.State.Suspended {
		return Verdicts{Outcome: Permitted, Reason: e.State.SuspendWhy}
	}
	if e.Window == nil || e.Judge == nil {
		return e.suspend("goal mode is not wired up: it has no transcript to read or no way to ask")
	}

	window, err := e.Window()
	if err != nil {
		return e.suspend("could not read the transcript to evaluate against (" + err.Error() + ") — /goal to retry")
	}

	verdict, comp, err := e.ask(ctx, RenderWindow(window, EvalTokens))
	if err != nil {
		// Cancelled mid-call is a cancellation, not an evaluator failure:
		// suspending over the user's own stop makes them re-arm the goal.
		if ctx.Err() != nil {
			return Verdicts{Outcome: Permitted, Reason: "the run was cancelled", Completion: comp}
		}
		out := e.suspend(err.Error())
		out.Completion = comp
		return out
	}

	e.State.Iterations++
	switch {
	case verdict.Met:
		return Verdicts{Outcome: Met, Reason: verdict.Reason, Completion: comp}
	case verdict.Impossible:
		return Verdicts{Outcome: Failed, Reason: verdict.Reason, Completion: comp}
	}

	if verdict.Reason == e.State.LastReason {
		e.State.ReasonRun++
	} else {
		e.State.LastReason, e.State.ReasonRun = verdict.Reason, 1
	}
	if e.State.ReasonRun >= repetitionLimit {
		return Verdicts{
			Outcome:    Failed,
			Reason:     fmt.Sprintf("the evaluator reported the same thing %d times running: %s", e.State.ReasonRun, verdict.Reason),
			Completion: comp,
		}
	}
	return Verdicts{
		Outcome:    Blocked,
		Reason:     verdict.Reason,
		Directive:  Directive(e.State.Condition, verdict.Reason),
		Completion: comp,
	}
}

// ask makes the evaluator call, with retry on an unusable answer. The retry
// sends the identical request: the failure it is for is a model that wrapped its
// object in prose, not a request that was wrong. A failure suspends, and
// the message names which of the failures it was.
func (e *Evaluator) ask(ctx context.Context, window string) (Verdict, *agentic.Completion, error) {
	user := EvalUser(e.State.Condition, window)

	var last *agentic.Completion
	var callErr, parseErr error
	for attempt := 0; attempt < 2; attempt++ {
		comp, err := e.Judge(ctx, EvalSystem, user)
		if comp != nil {
			last = comp
		}
		if err != nil {
			callErr = err
			if ctx.Err() != nil {
				return Verdict{}, last, err
			}
			continue
		}
		verdict, err := ParseVerdict(comp.Message.Content)
		if err == nil {
			return verdict, last, nil
		}
		parseErr = err
	}

	switch {
	case parseErr != nil && callErr != nil:
		return Verdict{}, last, fmt.Errorf("evaluator call failed (%v) and its other answer was unparseable (%v) — /goal to retry", callErr, parseErr)
	case parseErr != nil:
		return Verdict{}, last, errors.New("evaluator returned unparseable output twice")
	}
	return Verdict{}, last, fmt.Errorf("evaluator call failed (%v) — /goal to retry", callErr)
}

// suspend records the failure and fails open. The goal is kept, counters and
// all: a failure to evaluate is not a verdict on the condition.
func (e *Evaluator) suspend(why string) Verdicts {
	e.State.Suspended, e.State.SuspendWhy = true, why
	return Verdicts{Outcome: Suspended, Reason: why}
}
