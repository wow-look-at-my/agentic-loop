// Package goal is goal mode: a stopping condition the user states in words, and
// a stop policy that refuses to end a run until it holds.
//
// Every string this package emits is contract and is pinned by a test. A host
// shows these words to its user and sends the directive to its model, so
// hosts running goal mode behave the same. Do not "improve" them.
//
// The host owns seams and nothing else: where the state is stored, where
// the transcript comes from (Evaluator.Window), and how a model call is made
// (Evaluator.Judge). Everything else -- the parse, the counters, the evaluator
// prompt, the verdict, the notices -- is here, so a host cannot get a
// different answer from the same condition. Depth: docs/goal.md.
package goal

import (
	"fmt"
	"strings"
	"time"
)

// NoticeKind marks a stored goal notice: never in a prompt, never in a window.
const NoticeKind = "goal_notice"

// BriefingKind marks the stored Briefing: the goal message the model reads.
const BriefingKind = "goal_briefing"

// MaxCondition caps a condition at characters: it must read at a glance.
const MaxCondition = 4000

// State is the active goal; the counters are its bound. See docs/goal.md.
//
// Persisting it is the HOST's, and there is deliberately no Encode/Decode here:
// a library that ships a storage format decides what a row in somebody else's
// database holds, which is backwards. The fields are plain and exported so a
// host can map them onto its own columns, key-value rows, or file. The json
// tags are what a host maps BY, so an unset SetAt is omitzero rather than the
// year a time otherwise writes into a column that means "no goal".
type State struct {
	Condition string    `json:"condition"`
	SetAt     time.Time `json:"set_at,omitzero"`
	// Scope is the host's own opaque marker for where the goal started.
	Scope string `json:"scope,omitempty"`
	// Iterations is how many stops have been evaluated so far.
	Iterations int `json:"iterations"`
	// LastReason is the previous block's reason; ReasonRun counts it running.
	LastReason string `json:"last_reason,omitempty"`
	ReasonRun  int    `json:"reason_run,omitempty"`
	// Suspended is set by an honest failure to evaluate; it blocks nothing.
	Suspended  bool   `json:"suspended,omitempty"`
	SuspendWhy string `json:"suspend_why,omitempty"`
}

// Kind is what a parsed `/goal` invocation asks for.
type Kind uint8

const (
	// Show is `/goal` with no argument: report the active goal.
	Show Kind = iota
	// Set is `/goal <condition>`: set or replace it.
	Set
	// Clear is `/goal clear`: remove it early.
	Clear
)

// Command is a parsed `/goal` invocation.
type Command struct {
	Kind      Kind
	Condition string // set only when Kind is Set
}

// Parse reads the argument of `/goal`: everything after the command word,
// verbatim. Newlines survive, because a pasted checklist is a legitimate
// condition. The cap counts characters, the unit the notices quote and a pane
// wraps; len() would refuse a shorter condition for containing an accent.
func Parse(arg string) (Command, error) {
	condition := strings.TrimSpace(arg)
	switch {
	case condition == "":
		return Command{Kind: Show}, nil
	case condition == "clear":
		return Command{Kind: Clear}, nil
	}
	if n := len([]rune(condition)); n > MaxCondition {
		return Command{}, fmt.Errorf("goal condition is limited to %d characters (got %d)", MaxCondition, n)
	}
	return Command{Kind: Set, Condition: condition}, nil
}

// NoBound is SetNotice's line for a host that caps nothing.
const NoBound = "No spend or time bound: it runs until the condition holds, or until you clear it."

// SetNotice is what a user is shown when a goal is set. Its line NAMES
// THE BOUND, which only the host knows; empty is NoBound.
func SetNotice(condition, bound string) string {
	if bound == "" {
		bound = NoBound
	}
	return "goal set: " + quote(condition) + "\n" +
		"The turn will not end until this holds. /goal clear to stop, /goal to amend.\n" +
		bound
}

// Briefing is what the MODEL is told, when the goal is set. It is a notice
// in the transcript and a user-role message in the prompt.
func Briefing(condition string) string {
	return "A goal condition is now active for this session: " + condition + "\n\n" +
		"Treat the condition itself as your instruction. Start (or continue) working toward\n" +
		"it now — do not pause to ask the user what to do. When you try to end your turn the\n" +
		"condition is checked; if it does not hold you will be told why and asked to keep\n" +
		"going. It clears itself the moment it holds."
}

// Directive is the model-facing form of a blocked stop. Its last paragraph is
// what makes a queued interjection safe: a mid-run message never edits the
// condition, so a conflict surfaces as a visible stop with both texts quoted
// rather than as a run grinding against an instruction the user believes they
// changed. Depth: docs/goal.md.
func Directive(condition, reason string) string {
	return "[goal not met] " + reason + "\n\n" +
		"The goal condition is still: " + condition + "\n\n" +
		"Keep working toward it. Do not stop to ask whether to continue — the condition\n" +
		"itself is the instruction. If you now believe the condition cannot be satisfied,\n" +
		"say exactly what you tried and why it cannot be done, and stop; do not stop silently.\n\n" +
		"If the user has since told you something that conflicts with this condition, do not\n" +
		"try to satisfy both and do not guess which one they meant. Say which message conflicts\n" +
		"with which part of the condition, and stop."
}

// BlockNotice is the line a blocked stop puts in the transcript. An
// unpriced run omits spend rather than reporting: a total that silently
// excludes an unpriced call is the wrong number nobody checks.
func BlockNotice(iterations int, reason string, spend string, elapsed time.Duration) string {
	out := fmt.Sprintf("goal %d · not met: %s", iterations, reason)
	if spend != "" {
		out += " · " + spend
	}
	return out + " · " + Elapsed(elapsed)
}

// MetNotice is what the user sees when the goal clears itself.
func MetNotice(s *State, reason, spend string, elapsed time.Duration) string {
	head := fmt.Sprintf("goal met after %d iterations", s.Iterations)
	if spend != "" {
		head += " · " + spend
	}
	return head + " · " + Elapsed(elapsed) + "\n" +
		"  " + quote(s.Condition) + "\n" +
		"  → " + reason
}

// FailedNotice is what the user sees when the condition can never hold.
func FailedNotice(s *State, reason, spend string, elapsed time.Duration) string {
	head := fmt.Sprintf("goal failed after %d iterations", s.Iterations)
	if spend != "" {
		head += " · " + spend
	}
	return head + " · " + Elapsed(elapsed) + "\n  " + reason
}

// ClearedNotice is what `/goal clear` reports.
func ClearedNotice(s *State, spend string, elapsed time.Duration) string {
	out := fmt.Sprintf("goal cleared after %d iterations", s.Iterations)
	if spend != "" {
		out += " · " + spend
	}
	return out + " · " + Elapsed(elapsed)
}

// ShowNotice answers a bare `/goal`: the condition, its counters, and how to
// amend it -- a user who typed `/goal` was asking to edit the goal.
func ShowNotice(s *State, spend string, elapsed time.Duration) string {
	if s == nil {
		return "no goal is set — try: /goal all tests pass and go vet is clean"
	}
	out := "goal: " + quote(s.Condition) + "\n" +
		fmt.Sprintf("  %d iterations", s.Iterations)
	if spend != "" {
		out += " · " + spend
	}
	out += " · " + Elapsed(elapsed)
	if s.Suspended {
		out += "\n  suspended: " + s.SuspendWhy
	}
	return out + "\n  /goal <condition> to replace it, /goal clear to remove it."
}

// SuspendNotice is an honest failure: the stop is permitted, the goal is kept.
func SuspendNotice(why string) string { return "goal suspended: " + why }

// Elapsed renders a duration the way a goal notice does: "8m41s", "22m", "6s".
func Elapsed(d time.Duration) string {
	if d < 0 {
		d = 0
	}
	d = d.Round(time.Second)
	h := int(d / time.Hour)
	m := int(d/time.Minute) % 60
	s := int(d/time.Second) % 60
	switch {
	case h > 0 && m > 0:
		return fmt.Sprintf("%dh%dm", h, m)
	case h > 0:
		return fmt.Sprintf("%dh", h)
	case m > 0 && s > 0:
		return fmt.Sprintf("%dm%ds", m, s)
	case m > 0:
		return fmt.Sprintf("%dm", m)
	default:
		return fmt.Sprintf("%ds", s)
	}
}

// quote wraps a condition for a notice; a multi-line keeps its newlines.
func quote(s string) string { return `"` + s + `"` }
