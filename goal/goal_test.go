package goal_test

import (
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/agentic-loop/goal"
	"github.com/wow-look-at-my/go-containers/set"
)

func TestParseShowSetAndClear(t *testing.T) {
	cmd, err := goal.Parse("")
	require.NoError(t, err)
	assert.Equal(t, goal.Show, cmd.Kind)

	cmd, err = goal.Parse("   ")
	require.NoError(t, err)
	assert.Equal(t, goal.Show, cmd.Kind, "an argument that is only whitespace is Show")

	cmd, err = goal.Parse("clear")
	require.NoError(t, err)
	assert.Equal(t, goal.Clear, cmd.Kind)

	cmd, err = goal.Parse("  the four auth tests pass  ")
	require.NoError(t, err)
	assert.Equal(t, goal.Set, cmd.Kind)
	assert.Equal(t, "the four auth tests pass", cmd.Condition)
}

func TestParseKeepsNewlinesInACondition(t *testing.T) {
	cmd, err := goal.Parse("- tests pass\n- vet is clean\n")
	require.NoError(t, err)
	assert.Equal(t, "- tests pass\n- vet is clean", cmd.Condition,
		"a pasted checklist is a legitimate condition and must not be flattened")
}

func TestParseCountsTheCapInCharactersNotBytes(t *testing.T) {
	// Every rune here is multi-byte: a byte count would refuse it well under the cap.
	within := string(make([]rune, 0, goal.MaxCondition))
	for i := 0; i < goal.MaxCondition; i++ {
		within += "é"
	}
	cmd, err := goal.Parse(within)
	require.NoError(t, err)
	assert.Equal(t, goal.Set, cmd.Kind)

	_, err = goal.Parse(within + "é")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "limited to 4000 characters")
}

func TestTheThreeMessageKindsAreDistinct(t *testing.T) {
	kinds := []string{goal.DirectiveKind, goal.NoticeKind, goal.BriefingKind}
	seen := set.New[string]()
	for _, k := range kinds {
		assert.NotEmpty(t, k)
		assert.False(t, seen.Contains(k), "a host switches on these, so no two may collide")
		seen.Add(k)
	}
}

func TestDecodeEmptyIsNoGoalAndCorruptIsAnError(t *testing.T) {
	state, err := goal.Decode(nil)
	require.NoError(t, err)
	assert.Nil(t, state)

	_, err = goal.Decode([]byte("{not json"))
	require.Error(t, err)

	_, err = goal.Decode([]byte(`{"condition":"  "}`))
	require.Error(t, err, "a stored goal with no condition is reported, never read as absent")
}

func TestEncodeDecodeRoundTripsTheCounters(t *testing.T) {
	in := &goal.State{
		Condition:  "tests pass",
		SetAt:      time.Date(2026, 8, 26, 3, 14, 0, 0, time.UTC),
		Scope:      "msg_42",
		Iterations: 5,
		LastReason: "two of four still fail",
		ReasonRun:  2,
	}
	raw, err := in.Encode()
	require.NoError(t, err)
	out, err := goal.Decode(raw)
	require.NoError(t, err)
	assert.Equal(t, in, out)
}

// The notices are contract: a host shows these words to its user.
func TestNoticesAreContract(t *testing.T) {
	assert.Equal(t,
		"goal set: \"tests pass\"\n"+
			"The turn will not end until this holds. /goal clear to stop, /goal to amend.\n"+
			"No spend or time bound: it runs until the condition holds, or until you clear it.",
		goal.SetNotice("tests pass", ""))

	assert.Equal(t,
		"goal set: \"tests pass\"\n"+
			"The turn will not end until this holds. /goal clear to stop, /goal to amend.\n"+
			"Bound: $5.00. It suspends the goal rather than clearing it.",
		goal.SetNotice("tests pass", "Bound: $5.00. It suspends the goal rather than clearing it."),
		"the third line names the bound, which only the host knows")

	assert.Equal(t,
		"goal 3 · not met: two of four still fail · $0.12 · 8m41s",
		goal.BlockNotice(3, "two of four still fail", "$0.12", 8*time.Minute+41*time.Second))

	assert.Equal(t,
		"goal 3 · not met: two of four still fail · 8m41s",
		goal.BlockNotice(3, "two of four still fail", "", 8*time.Minute+41*time.Second),
		"an unpriced run omits spend rather than reporting zero")

	state := &goal.State{Condition: "tests pass", Iterations: 4}
	assert.Equal(t,
		"goal met after 4 iterations · 22m\n  \"tests pass\"\n  → all four pass",
		goal.MetNotice(state, "all four pass", "", 22*time.Minute))
	assert.Equal(t,
		"goal failed after 4 iterations · 22m\n  the repo has no test suite",
		goal.FailedNotice(state, "the repo has no test suite", "", 22*time.Minute))
	assert.Equal(t,
		"goal cleared after 4 iterations · $1.00 · 22m",
		goal.ClearedNotice(state, "$1.00", 22*time.Minute))
	assert.Equal(t, "goal suspended: the evaluator is unreachable",
		goal.SuspendNotice("the evaluator is unreachable"))
}

func TestDirectiveNamesTheConditionAndForbidsASilentStop(t *testing.T) {
	d := goal.Directive("tests pass", "two of four still fail")
	assert.Contains(t, d, "[goal not met] two of four still fail")
	assert.Contains(t, d, "The goal condition is still: tests pass")
	assert.Contains(t, d, "do not stop silently")
	assert.Contains(t, d, "Say which message conflicts",
		"a conflicting interjection has to surface as a visible stop")
}

func TestBriefingTellsTheModelToStartWithoutAsking(t *testing.T) {
	b := goal.Briefing("tests pass")
	assert.Contains(t, b, "A goal condition is now active for this session: tests pass")
	assert.Contains(t, b, "do not pause to ask the user what to do")
}

func TestShowNoticeReportsTheStateAndHowToAmendIt(t *testing.T) {
	assert.Equal(t, "no goal is set — try: /goal all tests pass and go vet is clean",
		goal.ShowNotice(nil, "", 0))

	state := &goal.State{Condition: "tests pass", Iterations: 2}
	assert.Equal(t,
		"goal: \"tests pass\"\n  2 iterations · $0.40 · 1m2s\n"+
			"  /goal <condition> to replace it, /goal clear to remove it.",
		goal.ShowNotice(state, "$0.40", 62*time.Second))

	state.Suspended, state.SuspendWhy = true, "the evaluator is unreachable"
	assert.Contains(t, goal.ShowNotice(state, "", time.Second), "\n  suspended: the evaluator is unreachable")
}

func TestElapsedRendersMinutesAndSeconds(t *testing.T) {
	assert.Equal(t, "0s", goal.Elapsed(-time.Second))
	assert.Equal(t, "6s", goal.Elapsed(6*time.Second))
	assert.Equal(t, "22m", goal.Elapsed(22*time.Minute))
	assert.Equal(t, "8m41s", goal.Elapsed(8*time.Minute+41*time.Second))
	assert.Equal(t, "2h", goal.Elapsed(2*time.Hour))
	assert.Equal(t, "2h5m", goal.Elapsed(2*time.Hour+5*time.Minute+3*time.Second))
}
