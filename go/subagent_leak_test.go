package agentic

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The observed failure, verbatim in shape: a sub-agent's last turn is working
// notes followed by a raw tool-call envelope, because the backend did not parse
// the model's tool-call template. Handed up unchanged it reads as findings.
const leakedReport = `Both memtraffic.go and traffic.go define ` + "`estimateMemoryTraffic`" + ` — that's a new critical finding.
Let me check the descriptorGroup struct in dram.go and its ` + "`taps`" + ` field.

<｜｜DSML｜｜tool_calls>
<｜｜DSML｜｜invoke name="Native_Tools__grep">
<｜｜DSML｜｜parameter name="path" string="true">/workspace/pkg/analyzer</｜｜DSML｜｜parameter>
</｜｜DSML｜｜invoke>
</｜｜DSML｜｜tool_calls>`

func TestSubagentReportCutsALeakedToolCallEnvelope(t *testing.T) {
	res := subagentReport(leakedReport)

	assert.NotContains(t, res.Content, "DSML", "the envelope must not reach the orchestrator")
	assert.NotContains(t, res.Content, "invoke name=")
	assert.Contains(t, res.Content, "estimateMemoryTraffic", "the real prose survives")
	assert.Contains(t, res.Content, SubagentCutOffNote)
	assert.True(t, res.IsError, "an interrupted attempt is not an answer")
}

func TestSubagentReportRejectsAnEnvelopeOnlyAnswer(t *testing.T) {
	for _, opener := range leakedToolCallOpeners {
		res := subagentReport(opener + "\nname=grep\n")
		assert.Equal(t, SubagentNoReportText, res.Content, "opener %q", opener)
		assert.True(t, res.IsError, "opener %q", opener)
	}
}

// A leak that follows only a scrap of prose is still no report: two words of
// working notes are not findings.
func TestSubagentReportRejectsALeakAfterAScrap(t *testing.T) {
	res := subagentReport("Let me check.\n\n<tool_call>\n{\"name\":\"grep\"}\n")
	assert.Equal(t, SubagentNoReportText, res.Content)
	assert.True(t, res.IsError)
}

// The false positive that matters: a sub-agent asked ABOUT tool calling quotes
// these tokens in prose. Mangling a real report is the worse failure, so only a
// line that STARTS with an envelope counts.
func TestSubagentReportKeepsProseThatMentionsToolCallMarkup(t *testing.T) {
	report := "The backend emits `<tool_call>` blocks when the template is unparsed, " +
		"and pkg/parse handles the <|tool_calls|> form mid-line as literal text. " +
		"Nothing else in the repository looks at these markers."
	res := subagentReport(report)

	assert.Equal(t, report, res.Content)
	assert.False(t, res.IsError)
	assert.NotContains(t, res.Content, SubagentCutOffNote)
}

func TestSubagentReportPassesACleanAnswerThrough(t *testing.T) {
	res := subagentReport("\n  Every path routes through Middleware(); no unauthenticated handlers.  \n")
	assert.Equal(t, "Every path routes through Middleware(); no unauthenticated handlers.", res.Content)
	assert.False(t, res.IsError)
}

func TestSplitLeakedToolCallsCutsAtTheLineStart(t *testing.T) {
	clean, leaked := splitLeakedToolCalls("findings here\n\n   <tool_call>\nrest")
	require.True(t, leaked)
	assert.Equal(t, "findings here", clean)

	clean, leaked = splitLeakedToolCalls("no envelope at all")
	assert.False(t, leaked)
	assert.Equal(t, "no envelope at all", clean)

	// An envelope on the very first line leaves nothing.
	clean, leaked = splitLeakedToolCalls("<|tool_calls|>\nwhatever")
	require.True(t, leaked)
	assert.Equal(t, "", strings.TrimSpace(clean))
}
