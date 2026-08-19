package subagent

import (
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
)

// A sub-agent's report is whatever text its last turn produced. That is only
// meaningful if the last turn was an ANSWER — and sometimes it is not: a model
// whose chat template the backend did not fully parse emits its tool calls as
// TEXT, so the run ends with a message that reads like working notes followed
// by a raw call envelope:
//
//	Both files define estimateMemoryTraffic -- that's a new critical finding.
//	Let me check the descriptorGroup struct in dram.go.
//
//	<|tool_calls|>
//	<|invoke name="grep"><|parameter name="pattern">type descriptorGroup...
//
// Handed up unchanged, that is worse than a failure: the orchestrator reads
// "critical finding" as a conclusion, and the markup as part of the answer.
// The sub-agent had not finished — it was interrupted, usually by running out
// of turns while still working.
//
// So a report that ends in a leaked call envelope is cut at the envelope and
// labelled as partial, and one that is nothing BUT an envelope is reported as
// no report at all.

// leakedToolCallOpeners are the envelope tokens seen from backends that fail
// to parse a model's tool-call template. They are matched only at the START of
// a line: prose ABOUT tool calls quotes these mid-sentence (this repository's
// own documentation does), and mangling a legitimate report is the worse
// failure of the two.
var leakedToolCallOpeners = []string{
	"<|tool_calls|>",
	"<|tool_call|>",
	"<tool_call>",
	"<tool_calls>",
	"<|python_tag|>",
	"<function=",
	"<｜tool▁calls▁begin｜>",
	"<｜｜DSML｜｜tool_calls>",
	"<｜｜DSML｜｜invoke",
}

// SubagentCutOffNote is appended to a report that was cut short by a leaked
// tool-call envelope, so the orchestrator treats what survived as partial
// rather than as the sub-agent's conclusions.
const SubagentCutOffNote = "\n\n[cut short: the sub-agent tried to call another tool instead of answering, " +
	"so everything above is partial working notes, NOT its conclusions. Relaunch it with a narrower task " +
	"if you need the rest.]"

// SubagentNoReportText replaces a "report" that consisted only of a leaked
// tool-call envelope: there is nothing in it to pass on.
const SubagentNoReportText = "the sub-agent produced no report: it ended by trying to call another tool " +
	"instead of answering. Relaunch it with a narrower task."

// splitLeakedToolCalls returns the report text with any trailing leaked
// tool-call envelope removed, and whether one was found. The cut runs from the
// opener to the end of the text: once a model starts emitting an envelope, the
// rest of the message is that envelope.
func splitLeakedToolCalls(s string) (clean string, leaked bool) {
	cut := -1
	for i := 0; i < len(s); {
		// Consider each line start, including the first.
		lineEnd := strings.IndexByte(s[i:], '\n')
		line := s[i:]
		if lineEnd >= 0 {
			line = s[i : i+lineEnd]
		}
		if hasLeakedOpener(strings.TrimLeft(line, " \t")) {
			cut = i
			break
		}
		if lineEnd < 0 {
			break
		}
		i += lineEnd + 1
	}
	if cut < 0 {
		return s, false
	}
	return strings.TrimRight(s[:cut], " \t\n\r"), true
}

// hasLeakedOpener reports whether a line begins with a tool-call envelope.
func hasLeakedOpener(line string) bool {
	for _, opener := range leakedToolCallOpeners {
		if strings.HasPrefix(line, opener) {
			return true
		}
	}
	return false
}

// subagentReport turns a finished run's final text into the tool result the
// orchestrator receives, refusing to pass an interrupted attempt off as an
// answer. minReportRunes is the floor below which what survived a cut is not
// worth reporting as findings.
const minReportRunes = 40

func subagentReport(final string) agentic.ToolResult {
	answer := strings.TrimSpace(final)
	clean, leaked := splitLeakedToolCalls(answer)
	if !leaked {
		return agentic.ToolResult{Content: answer}
	}
	clean = strings.TrimSpace(clean)
	if len([]rune(clean)) < minReportRunes {
		return agentic.ToolResult{Content: SubagentNoReportText, IsError: true}
	}
	return agentic.ToolResult{Content: clean + SubagentCutOffNote, IsError: true}
}
