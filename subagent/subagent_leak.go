package subagent

import (
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
)

// A report ending in a leaked tool-call envelope is cut at the envelope and labelled partial.

// leakedToolCallOpeners are envelope tokens matched only at the start of a line.
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

// subagentReport turns the final text into the tool result, never passing an interrupted attempt on.
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
