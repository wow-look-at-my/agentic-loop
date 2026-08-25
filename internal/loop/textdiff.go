package loop

import (
	"fmt"
	"strings"
)

// A pure, line-based unified-diff engine producing standard hunks and size wording.
const (
	diffContextLines = 3
	// diffMaxLines is the per-side guard; beyond it the file renders as one replace hunk.
	diffMaxLines = 20_000
	// diffMaxCells caps the LCS dynamic program; a bigger middle is one replace hunk.
	diffMaxCells = 4_000_000
)

// unifiedDiff renders the unified diff; fromLabel/toLabel become the ---/+++ header lines.
func unifiedDiff(fromLabel, toLabel, oldText, newText string) string {
	if oldText == newText {
		return ""
	}
	a := splitDiffLines(oldText)
	b := splitDiffLines(newText)

	header := "--- " + fromLabel + "\n+++ " + toLabel + "\n"
	if len(a) > diffMaxLines || len(b) > diffMaxLines {
		note := fmt.Sprintf("(diff too large for a line diff: %d -> %d lines; showing a whole-file replacement)\n", len(a), len(b))
		return header + note + renderHunks(replaceOps(a, b))
	}
	body := renderHunks(diffOps(a, b))
	if body == "" {
		// The contents differ only by a trailing newline — no visible line change.
		return ""
	}
	return header + body
}

// splitDiffLines splits content into lines; a trailing newline adds no line.
func splitDiffLines(s string) []string {
	if s == "" {
		return nil
	}
	s = strings.TrimSuffix(s, "\n")
	return strings.Split(s, "\n")
}

// diffOp is one line of the edit script: kind ' ' (context), '-' (deleted
// from old), or '+' (added in new).
type diffOp struct {
	kind byte
	text string
}

// diffOps computes the line-level edit script between a and b: common prefix
// and suffix are matched first, and the remaining middle is aligned with an
// LCS dynamic program when it fits the budget (one replace block otherwise).
func diffOps(a, b []string) []diffOp {
	// Common prefix.
	p := 0
	for p < len(a) && p < len(b) && a[p] == b[p] {
		p++
	}
	// Common suffix (of the parts past the prefix).
	s := 0
	for s < len(a)-p && s < len(b)-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	ma, mb := a[p:len(a)-s], b[p:len(b)-s]

	var middle []diffOp
	if len(ma)*len(mb) > diffMaxCells {
		middle = replaceOps(ma, mb)
	} else {
		middle = lcsOps(ma, mb)
	}

	ops := make([]diffOp, 0, p+len(middle)+s)
	for _, l := range a[:p] {
		ops = append(ops, diffOp{kind: ' ', text: l})
	}
	ops = append(ops, middle...)
	for _, l := range a[len(a)-s:] {
		ops = append(ops, diffOp{kind: ' ', text: l})
	}
	return ops
}

// replaceOps renders a as fully deleted and b as fully added — the fallback
// shape when a minimal diff would be too expensive.
func replaceOps(a, b []string) []diffOp {
	ops := make([]diffOp, 0, len(a)+len(b))
	for _, l := range a {
		ops = append(ops, diffOp{kind: '-', text: l})
	}
	for _, l := range b {
		ops = append(ops, diffOp{kind: '+', text: l})
	}
	return ops
}

// lcsOps aligns a and b by their longest common subsequence and emits the
// resulting edit script (deletions before additions within each changed run).
func lcsOps(a, b []string) []diffOp {
	n, m := len(a), len(b)
	if n == 0 || m == 0 {
		return replaceOps(a, b)
	}
	// dp[i][j] = LCS length of a[i:], b[j:].
	dp := make([][]int32, n+1)
	for i := range dp {
		dp[i] = make([]int32, m+1)
	}
	for i := n - 1; i >= 0; i-- {
		for j := m - 1; j >= 0; j-- {
			if a[i] == b[j] {
				dp[i][j] = dp[i+1][j+1] + 1
			} else if dp[i+1][j] >= dp[i][j+1] {
				dp[i][j] = dp[i+1][j]
			} else {
				dp[i][j] = dp[i][j+1]
			}
		}
	}
	var ops []diffOp
	i, j := 0, 0
	for i < n && j < m {
		switch {
		case a[i] == b[j]:
			ops = append(ops, diffOp{kind: ' ', text: a[i]})
			i++
			j++
		case dp[i+1][j] >= dp[i][j+1]:
			ops = append(ops, diffOp{kind: '-', text: a[i]})
			i++
		default:
			ops = append(ops, diffOp{kind: '+', text: b[j]})
			j++
		}
	}
	for ; i < n; i++ {
		ops = append(ops, diffOp{kind: '-', text: a[i]})
	}
	for ; j < m; j++ {
		ops = append(ops, diffOp{kind: '+', text: b[j]})
	}
	return ops
}

// renderHunks groups an edit script into unified hunks with diffContextLines
// lines of context, merging hunks whose context would touch or overlap. Counts
// are always written explicitly ("-3,1" rather than "-3") for consistency.
func renderHunks(ops []diffOp) string {
	if len(ops) == 0 {
		return ""
	}
	// Indices of ops that are changes.
	var changes []int
	for idx, op := range ops {
		if op.kind != ' ' {
			changes = append(changes, idx)
		}
	}
	if len(changes) == 0 {
		return ""
	}

	var out strings.Builder
	// oldAt[i] / newAt[i] = number of old/new lines among ops[:i].
	oldAt := make([]int, len(ops)+1)
	newAt := make([]int, len(ops)+1)
	for idx, op := range ops {
		oldAt[idx+1] = oldAt[idx]
		newAt[idx+1] = newAt[idx]
		if op.kind != '+' {
			oldAt[idx+1]++
		}
		if op.kind != '-' {
			newAt[idx+1]++
		}
	}

	for ci := 0; ci < len(changes); {
		start := changes[ci] - diffContextLines
		if start < 0 {
			start = 0
		}
		// Extend the hunk over changes whose context regions would touch or overlap.
		cj := ci
		for cj+1 < len(changes) && changes[cj+1]-changes[cj] <= 2*diffContextLines {
			cj++
		}
		end := changes[cj] + diffContextLines + 1
		if end > len(ops) {
			end = len(ops)
		}

		oldCount := oldAt[end] - oldAt[start]
		newCount := newAt[end] - newAt[start]
		oldStart := oldAt[start] + 1
		if oldCount == 0 {
			oldStart = oldAt[start]
		}
		newStart := newAt[start] + 1
		if newCount == 0 {
			newStart = newAt[start]
		}
		fmt.Fprintf(&out, "@@ -%d,%d +%d,%d @@\n", oldStart, oldCount, newStart, newCount)
		for _, op := range ops[start:end] {
			out.WriteByte(op.kind)
			out.WriteString(op.text)
			out.WriteByte('\n')
		}
		ci = cj + 1
	}
	return strings.TrimSuffix(out.String(), "\n")
}

// Plural renders "1 <one>" or "N <many>".
func Plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return fmt.Sprintf("%d %s", n, many)
}

// HumanSize renders a byte count the way a model reads it in a change summary.
func HumanSize(n int64) string {
	switch {
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MB", float64(n)/(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KB", float64(n)/(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// UnifiedDiff renders the unified diff between oldText and newText.
func UnifiedDiff(fromLabel, toLabel, oldText, newText string) string {
	return unifiedDiff(fromLabel, toLabel, oldText, newText)
}

// CountLineChanges counts added and removed lines between two texts, using the
// same edit script UnifiedDiff renders so a summary and its diff agree.
func CountLineChanges(before, after string) (added, removed int) {
	for _, op := range diffOps(splitDiffLines(before), splitDiffLines(after)) {
		switch op.kind {
		case '+':
			added++
		case '-':
			removed++
		}
	}
	return added, removed
}

// CountLines counts the lines in content, matching splitDiffLines' convention.
func CountLines(s string) int { return len(splitDiffLines(s)) }
