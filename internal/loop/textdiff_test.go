package loop

import (
	"fmt"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestUnifiedDiffTable(t *testing.T) {
	tests := []struct {
		name     string
		from, to string
		old, new string
		want     string
	}{
		{
			name: "identical",
			from: "a/f.txt", to: "b/f.txt",
			old:  "one\ntwo\n",
			new:  "one\ntwo\n",
			want: "",
		},
		{
			name: "identical empty",
			from: "a/f.txt", to: "b/f.txt",
			old: "", new: "",
			want: "",
		},
		{
			name: "trailing newline only difference",
			from: "a/f.txt", to: "b/f.txt",
			old: "x\n", new: "x",
			// Naive-but-consistent newline handling: a trailing-newline-only change shows no line diff.
			want: "",
		},
		{
			name: "added file",
			from: "/dev/null", to: "b/new.txt",
			old: "", new: "one\ntwo\n",
			want: "--- /dev/null\n" +
				"+++ b/new.txt\n" +
				"@@ -0,0 +1,2 @@\n" +
				"+one\n" +
				"+two",
		},
		{
			name: "deleted file",
			from: "a/old.txt", to: "/dev/null",
			old: "one\ntwo\n", new: "",
			want: "--- a/old.txt\n" +
				"+++ /dev/null\n" +
				"@@ -1,2 +0,0 @@\n" +
				"-one\n" +
				"-two",
		},
		{
			name: "single line edit with context",
			from: "a/f.txt", to: "b/f.txt",
			old: "l1\nl2\nl3\nl4\nl5\nl6\nl7\n",
			new: "l1\nl2\nl3\nL4\nl5\nl6\nl7\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,7 +1,7 @@\n" +
				" l1\n l2\n l3\n-l4\n+L4\n l5\n l6\n l7",
		},
		{
			name: "insertion between lines",
			from: "a/f.txt", to: "b/f.txt",
			old: "a\nb\n",
			new: "a\nx\nb\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,2 +1,3 @@\n" +
				" a\n+x\n b",
		},
		{
			name: "deletion between lines",
			from: "a/f.txt", to: "b/f.txt",
			old: "a\nx\nb\n",
			new: "a\nb\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,3 +1,2 @@\n" +
				" a\n-x\n b",
		},
		{
			name: "append at end",
			from: "a/f.txt", to: "b/f.txt",
			old: "a\nb\n",
			new: "a\nb\nc\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,2 +1,3 @@\n" +
				" a\n b\n+c",
		},
		{
			name: "two distant changes make two hunks",
			from: "a/f.txt", to: "b/f.txt",
			old: "l1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nl15\n",
			new: "L1\nl2\nl3\nl4\nl5\nl6\nl7\nl8\nl9\nl10\nl11\nl12\nl13\nl14\nL15\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,4 +1,4 @@\n" +
				"-l1\n+L1\n l2\n l3\n l4\n" +
				"@@ -12,4 +12,4 @@\n" +
				" l12\n l13\n l14\n-l15\n+L15",
		},
		{
			name: "nearby changes merge into one hunk",
			from: "a/f.txt", to: "b/f.txt",
			old: "l1\nl2\nl3\nl4\nl5\nl6\nl7\n",
			new: "L1\nl2\nl3\nl4\nl5\nl6\nL7\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,7 +1,7 @@\n" +
				"-l1\n+L1\n l2\n l3\n l4\n l5\n l6\n-l7\n+L7",
		},
		{
			name: "everything replaced",
			from: "a/f.txt", to: "b/f.txt",
			old: "a\nb\n",
			new: "x\ny\nz\n",
			want: "--- a/f.txt\n" +
				"+++ b/f.txt\n" +
				"@@ -1,2 +1,3 @@\n" +
				"-a\n-b\n+x\n+y\n+z",
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := unifiedDiff(tc.from, tc.to, tc.old, tc.new)
			assert.Equal(t, tc.want, got)
		})
	}
}

// Past diffMaxLines per side, no line diff is attempted: the output is a note
// plus whole-file replace hunk.
func TestUnifiedDiffLargeFallback(t *testing.T) {
	var a, b strings.Builder
	for i := 0; i < diffMaxLines+1; i++ {
		fmt.Fprintf(&a, "line %d\n", i)
		fmt.Fprintf(&b, "line %d\n", i)
	}
	b.WriteString("extra\n")

	got := unifiedDiff("a/big.txt", "b/big.txt", a.String(), b.String())
	require.NotEmpty(t, got)
	lines := strings.SplitN(got, "\n", 5)
	assert.Equal(t, "--- a/big.txt", lines[0])
	assert.Equal(t, "+++ b/big.txt", lines[1])
	assert.Contains(t, lines[2], "diff too large")
	assert.Equal(t, fmt.Sprintf("@@ -1,%d +1,%d @@", diffMaxLines+1, diffMaxLines+2), lines[3])
	// Whole-file replacement: every old line deleted, every new line added.
	assert.Contains(t, got, "-line 0\n")
	assert.Contains(t, got, "+extra")
}

// A fully-disjoint middle bigger than the LCS budget falls back to a replace
// block but still renders a valid single hunk.
func TestUnifiedDiffLCSBudgetFallback(t *testing.T) {
	n := 2100 //* > diffMaxCells, both sides < diffMaxLines
	var a, b strings.Builder
	for i := 0; i < n; i++ {
		fmt.Fprintf(&a, "old %d\n", i)
		fmt.Fprintf(&b, "new %d\n", i)
	}
	got := unifiedDiff("a/f", "b/f", a.String(), b.String())
	require.NotEmpty(t, got)
	assert.NotContains(t, got, "diff too large", "per-side guard must not trip")
	assert.Contains(t, got, fmt.Sprintf("@@ -1,%d +1,%d @@", n, n))
	assert.Contains(t, got, "-old 0\n")
	assert.Contains(t, got, "+new 2099")
}

// The LCS path finds minimal scripts: unchanged lines between edits stay
// context even when the changed regions have different lengths.
func TestUnifiedDiffUnevenEdit(t *testing.T) {
	old := "keep1\ndrop1\ndrop2\nkeep2\n"
	niw := "keep1\nadd1\nkeep2\n"
	got := unifiedDiff("a/f", "b/f", old, niw)
	want := "--- a/f\n" +
		"+++ b/f\n" +
		"@@ -1,4 +1,3 @@\n" +
		" keep1\n-drop1\n-drop2\n+add1\n keep2"
	assert.Equal(t, want, got)
}

func TestSplitDiffLines(t *testing.T) {
	assert.Nil(t, splitDiffLines(""))
	assert.Equal(t, []string{"a"}, splitDiffLines("a"))
	assert.Equal(t, []string{"a"}, splitDiffLines("a\n"))
	assert.Equal(t, []string{"a", "b"}, splitDiffLines("a\nb"))
	assert.Equal(t, []string{"a", "", "b"}, splitDiffLines("a\n\nb\n"))
}
