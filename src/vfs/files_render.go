package vfs

import (
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
	"path"
	"sort"
	"strings"
)

// How a file tool's answer READS. The rendering is the tool: a listing the
// model cannot scan, or an empty grep that does not say what it proves, is a
// worse failure than a wrong path.

// --- rendering --------------------------------------------------------------

// renderListing formats a directory listing: the canonical path, an optional
// annotation, then directories before files.
func renderListing(where string, l Listing) string {
	header := where
	if l.Note != "" {
		header += "  (" + l.Note + ")"
	}
	if len(l.Entries) == 0 {
		return header + "\n(empty directory)"
	}
	entries := l.Entries
	sort.SliceStable(entries, func(i, j int) bool {
		if entries[i].Dir != entries[j].Dir {
			return entries[i].Dir // directories first
		}
		return entries[i].Name < entries[j].Name
	})
	truncated := l.Truncated
	if len(entries) > ListMaxEntries {
		entries, truncated = entries[:ListMaxEntries], true
	}
	var b strings.Builder
	b.WriteString(header + "\n")
	for _, en := range entries {
		kind := "file"
		name := en.Name
		switch {
		case en.Kind != "":
			kind = en.Kind
		case en.Dir:
			kind, name = "dir", en.Name+"/"
		}
		fmt.Fprintf(&b, "%-5s %s", kind, name)
		if !en.Dir && en.Kind == "" {
			fmt.Fprintf(&b, " (%s)", agentic.HumanSize(en.Size))
		}
		if en.Note != "" {
			b.WriteString("  " + en.Note)
		}
		b.WriteByte('\n')
	}
	if truncated {
		fmt.Fprintf(&b, "(listing truncated; narrow the path or use %s)", FindFilesToolName)
	}
	return strings.TrimRight(b.String(), "\n")
}

// renderGrep renders hits grouped by file. Every path is a full virtual path
// carrying its line number, so a hit feeds straight back into read_file.
func renderGrep(where, pattern string, globs []string, res GrepResult) string {
	var b strings.Builder
	fmt.Fprintf(&b, "grep %q in %s", pattern, where)
	if len(globs) > 0 {
		fmt.Fprintf(&b, " (%s)", strings.Join(globs, ", "))
	}
	if len(res.Hits) == 0 {
		// An empty result has to state what it proves. Every line of every file
		// in scope was read, so the text really is absent from that scope --
		// unless coverage was partial, which the note then says.
		b.WriteString(": no matches.\nEvery line of every file in scope was searched, so the text is genuinely absent from it — this is a real negative, not a search that gave up.")
		if res.Note != "" {
			b.WriteString("\n" + res.Note)
		}
		return b.String()
	}
	fmt.Fprintf(&b, ": %s in %s\n", agentic.Plural(len(res.Hits), "matching line", "matching lines"), agentic.Plural(res.Files, "file", "files"))
	current := ""
	for _, h := range res.Hits {
		if h.Path != current {
			current = h.Path
			fmt.Fprintf(&b, "\n%s\n", h.Path)
		}
		fmt.Fprintf(&b, "%7d: %s\n", h.Line, h.Text)
	}
	if res.Truncated {
		fmt.Fprintf(&b, "\n(stopped at %d matching lines — more exist. Narrow the path or \"glob\", or raise \"limit\" to at most %d.)",
			len(res.Hits), GrepMaxLimit)
	}
	if res.Note != "" {
		b.WriteString("\n" + res.Note)
	}
	return strings.TrimRight(b.String(), "\n")
}

// SliceLines applies read_file's line window. A whole file is the default, but
// reading one function out of a large file should not cost the whole file: one
// unwindowed read of a 60,000-character source file added roughly 18,000
// tokens to a single turn.
//
// offset is 1-based and inclusive, matching the line numbers grep hands back.
func SliceLines(content string, offset, limit int) (body, note string) {
	if offset <= 0 && limit <= 0 {
		return content, ""
	}
	lines := strings.Split(content, "\n")
	// A trailing newline yields a final empty element that is not a line.
	if n := len(lines); n > 0 && lines[n-1] == "" {
		lines = lines[:n-1]
	}
	total := len(lines)
	start := offset - 1
	if start < 0 {
		start = 0
	}
	if start >= total {
		return "", fmt.Sprintf("(this file has %d lines; line %d is past its end)", total, offset)
	}
	end := total
	if limit > 0 && start+limit < end {
		end = start + limit
	}
	shown := lines[start:end]
	note = fmt.Sprintf("(lines %d-%d of %d", start+1, end, total)
	if end < total {
		note += fmt.Sprintf("; %d more follow — re-read with offset %d", total-end, end+1)
	}
	return strings.Join(shown, "\n"), note + ")"
}

// MatchesPattern reports whether a path matches a find_files pattern. A pattern
// carrying glob metacharacters is matched against both the base name and the
// full path (so *.go and src/*.go both work); anything else is a
// case-insensitive substring of the path. Exported for folders doing their own
// filtering with the same rule.
func MatchesPattern(p, pattern string) bool {
	if !strings.ContainsAny(pattern, "*?[") {
		return strings.Contains(strings.ToLower(p), strings.ToLower(pattern))
	}
	if ok, err := path.Match(pattern, path.Base(p)); err == nil && ok {
		return true
	}
	ok, err := path.Match(pattern, p)
	return err == nil && ok
}

// WithinDir reports whether a relative path lies below dir (dir "" = the root),
// and returns its path relative to dir.
func WithinDir(p, dir string) (string, bool) {
	if dir == "" {
		return p, true
	}
	if !strings.HasPrefix(p, dir+"/") {
		return "", false
	}
	return strings.TrimPrefix(p, dir+"/"), true
}

// WithinScope reports whether p falls inside a SEARCH scope, and returns the
// name the globs are matched against.
//
// A search scope is a path, not a directory: grep names either a subtree or one
// exact file, and both must work. WithinDir alone answers no for the file case
// (a file is not a child of itself), which makes every single-file grep report
// a false absence -- the searched file is skipped, and an empty result is then
// presented as proof the text is not there.
func WithinScope(p, scope string) (string, bool) {
	if p == scope {
		return path.Base(p), true
	}
	return WithinDir(p, scope)
}

// SplitGlobs parses grep's comma-separated glob argument.
func SplitGlobs(s string) []string {
	var out []string
	for _, g := range strings.Split(s, ",") {
		if g = strings.TrimSpace(g); g != "" {
			out = append(out, g)
		}
	}
	return out
}

// clampInt applies a default for a non-positive value, then bounds the result.
func clampInt(v, def, lo, hi int) int {
	if v <= 0 {
		v = def
	}
	return max(lo, min(v, hi))
}
