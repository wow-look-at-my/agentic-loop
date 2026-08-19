package agentic

import (
	"strings"
	"unicode/utf8"
)

// TruncateRunes caps s at maxRunes runes, reporting whether it truncated. It
// is exported because a host serving its own file reads must cut them the same
// way the tools do -- a differently-capped read is a differently-sized answer.
func TruncateRunes(s string, maxRunes int) (string, bool) {
	if maxRunes <= 0 || utf8.RuneCountInString(s) <= maxRunes {
		return s, false
	}
	var b strings.Builder
	count := 0
	for _, r := range s {
		if count >= maxRunes {
			break
		}
		b.WriteRune(r)
		count++
	}
	return strings.TrimSpace(b.String()), true
}
