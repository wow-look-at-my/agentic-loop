package commonai

import (
	"fmt"
	"io"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The writer is hand-rolled because encoding/xml cannot emit &#; or control namespace prefixes.

// Namespaces. The core vocabulary and per dialect, so provider-specific
// data is namespaced rather than guessed at by name.
const (
	NS          = "https://github.com/wow-look-at-my/common-ai-api/schema/v1"
	NSAnthropic = NS + "/anthropic"
	NSOpenAI    = NS + "/openai"
	NSResponses = NS + "/responses"
)

// Namespace prefixes, as the writer emits them.
const (
	prefixAnthropic = "anthropic"
	prefixOpenAI    = "openai"
	prefixResponses = "responses"
)

// xmlDecl is the declaration every document starts with; XML is required.
const xmlDecl = `<?xml version="1.1" encoding="UTF-8"?>` + "\n"

// writer emits XML to an io.Writer, tracking the error so callers can
// write a whole document and check.
type writer struct {
	w    io.Writer
	err  error
	open []string
}

// newWriter starts a document, emitting the XML declaration.
func newWriter(w io.Writer) *writer {
	xw := &writer{w: w}
	xw.raw(xmlDecl)
	return xw
}

// raw writes literal bytes, skipping an error has been seen.
func (x *writer) raw(s string) {
	if x.err != nil {
		return
	}
	if _, err := io.WriteString(x.w, s); err != nil {
		x.err = err
	}
}

// attr is attribute, already namespaced if it needs to be.
type attr struct {
	name  string
	value string
}

// optAttr returns the attribute, or nothing when empty; absence is meaningful in this format.
func optAttr(name, value string) []attr {
	if value == "" {
		return nil
	}
	return []attr{{name: name, value: value}}
}

// optBoolAttr renders a tri-state boolean; a nil pointer writes no attribute.
func optBoolAttr(name string, v *bool) []attr {
	if v == nil {
		return nil
	}
	return []attr{{name: name, value: strconv.FormatBool(*v)}}
}

// intAttr renders an int attribute.
func intAttr(name string, v int) attr { return attr{name: name, value: strconv.Itoa(v)} }

// ptrIntAttr renders a tri-state int: nothing at all when it was never
// reported, which is how the format says "unknown" rather than "zero".
func ptrIntAttr(name string, v *int) []attr {
	if v == nil {
		return nil
	}
	return []attr{intAttr(name, *v)}
}

// ptrFloatAttr renders a tri-state float the same way.
func ptrFloatAttr(name string, v *float64) []attr {
	if v == nil {
		return nil
	}
	return []attr{{name: name, value: strconv.FormatFloat(*v, 'g', -1, 64)}}
}

// start opens an element with attributes.
func (x *writer) start(name string, attrs ...attr) {
	x.raw("<" + name)
	for _, a := range attrs {
		x.raw(" " + a.name + `="` + escapeAttr(a.value) + `"`)
	}
	x.raw(">")
	x.open = append(x.open, name)
}

// empty writes a self-closing element.
func (x *writer) empty(name string, attrs ...attr) {
	x.raw("<" + name)
	for _, a := range attrs {
		x.raw(" " + a.name + `="` + escapeAttr(a.value) + `"`)
	}
	x.raw("/>")
}

// end closes the innermost open element.
func (x *writer) end() {
	if len(x.open) == 0 {
		if x.err == nil {
			x.err = fmt.Errorf("commonai: closing an element that was never opened")
		}
		return
	}
	name := x.open[len(x.open)-1]
	x.open = x.open[:len(x.open)-1]
	x.raw("</" + name + ">")
}

// text writes character data.
func (x *writer) text(s string) { x.raw(escapeText(s)) }

// element writes a complete element with text content.
func (x *writer) element(name, content string, attrs ...attr) {
	if content == "" {
		x.empty(name, attrs...)
		return
	}
	x.start(name, attrs...)
	x.text(content)
	x.end()
}

// escapeText escapes character data; everything XML cannot carry becomes a character reference.
func escapeText(s string) string {
	return escape(s, false)
}

// escapeAttr escapes an attribute value, additionally quoting " and whitespace.
func escapeAttr(s string) string {
	return escape(s, true)
}

// escape is the escaper both forms share. A byte that is not valid UTF-8
// has no character to reference at all, so it is written as its own code
// point -- the only representable reading of a byte that should not be there.
func escape(s string, inAttr bool) string {
	if !needsEscape(s, inAttr) {
		return s
	}
	var b strings.Builder
	b.Grow(len(s) + len(s)/8)
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size <= 1 {
			// Not valid UTF-8: reference the byte itself, since dropping it loses content.
			b.WriteString("&#" + strconv.Itoa(int(s[i])) + ";")
			i++
			continue
		}
		i += size
		switch {
		case r == '&':
			b.WriteString("&amp;")
		case r == '<':
			b.WriteString("&lt;")
		case r == '>':
			b.WriteString("&gt;")
		case inAttr && r == '"':
			b.WriteString("&quot;")
		case inAttr && (r == '\n' || r == '\r' || r == '\t'):
			b.WriteString("&#" + strconv.Itoa(int(r)) + ";")
		case mustReference(r):
			b.WriteString("&#" + strconv.Itoa(int(r)) + ";")
		default:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// needsEscape reports whether s contains anything the escaper would change.
func needsEscape(s string, inAttr bool) bool {
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		if r == utf8.RuneError && size <= 1 {
			return true
		}
		i += size
		switch {
		case r == '&' || r == '<' || r == '>':
			return true
		case inAttr && (r == '"' || r == '\n' || r == '\r' || r == '\t'):
			return true
		case mustReference(r):
			return true
		}
	}
	return false
}

// mustReference reports whether a rune must be written as a character reference.
func mustReference(r rune) bool {
	if r == 0 {
		return true
	}
	return (r >= 0x1 && r <= 0x8) ||
		(r >= 0xB && r <= 0xC) ||
		(r >= 0xE && r <= 0x1F) ||
		(r >= 0x7F && r <= 0x84) ||
		(r >= 0x86 && r <= 0x9F)
}
