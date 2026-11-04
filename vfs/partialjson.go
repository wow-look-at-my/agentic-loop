// partialjson.go reads one string member off a JSON object that is still arriving.

package vfs

import "encoding/json"

// fieldState is what a prefix of a JSON object says about one member.
type fieldState int

const (
	// fieldPending: the member has not arrived whole. The next delta may complete it.
	fieldPending fieldState = iota
	// fieldFound: the member arrived whole, as a string.
	fieldFound
	// fieldAbsent: the object closed without the member, the member is not a string, or the text is not an object.
	fieldAbsent
)

// partialStringField reads the top-level string member key from prefix, the opening bytes of a JSON
// object a model is still writing. A member that arrived whole decodes; one cut off mid-value is pending.
func partialStringField(prefix, key string) (string, fieldState) {
	s := scanner{src: prefix}
	s.skipSpace()
	if s.eof() {
		return "", fieldPending
	}
	if s.src[s.pos] != '{' {
		return "", fieldAbsent
	}
	s.pos++
	for {
		s.skipSpace()
		if s.eof() {
			return "", fieldPending
		}
		switch s.src[s.pos] {
		case '}':
			return "", fieldAbsent
		case ',':
			s.pos++
			continue
		case '"':
		default:
			return "", fieldAbsent
		}
		name, done := s.scanString()
		if !done {
			return "", fieldPending
		}
		s.skipSpace()
		if s.eof() {
			return "", fieldPending
		}
		if s.src[s.pos] != ':' {
			return "", fieldAbsent
		}
		s.pos++
		s.skipSpace()
		if s.eof() {
			return "", fieldPending
		}
		var decoded string
		if json.Unmarshal([]byte(name), &decoded) != nil {
			return "", fieldAbsent
		}
		if decoded == key {
			if s.src[s.pos] != '"' {
				return "", fieldAbsent
			}
			raw, done := s.scanString()
			if !done {
				return "", fieldPending
			}
			var value string
			if json.Unmarshal([]byte(raw), &value) != nil {
				return "", fieldAbsent
			}
			return value, fieldFound
		}
		if !s.skipValue() {
			return "", fieldPending
		}
	}
}

// scanner walks JSON text that may end anywhere.
type scanner struct {
	src string
	pos int
}

func (s *scanner) eof() bool { return s.pos >= len(s.src) }

func (s *scanner) skipSpace() {
	for !s.eof() {
		switch s.src[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

// scanString reads the string at pos, quotes included. done is false when it is cut off.
func (s *scanner) scanString() (string, bool) {
	start := s.pos
	s.pos++
	for !s.eof() {
		switch s.src[s.pos] {
		case '\\':
			s.pos += 2
		case '"':
			s.pos++
			return s.src[start:s.pos], true
		default:
			s.pos++
		}
	}
	return "", false
}

// skipValue steps over one value of any kind. It reports false when the value is cut off.
func (s *scanner) skipValue() bool {
	if s.eof() {
		return false
	}
	switch s.src[s.pos] {
	case '"':
		_, done := s.scanString()
		return done
	case '{', '[':
		return s.skipNested()
	}
	// A number or a literal ends at a delimiter. One cut off at the end of the text is pending.
	for !s.eof() {
		switch s.src[s.pos] {
		case ',', '}', ']', ' ', '\t', '\n', '\r':
			return true
		}
		s.pos++
	}
	return false
}

// skipNested steps over an object or an array, strings included, by depth.
func (s *scanner) skipNested() bool {
	depth := 0
	for !s.eof() {
		switch s.src[s.pos] {
		case '"':
			if _, done := s.scanString(); !done {
				return false
			}
			continue
		case '{', '[':
			depth++
		case '}', ']':
			depth--
			if depth == 0 {
				s.pos++
				return true
			}
		}
		s.pos++
	}
	return false
}
