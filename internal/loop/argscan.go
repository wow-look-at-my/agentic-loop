// argscan.go reads the top-level members of a JSON object off a stream, as
// each completes.

package loop

import (
	"encoding/json"
	"strings"
)

// argMember is one top-level member of the arguments object, complete.
type argMember struct {
	key   string
	value json.RawMessage
}

type argScanState int

const (
	scanStart argScanState = iota
	scanKeyOrEnd
	scanKey
	scanColon
	scanValueStart
	scanValString
	scanValNested
	scanValScalar
	scanAfterValue
	scanDone
	scanDead
)

// argScanner is fed the arguments a delta at a time and reports each
// top-level member when its value ends. It reads every byte once.
type argScanner struct {
	state  argScanState
	key    strings.Builder
	value  strings.Builder
	esc    bool
	inStr  bool
	depth  int
	keyRaw string
}

// feed scans delta and returns the members it completed, in order.
func (s *argScanner) feed(delta string) []argMember {
	var out []argMember
	for i := 0; i < len(delta); i++ {
		if m, ok := s.step(delta[i]); ok {
			out = append(out, m)
		}
	}
	return out
}

// ended reports that no more members can arrive.
func (s *argScanner) ended() bool { return s.state == scanDone || s.state == scanDead }

func isJSONSpace(c byte) bool { return c == ' ' || c == '\t' || c == '\n' || c == '\r' }

func (s *argScanner) step(c byte) (argMember, bool) {
	switch s.state {
	case scanStart:
		switch {
		case isJSONSpace(c):
		case c == '{':
			s.state = scanKeyOrEnd
		default:
			s.state = scanDead
		}
	case scanKeyOrEnd:
		switch {
		case isJSONSpace(c), c == ',':
		case c == '"':
			s.key.Reset()
			s.key.WriteByte(c)
			s.esc = false
			s.state = scanKey
		case c == '}':
			s.state = scanDone
		default:
			s.state = scanDead
		}
	case scanKey:
		s.key.WriteByte(c)
		if s.esc {
			s.esc = false
		} else if c == '\\' {
			s.esc = true
		} else if c == '"' {
			s.keyRaw = s.key.String()
			s.state = scanColon
		}
	case scanColon:
		switch {
		case isJSONSpace(c):
		case c == ':':
			s.state = scanValueStart
		default:
			s.state = scanDead
		}
	case scanValueStart:
		if isJSONSpace(c) {
			return argMember{}, false
		}
		s.value.Reset()
		s.value.WriteByte(c)
		s.esc = false
		switch c {
		case '"':
			s.state = scanValString
		case '{', '[':
			s.depth, s.inStr = 1, false
			s.state = scanValNested
		default:
			s.state = scanValScalar
		}
	case scanValString:
		s.value.WriteByte(c)
		if s.esc {
			s.esc = false
		} else if c == '\\' {
			s.esc = true
		} else if c == '"' {
			return s.complete()
		}
	case scanValNested:
		s.value.WriteByte(c)
		switch {
		case s.esc:
			s.esc = false
		case s.inStr && c == '\\':
			s.esc = true
		case c == '"':
			s.inStr = !s.inStr
		case s.inStr:
		case c == '{' || c == '[':
			s.depth++
		case c == '}' || c == ']':
			s.depth--
			if s.depth == 0 {
				return s.complete()
			}
		}
	case scanValScalar:
		if c == ',' || c == '}' || isJSONSpace(c) {
			m, ok := s.complete()
			// The delimiter belongs to the object, so the object reads it next.
			s.step(c)
			return m, ok
		}
		s.value.WriteByte(c)
	case scanAfterValue:
		switch {
		case isJSONSpace(c):
		case c == ',':
			s.state = scanKeyOrEnd
		case c == '}':
			s.state = scanDone
		default:
			s.state = scanDead
		}
	}
	return argMember{}, false
}

// complete ends the member in progress. A key that does not decode stops the scan.
func (s *argScanner) complete() (argMember, bool) {
	s.state = scanAfterValue
	var key string
	if json.Unmarshal([]byte(s.keyRaw), &key) != nil {
		s.state = scanDead
		return argMember{}, false
	}
	return argMember{key: key, value: json.RawMessage(s.value.String())}, true
}
