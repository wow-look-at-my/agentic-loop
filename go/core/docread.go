package commonai

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
)

// A document is self-delimiting: it ends when its root element closes. That is
// what lets documents ride back-to-back down one connection with no framing
// layer -- no length prefix, no envelope, nothing to strip before parsing.
// ReadDocument is that rule, written once, because every transport that
// carries more than one document needs it.

// maxDocument caps a single document. A transcript is large but bounded, and a
// stream that never closes its root would otherwise be read until memory runs
// out.
const maxDocument = 64 << 20

// ReadDocument reads exactly one document from r, leaving whatever follows it
// for the next call. It returns io.EOF when the stream ends cleanly between
// documents, and an error when it ends in the middle of one -- a truncated
// document is not a document, and a reader that returned it as if it were
// would be handing the caller half an answer with no way to tell.
func ReadDocument(r *bufio.Reader) ([]byte, error) {
	var buf bytes.Buffer
	s := &docScanner{}
	for {
		b, err := r.ReadByte()
		if err != nil {
			if err == io.EOF && buf.Len() == 0 {
				return nil, io.EOF
			}
			if err == io.EOF {
				return nil, fmt.Errorf("commonai: the stream ended inside a document")
			}
			return nil, err
		}
		if buf.Len() == 0 && isSpaceByte(b) {
			// Whitespace between documents is not part of either.
			continue
		}
		buf.WriteByte(b)
		if buf.Len() > maxDocument {
			return nil, fmt.Errorf("commonai: document exceeds %d bytes", maxDocument)
		}
		if s.feed(b) {
			return buf.Bytes(), nil
		}
	}
}

func isSpaceByte(b byte) bool {
	return b == ' ' || b == '\t' || b == '\r' || b == '\n'
}

// docScanner tracks how deep in the element tree the bytes so far have got.
// It is a byte-at-a-time state machine rather than a parser: the question is
// only where the document ENDS, and answering it must not depend on the
// document being valid -- the validator says that afterwards, over the whole
// thing.
type docScanner struct {
	state   scanState
	depth   int
	started bool
	// tagKind is what the tag being read turned out to be, decided from the
	// bytes right after '<'.
	tagKind tagKind
	// lead holds the first few bytes of a tag, which is all it takes to tell a
	// comment from a CDATA section from a PI.
	lead []byte
	// quote is the attribute delimiter currently open, or 0.
	quote byte
}

type scanState int

const (
	scanOutside scanState = iota // in character data
	scanLead                     // just past '<', deciding what this is
	scanTag                      // inside a tag, reading name and attributes
	scanComment                  // inside <!-- -->
	scanCDATA                    // inside <![CDATA[ ]]>
	scanPI                       // inside <? ?>
)

type tagKind int

const (
	tagOpen tagKind = iota
	tagClose
	tagSelfClosing
)

// feed advances the scanner by one byte, reporting whether the document just
// ended.
func (s *docScanner) feed(b byte) bool {
	switch s.state {
	case scanOutside:
		if b == '<' {
			s.state, s.lead = scanLead, s.lead[:0]
		}
		return false

	case scanLead:
		s.lead = append(s.lead, b)
		switch {
		case s.lead[0] == '?':
			s.state = scanPI
			return false
		case s.lead[0] == '!':
			// A comment and a CDATA section are only distinguishable after a
			// few bytes, and both hold text that looks like markup -- which is
			// exactly why they have to be recognized here.
			switch {
			case bytes.Equal(s.lead, []byte("!--")):
				s.state = scanComment
			case bytes.Equal(s.lead, []byte("![CDATA[")):
				s.state = scanCDATA
			case isPrefixOf(s.lead, "!--"), isPrefixOf(s.lead, "![CDATA["):
				// Still deciding.
			default:
				// A declaration this format does not have; it ends like a tag.
				s.state, s.tagKind = scanTag, tagOpen
			}
		case s.lead[0] == '/':
			s.state, s.tagKind = scanTag, tagClose
		default:
			s.state, s.tagKind = scanTag, tagOpen
		}
		if b == '>' && s.state == scanTag {
			return s.closeTag()
		}
		return false

	case scanTag:
		if s.quote != 0 {
			if b == s.quote {
				s.quote = 0
			}
			return false
		}
		switch b {
		case '"', '\'':
			s.quote = b
		case '/':
			s.tagKind = tagSelfClosing
		case '>':
			return s.closeTag()
		default:
			if s.tagKind == tagSelfClosing && !isSpaceByte(b) {
				// A slash that was not the end of the tag after all.
				s.tagKind = tagOpen
			}
		}
		return false

	case scanComment:
		s.lead = append(s.lead, b)
		if bytes.HasSuffix(s.lead, []byte("-->")) {
			s.state = scanOutside
		}
		return false

	case scanCDATA:
		s.lead = append(s.lead, b)
		if bytes.HasSuffix(s.lead, []byte("]]>")) {
			s.state = scanOutside
		}
		return false

	case scanPI:
		s.lead = append(s.lead, b)
		if bytes.HasSuffix(s.lead, []byte("?>")) {
			s.state = scanOutside
		}
		return false
	}
	return false
}

// closeTag ends a tag and reports whether that closed the root element.
func (s *docScanner) closeTag() bool {
	s.state = scanOutside
	switch s.tagKind {
	case tagSelfClosing:
		if !s.started {
			// A self-closing root: the whole document is one element.
			s.started = true
			return true
		}
	case tagOpen:
		s.started = true
		s.depth++
	case tagClose:
		s.depth--
		if s.started && s.depth == 0 {
			return true
		}
	}
	return false
}

// isPrefixOf reports whether b is a proper prefix of s, which is how the
// scanner waits for enough bytes to tell a comment from a declaration.
func isPrefixOf(b []byte, s string) bool {
	return len(b) < len(s) && s[:len(b)] == string(b)
}
