package httpapi

import (
	"net/http"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// streamWriter turns a call's events into the response document down a chunked body as they arrive.
type streamWriter struct {
	w  http.ResponseWriter
	rw *commonai.ResponseWriter
	// sent counts the parts written from OnPart, so finish can write the tail of the completion.
	sent int
	// pending is text streamed into the open <text> element, so the finished TextPart is the same.
	pending string
	role    commonai.Role
}

func newStreamWriter(w http.ResponseWriter) *streamWriter {
	return &streamWriter{w: w, role: commonai.RoleAssistant}
}

// started reports whether any of the document has been written.
func (s *streamWriter) started() bool { return s.rw != nil }

// begin opens the response document and commits the.
func (s *streamWriter) begin(role commonai.Role) {
	if s.rw != nil {
		return
	}
	if role != "" {
		s.role = role
	}
	s.w.Header().Set("Content-Type", contentType)
	s.w.Header().Set("X-Content-Type-Options", "nosniff")
	s.rw = commonai.NewResponseWriter(s.w, s.role)
	s.flush()
}

// flush pushes what has been written so far to the client, which is what makes
func (s *streamWriter) flush() {
	if f, ok := s.w.(http.Flusher); ok {
		f.Flush()
	}
}

// events are the callbacks that write the document as the call runs.
func (s *streamWriter) events() *commonai.StreamEvents {
	return &commonai.StreamEvents{
		OnText: func(delta string) error {
			s.begin(s.role)
			defer s.flush()
			s.pending += delta
			return s.rw.Text(delta)
		},
		OnPart: func(p commonai.Part) error {
			s.begin(s.role)
			defer s.flush()
			s.sent++
			// The text of this part is already in the document, delta by
			// delta: it is the element still open.
			if tp, ok := p.(commonai.TextPart); ok && tp.Text == s.pending {
				s.pending = ""
				return nil
			}
			s.pending = ""
			return s.rw.Part(p)
		},
		OnUsage: func(u commonai.Usage) error {
			s.begin(s.role)
			defer s.flush()
			return s.rw.Usage(u)
		},
		OnTimings: func(t commonai.Timings) error {
			s.begin(s.role)
			defer s.flush()
			return s.rw.Timings(t)
		},
	}
}

// finish writes what the events never announced and closes the document: the
// parts of a call that did not stream at all (a server that ignored
// stream:true), and the tail of that was cut off mid-part.
func (s *streamWriter) finish(comp *commonai.Completion, callErr error) {
	defer s.flush()
	if comp == nil {
		if callErr != nil && s.rw != nil {
			_ = s.rw.Fail(callErr)
		}
		return
	}
	parts := comp.Message.EffectiveParts()
	for i := s.sent; i < len(parts); i++ {
		if tp, ok := parts[i].(commonai.TextPart); ok && tp.Text == s.pending {
			s.pending = ""
			continue
		}
		_ = s.rw.Part(parts[i])
	}
	if !comp.Streamed {
		// A buffered answer reported its usage with the body rather than
		// through the callbacks.
		for _, u := range comp.Usages {
			_ = s.rw.Usage(u)
		}
		for _, t := range comp.Timings {
			_ = s.rw.Timings(t)
		}
	}
	if callErr != nil {
		_ = s.rw.Fail(callErr)
		return
	}
	_ = s.rw.Close(comp.StopReason, comp.Streamed)
}
