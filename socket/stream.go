package socket

import (
	"io"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// streamWriter writes the response document as it runs; nothing is written until the first event.
type streamWriter struct {
	w  io.Writer
	rw *commonai.ResponseWriter
	// sent counts the parts written from OnPart, so finish can write the tail of the completion.
	sent int
	// pending is text streamed into the open <text> element, so the finished TextPart is the same one.
	pending string
	role    commonai.Role
}

func newStreamWriter(w io.Writer) *streamWriter {
	return &streamWriter{w: w, role: commonai.RoleAssistant}
}

// started reports whether any of the document has been written.
func (s *streamWriter) started() bool { return s.rw != nil }

// begin opens the response document.
func (s *streamWriter) begin(role commonai.Role) {
	if s.rw != nil {
		return
	}
	if role != "" {
		s.role = role
	}
	s.rw = commonai.NewResponseWriter(s.w, s.role)
}

// events are the callbacks that write the document as the call runs.
func (s *streamWriter) events() *commonai.StreamEvents {
	return &commonai.StreamEvents{
		OnText: func(delta string) error {
			s.begin(s.role)
			s.pending += delta
			return s.rw.Text(delta)
		},
		OnPart: func(p commonai.Part) error {
			s.begin(s.role)
			s.sent++
			if tp, ok := p.(commonai.TextPart); ok && tp.Text == s.pending {
				s.pending = ""
				return nil
			}
			s.pending = ""
			return s.rw.Part(p)
		},
		OnUsage: func(u commonai.Usage) error {
			s.begin(s.role)
			return s.rw.Usage(u)
		},
		OnTimings: func(t commonai.Timings) error {
			s.begin(s.role)
			return s.rw.Timings(t)
		},
	}
}

// finish writes what the events never announced and closes the document.
func (s *streamWriter) finish(comp *commonai.Completion, callErr error) {
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
