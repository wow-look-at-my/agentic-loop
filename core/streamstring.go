package commonai

import (
	"strings"
	"sync"
	"time"

	"github.com/wow-look-at-my/go-containers/event"
)

// Revision is one value a stream field held.
type Revision struct {
	Value  string
	Source RevisionSource
	At     time.Time
}

// StreamDelta is the argument of IStreamString.OnDelta.
type StreamDelta struct {
	event.Args
	Delta string
}

// StreamDone is the argument of IStreamString.OnDone.
type StreamDone struct {
	event.Args
	Value string
}

// IStreamString is a text field that fills while a model streams. It never shrinks: a value that
// a finish or a retry replaces stays in Revisions.
type IStreamString interface {
	String() string
	Len() uint
	Done() bool
	// OnDelta fires once per appended piece. Its listeners are allocated on first use.
	OnDelta() *event.Event[StreamDelta]
	OnDone() *event.Event[StreamDone]
	// Revisions is every value the field held, oldest first. The last entry is the current value.
	Revisions() []Revision
}

// StreamWriter writes a stream field. The session and the dialect decoders hold it; a host reads
// through IStreamString.
type StreamWriter interface {
	Append(delta string)
	// Finish ends the field. A final value that differs from what streamed becomes a new revision.
	Finish(final string)
	// Restart begins a retry: the current value becomes a retry revision and a fresh value starts.
	Restart()
	// Set replaces the value as a host edit, kept as a revision.
	Set(value string, src RevisionSource)
}

type streamEvents struct {
	delta event.Event[StreamDelta]
	done  event.Event[StreamDone]
}

type streamString struct {
	mu      sync.Mutex
	b       strings.Builder
	s       string
	done    bool
	source  RevisionSource
	started time.Time
	revs    []Revision
	ev      *streamEvents
}

// NewStreamString is an empty field that is still streaming.
func NewStreamString() IStreamString {
	return &streamString{started: time.Now()}
}

// FinishedString is a field that holds value and is done.
func FinishedString(value string) IStreamString {
	return &streamString{s: value, done: true, source: RevisionFinal, started: time.Now()}
}

// WriterOf gives the writer of a field made by this package.
func WriterOf(s IStreamString) StreamWriter {
	return s.(*streamString)
}

func (s *streamString) current() string {
	if s.done {
		return s.s
	}
	return s.b.String()
}

func (s *streamString) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.current()
}

func (s *streamString) Len() uint {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return uint(len(s.s))
	}
	return uint(s.b.Len())
}

func (s *streamString) Done() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.done
}

func (s *streamString) events() *streamEvents {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ev == nil {
		s.ev = &streamEvents{}
	}
	return s.ev
}

func (s *streamString) OnDelta() *event.Event[StreamDelta] { return &s.events().delta }
func (s *streamString) OnDone() *event.Event[StreamDone]   { return &s.events().done }

func (s *streamString) Revisions() []Revision {
	s.mu.Lock()
	defer s.mu.Unlock()
	out := make([]Revision, len(s.revs), len(s.revs)+1)
	copy(out, s.revs)
	return append(out, Revision{Value: s.current(), Source: s.source, At: s.started})
}

// pushRevision moves the current value into the history. The caller holds the lock.
func (s *streamString) pushRevision(src RevisionSource) {
	s.revs = append(s.revs, Revision{Value: s.current(), Source: src, At: s.started})
	s.b.Reset()
	s.s = ""
	s.started = time.Now()
}

func (s *streamString) Append(delta string) {
	if delta == "" {
		return
	}
	s.mu.Lock()
	if s.done {
		// A late piece after the end restarts the field rather than losing either value.
		s.revs = append(s.revs, Revision{Value: s.s, Source: s.source, At: s.started})
		s.b.Reset()
		s.b.WriteString(s.s)
		s.s, s.done, s.source = "", false, RevisionStreamed
	}
	s.b.WriteString(delta)
	ev := s.ev
	s.mu.Unlock()
	if ev != nil {
		_ = ev.delta.Invoke(StreamDelta{Delta: delta})
	}
}

func (s *streamString) Finish(final string) {
	s.mu.Lock()
	streamed := s.current()
	if !s.done && streamed != "" && final != streamed {
		s.revs = append(s.revs, Revision{Value: streamed, Source: RevisionStreamed, At: s.started})
		s.started = time.Now()
	}
	s.b = strings.Builder{}
	s.s, s.done, s.source = final, true, RevisionFinal
	ev := s.ev
	s.mu.Unlock()
	if ev != nil {
		_ = ev.done.Invoke(StreamDone{Value: final})
	}
}

// FinishStreamed ends the field on what streamed.
func FinishStreamed(s IStreamString) {
	WriterOf(s).Finish(s.String())
}

func (s *streamString) Restart() {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.current() == "" && len(s.revs) == 0 {
		return
	}
	s.pushRevision(RevisionRetry)
	s.done, s.source = false, RevisionStreamed
}

func (s *streamString) Set(value string, src RevisionSource) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pushRevision(s.source)
	s.s, s.done, s.source = value, true, src
}
