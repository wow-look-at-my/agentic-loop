// Package session stores conversations between calls, for the transports that
// keep state on the caller's behalf.
//
// A stored conversation IS a request that has not been sent: the same model,
// system prompt, tools and defaults, plus the transcript a next turn appends
// to. That is why it is one <conversation> document rather than a schema of
// its own -- a session a caller can read, edit and hand back is worth more
// than an opaque handle.
package session

import (
	"errors"
	"fmt"
	"sort"
	"sync"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
)

// ErrNotFound is returned for an id no store holds. It is a sentinel so a
// transport can answer 404 rather than guessing from an error string.
var ErrNotFound = errors.New("session: no conversation with that id")

// Store holds conversations by id. Implementations must be safe for
// concurrent use by multiple goroutines.
type Store interface {
	// Create stores req under a new id and returns it.
	Create(req commonai.Request) (string, error)
	// Put stores req under an id the CALLER chose, replacing whatever was
	// there. A transport whose clients name their own sessions needs this: a
	// store that only ever hands out its own ids cannot hold one, and the
	// alternative is a caller keeping a map from its names to ours.
	Put(id string, req commonai.Request) error
	// Get returns the stored conversation.
	Get(id string) (commonai.Request, error)
	// Append adds messages to the stored transcript and returns the
	// conversation as it now stands.
	Append(id string, msgs ...commonai.Message) (commonai.Request, error)
	// Delete removes a conversation. Deleting one that is not there is an
	// error: a caller that asked to delete something has a belief about what
	// exists, and silence would leave it standing.
	Delete(id string) error
	// List returns every id held, in a stable order.
	List() ([]string, error)
}

// validID rejects an id that could escape a file store's directory or collide
// with its naming. It is enforced by every store so a conversation created
// through one is addressable through another.
func validID(id string) error {
	if id == "" {
		return fmt.Errorf("session: an id is required")
	}
	for _, r := range id {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9':
		case r == '-', r == '_':
		default:
			return fmt.Errorf("session: id %q may only hold letters, digits, - and _", id)
		}
	}
	return nil
}

// Memory is a Store held in memory. It is the default for a server that has no
// business outliving its process.
type Memory struct {
	mu    sync.Mutex
	seq   int
	convs map[string]commonai.Request
}

// NewMemory returns an empty in-memory store.
func NewMemory() *Memory { return &Memory{convs: map[string]commonai.Request{}} }

// Create implements Store.
func (m *Memory) Create(req commonai.Request) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seq++
	id := fmt.Sprintf("c%d", m.seq)
	m.convs[id] = cloneRequest(req)
	return id, nil
}

// Put implements Store.
func (m *Memory) Put(id string, req commonai.Request) error {
	if err := validID(id); err != nil {
		return err
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	m.convs[id] = cloneRequest(req)
	return nil
}

// Get implements Store.
func (m *Memory) Get(id string) (commonai.Request, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.convs[id]
	if !ok {
		return commonai.Request{}, fmt.Errorf("%w: %q", ErrNotFound, id)
	}
	return cloneRequest(req), nil
}

// Append implements Store.
func (m *Memory) Append(id string, msgs ...commonai.Message) (commonai.Request, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	req, ok := m.convs[id]
	if !ok {
		return commonai.Request{}, fmt.Errorf("%w: %q", ErrNotFound, id)
	}
	req.Messages = append(req.Messages, msgs...)
	m.convs[id] = req
	return cloneRequest(req), nil
}

// Delete implements Store.
func (m *Memory) Delete(id string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if _, ok := m.convs[id]; !ok {
		return fmt.Errorf("%w: %q", ErrNotFound, id)
	}
	delete(m.convs, id)
	return nil
}

// List implements Store.
func (m *Memory) List() ([]string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	ids := make([]string, 0, len(m.convs))
	for id := range m.convs {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids, nil
}

// cloneRequest copies the parts a caller could otherwise mutate through the
// value it was handed. A store that returns its own slice hands out a way to
// change stored history without going through Append.
func cloneRequest(req commonai.Request) commonai.Request {
	out := req
	out.Messages = append([]commonai.Message(nil), req.Messages...)
	out.Tools = append([]commonai.ToolDecl(nil), req.Tools...)
	out.SystemParts = append([]commonai.Part(nil), req.SystemParts...)
	return out
}
