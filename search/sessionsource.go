package search

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

// SessionSource indexes an agentic-loop session.Store, so the transports that
// already keep conversations -- cai, and the http and socket servers -- get a
// searchable history without changing how any of them store anything.
type SessionSource struct {
	// Store is the conversations to index.
	Store session.Store
	// Owner scopes every conversation this source reports, and is what a
	// search must ask with. A single-user store leaves it empty; a host
	// running one store per tenant sets it per source.
	Owner string
}

// Revisions is the optimization a store can offer: a change marker per
// conversation without reading the transcripts.
//
// Without it, SessionSource has to read every conversation on every pass just
// to learn which ones moved, which is the whole corpus per pass. session.File
// implements this from the file's size and modification time, turning that
// read into a stat.
type Revisions interface {
	Revisions() (map[string]string, error)
}

// Conversations implements Source.
func (s *SessionSource) Conversations(_ context.Context) ([]Conversation, error) {
	if r, ok := s.Store.(Revisions); ok {
		revs, err := r.Revisions()
		if err != nil {
			return nil, fmt.Errorf("search: read session revisions: %w", err)
		}
		out := make([]Conversation, 0, len(revs))
		for id, rev := range revs {
			out = append(out, Conversation{ID: id, Owner: s.Owner, Revision: rev})
		}
		return out, nil
	}

	ids, err := s.Store.List()
	if err != nil {
		return nil, fmt.Errorf("search: list sessions: %w", err)
	}
	out := make([]Conversation, 0, len(ids))
	for _, id := range ids {
		req, err := s.Store.Get(id)
		if err != nil {
			return nil, fmt.Errorf("search: read session %q: %w", id, err)
		}
		out = append(out, Conversation{ID: id, Owner: s.Owner, Revision: transcriptRevision(req)})
	}
	return out, nil
}

// Messages implements Source.
func (s *SessionSource) Messages(_ context.Context, conversationID string) ([]Message, error) {
	req, err := s.Store.Get(conversationID)
	if err != nil {
		return nil, fmt.Errorf("search: read session %q: %w", conversationID, err)
	}
	out := make([]Message, 0, len(req.Messages))
	for pos, m := range req.Messages {
		out = append(out, Message{
			ID:      messageID(conversationID, pos, m),
			Role:    string(m.Role),
			Content: searchableText(m),
		})
	}
	return out, nil
}

// messageID is the stable identity the index keys a message's vectors by.
//
// A message the host already identified keeps that id. Otherwise the id is the
// conversation plus the message's POSITION, which is stable for the
// append-only transcript a turn produces -- appending message N+1 does not
// move any of the first N.
//
// The cost of that choice: INSERTING or removing a message mid-transcript
// shifts every position after it, so each of those messages looks new and is
// embedded again. session.Put replacing a transcript wholesale is the way that
// happens. It is the honest trade for a store whose messages have no ids of
// their own -- the alternative, hashing the content, silently merges two
// identical messages in one conversation into a single searchable row.
func messageID(conversationID string, pos int, m commonai.Message) string {
	if m.ID != "" {
		return m.ID
	}
	return conversationID + ":" + strconv.Itoa(pos)
}

// searchableText is the part of a message worth indexing: its text, in order.
//
// Thinking blocks are deliberately excluded. They carry provider tokens and
// are the model's working notes rather than anything it said, and a search
// that surfaced them would answer with reasoning the user never saw. Tool
// CALLS are excluded for the same reason -- the arguments are a machine
// protocol. A tool RESULT is ordinary content and is indexed: it is often
// exactly what someone is looking for.
func searchableText(m commonai.Message) string {
	var out []byte
	for _, p := range m.EffectiveParts() {
		t, ok := p.(commonai.TextPart)
		if !ok {
			continue
		}
		if len(out) > 0 {
			out = append(out, '\n')
		}
		out = append(out, t.Text...)
	}
	return string(out)
}

// transcriptRevision is the fallback change marker: a hash over the messages
// that would be indexed.
//
// It hashes the same text Messages returns, so a change the index cannot see
// -- a tool call's arguments, a thinking block's signature -- does not trigger
// a re-read that would produce identical rows.
func transcriptRevision(req commonai.Request) string {
	h := sha256.New()
	for _, m := range req.Messages {
		fmt.Fprintf(h, "%s\x00%s\x00", m.Role, searchableText(m))
	}
	return hex.EncodeToString(h.Sum(nil))
}
