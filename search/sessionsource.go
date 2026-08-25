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
	// Owner scopes every conversation this source reports; a search must ask with it.
	Owner string
}

// Revisions is the optional change marker a store can offer, avoiding a full read per pass.
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

// messageID is the stable vector key: a host message id if any, else conversation plus the message's position.
func messageID(conversationID string, pos int, m commonai.Message) string {
	if m.ID != "" {
		return m.ID
	}
	return conversationID + ":" + strconv.Itoa(pos)
}

// searchableText is the message's indexed text; thinking blocks and tool calls are excluded, results indexed.
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

// transcriptRevision hashes the indexed text, so a change the index cannot see doesn't trigger a re-read.
func transcriptRevision(req commonai.Request) string {
	h := sha256.New()
	for _, m := range req.Messages {
		fmt.Fprintf(h, "%s\x00%s\x00", m.Role, searchableText(m))
	}
	return hex.EncodeToString(h.Sum(nil))
}
