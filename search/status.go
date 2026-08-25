package search

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
)

// Status is what the index can honestly say about itself. It exists because
// the index is asynchronous: a search answers from whatever has been indexed
// so far, and without this the difference between "no message says that" and
// "the message that says it has not been indexed yet" is invisible.
type Status struct {
	// IndexedConversations / IndexedMessages is what the text index holds for this owner.
	IndexedConversations int64
	IndexedMessages      int64
	// StaleConversations is how many conversations sit at a revision the index has not read.
	StaleConversations int64
	// LastError is the last indexing failure, verbatim, or "" if the last pass succeeded.
	LastError string

	// Model is the embedding model this status was computed for, or "" if semantic search is off.
	Model string
	// EmbeddedMessages is how many messages have vectors under Model; PendingEmbeddings how many need one.
	EmbeddedMessages  int64
	PendingEmbeddings int64
	// TruncatedMessages is how many messages exceeded maxChunksPerMessage; the text half still covers them.
	TruncatedMessages int64
	// Dim is the width of the stored vectors, read from the rows rather than assumed.
	Dim int64
}

// Status reports the index's state for one owner and embedding model. model
// may be "" for a caller with no embedding endpoint; the embedding fields are
// then left zero rather than computed against a model nobody chose.
func (i *Index) Status(ctx context.Context, src Source, owner, model string) (Status, error) {
	var s Status

	if err := i.sql.QueryRowContext(ctx, `
		SELECT
			(SELECT COUNT(*) FROM indexed_conversations WHERE owner = ?),
			(SELECT COUNT(*) FROM indexed_messages WHERE owner = ?)`,
		owner, owner,
	).Scan(&s.IndexedConversations, &s.IndexedMessages); err != nil {
		return s, fmt.Errorf("search: count indexed rows: %w", err)
	}

	stale, err := i.staleCount(ctx, src, owner)
	if err != nil {
		return s, err
	}
	s.StaleConversations = stale

	if s.LastError, err = i.meta(ctx, metaLastError); err != nil {
		return s, err
	}

	if model == "" {
		return s, nil
	}
	s.Model = model

	if err := i.sql.QueryRowContext(ctx, `
		SELECT
			COUNT(*),
			COALESCE(SUM(CASE WHEN st.chunks_total > st.chunks THEN 1 ELSE 0 END), 0)
		FROM embed_status st
		JOIN indexed_messages m ON m.message_id = st.message_id
		WHERE m.owner = ? AND st.model = ?`, owner, model,
	).Scan(&s.EmbeddedMessages, &s.TruncatedMessages); err != nil {
		return s, fmt.Errorf("search: count embedded messages: %w", err)
	}

	if err := i.sql.QueryRowContext(ctx, `
		SELECT COUNT(*)
		FROM indexed_messages m
		WHERE m.owner = ?
		  AND m.content <> ''
		  AND NOT EXISTS (
		      SELECT 1 FROM embed_status s
		      WHERE s.message_id = m.message_id AND s.model = ?
		  )`, owner, model,
	).Scan(&s.PendingEmbeddings); err != nil {
		return s, fmt.Errorf("search: count pending embeddings: %w", err)
	}

	err = i.sql.QueryRowContext(ctx,
		`SELECT dim FROM embeddings WHERE model = ? LIMIT 1`, model).Scan(&s.Dim)
	if err != nil && !errors.Is(err, sql.ErrNoRows) {
		return s, fmt.Errorf("search: read vector dimension: %w", err)
	}
	return s, nil
}

// staleCount is how many of the owner's conversations sit at a revision the
// index has not read -- including ones it has never seen at all.
//
// It asks the source rather than the index, because the whole question is
// about what the index does not know yet.
func (i *Index) staleCount(ctx context.Context, src Source, owner string) (int64, error) {
	convs, err := src.Conversations(ctx)
	if err != nil {
		return 0, fmt.Errorf("search: list conversations for status: %w", err)
	}
	known, err := i.knownRevisions(ctx)
	if err != nil {
		return 0, err
	}
	var n int64
	for _, c := range convs {
		if c.Owner != owner {
			continue
		}
		if at, ok := known[c.ID]; !ok || at != c.Revision {
			n++
		}
	}
	return n, nil
}
