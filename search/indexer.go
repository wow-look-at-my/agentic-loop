package search

import (
	"context"
	"fmt"

	"github.com/wow-look-at-my/go-containers/set"
)

// Conversation identifies one conversation and says whether its transcript has
// moved since the index last read it.
type Conversation struct {
	// ID is the host's own conversation id.
	ID string
	// Owner scopes searches. A single-user host leaves it empty; what matters
	// is that it is the SAME value the host later searches with, because the
	// match is exact and there is no wildcard.
	Owner string
	// Revision changes whenever the conversation's messages change, and is
	// otherwise opaque to the index. A message count is enough for an
	// append-only store; a hash, an mtime or a version number all work. A
	// revision that fails to change when the transcript does is the one way to
	// make this index quietly wrong, so a store that cannot answer honestly
	// should return something that always differs and pay the re-read.
	Revision string
}

// Message is one transcript entry as the index needs it.
type Message struct {
	// ID must be STABLE across re-reads of the same conversation, because it
	// is what the embeddings are keyed by: an id that changes re-embeds the
	// message, and one that is reused for different text attaches the wrong
	// vector to it. A host with real message ids should pass them. One without
	// can derive an id from the conversation and position -- see
	// SessionSource, and what that costs when a message is inserted rather
	// than appended.
	ID        string
	Role      string
	Content   string
	CreatedAt string
}

// Source is the host's conversations, as the index reads them. It is an
// interface so the index can sit over a directory of XML files, a SQL store,
// or anything else, and so it can be tested against a corpus in memory.
type Source interface {
	// Conversations returns every conversation that currently exists, with the
	// revision each is at.
	Conversations(ctx context.Context) ([]Conversation, error)
	// Messages returns one conversation's transcript, in order.
	Messages(ctx context.Context, conversationID string) ([]Message, error)
}

// Ingest brings the index up to the source's current state and returns how
// many conversations it re-read.
//
// It compares each conversation's revision against the one recorded when it
// was last indexed, and re-reads only those that have moved. A conversation
// that has moved is re-indexed WHOLE -- its rows are replaced, not appended to
// -- because Revision is opaque: the index knows the transcript changed and
// nothing about how, and a store whose Put replaces a conversation outright
// can change any part of it.
//
// Re-indexing whole does not re-embed whole. The vectors are keyed by message
// id and are never touched here, so a conversation that gained one message
// costs one embedding, not a transcript's worth.
//
// Conversations the source no longer lists are removed, along with their
// vectors: a deleted conversation is not coming back, so keeping them would be
// storage nothing can ever join to.
func (i *Index) Ingest(ctx context.Context, src Source) (int, error) {
	convs, err := src.Conversations(ctx)
	if err != nil {
		return 0, fmt.Errorf("search: list conversations: %w", err)
	}
	known, err := i.knownRevisions(ctx)
	if err != nil {
		return 0, err
	}

	live := set.New[string]()
	changed := 0
	for _, c := range convs {
		live.Add(c.ID)
		if at, ok := known[c.ID]; ok && at == c.Revision {
			continue
		}
		msgs, err := src.Messages(ctx, c.ID)
		if err != nil {
			return changed, fmt.Errorf("search: read conversation %q: %w", c.ID, err)
		}
		if err := i.reindex(ctx, c, msgs); err != nil {
			return changed, err
		}
		changed++
	}

	for id := range known {
		if live.Contains(id) {
			continue
		}
		if err := i.forget(ctx, id); err != nil {
			return changed, err
		}
	}
	return changed, i.dropOrphanedVectors(ctx)
}

// knownRevisions returns the revision each indexed conversation was last read
// at.
func (i *Index) knownRevisions(ctx context.Context) (map[string]string, error) {
	rows, err := i.sql.QueryContext(ctx, `SELECT conversation_id, revision FROM indexed_conversations`)
	if err != nil {
		return nil, fmt.Errorf("search: read indexed revisions: %w", err)
	}
	defer func() { _ = rows.Close() }()

	out := map[string]string{}
	for rows.Next() {
		var id, rev string
		if err := rows.Scan(&id, &rev); err != nil {
			return nil, fmt.Errorf("search: scan indexed revision: %w", err)
		}
		out[id] = rev
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate indexed revisions: %w", err)
	}
	return out, nil
}

// reindex replaces one conversation's indexed messages and records the
// revision they were read at, in a single transaction. The revision landing in
// the same commit as the rows it describes is what makes a crash mid-pass
// re-read that conversation rather than skip it.
func (i *Index) reindex(ctx context.Context, c Conversation, msgs []Message) (err error) {
	tx, err := i.sql.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("search: begin reindex: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	if _, err = tx.ExecContext(ctx,
		`DELETE FROM indexed_messages WHERE conversation_id = ?`, c.ID); err != nil {
		return fmt.Errorf("search: clear conversation %q: %w", c.ID, err)
	}
	for pos, m := range msgs {
		if m.ID == "" {
			return fmt.Errorf("search: conversation %q message %d has no id; an index keyed by nothing cannot hold vectors", c.ID, pos)
		}
		if _, err = tx.ExecContext(ctx,
			`INSERT INTO indexed_messages
			   (message_id, conversation_id, owner, role, content, position, created_at)
			 VALUES (?, ?, ?, ?, ?, ?, ?)
			 ON CONFLICT(message_id) DO UPDATE SET
			   conversation_id = excluded.conversation_id,
			   owner           = excluded.owner,
			   role            = excluded.role,
			   content         = excluded.content,
			   position        = excluded.position,
			   created_at      = excluded.created_at`,
			m.ID, c.ID, c.Owner, m.Role, m.Content, pos, m.CreatedAt,
		); err != nil {
			return fmt.Errorf("search: index message %q: %w", m.ID, err)
		}
	}
	if _, err = tx.ExecContext(ctx,
		`INSERT INTO indexed_conversations (conversation_id, owner, revision)
		 VALUES (?, ?, ?)
		 ON CONFLICT(conversation_id) DO UPDATE SET
		   owner = excluded.owner, revision = excluded.revision`,
		c.ID, c.Owner, c.Revision,
	); err != nil {
		return fmt.Errorf("search: record revision for %q: %w", c.ID, err)
	}
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("search: commit reindex: %w", err)
	}
	return nil
}

// forget removes a conversation the source no longer has, and the vectors of
// its messages.
func (i *Index) forget(ctx context.Context, conversationID string) (err error) {
	tx, err := i.sql.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("search: begin forget: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	const scope = `SELECT message_id FROM indexed_messages WHERE conversation_id = ?`
	for _, stmt := range []string{
		`DELETE FROM embeddings WHERE message_id IN (` + scope + `)`,
		`DELETE FROM embed_status WHERE message_id IN (` + scope + `)`,
		`DELETE FROM indexed_messages WHERE conversation_id = ?`,
		`DELETE FROM indexed_conversations WHERE conversation_id = ?`,
	} {
		if _, err = tx.ExecContext(ctx, stmt, conversationID); err != nil {
			return fmt.Errorf("search: forget conversation %q: %w", conversationID, err)
		}
	}
	if err = tx.Commit(); err != nil {
		return fmt.Errorf("search: commit forget: %w", err)
	}
	return nil
}

// dropOrphanedVectors deletes vectors whose message is no longer indexed.
//
// A conversation that is re-indexed after an EDIT rather than an append leaves
// these behind: the replaced message's id is gone from indexed_messages, but
// its vector is keyed by that id and nothing above deletes it. Without this
// they accumulate for the life of the index, and every one is paid-for storage
// that no query can reach.
func (i *Index) dropOrphanedVectors(ctx context.Context) error {
	for _, table := range []string{"embeddings", "embed_status"} {
		if _, err := i.sql.ExecContext(ctx, `DELETE FROM `+table+`
			WHERE message_id NOT IN (SELECT message_id FROM indexed_messages)`); err != nil {
			return fmt.Errorf("search: drop orphaned rows from %s: %w", table, err)
		}
	}
	return nil
}
