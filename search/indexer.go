package search

import (
	"context"
	"fmt"

	"github.com/wow-look-at-my/go-containers/set"
)

// Conversation identifies conversation and says whether its transcript has
// moved since the index last read it.
type Conversation struct {
	// ID is the host's own conversation id.
	ID string
	// Owner scopes searches; it must be the SAME value the host later searches with.
	Owner string
	// Revision changes whenever the messages change; a revision that fails to change makes the index quietly wrong.
	Revision string
}

// Message is transcript entry as the index needs it.
type Message struct {
	// ID must be STABLE across re-reads, because it is what the embeddings are keyed by.
	ID        string
	Role      string
	Content   string
	CreatedAt string
}

// Source is the host's conversations, as the index reads them. It is an
// interface so the index can sit over a directory of XML files, a SQL store,
// or anything else, and so it can be tested against a corpus in memory.
type Source interface {
	// Conversations returns every conversation that currently exists, with its revision.
	Conversations(ctx context.Context) ([]Conversation, error)
	// Messages returns conversation's transcript, in order.
	Messages(ctx context.Context, conversationID string) ([]Message, error)
}

// Ingest brings the index up to the source's current state, re-reading only conversations whose revision moved.
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

// reindex replaces conversation's indexed messages and records the
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
func (i *Index) dropOrphanedVectors(ctx context.Context) error {
	for _, table := range []string{"embeddings", "embed_status"} {
		if _, err := i.sql.ExecContext(ctx, `DELETE FROM `+table+`
			WHERE message_id NOT IN (SELECT message_id FROM indexed_messages)`); err != nil {
			return fmt.Errorf("search: drop orphaned rows from %s: %w", table, err)
		}
	}
	return nil
}
