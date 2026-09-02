package search

import (
	"context"
	"fmt"
	"time"
)

const (
	// chunkRunes is the window embedding covers; too wide a window averages unrelated topics into a direction that matches none.
	chunkRunes = 1200
	// chunkOverlap is how much of the previous window each chunk repeats, so a straddling passage is whole in chunk.
	chunkOverlap = 120
	// maxChunksPerMessage caps what message can cost; the cap is recorded per message (embed_status.chunks_total).
	maxChunksPerMessage = 16
	// embedBatchSize is how many chunks go in request; a message's chunks never span batches.
	embedBatchSize = 64
)

// Embedder turns text into vectors; retrieval is asymmetric, so passages and queries embed separately.
type Embedder interface {
	// EmbedDocuments embeds text being STORED; it must return exactly vector per input, in order.
	EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
	// EmbedQuery embeds search query.
	EmbedQuery(ctx context.Context, text string) ([]float32, error)
}

// chunkContent splits content into overlapping rune windows, returning the chunks to embed and the total needed (total > len(chunks) is truncation).
func chunkContent(content string) (chunks []string, total int) {
	r := []rune(content)
	if len(r) == 0 {
		return nil, 0
	}
	if len(r) <= chunkRunes {
		return []string{content}, 1
	}
	stride := chunkRunes - chunkOverlap
	for start := 0; start < len(r); start += stride {
		end := min(start+chunkRunes, len(r))
		total++
		if len(chunks) < maxChunksPerMessage {
			chunks = append(chunks, string(r[start:end]))
		}
		if end == len(r) {
			break
		}
	}
	return chunks, total
}

// pending is message awaiting embedding.
type pending struct {
	id      string
	content string
}

// PendingForModel returns up to limit of the owner's messages that have no
// embedding under model, NEWEST.
//
// Newest is the load-bearing part of the ordering. A first-time backfill
// over a long history drains over minutes, and during that time the covered
// half should be the half most likely to be searched. It also means a caller
// who never lets it finish still has a useful index.
func (i *Index) PendingForModel(ctx context.Context, owner, model string, limit int) ([]pending, error) {
	rows, err := i.sql.QueryContext(ctx, `
		SELECT m.message_id, m.content
		FROM indexed_messages m
		WHERE m.owner = ?
		  AND m.content <> ''
		  AND NOT EXISTS (
		      SELECT 1 FROM embed_status s
		      WHERE s.message_id = m.message_id AND s.model = ?
		  )
		ORDER BY m.created_at DESC
		LIMIT ?`, owner, model, limit)
	if err != nil {
		return nil, fmt.Errorf("search: list unembedded messages: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []pending
	for rows.Next() {
		var p pending
		if err := rows.Scan(&p.id, &p.content); err != nil {
			return nil, fmt.Errorf("search: scan unembedded message: %w", err)
		}
		out = append(out, p)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate unembedded messages: %w", err)
	}
	return out, nil
}

// batch is a set of whole messages whose chunks fit in embedding request.
type batch struct {
	texts []string
	msgs  []batchMessage
}

type batchMessage struct {
	id     string
	chunks int
	total  int
}

// EmbedPending embeds up to limit of the owner's unembedded messages with
// model and returns how many messages it finished.
//
// A request that fails aborts the pass with that error rather than skipping
// the batch: the messages stay unembedded, so the next pass picks them up
// again. There is no attempt counter and nothing is marked permanently failed
// -- a message that cannot be embedded today is retried on whatever cadence
// the caller runs, and the reason is reported through Status rather than
// buried.
func (i *Index) EmbedPending(ctx context.Context, owner, model string, e Embedder, limit int) (int, error) {
	msgs, err := i.PendingForModel(ctx, owner, model, limit)
	if err != nil {
		return 0, err
	}

	done := 0
	cur := batch{}
	flush := func() error {
		if len(cur.texts) == 0 {
			return nil
		}
		n, err := i.embedBatch(ctx, model, e, cur)
		done += n
		cur = batch{}
		return err
	}

	for _, m := range msgs {
		chunks, total := chunkContent(m.content)
		if len(chunks) == 0 {
			continue
		}
		if len(cur.texts)+len(chunks) > embedBatchSize {
			if err := flush(); err != nil {
				return done, err
			}
		}
		cur.texts = append(cur.texts, chunks...)
		cur.msgs = append(cur.msgs, batchMessage{id: m.id, chunks: len(chunks), total: total})
	}
	if err := flush(); err != nil {
		return done, err
	}
	return done, nil
}

// embedBatch makes embedding request and writes every vector it returned,
// with the per-message embed_status rows, in transaction. That is what
// lets the pending query trust embed_status: a message either has its full set
// of chunks stored, or it has no record at all and is picked up again.
func (i *Index) embedBatch(ctx context.Context, model string, e Embedder, b batch) (n int, err error) {
	vecs, err := e.EmbedDocuments(ctx, b.texts)
	if err != nil {
		return 0, fmt.Errorf("search: embed %d chunks with %q: %w", len(b.texts), model, err)
	}
	// A provider returning a different number of vectors than inputs cannot be matched back to their messages.
	if len(vecs) != len(b.texts) {
		return 0, fmt.Errorf("search: %q returned %d vectors for %d inputs", model, len(vecs), len(b.texts))
	}

	tx, err := i.sql.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("search: begin embed write: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	now := time.Now().UTC().Format(time.RFC3339Nano)
	at := 0
	for _, m := range b.msgs {
		for c := range m.chunks {
			v := vecs[at+c]
			blob, encErr := encodeVector(v)
			if encErr != nil {
				err = fmt.Errorf("search: message %q chunk %d: %w", m.id, c, encErr)
				return 0, err
			}
			if _, err = tx.ExecContext(ctx,
				`INSERT INTO embeddings (message_id, chunk_index, model, dim, vector, created_at)
				 VALUES (?, ?, ?, ?, ?, ?)
				 ON CONFLICT(message_id, chunk_index, model) DO UPDATE SET
				   dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at`,
				m.id, c, model, len(v), blob, now,
			); err != nil {
				return 0, fmt.Errorf("search: store vector for %q: %w", m.id, err)
			}
		}
		if _, err = tx.ExecContext(ctx,
			`INSERT INTO embed_status (message_id, model, chunks, chunks_total, created_at)
			 VALUES (?, ?, ?, ?, ?)
			 ON CONFLICT(message_id, model) DO UPDATE SET
			   chunks = excluded.chunks, chunks_total = excluded.chunks_total,
			   created_at = excluded.created_at`,
			m.id, model, m.chunks, m.total, now,
		); err != nil {
			return 0, fmt.Errorf("search: record embed status for %q: %w", m.id, err)
		}
		at += m.chunks
	}

	if err = tx.Commit(); err != nil {
		return 0, fmt.Errorf("search: commit embed write: %w", err)
	}
	return len(b.msgs), nil
}

// DropModel removes every vector stored under model and returns how many
// messages it un-embedded. It is what changing embedding model costs: vectors
// from models are not comparable, so the old ones can never answer a query
// again and are storage with no reader.
func (i *Index) DropModel(ctx context.Context, model string) (n int, err error) {
	tx, err := i.sql.BeginTx(ctx, nil)
	if err != nil {
		return 0, fmt.Errorf("search: begin drop model: %w", err)
	}
	defer func() {
		if err != nil {
			_ = tx.Rollback()
		}
	}()

	if _, err = tx.ExecContext(ctx, `DELETE FROM embeddings WHERE model = ?`, model); err != nil {
		return 0, fmt.Errorf("search: drop vectors for %q: %w", model, err)
	}
	res, err := tx.ExecContext(ctx, `DELETE FROM embed_status WHERE model = ?`, model)
	if err != nil {
		return 0, fmt.Errorf("search: drop embed status for %q: %w", model, err)
	}
	affected, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("search: drop model %q: %w", model, err)
	}
	if err = tx.Commit(); err != nil {
		return 0, fmt.Errorf("search: commit drop model: %w", err)
	}
	return int(affected), nil
}

// ModelsInUse lists every embedding model that currently has vectors stored.
// It is what makes a model switch reportable: a model here that nobody is
// asking with any more is storage being paid for and never read.
func (i *Index) ModelsInUse(ctx context.Context) ([]string, error) {
	rows, err := i.sql.QueryContext(ctx, `SELECT DISTINCT model FROM embeddings ORDER BY model`)
	if err != nil {
		return nil, fmt.Errorf("search: list stored models: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []string
	for rows.Next() {
		var m string
		if err := rows.Scan(&m); err != nil {
			return nil, fmt.Errorf("search: scan stored model: %w", err)
		}
		out = append(out, m)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate stored models: %w", err)
	}
	return out, nil
}
