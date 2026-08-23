package search

import (
	"context"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestIngestReadsOnlyTheConversationsThatMoved(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "first message"))
	src.put("c2", "u1", msg("m2", "user", "second message"))

	n, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 2, n)

	// A pass with nothing moved re-reads nothing: the recorded revisions match.
	n, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Zero(t, n)

	src.put("c2", "u1", msg("m2", "user", "second message"), msg("m3", "user", "a reply"))
	n, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 1, n, "only the conversation whose revision moved is re-read")

	status, err := idx.Status(ctx, src, "u1", "")
	require.NoError(t, err)
	assert.Equal(t, int64(2), status.IndexedConversations)
	assert.Equal(t, int64(3), status.IndexedMessages)
	assert.Zero(t, status.StaleConversations)
}

func TestStatusReportsWhatTheIndexHasNotReadYet(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "hello"))
	src.put("c2", "u1", msg("m2", "user", "hello"))

	status, err := idx.Status(ctx, src, "u1", "")
	require.NoError(t, err)
	assert.Equal(t, int64(2), status.StaleConversations,
		"an index that has read nothing must say so, not report an empty corpus")

	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	status, err = idx.Status(ctx, src, "u1", "")
	require.NoError(t, err)
	assert.Zero(t, status.StaleConversations)
}

func TestReindexingAConversationDoesNotReEmbedIt(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "the original message"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	emb := &bagEmbedder{dim: 16}
	n, err := idx.EmbedPending(ctx, "u1", "m", emb, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	afterFirst := emb.calls

	// Appending a message re-indexes the whole conversation, but only the new
	// message needs a vector. Re-embedding the transcript on every append is
	// exactly the bill this keying exists to avoid.
	src.put("c1", "u1", msg("m1", "user", "the original message"), msg("m2", "assistant", "a new reply"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)

	pendingBefore, err := idx.PendingForModel(ctx, "u1", "m", 10)
	require.NoError(t, err)
	require.Len(t, pendingBefore, 1)
	assert.Equal(t, "m2", pendingBefore[0].id)

	n, err = idx.EmbedPending(ctx, "u1", "m", emb, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, n)
	assert.Equal(t, afterFirst+1, emb.calls, "one request for one new message")
}

func TestADeletedConversationLeavesTheIndexWithItsVectors(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "keep this one"))
	src.put("c2", "u1", msg("m2", "user", "delete this one"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "m", &bagEmbedder{dim: 16}, 10)
	require.NoError(t, err)

	src.delete("c2")
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)

	status, err := idx.Status(ctx, src, "u1", "m")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.IndexedConversations)
	assert.Equal(t, int64(1), status.IndexedMessages)
	assert.Equal(t, int64(1), status.EmbeddedMessages,
		"a deleted conversation's vectors go with it: nothing can ever join to them again")
}

// An EDIT, rather than an append, retires a message id. Its vector is keyed by
// that id and no conversation-level delete reaches it, so without the orphan
// sweep it would be paid-for storage no query can read, forever.
func TestAReplacedMessageDoesNotLeaveItsVectorBehind(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "the first wording"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "m", &bagEmbedder{dim: 16}, 10)
	require.NoError(t, err)

	// The transcript is replaced: m1 is gone and m9 is in its place.
	src.put("c1", "u1", msg("m9", "user", "a different wording entirely"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)

	var orphans int
	require.NoError(t, idx.sql.QueryRowContext(ctx,
		`SELECT COUNT(*) FROM embeddings
		 WHERE message_id NOT IN (SELECT message_id FROM indexed_messages)`).Scan(&orphans))
	assert.Zero(t, orphans)

	status, err := idx.Status(ctx, src, "u1", "m")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.PendingEmbeddings, "the new wording still needs its own vector")
	assert.Zero(t, status.EmbeddedMessages)
}

func TestAMessageWithoutAnIDIsRefused(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", Message{Role: "user", Content: "no id"})
	_, err := idx.Ingest(ctx, src)
	require.Error(t, err, "an index keyed by nothing cannot hold vectors, so this must fail loudly")
	assert.Contains(t, err.Error(), "no id")
}

// The whole reason the two halves carry separate versions: rebuilding the text
// index is free, and re-embedding a history is not. A text-side bump must not
// send every caller back to their provider's billing page.
func TestRebuildingTheTextIndexKeepsThePaidForVectors(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "search.db")

	idx, err := OpenEphemeral(ctx, path)
	require.NoError(t, err)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "durable content"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	emb := &bagEmbedder{dim: 8}
	_, err = idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)

	// Simulate the next release bumping ftsSchemaVersion.
	require.NoError(t, idx.setMeta(ctx, metaFTSVersion, "999"))
	require.NoError(t, idx.Close())

	idx, err = OpenEphemeral(ctx, path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = idx.Close() })

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Zero(t, status.IndexedMessages)
	assert.Equal(t, int64(1), status.StaleConversations)

	n, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	// The vectors were never dropped, and re-indexing the same message id
	// re-adopts them: nothing is embedded a second time.
	before := emb.calls
	got, err := idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)
	assert.Zero(t, got)
	assert.Equal(t, before, emb.calls)

	status, err = idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.EmbeddedMessages)
}

func TestBumpingTheVectorSchemaDiscardsTheVectors(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "search.db")

	idx, err := OpenEphemeral(ctx, path)
	require.NoError(t, err)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "content"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8}, 10)
	require.NoError(t, err)
	require.NoError(t, idx.setMeta(ctx, metaEmbedVersion, "999"))
	require.NoError(t, idx.Close())

	idx, err = OpenEphemeral(ctx, path)
	require.NoError(t, err)
	t.Cleanup(func() { _ = idx.Close() })

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.IndexedMessages, "the text half is untouched")
	assert.Zero(t, status.EmbeddedMessages)
	assert.Equal(t, int64(1), status.PendingEmbeddings)
}

func TestLastErrorIsReadableFromStatus(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	require.NoError(t, idx.RecordError(ctx, "embedding messages: endpoint refused the connection"))

	status, err := idx.Status(ctx, src, "u1", "")
	require.NoError(t, err)
	assert.Contains(t, status.LastError, "endpoint refused")

	require.NoError(t, idx.RecordError(ctx, ""))
	status, err = idx.Status(ctx, src, "u1", "")
	require.NoError(t, err)
	assert.Empty(t, status.LastError)
}

func TestACorruptSchemaVersionIsReportedNotReadAsZero(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "search.db")
	idx, err := OpenEphemeral(ctx, path)
	require.NoError(t, err)
	require.NoError(t, idx.setMeta(ctx, metaFTSVersion, "not-a-number"))
	require.NoError(t, idx.Close())

	_, err = OpenEphemeral(ctx, path)
	require.Error(t, err, "reading a corrupt version as 0 would skip a rebuild the index needs")
}

// oldForeignFTSSchema is the text half as ANOTHER implementation of this index
// wrote it, recording the same fts_schema_version this package is on. It is
// simple-llm-ui's internal/search, which this package was extracted from: the
// owner column was user_id, there was no position and no indexed_conversations.
const oldForeignFTSSchema = `
DROP TRIGGER IF EXISTS indexed_messages_ai;
DROP TRIGGER IF EXISTS indexed_messages_ad;
DROP TRIGGER IF EXISTS indexed_messages_au;
DROP TABLE IF EXISTS messages_fts;
DROP TABLE IF EXISTS indexed_messages;
DROP TABLE IF EXISTS indexed_conversations;

CREATE TABLE indexed_messages (
	message_id      TEXT PRIMARY KEY,
	conversation_id TEXT NOT NULL,
	user_id         TEXT NOT NULL,
	role            TEXT NOT NULL,
	content         TEXT NOT NULL,
	created_at      TEXT NOT NULL
);
CREATE INDEX idx_indexed_user ON indexed_messages(user_id);
CREATE VIRTUAL TABLE messages_fts USING fts5(
	content,
	content='indexed_messages',
	content_rowid='rowid',
	tokenize='unicode61 remove_diacritics 2'
);
INSERT INTO indexed_messages VALUES ('m1','c1','u1','user','durable content','2026-01-01T00:00:00Z');
`

func TestAForeignIndexAtTheSameVersionIsRebuiltNotRefused(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "search.db")

	idx, err := OpenEphemeral(ctx, path)
	require.NoError(t, err)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "durable content"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	emb := &bagEmbedder{dim: 8}
	_, err = idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)

	// The file now looks like one an earlier, differently-shaped index left
	// behind, still claiming this package's current version.
	_, err = idx.sql.ExecContext(ctx, oldForeignFTSSchema)
	require.NoError(t, err)
	require.NoError(t, idx.Close())

	// Opening it must rebuild the text half rather than fail. The vectors are
	// in the shape this package writes, so they are kept: they cost money.
	idx, err = OpenEphemeral(ctx, path)
	require.NoError(t, err, "a rebuildable index must never make the host fail to start")
	t.Cleanup(func() { _ = idx.Close() })

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Zero(t, status.IndexedMessages, "the foreign rows are gone")
	assert.Equal(t, int64(1), status.StaleConversations, "and the conversation is read again")

	n, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	before := emb.calls
	got, err := idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)
	assert.Zero(t, got, "the vectors survived the rebuild")
	assert.Equal(t, before, emb.calls)
}

func TestAForeignVectorShapeIsRebuiltNotRefused(t *testing.T) {
	ctx := context.Background()
	path := filepath.Join(t.TempDir(), "search.db")

	idx, err := OpenEphemeral(ctx, path)
	require.NoError(t, err)
	_, err = idx.sql.ExecContext(ctx, `
		DROP TABLE embeddings;
		CREATE TABLE embeddings (message_id TEXT PRIMARY KEY, blob BLOB NOT NULL);`)
	require.NoError(t, err)
	require.NoError(t, idx.Close())

	idx, err = OpenEphemeral(ctx, path)
	require.NoError(t, err, "vectors this package cannot read are rebuilt, not fatal")
	t.Cleanup(func() { _ = idx.Close() })

	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "durable content"))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8}, 10)
	require.NoError(t, err)
}
