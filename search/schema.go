package search

// The index is derived data, rebuildable from the conversations; its halves are versioned separately.

const (
	// ftsSchemaVersion covers the text tables; a bump drops them and re-indexes.
	ftsSchemaVersion = 1
	// embedSchemaVersion covers the embeddings tables; a bump DISCARDS every stored vector.
	embedSchemaVersion = 1
)

// metaKey* name the rows of the meta table.
const (
	// metaFTSVersion / metaEmbedVersion hold the applied schema versions.
	metaFTSVersion   = "fts_schema_version"
	metaEmbedVersion = "embed_schema_version"
	// metaLastError holds the last indexing failure verbatim, so Status can report why.
	metaLastError = "last_error"
)

// ftsSchema builds the text half.
//
// indexed_messages carries its own copy of the content because an
// external-content FTS5 table has to read from a table in the SAME database,
// and the host's conversations are not in this -- they may not be in a
// database at all.
//
// It also carries owner, so scoping a search to whoever owns the conversation
// is a predicate on the row that answers the query. That is the only place the
// check cannot be forgotten.
//
// indexed_conversations is what makes re-indexing incremental: it remembers
// the revision each conversation had when it was last read, so a pass re-reads
// only the conversations whose transcripts have moved.
const ftsSchema = `
CREATE TABLE IF NOT EXISTS indexed_conversations (
	conversation_id TEXT PRIMARY KEY,
	owner           TEXT NOT NULL,
	revision        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexed_messages (
	message_id      TEXT PRIMARY KEY,
	conversation_id TEXT NOT NULL,
	owner           TEXT NOT NULL,
	role            TEXT NOT NULL,
	content         TEXT NOT NULL,
	position        INTEGER NOT NULL,
	created_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_indexed_owner ON indexed_messages(owner);
CREATE INDEX IF NOT EXISTS idx_indexed_conversation ON indexed_messages(conversation_id);

CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
	content,
	content='indexed_messages',
	content_rowid='rowid',
	tokenize='unicode61 remove_diacritics 2'
);

CREATE TRIGGER IF NOT EXISTS indexed_messages_ai AFTER INSERT ON indexed_messages BEGIN
	INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
CREATE TRIGGER IF NOT EXISTS indexed_messages_ad AFTER DELETE ON indexed_messages BEGIN
	INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;
CREATE TRIGGER IF NOT EXISTS indexed_messages_au AFTER UPDATE ON indexed_messages BEGIN
	INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
	INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;
`

// ftsTables and embedTables are the columns each half's tables have at the
// versions above. applySchema reads the file's real shape against them and
// rebuilds a half that does not match, so a recorded version that describes
// some OTHER implementation's tables costs a re-index instead of failing every
// open. Keep each list in step with the CREATE statements beside it.
var (
	ftsTables = map[string][]string{
		"indexed_conversations": {"conversation_id", "owner", "revision"},
		"indexed_messages":      {"message_id", "conversation_id", "owner", "role", "content", "position", "created_at"},
	}
	embedTables = map[string][]string{
		"embeddings":   {"message_id", "chunk_index", "model", "dim", "vector", "created_at"},
		"embed_status": {"message_id", "model", "chunks", "chunks_total", "created_at"},
	}
)

// dropFTSSchema tears the text half down for a version bump. The triggers go
//: dropping indexed_messages while its delete trigger still exists would
// fire that trigger for every row into an FTS table that is about to be
// dropped anyway.
const dropFTSSchema = `
DROP TRIGGER IF EXISTS indexed_messages_ai;
DROP TRIGGER IF EXISTS indexed_messages_ad;
DROP TRIGGER IF EXISTS indexed_messages_au;
DROP TABLE IF EXISTS messages_fts;
DROP TABLE IF EXISTS indexed_messages;
DROP TABLE IF EXISTS indexed_conversations;
`

// embedSchema builds the vector half; model is part of the key since vectors from models aren't comparable.
const embedSchema = `
CREATE TABLE IF NOT EXISTS embeddings (
	message_id  TEXT NOT NULL,
	chunk_index INTEGER NOT NULL,
	model       TEXT NOT NULL,
	dim         INTEGER NOT NULL,
	vector      BLOB NOT NULL,
	created_at  TEXT NOT NULL,
	PRIMARY KEY (message_id, chunk_index, model)
);
CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model);

CREATE TABLE IF NOT EXISTS embed_status (
	message_id   TEXT NOT NULL,
	model        TEXT NOT NULL,
	chunks       INTEGER NOT NULL,
	chunks_total INTEGER NOT NULL,
	created_at   TEXT NOT NULL,
	PRIMARY KEY (message_id, model)
);
`

// dropEmbedSchema discards every stored vector.
const dropEmbedSchema = `
DROP TABLE IF EXISTS embed_status;
DROP TABLE IF EXISTS embeddings;
`

// metaSchema is never versioned: it is what RECORDS the versions.
const metaSchema = `
CREATE TABLE IF NOT EXISTS meta (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
);
`
