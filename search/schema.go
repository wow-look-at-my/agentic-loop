package search

// The index is a database of DERIVED data: everything in it is rebuildable
// from the conversations in whatever store the host keeps, plus (for the
// vectors) calls to an embedding model. Deleting the file is a supported
// repair -- Ingest refills it.
//
// That is why it is a file of its own rather than tables inside the host's
// store. An FTS5 index is roughly the size of the text it indexes, so keeping
// it separate keeps the source of truth small enough to copy around, and a
// corrupt index is then a file to delete rather than a store to repair. It
// also means the host's conversations do not have to be in SQLite at all: a
// directory of XML files (session.File) indexes exactly as well.
//
// The two halves are versioned SEPARATELY because their rebuild costs are not
// comparable. Rebuilding the text index is free -- it is a re-read of what the
// host already has. Rebuilding the vectors costs money at the caller's own
// embedding endpoint. So ftsSchemaVersion can be bumped whenever the tokenizer
// or the column set changes, and the vectors survive it: embeddings are keyed
// by message id, which outlives the row that indexed_messages held.

const (
	// ftsSchemaVersion covers indexed_conversations, indexed_messages,
	// messages_fts and their triggers. A bump drops them and re-indexes.
	ftsSchemaVersion = 1
	// embedSchemaVersion covers the embeddings tables. A bump DISCARDS every
	// stored vector, and every caller re-pays for their history, so it changes
	// only when the stored bytes genuinely cannot be read the old way.
	embedSchemaVersion = 1
)

// metaKey* name the rows of the meta table.
const (
	// metaFTSVersion / metaEmbedVersion hold the applied schema versions.
	metaFTSVersion   = "fts_schema_version"
	metaEmbedVersion = "embed_schema_version"
	// metaLastError holds the last indexing failure, verbatim, so Status can
	// report WHY the index is behind instead of only that it is.
	metaLastError = "last_error"
)

// ftsSchema builds the text half.
//
// indexed_messages carries its own copy of the content because an
// external-content FTS5 table has to read from a table in the SAME database,
// and the host's conversations are not in this one -- they may not be in a
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

// dropFTSSchema tears the text half down for a version bump. The triggers go
// first: dropping indexed_messages while its delete trigger still exists would
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

// embedSchema builds the vector half. A row is one CHUNK of one message under
// one model: a long message is split (see chunkContent), because a single
// embedding of a 40 KB tool result describes nothing in particular.
//
// model is part of the key rather than a setting of the index because a caller
// can change models, and vectors from two models are not comparable. A query
// filters by the model it is asking with, so a vector from any other one is
// invisible rather than wrong.
//
// embed_status is the record of what was actually embedded, and it is what
// "this message is done" means -- not the presence of a vector row. The two
// differ for a long message: chunks is how many chunks were embedded and
// chunks_total how many the content needed, so a message capped at
// maxChunksPerMessage is visibly partial rather than quietly complete. The
// text index still covers every word of it, so the cap narrows the semantic
// half alone.
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
