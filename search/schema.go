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

// ftsTables and embedTables are the columns each half's tables have at the
// versions above; keep each list in step with the statements in sql/. The text
// half carries its own copy of the content, since an external-content FTS5
// table reads from a table in the SAME database. It also carries owner, so a
// search is scoped on the row that answers it. indexed_conversations makes a
// pass incremental by remembering the revision it last read.
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
