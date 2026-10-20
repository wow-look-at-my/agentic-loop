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
