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
