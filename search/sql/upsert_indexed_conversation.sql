INSERT INTO indexed_conversations (conversation_id, owner, revision)
VALUES (?, ?, ?)
ON CONFLICT(conversation_id) DO UPDATE SET
  owner = excluded.owner, revision = excluded.revision
