INSERT INTO indexed_messages
  (message_id, conversation_id, owner, role, content, position, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id) DO UPDATE SET
  conversation_id = excluded.conversation_id,
  owner           = excluded.owner,
  role            = excluded.role,
  content         = excluded.content,
  position        = excluded.position,
  created_at      = excluded.created_at
