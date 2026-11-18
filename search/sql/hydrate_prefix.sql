SELECT message_id, conversation_id, role, content, position, created_at
FROM indexed_messages
WHERE owner = ? AND message_id IN (
