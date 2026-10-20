SELECT message_id, conversation_id, role, content, position, created_at
FROM indexed_messages
WHERE owner = ? AND content LIKE '%' || ? || '%' ESCAPE '\'
ORDER BY created_at DESC
LIMIT ?
