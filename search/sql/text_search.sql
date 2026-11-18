SELECT m.message_id, bm25(messages_fts)
FROM messages_fts
JOIN indexed_messages m ON m.rowid = messages_fts.rowid
WHERE messages_fts MATCH ? AND m.owner = ?
ORDER BY bm25(messages_fts), m.created_at DESC
LIMIT ?
