SELECT e.message_id, e.vector
FROM embeddings e
JOIN indexed_messages m ON m.message_id = e.message_id
WHERE m.owner = ? AND e.model = ?
