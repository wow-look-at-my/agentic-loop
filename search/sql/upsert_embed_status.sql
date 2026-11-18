INSERT INTO embed_status (message_id, model, chunks, chunks_total, created_at)
VALUES (?, ?, ?, ?, ?)
ON CONFLICT(message_id, model) DO UPDATE SET
  chunks = excluded.chunks, chunks_total = excluded.chunks_total,
  created_at = excluded.created_at
