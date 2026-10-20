INSERT INTO embeddings (message_id, chunk_index, model, dim, vector, created_at)
VALUES (?, ?, ?, ?, ?, ?)
ON CONFLICT(message_id, chunk_index, model) DO UPDATE SET
  dim = excluded.dim, vector = excluded.vector, created_at = excluded.created_at
