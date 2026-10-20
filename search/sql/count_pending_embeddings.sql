SELECT COUNT(*)
FROM indexed_messages m
WHERE m.owner = ?
  AND m.content <> ''
  AND NOT EXISTS (
      SELECT 1 FROM embed_status s
      WHERE s.message_id = m.message_id AND s.model = ?
  )
