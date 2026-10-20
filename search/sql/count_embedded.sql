SELECT
	COUNT(*),
	COALESCE(SUM(CASE WHEN st.chunks_total > st.chunks THEN 1 ELSE 0 END), 0)
FROM embed_status st
JOIN indexed_messages m ON m.message_id = st.message_id
WHERE m.owner = ? AND st.model = ?
