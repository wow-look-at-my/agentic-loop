SELECT
	(SELECT COUNT(*) FROM indexed_conversations WHERE owner = ?),
	(SELECT COUNT(*) FROM indexed_messages WHERE owner = ?)
