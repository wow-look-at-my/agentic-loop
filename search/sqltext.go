package search

import _ "embed"

// The statements live beside the package in sql/, one to a file.

//go:embed sql/fts_schema.sql
var ftsSchema string

//go:embed sql/drop_fts_schema.sql
var dropFTSSchema string

//go:embed sql/embed_schema.sql
var embedSchema string

//go:embed sql/drop_embed_schema.sql
var dropEmbedSchema string

//go:embed sql/meta_schema.sql
var metaSchema string

//go:embed sql/pending_unembedded.sql
var pendingUnembeddedSQL string

//go:embed sql/upsert_embedding.sql
var upsertEmbeddingSQL string

//go:embed sql/upsert_embed_status.sql
var upsertEmbedStatusSQL string

//go:embed sql/text_search.sql
var textSearchSQL string

//go:embed sql/vector_scan.sql
var vectorScanSQL string

//go:embed sql/substring_search.sql
var substringSearchSQL string

//go:embed sql/hydrate_prefix.sql
var hydratePrefixSQL string

//go:embed sql/upsert_indexed_message.sql
var upsertIndexedMessageSQL string

//go:embed sql/upsert_indexed_conversation.sql
var upsertIndexedConversationSQL string

//go:embed sql/count_indexed_rows.sql
var countIndexedRowsSQL string

//go:embed sql/count_embedded.sql
var countEmbeddedSQL string

//go:embed sql/count_pending_embeddings.sql
var countPendingEmbeddingsSQL string
