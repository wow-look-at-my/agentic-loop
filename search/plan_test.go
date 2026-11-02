package search

import (
	"context"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The pending-embedding queries run on every indexer pass. Without an index on the non-empty rows, each pass reads
// one table page per indexed message. This happens for every owner with an embedding model, even when nothing is
// pending.
func TestThePendingQueriesReadTheIndexNotTheTable(t *testing.T) {
	idx := testIndex(t)
	for name, q := range map[string]string{"pending": pendingUnembeddedSQL, "count": countPendingEmbeddingsSQL} {
		rows, err := idx.sql.QueryContext(context.Background(), "EXPLAIN QUERY PLAN "+q, "owner", "model", 10)
		if name == "count" {
			rows, err = idx.sql.QueryContext(context.Background(), "EXPLAIN QUERY PLAN "+q, "owner", "model")
		}
		require.NoError(t, err, name)
		var plan []string
		for rows.Next() {
			var id, parent, notused int
			var detail string
			require.NoError(t, rows.Scan(&id, &parent, &notused, &detail))
			plan = append(plan, detail)
		}
		require.NoError(t, rows.Close())
		joined := strings.Join(plan, "\n")
		assert.Contains(t, joined, "idx_indexed_pending", "%s query plan:\n%s", name, joined)
	}
}
