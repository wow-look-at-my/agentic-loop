package search

import (
	"context"
	"fmt"
	"math/rand/v2"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// What the semantic half costs is the question that decides whether this
// package needs a vector index at all, so it is measured here rather than
// asserted in prose. The numbers this produces, and what they mean for a real
// corpus, are written up in docs/search.md.
//
// go-toolchain does not run benchmarks. To reproduce:
//
//	go test -run '^$' -bench BenchmarkVectorScan -benchtime 10x./search/

// fillVectors writes n vectors of the given width, all under owner and
// model, and returns the index holding them.
func fillVectors(tb testing.TB, dir string, n, dim int) *Index {
	tb.Helper()
	ctx := context.Background()
	idx, err := OpenEphemeral(ctx, filepath.Join(dir, "bench.db"))
	require.NoError(tb, err)

	src := newFakeSource()
	msgs := make([]Message, n)
	for i := range n {
		msgs[i] = msg(fmt.Sprintf("m%d", i), "user", fmt.Sprintf("message number %d", i))
	}
	src.put("c1", "u1", msgs...)
	_, err = idx.Ingest(ctx, src)
	require.NoError(tb, err)

	// Written straight in rather than through EmbedPending, because the subject is the SCAN.
	rng := rand.New(rand.NewPCG(1, 2))
	tx, err := idx.sql.BeginTx(ctx, nil)
	require.NoError(tb, err)
	for i := range n {
		v := make([]float32, dim)
		for d := range v {
			v[d] = float32(rng.NormFloat64())
		}
		blob, err := encodeVector(v)
		require.NoError(tb, err)
		_, err = tx.ExecContext(ctx,
			`INSERT INTO embeddings (message_id, chunk_index, model, dim, vector, created_at)
			 VALUES (?, 0, 'bench', ?, ?, '2026-01-01T00:00:00Z')`,
			fmt.Sprintf("m%d", i), dim, blob)
		require.NoError(tb, err)
	}
	require.NoError(tb, tx.Commit())
	return idx
}

// fixedEmbedder answers every query with the same vector, so a scan benchmark measures the scan.
type fixedEmbedder struct{ dim int }

func (f fixedEmbedder) EmbedQuery(context.Context, string) ([]float32, error) {
	v := make([]float32, f.dim)
	v[0] = 1
	return v, nil
}

func (f fixedEmbedder) EmbedDocuments(_ context.Context, texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i := range out {
		v := make([]float32, f.dim)
		v[0] = 1
		out[i] = v
	}
	return out, nil
}

func BenchmarkVectorScan(b *testing.B) {
	for _, tc := range []struct{ vectors, dim int }{
		{1_000, 768},
		{10_000, 768},
		{50_000, 768},
		{10_000, 1536},
		{10_000, 3072},
	} {
		b.Run(fmt.Sprintf("%dx%d", tc.vectors, tc.dim), func(b *testing.B) {
			idx := fillVectors(b, b.TempDir(), tc.vectors, tc.dim)
			defer func() { _ = idx.Close() }()
			q := Query{Owner: "u1", Text: "anything", Limit: 20,
				Model: "bench", Embedder: fixedEmbedder{dim: tc.dim}}
			b.ResetTimer()
			for b.Loop() {
				_, err := idx.searchSemantic(context.Background(), q, "anything", 20)
				require.NoError(b, err)
			}
		})
	}
}

// The scan is linear in vectors x dimensions, so a regression that made it
// quadratic -- decoding each blob into a fresh slice, say, or re-normalizing
// the query per row -- would show up as a multiple, not a few percent. The
// bound is deliberately far above the measured cost: it is an alarm for a
// change in the SHAPE of the cost, not a performance target to tune against.
func TestVectorScanStaysLinear(t *testing.T) {
	if testing.Short() {
		t.Skip("builds a 10k-vector index")
	}
	ctx := context.Background()
	const dim = 768
	idx := fillVectors(t, t.TempDir(), 10_000, dim)
	defer func() { _ = idx.Close() }()

	q := Query{Owner: "u1", Text: "anything", Limit: 20,
		Model: "bench", Embedder: fixedEmbedder{dim: dim}}

	start := time.Now()
	hits, err := idx.searchSemantic(ctx, q, "anything", 20)
	elapsed := time.Since(start)
	require.NoError(t, err)
	require.Len(t, hits, 20)

	t.Logf("scanned 10000 vectors of %d dimensions in %s", dim, elapsed)
	require.Less(t, elapsed, 5*time.Second,
		"a scan of 10k vectors taking seconds means the cost stopped being linear in vectors x dimensions")
}
