package search

import (
	"context"
	"math"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- vectors -------------------------------------------------------------

func TestEncodeVectorNormalizes(t *testing.T) {
	// A vector and the same vector scaled up land on the same direction, which
	// is what makes the stored dot product a cosine similarity.
	small, err := encodeVector([]float32{3, 4})
	require.NoError(t, err)
	large, err := encodeVector([]float32{300, 400})
	require.NoError(t, err)
	assert.Equal(t, small, large)

	// Dotting against a basis vector reads one component back out: 3/5.
	got, ok := dotBlob([]float32{1, 0}, small)
	require.True(t, ok)
	assert.InDelta(t, 0.6, got, 1e-6)
}

func TestEncodeVectorRefusesWhatCannotBeSearched(t *testing.T) {
	// Each of these would store a row that counts as embedded and can never
	// match anything, which is worse than refusing it.
	for name, v := range map[string][]float32{
		"empty":     {},
		"all zeros": {0, 0, 0},
		"NaN":       {1, float32(math.NaN())},
		"infinite":  {1, float32(math.Inf(1))},
	} {
		_, err := encodeVector(v)
		assert.Error(t, err, name)
	}
}

func TestDotBlobRejectsAMismatchedDimension(t *testing.T) {
	blob, err := encodeVector([]float32{1, 0, 0})
	require.NoError(t, err)
	_, ok := dotBlob([]float32{1, 0}, blob)
	assert.False(t, ok, "a vector of another width is not comparable, not merely a low score")
}

func TestNormalizeRejectsAVectorWithNoDirection(t *testing.T) {
	_, ok := normalize([]float32{0, 0})
	assert.False(t, ok, "scoring against it would rank the corpus at 0 and present that as a result")
	unit, ok := normalize([]float32{0, 2})
	require.True(t, ok)
	assert.InDelta(t, 1.0, float64(unit[1]), 1e-6)
}

// --- chunking ------------------------------------------------------------

func TestChunkContentOverlapsAndReportsTruncation(t *testing.T) {
	short := strings.Repeat("a", 10)
	chunks, total := chunkContent(short)
	assert.Equal(t, []string{short}, chunks)
	assert.Equal(t, 1, total)

	chunks, total = chunkContent("")
	assert.Empty(t, chunks)
	assert.Zero(t, total)

	// Consecutive chunks share chunkOverlap runes, so a phrase on a boundary
	// is whole inside one of them.
	long := strings.Repeat("b", chunkRunes*2)
	chunks, total = chunkContent(long)
	require.GreaterOrEqual(t, len(chunks), 2)
	assert.Equal(t, len(chunks), total, "nothing is truncated at this length")
	assert.Len(t, []rune(chunks[0]), chunkRunes)

	// Past the cap, what was embedded and what the content needed differ --
	// and the difference is reported rather than dropped.
	huge := strings.Repeat("c", chunkRunes*(maxChunksPerMessage+10))
	chunks, total = chunkContent(huge)
	assert.Len(t, chunks, maxChunksPerMessage)
	assert.Greater(t, total, maxChunksPerMessage)
}

// --- query building ------------------------------------------------------

func TestFTSQueryQuotesEveryTermAndPrefixesOnlyMidWord(t *testing.T) {
	// A half-typed final word gets the prefix match a search box needs.
	assert.Equal(t, `"hello" "wor"*`, ftsQuery("hello wor"))
	// A query that ended in a separator has said where the word stops.
	assert.Equal(t, `"100"`, ftsQuery("100%"))
	assert.Equal(t, `"a" "b"`, ftsQuery("a_b "))
	// FTS5's operators are text, not syntax.
	assert.Equal(t, `"NEAR" "miss"*`, ftsQuery("NEAR miss"))
	assert.Equal(t, `"c"`, ftsQuery("c++"))
	// A double quote is a separator, so it ends a term rather than landing
	// inside one -- which is why no term needs escaping in the wrapper.
	assert.Equal(t, `"say" "quoted" "x"*`, ftsQuery(`say "quoted"x`))
	// Nothing to tokenize at all.
	assert.Equal(t, "", ftsQuery("???"))
	assert.Equal(t, "", ftsQuery(""))
}

func TestFuseRanksByPositionInBothLists(t *testing.T) {
	// A result both halves agree on outranks one that only leads a single list.
	text := []candidate{{messageID: "a", score: -5}, {messageID: "b", score: -1}}
	semantic := []candidate{{messageID: "c", score: 0.9}, {messageID: "b", score: 0.8}}
	got := fuse(text, semantic)
	require.Len(t, got, 3)
	assert.Equal(t, "b", got[0].MessageID, "agreed-on results win, which is the point of fusing")
	assert.True(t, got[0].Text && got[0].Semantic)
}

// --- search --------------------------------------------------------------

// seeded builds an index over a small corpus owned by u1, plus one message
// with the same words belonging to somebody else.
func seeded(t *testing.T) (*Index, *fakeSource) {
	t.Helper()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1",
		msg("m1", "user", "the deployment pipeline failed again"),
		msg("m2", "assistant", "check the database migration first"))
	src.put("c2", "u1", msg("m3", "user", "database migration rollback notes"))
	src.put("c3", "u2", msg("m4", "user", "database migration rollback notes"))
	_, err := idx.Ingest(context.Background(), src)
	require.NoError(t, err)
	return idx, src
}

func TestSearchScopesToTheOwner(t *testing.T) {
	idx, _ := seeded(t)
	hits, mode, err := idx.Search(context.Background(), Query{
		Owner: "u1", Text: "database migration", Limit: 10,
	})
	require.NoError(t, err)
	assert.Equal(t, ModeText, mode)
	require.NotEmpty(t, hits)
	for _, h := range hits {
		assert.NotEqual(t, "m4", h.MessageID, "another owner's identical message must never be returned")
	}
}

func TestSearchCarriesEnoughToJumpToTheHit(t *testing.T) {
	idx, _ := seeded(t)
	hits, _, err := idx.Search(context.Background(), Query{Owner: "u1", Text: "pipeline", Limit: 10})
	require.NoError(t, err)
	require.Len(t, hits, 1)
	assert.Equal(t, "c1", hits[0].ConversationID)
	assert.Equal(t, 0, hits[0].Position)
	assert.Equal(t, "user", hits[0].Role)
	assert.True(t, hits[0].Text)
	assert.False(t, hits[0].Semantic)
}

func TestSearchFallsBackToSubstringForAFragmentInsideAWord(t *testing.T) {
	idx, _ := seeded(t)
	// "ploy" is inside "deployment" but is not a token of it, so FTS5 cannot
	// match it and the literal scan is what answers.
	hits, mode, err := idx.Search(context.Background(), Query{Owner: "u1", Text: "ploy", Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, ModeSubstring, mode)
	require.Len(t, hits, 1)
	assert.Equal(t, "m1", hits[0].MessageID)
}

func TestSearchTreatsQueryOperatorsAsText(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1",
		msg("m1", "user", "the NEAR miss"),
		msg("m2", "user", "written in c++ today"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	// Every one of these is a parse error against an unquoted MATCH, so this
	// is the test that fails first if the quoting in ftsQuery is dropped.
	for _, q := range []string{"NEAR miss", "c++", `"quoted`, "a AND b OR NOT c", "(unbalanced", "-minus"} {
		_, _, err := idx.Search(ctx, Query{Owner: "u1", Text: q, Limit: 10})
		require.NoError(t, err, "query %q must not be a parse error", q)
	}
}

func TestSemanticSearchFindsAMessageWithoutTheQuerysWords(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1",
		msg("m1", "user", "alpha beta gamma"),
		msg("m2", "user", "totally unrelated wording"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	emb := &bagEmbedder{dim: 32}
	n, err := idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)
	assert.Equal(t, 2, n)

	hits, mode, err := idx.Search(ctx, Query{
		Owner: "u1", Text: "gamma", Limit: 10, Model: "up/model", Embedder: emb,
	})
	require.NoError(t, err)
	require.NotEmpty(t, hits)
	assert.Equal(t, "m1", hits[0].MessageID)
	assert.Contains(t, []Mode{ModeHybrid, ModeSemantic}, mode)
	assert.True(t, hits[0].Semantic)
}

func TestSearchWithNoEmbedderIsTextOnlyAndNotAnError(t *testing.T) {
	idx, _ := seeded(t)
	hits, mode, err := idx.Search(context.Background(), Query{
		Owner: "u1", Text: "pipeline", Limit: 10, Model: "up/model", Embedder: nil,
	})
	require.NoError(t, err, "a caller with no embedding endpoint searches text, it does not fail")
	assert.Equal(t, ModeText, mode)
	assert.Len(t, hits, 1)
}

func TestAnEmptyQueryOrLimitReturnsNothingRatherThanEverything(t *testing.T) {
	idx, _ := seeded(t)
	for _, q := range []Query{
		{Owner: "u1", Text: "   ", Limit: 10},
		{Owner: "u1", Text: "database", Limit: 0},
		{Owner: "u1", Text: "database", Limit: -1},
	} {
		hits, _, err := idx.Search(context.Background(), q)
		require.NoError(t, err)
		assert.Empty(t, hits)
	}
}

// Vectors of another width under the same model name mean the model changed
// dimensions. Skipping them would shrink the searchable corpus with no sign
// anywhere, so the search says so instead.
func TestAStoredVectorOfTheWrongWidthIsReportedNotSkipped(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "content"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8}, 10)
	require.NoError(t, err)

	// The same model now answers with a different width.
	_, _, err = idx.Search(ctx, Query{
		Owner: "u1", Text: "content", Limit: 10,
		Model: "up/model", Embedder: &bagEmbedder{dim: 16},
	})
	require.Error(t, err)
	assert.Contains(t, err.Error(), "different dimension")
}
