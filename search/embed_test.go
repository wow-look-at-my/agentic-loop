package search

import (
	"context"
	"errors"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEmbedPendingIsResumableAndRecordsTruncation(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1",
		msg("m1", "user", "short one"),
		msg("m2", "user", strings.Repeat("word ", chunkRunes*(maxChunksPerMessage+4)/5)))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	emb := &bagEmbedder{dim: 8}
	n, err := idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)
	assert.Equal(t, 2, n)

	// Nothing is left pending, so a second pass makes no request at all.
	before := emb.calls
	n, err = idx.EmbedPending(ctx, "u1", "up/model", emb, 10)
	require.NoError(t, err)
	assert.Zero(t, n)
	assert.Equal(t, before, emb.calls, "an already-embedded corpus must not be re-billed")

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Equal(t, int64(2), status.EmbeddedMessages)
	assert.Zero(t, status.PendingEmbeddings)
	assert.Equal(t, int64(1), status.TruncatedMessages,
		"the message past the chunk cap is covered only in part, and the status has to say so")
	assert.Equal(t, int64(8), status.Dim)
}

func TestEmbedFailureLeavesTheMessagePendingRatherThanMarkedDone(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "will fail at first"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	failing := &bagEmbedder{dim: 8, fail: errors.New("embedding endpoint is down")}
	_, err = idx.EmbedPending(ctx, "u1", "up/model", failing, 10)
	require.Error(t, err)

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.PendingEmbeddings,
		"a failed embedding is retried on the next pass, never recorded as covered")
	assert.Zero(t, status.EmbeddedMessages)

	// And it succeeds once the endpoint does; there is no attempt counter to have run out.
	n, err := idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8}, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, n)
}

// A provider that returns fewer vectors than it was given inputs cannot have
// them lined back up with the messages that produced them. Storing the overlap
// would attach one message's vector to another's id, which is a wrong answer
// that looks like a right one forever after.
func TestAShortVectorBatchIsRefusedRatherThanPartiallyStored(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "one"), msg("m2", "user", "two"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	_, err = idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8, short: true}, 10)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "vectors for")

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Zero(t, status.EmbeddedMessages)
	assert.Equal(t, int64(2), status.PendingEmbeddings)
}

func TestAnEmptyMessageIsNeverPending(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", ""), msg("m2", "user", "real content"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	n, err := idx.EmbedPending(ctx, "u1", "up/model", &bagEmbedder{dim: 8}, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	status, err := idx.Status(ctx, src, "u1", "up/model")
	require.NoError(t, err)
	assert.Zero(t, status.PendingEmbeddings,
		"an empty message has nothing to embed, so it must not sit in the backlog forever")
}

func TestDropModelRemovesOnlyThatModelsVectors(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "content here"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	emb := &bagEmbedder{dim: 8}
	_, err = idx.EmbedPending(ctx, "u1", "up/old", emb, 10)
	require.NoError(t, err)
	_, err = idx.EmbedPending(ctx, "u1", "up/new", emb, 10)
	require.NoError(t, err)

	models, err := idx.ModelsInUse(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"up/new", "up/old"}, models)

	n, err := idx.DropModel(ctx, "up/old")
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	models, err = idx.ModelsInUse(ctx)
	require.NoError(t, err)
	assert.Equal(t, []string{"up/new"}, models)

	status, err := idx.Status(ctx, src, "u1", "up/new")
	require.NoError(t, err)
	assert.Equal(t, int64(1), status.EmbeddedMessages)
}

// A message's chunks never span two batches, so a batch that lands is a whole
// number of finished messages -- which is what lets embed_status be trusted as
// "this message is done".
func TestAMessagesChunksNeverSpanTwoRequests(t *testing.T) {
	ctx := context.Background()
	idx := testIndex(t)
	src := newFakeSource()
	// Each message needs several chunks; together they exceed one batch.
	long := strings.Repeat("word ", chunkRunes*6/5)
	perMessage, _ := chunkContent(long)
	count := embedBatchSize/len(perMessage) + 2
	var msgs []Message
	for i := range count {
		msgs = append(msgs, msg("m"+strconv.Itoa(i), "user", long))
	}
	src.put("c1", "u1", msgs...)
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	rec := &batchRecorder{inner: &bagEmbedder{dim: 8}}
	n, err := idx.EmbedPending(ctx, "u1", "up/model", rec, 100)
	require.NoError(t, err)
	assert.Equal(t, count, n)
	require.Greater(t, len(rec.sizes), 1, "this corpus must take more than one batch to be a test")
	for _, size := range rec.sizes {
		assert.LessOrEqual(t, size, embedBatchSize)
	}
}

// batchRecorder records the size of each request the index makes.
type batchRecorder struct {
	inner Embedder
	sizes []int
}

func (b *batchRecorder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	b.sizes = append(b.sizes, len(texts))
	return b.inner.EmbedDocuments(ctx, texts)
}

func (b *batchRecorder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	return b.inner.EmbedQuery(ctx, text)
}
