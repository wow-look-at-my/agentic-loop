package search

import (
	"context"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/session"
)

func TestSessionSourceIndexesAStoredConversation(t *testing.T) {
	ctx := context.Background()
	store := session.NewMemory()
	id, err := store.Create(commonai.Request{
		Model: "m",
		Messages: []commonai.Message{
			{Role: commonai.RoleUser, Content: "how do I rotate the signing key"},
			{Role: commonai.RoleAssistant, Content: "run the rotate command"},
		},
	})
	require.NoError(t, err)

	src := &SessionSource{Store: store}
	idx := testIndex(t)
	n, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	hits, mode, err := idx.Search(ctx, Query{Text: "signing key", Limit: 10})
	require.NoError(t, err)
	assert.Equal(t, ModeText, mode)
	require.Len(t, hits, 1)
	assert.Equal(t, id, hits[0].ConversationID)
	assert.Equal(t, 0, hits[0].Position)
	assert.Equal(t, string(commonai.RoleUser), hits[0].Role)
}

func TestSessionSourceReIndexesOnlyWhatChanged(t *testing.T) {
	ctx := context.Background()
	store := session.NewMemory()
	id, err := store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "first"},
	}})
	require.NoError(t, err)
	_, err = store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "untouched"},
	}})
	require.NoError(t, err)

	src := &SessionSource{Store: store}
	idx := testIndex(t)
	n, err := idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 2, n)

	n, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Zero(t, n, "an unchanged store is not re-read")

	_, err = store.Append(id, commonai.Message{Role: commonai.RoleAssistant, Content: "a reply"})
	require.NoError(t, err)
	n, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	hits, _, err := idx.Search(ctx, Query{Text: "reply", Limit: 10})
	require.NoError(t, err)
	require.Len(t, hits, 1)
	assert.Equal(t, 1, hits[0].Position)
}

func TestSessionSourceDropsADeletedConversation(t *testing.T) {
	ctx := context.Background()
	store := session.NewMemory()
	id, err := store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "doomed content"},
	}})
	require.NoError(t, err)

	src := &SessionSource{Store: store}
	idx := testIndex(t)
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)

	require.NoError(t, store.Delete(id))
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)

	hits, _, err := idx.Search(ctx, Query{Text: "doomed", Limit: 10})
	require.NoError(t, err)
	assert.Empty(t, hits)
}

// Thinking is the model's working notes and carries provider tokens; a tool
// call's arguments are a machine protocol. Neither is anything the user saw,
// so a search must not answer with them. Text -- including a tool RESULT -- is
// exactly what someone is looking for.
func TestSearchableTextIsWhatTheUserActuallySaw(t *testing.T) {
	m := commonai.Message{
		Role: commonai.RoleAssistant,
		Parts: []commonai.Part{
			commonai.ThinkingPart{Text: "secretly deliberating"},
			commonai.TextPart{Text: "the visible answer"},
			commonai.ToolCallPart{ID: "1", Name: "grep", Arguments: `{"q":"hidden"}`},
			commonai.TextPart{Text: "and a second paragraph"},
		},
	}
	got := searchableText(m)
	assert.Equal(t, "the visible answer\nand a second paragraph", got)
	assert.NotContains(t, got, "deliberating")
	assert.NotContains(t, got, "hidden")
}

// The revision must move when the indexed text moves, or the index goes
// quietly stale -- and must NOT move for a change the index cannot see, or
// every pass re-reads the corpus to write identical rows.
func TestTranscriptRevisionTracksTheIndexedTextOnly(t *testing.T) {
	base := commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "hello"},
	}}
	rev := transcriptRevision(base)

	same := commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "hello", Thinking: []commonai.ThinkingBlock{{Text: "notes"}}},
	}}
	assert.Equal(t, rev, transcriptRevision(same),
		"a thinking block is not indexed, so it must not trigger a re-read")

	changed := commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "hello there"},
	}}
	assert.NotEqual(t, rev, transcriptRevision(changed))

	appended := commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "hello"},
		{Role: commonai.RoleAssistant, Content: "hi"},
	}}
	assert.NotEqual(t, rev, transcriptRevision(appended))
}

// A host that already identifies its messages keeps those ids, so the vectors
// follow the message rather than its position.
func TestAHostAssignedMessageIDIsUsedAsIs(t *testing.T) {
	ctx := context.Background()
	store := session.NewMemory()
	id, err := store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, ID: "durable-id", Content: "content"},
	}})
	require.NoError(t, err)

	msgs, err := (&SessionSource{Store: store}).Messages(ctx, id)
	require.NoError(t, err)
	require.Len(t, msgs, 1)
	assert.Equal(t, "durable-id", msgs[0].ID)
}

// session.File answers revisions from a stat rather than by reading and
// validating every document, which is the difference between a stat per
// conversation and the whole store on every indexing pass.
func TestFileStoreRevisionsAvoidReadingEveryDocument(t *testing.T) {
	ctx := context.Background()
	store, err := session.NewFile(t.TempDir())
	require.NoError(t, err)
	id, err := store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "stored on disk"},
	}})
	require.NoError(t, err)

	var _ Revisions = store // the optimization is wired, not just present

	src := &SessionSource{Store: store}
	convs, err := src.Conversations(ctx)
	require.NoError(t, err)
	require.Len(t, convs, 1)
	assert.Equal(t, id, convs[0].ID)
	assert.NotEmpty(t, convs[0].Revision)

	idx := testIndex(t)
	_, err = idx.Ingest(ctx, src)
	require.NoError(t, err)
	hits, _, err := idx.Search(ctx, Query{Text: "stored", Limit: 10})
	require.NoError(t, err)
	require.Len(t, hits, 1)

	// The marker moves when the document does.
	before := convs[0].Revision
	_, err = store.Append(id, commonai.Message{Role: commonai.RoleAssistant, Content: "appended"})
	require.NoError(t, err)
	convs, err = src.Conversations(ctx)
	require.NoError(t, err)
	require.Len(t, convs, 1)
	assert.NotEqual(t, before, convs[0].Revision)
}

func TestSessionSourceScopesByOwner(t *testing.T) {
	ctx := context.Background()
	store := session.NewMemory()
	_, err := store.Create(commonai.Request{Messages: []commonai.Message{
		{Role: commonai.RoleUser, Content: "tenant content"},
	}})
	require.NoError(t, err)

	idx := testIndex(t)
	_, err = idx.Ingest(ctx, &SessionSource{Store: store, Owner: "tenant-a"})
	require.NoError(t, err)

	hits, _, err := idx.Search(ctx, Query{Owner: "tenant-a", Text: "tenant", Limit: 10})
	require.NoError(t, err)
	assert.Len(t, hits, 1)

	// The match is exact and there is no wildcard, so asking as anyone else --
	// including the empty owner -- reaches nothing.
	hits, _, err = idx.Search(ctx, Query{Owner: "", Text: "tenant", Limit: 10})
	require.NoError(t, err)
	assert.Empty(t, hits)
}
