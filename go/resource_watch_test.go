package agentic

import (
	"context"
	"encoding/base64"
	"errors"
	"strconv"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeSource is a resource source the test mutates between polls -- which is
// the thing under test, so it behaves like a live source (a listing and a read
// answered from current state) rather than a canned before/after pair.
type fakeSource struct {
	id, name string
	mu       sync.Mutex
	files    map[string]string
	blobs    map[string][]byte
	listErr  error
	readErr  map[string]bool
	// cap, when >0, is how many resources this source claims to have beyond
	// what it lists, so the watcher sees a truncated listing.
	claims int
}

func newFakeSource(id, name string, files map[string]string) *fakeSource {
	return &fakeSource{id: id, name: name, files: files, blobs: map[string][]byte{}, readErr: map[string]bool{}}
}

func (f *fakeSource) ID() string   { return f.id }
func (f *fakeSource) Name() string { return f.name }

func (f *fakeSource) set(uri, text string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.files[uri] = text
}

func (f *fakeSource) remove(uri string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	delete(f.files, uri)
}

func (f *fakeSource) List(_ context.Context, max int) ([]Resource, bool, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.listErr != nil {
		return nil, false, f.listErr
	}
	uris := make([]string, 0, len(f.files)+len(f.blobs))
	for uri := range f.files {
		uris = append(uris, uri)
	}
	for uri := range f.blobs {
		uris = append(uris, uri)
	}
	sortStrings(uris)
	truncated := f.claims > 0
	if len(uris) > max {
		uris, truncated = uris[:max], true
	}
	out := make([]Resource, 0, len(uris))
	for _, uri := range uris {
		out = append(out, Resource{URI: uri, Name: uri, Title: strings.ToUpper(uri)})
	}
	return out, truncated, nil
}

func (f *fakeSource) Read(_ context.Context, uri string) ([]ResourceContent, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.readErr[uri] {
		return nil, errors.New("read refused")
	}
	if b, ok := f.blobs[uri]; ok {
		return []ResourceContent{{Blob: base64.StdEncoding.EncodeToString(b), MimeType: "application/octet-stream"}}, nil
	}
	text, ok := f.files[uri]
	if !ok {
		return nil, errors.New("no such resource")
	}
	return []ResourceContent{{Text: text, MimeType: "text/plain"}}, nil
}

func sortStrings(s []string) {
	for i := 1; i < len(s); i++ {
		for j := i; j > 0 && s[j] < s[j-1]; j-- {
			s[j], s[j-1] = s[j-1], s[j]
		}
	}
}

// memSnapshots is an in-memory ResourceSnapshots, minting sequential ids.
type memSnapshots struct {
	mu      sync.Mutex
	snaps   map[string]ResourceSnapshot
	changes []StoredResourceChange
	seq     int
	failPut bool
}

func newMemSnapshots() *memSnapshots {
	return &memSnapshots{snaps: map[string]ResourceSnapshot{}}
}

func (m *memSnapshots) ListSnapshots(context.Context) ([]ResourceSnapshot, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]ResourceSnapshot, 0, len(m.snaps))
	for _, s := range m.snaps {
		out = append(out, s)
	}
	return out, nil
}

func (m *memSnapshots) PutSnapshot(_ context.Context, snap ResourceSnapshot) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.failPut {
		return errors.New("disk on fire")
	}
	m.snaps[snap.SourceID+"\x00"+snap.URI] = snap
	return nil
}

func (m *memSnapshots) DeleteSnapshot(_ context.Context, sourceID, uri string) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.snaps, sourceID+"\x00"+uri)
	return nil
}

func (m *memSnapshots) RecordChange(_ context.Context, rec ResourceChangeRecord) (string, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.seq++
	id := "chg_" + strconv.Itoa(m.seq)
	m.changes = append(m.changes, StoredResourceChange{ResourceChangeRecord: rec, ID: id, CapturedAt: "now"})
	return id, nil
}

func (m *memSnapshots) GetChange(_ context.Context, id string) (StoredResourceChange, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	for _, c := range m.changes {
		if c.ID == id {
			return c, nil
		}
	}
	return StoredResourceChange{}, ErrNoResourceChange
}

func (m *memSnapshots) RecentChanges(_ context.Context, limit int) ([]StoredResourceChange, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := m.changes
	if len(out) > limit {
		out = out[len(out)-limit:]
	}
	return out, nil
}

func watcherOver(src ResourceSource, snaps ResourceSnapshots) ResourceWatcher {
	return NewResourceWatcher(ResourceWatchConfig{Sources: []ResourceSource{src}, Snapshots: snaps})
}

// The first pass is a BASELINE: everything is new because the watch is new, not
// because anything moved, and the wording has to say so.
func TestFirstPassIsABaseline(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "one\ntwo\n"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	require.NotNil(t, w)

	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	assert.True(t, poll.Baseline)
	require.Len(t, poll.Changes, 1)
	assert.Equal(t, ResourceAdded, poll.Changes[0].Kind)
	assert.Equal(t, "FILE://A", poll.Changes[0].Label, "the title is the label when there is one")

	text := FormatResourceNotice(poll, "mcp_resource_diff")
	assert.Contains(t, text, "now being watched")
	assert.NotContains(t, text, "changed since the last turn")
}

// An unchanged resource says nothing at all: silence is the common case, and
// re-announcing it every turn would cost the model a line per turn forever.
func TestUnchangedResourcesAreSilent(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "same"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)

	_, err := w.Poll(context.Background())
	require.NoError(t, err)
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	assert.True(t, poll.Empty())
	assert.False(t, poll.Baseline, "the second pass has a stored baseline")
}

// A modification is reported with the shape of the change, and its id resolves
// to the before/after captured AT THAT MOMENT -- not to the current state.
func TestAChangeIdKeepsAnsweringItsOwnChange(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "one\ntwo\n"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	_, err := w.Poll(context.Background())
	require.NoError(t, err)

	src.set("file://a", "one\ntwo\nthree\n")
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Changes, 1)
	assert.Equal(t, ResourceModified, poll.Changes[0].Kind)
	assert.Contains(t, poll.Changes[0].Summary, "+1 -0 lines")
	firstID := poll.Changes[0].ChangeID

	// Move it again; the first id must still answer with the first diff.
	src.set("file://a", "totally different\n")
	_, err = w.Poll(context.Background())
	require.NoError(t, err)

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": firstID})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "+three")
	assert.NotContains(t, res.Content, "totally different")
}

// A resource that vanished is a change the model must be told about: it was
// told the resource existed.
func TestARemovedResourceIsReported(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x", "file://b": "y"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	_, err := w.Poll(context.Background())
	require.NoError(t, err)

	src.remove("file://b")
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Changes, 1)
	assert.Equal(t, ResourceRemoved, poll.Changes[0].Kind)
	assert.Equal(t, "file://b", poll.Changes[0].URI)
}

// A source that could not be listed makes its resources UNKNOWN, never
// removed: absent-because-unreachable is not absent, and reporting a removal
// would tell the model something false.
func TestAnUnreachableSourceIsAWarningNotARemoval(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	_, err := w.Poll(context.Background())
	require.NoError(t, err)

	src.listErr = errors.New("connection refused")
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	assert.Empty(t, poll.Changes, "nothing may be reported removed")
	require.Len(t, poll.Warnings, 1)
	assert.Contains(t, poll.Warnings[0], "could not be listed")

	// A persistent fault is reported ONCE, not re-announced every turn.
	poll, err = w.Poll(context.Background())
	require.NoError(t, err)
	assert.Empty(t, poll.Warnings)

	// ... and reported again when it recurs after clearing.
	src.listErr = nil
	_, err = w.Poll(context.Background())
	require.NoError(t, err)
	src.listErr = errors.New("connection refused")
	poll, err = w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Warnings, 1)
}

// An unreadable resource is a warning too -- its state is unknown, and the
// notice says so rather than leaving silence to read as "unchanged".
func TestAnUnreadableResourceIsAWarning(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x"})
	src.readErr["file://a"] = true
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)

	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Warnings, 1)
	assert.Contains(t, poll.Warnings[0], "could not be read")

	text := FormatResourceNotice(poll, "mcp_resource_diff")
	assert.Contains(t, text, "could not account for everything")
	assert.Contains(t, text, "unknown rather than unchanged")
}

// A listing cut at the cap is announced: an unwatched resource must never look
// like a watched one that never changes.
func TestATruncatedListingIsAnnounced(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x", "file://b": "y"})
	snaps := newMemSnapshots()
	w := NewResourceWatcher(ResourceWatchConfig{
		Sources: []ResourceSource{src}, Snapshots: snaps, MaxResources: 1,
	})
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Warnings, 1)
	assert.Contains(t, poll.Warnings[0], "advertises more than 1 resources")
	assert.Len(t, poll.Changes, 1, "only what was listed is watched")
}

// Binary content is watched by hash and size, never captured: base64 bytes are
// useless as model context and ruinous to prompt caching.
func TestBinaryResourcesAreWatchedNotCaptured(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{})
	src.blobs["file://img"] = []byte{0, 1, 2, 3, 4, 5, 6, 7}
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)

	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Changes, 1)
	assert.Contains(t, poll.Changes[0].Summary, "binary")
	assert.Contains(t, poll.Changes[0].Note, "watched by hash and size only")

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": poll.Changes[0].ChangeID})))
	require.NoError(t, err)
	assert.Contains(t, res.Content, "watched by hash and size only")
	assert.NotContains(t, res.Content, "@@", "a binary change is never rendered as a diff")
}

// Content past the cap is captured prefix-only, and every reader says so --
// a change past the cut is neither visible nor detected.
func TestTruncatedCaptureCarriesItsCaveat(t *testing.T) {
	long := strings.Repeat("x", 4096)
	src := newFakeSource("s1", "docs", map[string]string{"file://a": long})
	snaps := newMemSnapshots()
	w := NewResourceWatcher(ResourceWatchConfig{
		Sources: []ResourceSource{src}, Snapshots: snaps, MaxBytes: 128,
	})
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Changes, 1)
	assert.Contains(t, poll.Changes[0].Note, "was cut at")

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": poll.Changes[0].ChangeID})))
	require.NoError(t, err)
	assert.Contains(t, res.Content, "cut at the host's per-resource cap")
}

// Only a STORAGE failure fails a pass: a broken source is a warning, but a
// change nobody recorded must never be reported as recorded.
func TestAStorageFailureFailsThePass(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x"})
	snaps := newMemSnapshots()
	snaps.failPut = true
	w := watcherOver(src, snaps)

	_, err := w.Poll(context.Background())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "updating a watched resource")
}

// Nothing to watch, or nowhere to keep it, costs nothing at all.
func TestNoSourcesMeansNoWatcher(t *testing.T) {
	assert.Nil(t, NewResourceWatcher(ResourceWatchConfig{Snapshots: newMemSnapshots()}))
	assert.Nil(t, NewResourceWatcher(ResourceWatchConfig{Sources: []ResourceSource{newFakeSource("s", "n", nil)}}))
	assert.Nil(t, NewResourceDiffTool(nil))
}

// An unknown id is answered with the REAL ids, not a dead end.
func TestUnknownChangeIdListsTheRealOnes(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "x"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	real := poll.Changes[0].ChangeID

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(`{"change_id":"nope"}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, real, "the model is shown ids it could have used")

	res, err = tool.Execute(context.Background(), []byte(`{}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "change_id is required")

	res, err = tool.Execute(context.Background(), []byte(`{`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "could not parse arguments")
}

// full=true returns the captured content instead of a diff, and a newly added
// resource returns its contents rather than a diff against nothing.
func TestFullAndAddedReturnContent(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "hello\n"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	added := poll.Changes[0].ChangeID

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": added})))
	require.NoError(t, err)
	assert.Contains(t, res.Content, "newly available")
	assert.Contains(t, res.Content, "hello")

	src.set("file://a", "goodbye\n")
	poll, err = w.Poll(context.Background())
	require.NoError(t, err)
	res, err = tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": poll.Changes[0].ChangeID, "full": true})))
	require.NoError(t, err)
	assert.Contains(t, res.Content, "Full captured content")
	assert.Contains(t, res.Content, "goodbye")
}

// The removal diff still has the last known contents: a resource that is gone
// is exactly the one a model cannot go and read for itself.
func TestARemovalKeepsItsLastContents(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "kept text\n"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	_, err := w.Poll(context.Background())
	require.NoError(t, err)

	src.remove("file://a")
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)
	require.Len(t, poll.Changes, 1)

	tool := NewResourceDiffTool(snaps)
	res, err := tool.Execute(context.Background(), []byte(jsonMust(jsonObj{"change_id": poll.Changes[0].ChangeID})))
	require.NoError(t, err)
	assert.Contains(t, res.Content, "no longer advertised")
	assert.Contains(t, res.Content, "kept text")
}

// The notice is a POINTER: it names what moved and hands over ids, and never
// carries the content itself.
func TestTheNoticeCarriesIdsNotContent(t *testing.T) {
	src := newFakeSource("s1", "docs", map[string]string{"file://a": "secret payload\n"})
	snaps := newMemSnapshots()
	w := watcherOver(src, snaps)
	poll, err := w.Poll(context.Background())
	require.NoError(t, err)

	text := FormatResourceNotice(poll, "srv__mcp_resource_diff")
	assert.Contains(t, text, "automated notice")
	assert.Contains(t, text, poll.Changes[0].ChangeID)
	assert.Contains(t, text, "srv__mcp_resource_diff", "the model is told the name it was actually given")
	assert.NotContains(t, text, "secret payload")
}
