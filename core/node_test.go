package commonai

import (
	"encoding/base64"
	"encoding/json"
	"reflect"
	"runtime"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestEncryptedReasoningGivesBackTheExactWireText(t *testing.T) {
	raw := []byte{0, 1, 2, 250, 251, 252, 253, 254, 255, 62, 63}
	for _, wire := range []string{
		base64.StdEncoding.EncodeToString(raw),
		base64.RawStdEncoding.EncodeToString(raw),
		base64.URLEncoding.EncodeToString(raw),
		base64.RawURLEncoding.EncodeToString(raw),
		"poopy",
		"not base64 at all!",
		"QUJD\n",
		"a",
	} {
		e := NewEncryptedReasoning(wire)
		require.NotNil(t, e, wire)
		assert.Equal(t, wire, e.String(), "%q decoded as %v", wire, e.Encoding)
		assert.Equal(t, uint(len(wire)), e.Len(), wire)
	}
	assert.Nil(t, NewEncryptedReasoning(""))
	assert.Equal(t, "", (*EncryptedReasoning)(nil).String())
	std := NewEncryptedReasoning(base64.StdEncoding.EncodeToString(raw))
	assert.Equal(t, ReasoningBase64Std, std.Encoding)
	assert.Equal(t, raw, std.Data, "base64 is held as bytes")
}

func TestEveryEnumHasAName(t *testing.T) {
	for v := range uint8(NodeExtension) + 1 {
		assert.NotContains(t, NodeKind(v).String(), "NodeKind(")
	}
	assert.Contains(t, NodeKind(200).String(), "NodeKind(200)")
	for v := range uint8(CallSuperseded) + 1 {
		assert.NotContains(t, CallStatus(v).String(), "(")
	}
	for v := range uint8(RevisionHostEdit) + 1 {
		assert.NotContains(t, RevisionSource(v).String(), "(")
	}
	for v := range uint8(SeverityChronic) + 1 {
		assert.NotContains(t, Severity(v).String(), "(")
	}
	for v := range uint8(FieldArguments) + 1 {
		assert.NotContains(t, FieldID(v).String(), "(")
	}
	for v := range uint8(MisstepEmptyCompletion) + 1 {
		assert.NotContains(t, MisstepKind(v).String(), "(")
	}
	for v := range uint8(ReasoningVerbatim) + 1 {
		assert.NotContains(t, ReasoningEncoding(v).String(), "(")
	}
}

func TestStreamStringNeverLosesAValue(t *testing.T) {
	s := NewStreamString()
	var seen []string
	cb := func(d StreamDelta) error { seen = append(seen, d.Delta); return nil }
	s.OnDelta().Subscribe(&cb)
	w := WriterOf(s)
	w.Append("hel")
	w.Append("lo")
	assert.Equal(t, "hello", s.String())
	assert.False(t, s.Done())
	assert.Equal(t, []string{"hel", "lo"}, seen)

	w.Restart()
	assert.Equal(t, "", s.String())
	w.Append("hi")
	w.Finish("hi there")
	assert.True(t, s.Done())
	assert.Equal(t, "hi there", s.String())
	var values []string
	var sources []RevisionSource
	for _, r := range s.Revisions() {
		values = append(values, r.Value)
		sources = append(sources, r.Source)
	}
	assert.Equal(t, []string{"hello", "hi", "hi there"}, values)
	assert.Equal(t, []RevisionSource{RevisionRetry, RevisionStreamed, RevisionFinal}, sources)
	runtime.KeepAlive(&cb)

	same := NewStreamString()
	WriterOf(same).Append("x")
	FinishStreamed(same)
	assert.Len(t, same.Revisions(), 1, "a finish that matches what streamed adds no revision")

	edited := FinishedString("a")
	WriterOf(edited).Set("b", RevisionHostEdit)
	assert.Equal(t, "b", edited.String())
	assert.Len(t, edited.Revisions(), 2)
}

func TestStreamStringIsSafeAcrossGoroutines(t *testing.T) {
	s := NewStreamString()
	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for range 100 {
				WriterOf(s).Append("x")
				_ = s.String()
			}
		}()
	}
	wg.Wait()
	assert.Equal(t, uint(800), s.Len())
}

type metaSample struct{ N int }

var (
	metaKeyOnce   sync.Once
	metaSampleKey MetaKey[metaSample]
	metaPtrKey    MetaKey[*metaSample]
)

func sampleKeys() {
	metaKeyOnce.Do(func() {
		metaSampleKey = RegisterMetaKey[metaSample]("7f2c9d6e-0000-4000-8000-000000000001", "sample", "core tests")
		metaPtrKey = RegisterMetaKey[*metaSample]("7f2c9d6e-0000-4000-8000-000000000002", "sample ptr", "core tests")
	})
}

func TestMetaIsTypedAndKeepsWhatItDoesNotKnow(t *testing.T) {
	sampleKeys()
	var m Meta
	_, ok := GetMeta(&m, metaSampleKey)
	assert.False(t, ok)
	SetMeta(&m, metaSampleKey, metaSample{N: 3})
	got, ok := GetMeta(&m, metaSampleKey)
	require.True(t, ok)
	assert.Equal(t, 3, got.N)

	SetMeta(&m, metaPtrKey, &metaSample{N: 4})
	SetMeta(&m, metaPtrKey, nil)
	_, ok = GetMeta(&m, metaPtrKey)
	assert.False(t, ok, "nil deletes")

	entries, err := m.MarshalEntries()
	require.NoError(t, err)
	entries["unknown-key"] = json.RawMessage(`{"kept":true}`)
	var back Meta
	back.LoadEntries(entries)
	got, ok = GetMeta(&back, metaSampleKey)
	require.True(t, ok)
	assert.Equal(t, 3, got.N, "a stored value decodes into its type")
	again, err := back.MarshalEntries()
	require.NoError(t, err)
	assert.JSONEq(t, `{"kept":true}`, string(again["unknown-key"]), "an unregistered entry is written back unchanged")
	ids := []string{}
	for id := range back.All() {
		ids = append(ids, id)
	}
	assert.Len(t, ids, 2)

	unregistered := MetaKey[int]{id: "nope"}
	assert.Panics(t, func() { SetMeta(&m, unregistered, 1) })
	assert.Panics(t, func() { _, _ = GetMeta(&m, unregistered) })
	assert.Panics(t, func() { SetMeta(&m, metaSampleKey, metaSample{}); SetMeta(&m, MetaKey[func()]{id: metaSampleKey.ID()}, func() {}) })
	assert.Panics(t, func() { RegisterMetaKey[int](metaSampleKey.ID(), "dup", "tests") })
	assert.NotEmpty(t, RegisteredMetaKeys())
}

type bookmark struct {
	MessageBase
	Label string
}

func (*bookmark) NodeKind() NodeKind { return NodeExtension }
func (*bookmark) Final() bool        { return true }

type unregisteredNode struct{ MessageBase }

func (*unregisteredNode) NodeKind() NodeKind { return NodeExtension }
func (*unregisteredNode) Final() bool        { return true }

var bookmarkOnce sync.Once

func TestTreeKeepsEveryBranch(t *testing.T) {
	tr := NewTree()
	u := NewUserText("hi")
	require.NoError(t, tr.Add(u, nil))
	a1 := NewAssistantText("one")
	a2 := NewAssistantText("two")
	require.NoError(t, tr.Add(a1, u))
	require.NoError(t, tr.Add(a2, u), "a regenerate is a sibling")
	assert.Len(t, u.Children(), 2)
	assert.Equal(t, []IMessage{u, a2}, tr.PathTo(a2))
	assert.Equal(t, a1, tr.Node(a1.ID()))
	assert.Equal(t, uint(3), tr.Len())
	assert.Error(t, tr.Add(a1, u), "a node has one parent")
	assert.ErrorIs(t, tr.Add(NewUserText("x"), NewUserText("stranger")), ErrNotInTree)

	var order []IMessage
	for n := range tr.Nodes() {
		order = append(order, n)
	}
	assert.Equal(t, []IMessage{u, a1, a2}, order)

	bookmarkOnce.Do(func() { RegisterMessageType[*bookmark]("4b1f0c2a-0000-4000-8000-0000000000aa", "bookmark") })
	require.NoError(t, tr.Add(&bookmark{Label: "here"}, a2))
	e, ok := ExtensionOf(&bookmark{})
	require.True(t, ok)
	assert.Equal(t, reflect.TypeFor[*bookmark](), e.Type)
	assert.Panics(t, func() { _ = tr.Add(&unregisteredNode{}, a2) })
}

func TestTextOf(t *testing.T) {
	assert.Equal(t, "", TextOf(nil))
	assert.Equal(t, "a", TextOf([]IContent{Text("a")}))
	assert.Equal(t, "ab", TextOf([]IContent{Text("a"), &ImageContent{Src: "x"}, Text("b")}))
}

func TestBlobStoreHoldsRawBytesOnDisk(t *testing.T) {
	s, err := NewBlobStore(t.TempDir())
	require.NoError(t, err)
	payload := strings.Repeat("blob", 1000)
	ref, err := s.PutBase64("image/png", base64.StdEncoding.EncodeToString([]byte(payload)))
	require.NoError(t, err)
	assert.Equal(t, uint64(len(payload)), ref.Size)
	data, release, err := ref.Bytes()
	require.NoError(t, err)
	assert.Equal(t, payload, string(data))
	release()
	release()
	b64, err := ref.Base64()
	require.NoError(t, err)
	assert.Equal(t, base64.StdEncoding.EncodeToString([]byte(payload)), b64)
	again, err := s.Ref(ref.Hash, "image/png")
	require.NoError(t, err)
	assert.Equal(t, ref.Size, again.Size)
	_, err = s.Ref("missing", "image/png")
	assert.Error(t, err)
	_, _, err = (&BlobRef{}).Bytes()
	assert.ErrorIs(t, err, errNoStore)
}
