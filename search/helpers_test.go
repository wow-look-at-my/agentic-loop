package search

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

// fakeSource is a corpus in memory. A conversation's revision is its message
// count plus a counter the test bumps by hand, so a test can say "this changed"
// without the source having to model how.
type fakeSource struct {
	convs map[string]*fakeConv
	order []string
}

type fakeConv struct {
	owner string
	rev   string
	msgs  []Message
}

func newFakeSource() *fakeSource { return &fakeSource{convs: map[string]*fakeConv{}} }

// put replaces a conversation's transcript and moves its revision.
func (f *fakeSource) put(id, owner string, msgs ...Message) {
	c, ok := f.convs[id]
	if !ok {
		c = &fakeConv{owner: owner}
		f.convs[id] = c
		f.order = append(f.order, id)
	}
	c.owner = owner
	c.msgs = msgs
	c.rev = strings.Repeat("r", len(msgs)+1)
}

func (f *fakeSource) delete(id string) {
	delete(f.convs, id)
	var kept []string
	for _, o := range f.order {
		if o != id {
			kept = append(kept, o)
		}
	}
	f.order = kept
}

func (f *fakeSource) Conversations(context.Context) ([]Conversation, error) {
	out := make([]Conversation, 0, len(f.order))
	for _, id := range f.order {
		c := f.convs[id]
		out = append(out, Conversation{ID: id, Owner: c.owner, Revision: c.rev})
	}
	return out, nil
}

func (f *fakeSource) Messages(_ context.Context, id string) ([]Message, error) {
	c, ok := f.convs[id]
	if !ok {
		return nil, nil
	}
	return c.msgs, nil
}

// msg is a Message with an id and text, which is all most tests need.
func msg(id, role, content string) Message {
	return Message{ID: id, Role: role, Content: content, CreatedAt: "2026-01-01T00:00:00Z"}
}

// bagEmbedder is a deterministic stand-in for an embedding model: it projects
// a text's words onto a fixed number of buckets. Two texts sharing words point
// in a similar direction, which is the only property the semantic half relies
// on, and it needs no network and no key.
type bagEmbedder struct {
	dim  int
	fail error
	// calls counts requests, so a test can assert the batching rather than
	// assume it.
	calls int
	// short makes it return one fewer vector than it was given inputs.
	short bool
}

func (b *bagEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	b.calls++
	if b.fail != nil {
		return nil, b.fail
	}
	out := make([][]float32, 0, len(texts))
	for _, t := range texts {
		v := make([]float32, b.dim)
		for _, w := range strings.Fields(strings.ToLower(t)) {
			h := 0
			for _, r := range w {
				h = h*31 + int(r)
			}
			v[((h%b.dim)+b.dim)%b.dim]++
		}
		// A text with no words would be a zero vector, which encodeVector
		// rightly refuses; give it one fixed direction instead.
		if strings.TrimSpace(t) == "" {
			v[0] = 1
		}
		out = append(out, v)
	}
	if b.short && len(out) > 0 {
		out = out[:len(out)-1]
	}
	return out, nil
}

func testIndex(t *testing.T) *Index {
	t.Helper()
	idx, err := OpenEphemeral(context.Background(), filepath.Join(t.TempDir(), "search.db"))
	require.NoError(t, err)
	t.Cleanup(func() { _ = idx.Close() })
	return idx
}
