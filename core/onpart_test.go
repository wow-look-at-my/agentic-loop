package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// collectParts records the parts a call announced as finished.
func collectParts(got *[]Part) *StreamEvents {
	return &StreamEvents{OnPart: func(p Part) error {
		*got = append(*got, p)
		return nil
	}}
}

// kinds is the shape of a parts list, which is what the ordering claims are
// actually about.
func kinds(parts []Part) []PartKind {
	out := make([]PartKind, 0, len(parts))
	for _, p := range parts {
		out = append(out, p.Kind())
	}
	return out
}

// sseServer replays SSE payloads as event stream.
func sseServer(t *testing.T, payloads ...string) *httptest.Server {
	t.Helper()
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		for _, p := range payloads {
			_, _ = w.Write([]byte("data: " + p + "\n\n"))
		}
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)
	return srv
}

// A part is announced, in the order it occupies in the finished message.
// A host writing the answer out as it arrives depends on both.
func TestOnPartDeliversEveryPartOnceInOrder(t *testing.T) {
	srv := sseServer(t,
		`{"choices":[{"delta":{"reasoning":"weigh"}}]}`,
		`{"choices":[{"delta":{"reasoning":"ing"}}]}`,
		`{"choices":[{"delta":{"content":"ans"}}]}`,
		`{"choices":[{"delta":{"content":"wer"}}]}`,
		`{"choices":[{"delta":{"tool_calls":[{"index":0,"id":"c1","function":{"name":"grep","arguments":"{}"}}]},"finish_reason":"tool_calls"}]}`,
	)
	var got []Part
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, collectParts(&got))
	require.NoError(t, err)

	assert.Equal(t, kinds(comp.Message.EffectiveParts()), kinds(got),
		"what was announced is what the call returned")
	assert.Equal(t, []PartKind{PartKindThinking, PartKindText, PartKindToolCall}, kinds(got))
	assert.Equal(t, "weighing", got[0].(ThinkingPart).Text)
	assert.Equal(t, "answer", got[1].(TextPart).Text)
}

// The signature is the reason this event exists: it arrives after the text it
// belongs to, so only the finished block carries it.
func TestOnPartCarriesTheThinkingSignature(t *testing.T) {
	srv := sseServer(t,
		`{"type":"message_start","message":{"usage":{"input_tokens":1}}}`,
		`{"type":"content_block_start","index":0,"content_block":{"type":"thinking"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"weighing it"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"here"}}`,
		`{"type":"content_block_stop","index":1}`,
		`{"type":"message_delta","delta":{"stop_reason":"end_turn"}}`,
		`{"type":"message_stop"}`,
	)
	var got []Part
	comp, err := anProvider(t, srv.URL).
		Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, collectParts(&got))
	require.NoError(t, err)

	require.Len(t, got, 2)
	assert.Equal(t, kinds(comp.Message.EffectiveParts()), kinds(got))
	tp, ok := got[0].(ThinkingPart)
	require.True(t, ok)
	assert.Equal(t, "sig-1", tp.Signature, "the block was announced only once it had one")
	assert.Equal(t, "weighing it", tp.Text)
}

// A block still filling when the connection drops was never announced, but it
// is still in the completion: a host writes what it was told plus whatever the
// completion holds beyond that.
func TestOnPartSkipsABlockThatNeverFinished(t *testing.T) {
	srv := sseServer(t,
		`{"type":"content_block_start","index":0,"content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"first"}}`,
		`{"type":"content_block_stop","index":0}`,
		`{"type":"content_block_start","index":1,"content_block":{"type":"text"}}`,
		`{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"cut off"}}`,
	)
	var got []Part
	comp, err := anProvider(t, srv.URL).
		Complete(context.Background(), Request{Model: "m", MaxTokens: 64}, collectParts(&got))
	require.NoError(t, err)

	require.Len(t, got, 1, "only the block that stopped was announced")
	parts := comp.Message.EffectiveParts()
	require.Len(t, parts, 2, "the cut block's text is still output the caller saw")
	assert.Equal(t, "cut off", parts[1].(TextPart).Text)
}

// An error from the callback aborts the call and stays the caller's own.
func TestOnPartErrorAbortsTheCall(t *testing.T) {
	srv := sseServer(t, `{"choices":[{"delta":{"content":"hi"},"finish_reason":"stop"}]}`)
	sentinel := assert.AnError
	ev := &StreamEvents{OnPart: func(Part) error { return sentinel }}
	comp, err := oaProvider(t, srv.URL).Complete(context.Background(), Request{Model: "m"}, ev)
	require.ErrorIs(t, err, sentinel)
	assert.False(t, IsTransient(err), "a failed sink is never re-sent")
	require.NotNil(t, comp, "what already arrived comes back with the failure")
}
