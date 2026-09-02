package commonai

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// invalidSigBody names content block of wire message, the assistant turn every case below builds.
const invalidSigBody = `{"type":"error","error":{"type":"invalid_request_error",` +
	"\"message\":\"messages.1.content.0: Invalid `signature` in `thinking` block\"}}"

func replayableAssistantMsgs(sig string) []Message {
	return []Message{
		{Role: RoleUser, Content: "hi"},
		{Role: RoleAssistant, Content: "answer", Thinking: []ThinkingBlock{{Text: "thoughts", Signature: sig}}},
	}
}

func TestThinkingSignatureRepairStripsAndRetries(t *testing.T) {
	inner := &scriptProvider{steps: []scriptStep{
		{err: &APIError{Status: 400, Body: invalidSigBody}},
		{comp: assistantComp("recovered")},
	}}
	r := NewThinkingSignatureRepair(inner)
	req := Request{Model: "m", Messages: replayableAssistantMsgs("bad-sig")}

	comp, err := r.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	assert.Equal(t, "recovered", comp.Message.Content)
	require.Len(t, inner.reqs, 2, "retried exactly once")
	assert.Equal(t, "bad-sig", inner.reqs[0].Messages[1].Thinking[0].Signature)
	assert.Empty(t, inner.reqs[1].Messages[1].Thinking[0].Signature, "the rejected signature was stripped")
	assert.Equal(t, "bad-sig", req.Messages[1].Thinking[0].Signature, "the caller's own message was never mutated")

	// Remembered: a later call strips it up front, no failing round trip.
	inner.steps = []scriptStep{{comp: assistantComp("again")}}
	_, err = r.Complete(context.Background(), req, nil)
	require.NoError(t, err)
	require.Len(t, inner.reqs, 3)
	assert.Empty(t, inner.reqs[2].Messages[1].Thinking[0].Signature)
}

func TestThinkingSignatureRepairNoMatchNoRetry(t *testing.T) {
	apiErr := &APIError{Status: 400, Body: "totally different failure"}
	inner := &scriptProvider{steps: []scriptStep{{err: apiErr}}}
	r := NewThinkingSignatureRepair(inner)
	_, err := r.Complete(context.Background(), Request{Model: "m", Messages: replayableAssistantMsgs("bad-sig")}, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, apiErr)
	assert.Len(t, inner.reqs, 1, "an error naming no thinking block never retries")
}

func TestThinkingSignatureRepairNeverOnCancel(t *testing.T) {
	inner := &scriptProvider{steps: []scriptStep{{err: context.Canceled}}}
	r := NewThinkingSignatureRepair(inner)
	_, err := r.Complete(context.Background(), Request{Model: "m", Messages: replayableAssistantMsgs("bad-sig")}, nil)
	assert.ErrorIs(t, err, context.Canceled)
	assert.Len(t, inner.reqs, 1)
}

func TestThinkingSignatureRepairNeverAfterDelivery(t *testing.T) {
	apiErr := &APIError{Status: 400, Body: invalidSigBody}
	partial := &Completion{Message: Message{Role: RoleAssistant, Content: "half a token"}}
	inner := &scriptProvider{steps: []scriptStep{
		{comp: partial, err: apiErr, emit: func(ev *StreamEvents) { _ = ev.EmitText("half a token") }},
	}}
	r := NewThinkingSignatureRepair(inner)
	_, err := r.Complete(context.Background(), Request{Model: "m", Messages: replayableAssistantMsgs("bad-sig")}, nil)
	require.Error(t, err)
	assert.Len(t, inner.reqs, 1, "a call that already streamed is never re-sent")
}

func TestThinkingSignatureRepairSecondFailureSurfaces(t *testing.T) {
	first := &APIError{Status: 400, Body: invalidSigBody}
	second := &APIError{Status: 400, Body: "still broken"}
	inner := &scriptProvider{steps: []scriptStep{{err: first}, {err: second}}}
	r := NewThinkingSignatureRepair(inner)
	_, err := r.Complete(context.Background(), Request{Model: "m", Messages: replayableAssistantMsgs("bad-sig")}, nil)
	require.Error(t, err)
	assert.ErrorIs(t, err, second, "at most one strip-retry; the second failure surfaces")
	assert.Len(t, inner.reqs, 2)
}

// TestThinkingSignatureRepairOverAnthropicProvider proves the wrapper reads
// the SAME index Anthropic's own error names off the real wire body, not a
// hand-computed, by driving the real dialect encoder end to end.
func TestThinkingSignatureRepairOverAnthropicProvider(t *testing.T) {
	hits := 0
	var secondBody []byte
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits++
		if hits == 1 {
			http.Error(w, invalidSigBody, http.StatusBadRequest)
			return
		}
		buf := make([]byte, 8192)
		n, _ := r.Body.Read(buf)
		secondBody = append([]byte(nil), buf[:n]...)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`event: content_block_delta` + "\n" +
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"ok"}}` + "\n\n" +
			`event: message_stop` + "\n" + `data: {"type":"message_stop"}` + "\n\n"))
	}))
	defer srv.Close()

	p := NewThinkingSignatureRepair(anProvider(t, srv.URL))
	comp, err := p.Complete(context.Background(), Request{
		Model:     "m",
		MaxTokens: 16,
		Messages:  replayableAssistantMsgs("real-bad-signature"),
	}, nil)
	require.NoError(t, err)
	assert.Equal(t, "ok", comp.Message.Content)
	assert.Equal(t, 2, hits)
	assert.NotContains(t, string(secondBody), "real-bad-signature")
}
