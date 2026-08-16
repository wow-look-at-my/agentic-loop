package socket

import (
	"bufio"
	"context"
	"crypto/rand"
	"encoding/base64"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"path/filepath"
	"testing"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"
	"github.com/wow-look-at-my/agentic-loop/go/session"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// fakeProvider answers with a scripted completion, emitting events first so
// the streamed path is what the tests exercise.
type fakeProvider struct {
	text string
	err  error
	reqs []commonai.Request
}

func (p *fakeProvider) Complete(_ context.Context, req commonai.Request, ev *commonai.StreamEvents) (*commonai.Completion, error) {
	p.reqs = append(p.reqs, req)
	if p.err != nil {
		return nil, p.err
	}
	if err := ev.OnText(p.text); err != nil {
		return nil, err
	}
	if err := ev.OnPart(commonai.TextPart{Text: p.text}); err != nil {
		return nil, err
	}
	if err := ev.OnUsage(commonai.Usage{PromptTokens: 4, CompletionTokens: 1, TotalTokens: 5}); err != nil {
		return nil, err
	}
	return &commonai.Completion{
		Message:    commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: p.text}),
		Usages:     []commonai.Usage{{PromptTokens: 4, CompletionTokens: 1, TotalTokens: 5}},
		Streamed:   true,
		StopReason: commonai.StopEndTurn,
	}, nil
}

func askDoc(t *testing.T, text string) []byte {
	t.Helper()
	data, err := commonai.EncodeRequestBytes(commonai.Request{
		Model:    "m",
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: text})},
	})
	require.NoError(t, err)
	return data
}

// listen starts a server on a unix socket in a temp dir and returns its path.
func listen(t *testing.T, cfg Config) string {
	t.Helper()
	s, err := NewServer(cfg)
	require.NoError(t, err)
	path := filepath.Join(t.TempDir(), "cai.sock")
	l, err := net.Listen("unix", path)
	require.NoError(t, err)
	ctx, cancel := context.WithCancel(context.Background())
	go func() { _ = s.Serve(ctx, l) }()
	t.Cleanup(func() {
		cancel()
		_ = l.Close()
	})
	return path
}

// Documents ride back-to-back in both directions: no framing layer, and the
// second call goes down the same connection as the first.
func TestUnixSocketCarriesDocumentsBackToBack(t *testing.T) {
	p := &fakeProvider{text: "hello"}
	conn, err := net.Dial("unix", listen(t, Config{Provider: p}))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))

	_, err = conn.Write(askDoc(t, "one"))
	require.NoError(t, err)
	_, err = conn.Write(askDoc(t, "two"))
	require.NoError(t, err)

	r := bufio.NewReader(conn)
	for range 2 {
		data, err := commonai.ReadDocument(r)
		require.NoError(t, err)
		require.NoError(t, commonai.Validate(data), "document:\n%s", data)
		comp, err := commonai.DecodeResponse(data)
		require.NoError(t, err)
		assert.Equal(t, "hello", comp.Message.Content)
		assert.Equal(t, commonai.StopEndTurn, comp.StopReason)
		require.Len(t, comp.Usages, 1)
	}
	require.Len(t, p.reqs, 2)
	assert.Equal(t, "one", p.reqs[0].Messages[0].Content)
	assert.Equal(t, "two", p.reqs[1].Messages[0].Content)
}

// A <conversation> names its own session: the first turn starts it, the second
// sees what was said.
func TestUnixSocketKeepsAConversation(t *testing.T) {
	p := &fakeProvider{text: "hi"}
	store := session.NewMemory()
	conn, err := net.Dial("unix", listen(t, Config{Provider: p, Store: store}))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))

	turn := func(text string) {
		buf, err := encodeConversation(commonai.Request{
			Model:    "m",
			Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: text})},
		}, "work")
		require.NoError(t, err)
		_, err = conn.Write(buf)
		require.NoError(t, err)
	}
	r := bufio.NewReader(conn)

	turn("first")
	data, err := commonai.ReadDocument(r)
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)

	turn("second")
	data, err = commonai.ReadDocument(r)
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)

	require.Len(t, p.reqs, 2)
	assert.Len(t, p.reqs[0].Messages, 1)
	require.Len(t, p.reqs[1].Messages, 3, "the question, the answer, and the next question")
	assert.Equal(t, "hi", p.reqs[1].Messages[1].Content)
	assert.Equal(t, "second", p.reqs[1].Messages[2].Content)

	stored, err := store.Get("work")
	require.NoError(t, err)
	assert.Len(t, stored.Messages, 4)
}

// A call that fails before it produced anything is answered with an <error>
// document, and the connection is still good for the next call.
func TestUnixSocketAnswersAFailureWithAnErrorDocument(t *testing.T) {
	p := &fakeProvider{err: &commonai.APIError{Status: 401, Body: "no key"}}
	conn, err := net.Dial("unix", listen(t, Config{Provider: p}))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))

	_, err = conn.Write(askDoc(t, "one"))
	require.NoError(t, err)
	data, err := commonai.ReadDocument(bufio.NewReader(conn))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	got := commonai.DecodeError(data)
	require.Error(t, got)
	var ae *commonai.APIError
	require.ErrorAs(t, got, &ae)
	assert.Equal(t, 401, ae.Status)
}

func TestUnixSocketRejectsADocumentTheSchemaDoesNotAllow(t *testing.T) {
	conn, err := net.Dial("unix", listen(t, Config{Provider: &fakeProvider{text: "hi"}}))
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))

	_, err = conn.Write([]byte(`<?xml version="1.1"?><request xmlns="https://github.com/wow-look-at-my/common-ai-api/schema/v1"><nope/></request>`))
	require.NoError(t, err)
	data, err := commonai.ReadDocument(bufio.NewReader(conn))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "even a refusal is a document:\n%s", data)
	assert.Error(t, commonai.DecodeError(data))
}

// The websocket carries the same document, split across messages -- one per
// flush -- which concatenate into exactly what a unix-socket client reads.
func TestWebSocketCarriesTheSameDocument(t *testing.T) {
	p := &fakeProvider{text: "hello"}
	s, err := NewServer(Config{Provider: p})
	require.NoError(t, err)
	srv := httptest.NewServer(s.WebSocketHandler())
	defer srv.Close()

	ws := dialWebSocket(t, srv.URL)
	defer ws.Close()
	require.NoError(t, ws.send(askDoc(t, "one")))

	var doc []byte
	messages := 0
	for {
		payload, err := ws.receive()
		require.NoError(t, err)
		doc = append(doc, payload...)
		messages++
		if r := bufio.NewReader(newByteReader(doc)); documentComplete(r) {
			break
		}
	}
	assert.Greater(t, messages, 1, "the answer arrived in pieces, not all at once")
	require.NoError(t, commonai.Validate(doc), "document:\n%s", doc)
	comp, err := commonai.DecodeResponse(doc)
	require.NoError(t, err)
	assert.Equal(t, "hello", comp.Message.Content)
}

func TestServerRequiresAProvider(t *testing.T) {
	_, err := NewServer(Config{})
	require.Error(t, err)
	assert.True(t, commonai.IsBadRequest(err))
}

// --- test helpers ---

// encodeConversation renders a <conversation> turn.
func encodeConversation(req commonai.Request, id string) ([]byte, error) {
	var b byteWriter
	if err := commonai.EncodeConversation(&b, id, req); err != nil {
		return nil, err
	}
	return b.data, nil
}

type byteWriter struct{ data []byte }

func (b *byteWriter) Write(p []byte) (int, error) {
	b.data = append(b.data, p...)
	return len(p), nil
}

func newByteReader(b []byte) io.Reader { return &byteReader{b: b} }

type byteReader struct {
	b []byte
	i int
}

func (r *byteReader) Read(p []byte) (int, error) {
	if r.i >= len(r.b) {
		return 0, io.EOF
	}
	n := copy(p, r.b[r.i:])
	r.i += n
	return n, nil
}

// documentComplete reports whether the bytes so far are a whole document.
func documentComplete(r *bufio.Reader) bool {
	_, err := commonai.ReadDocument(r)
	return err == nil
}

// wsClient is a minimal websocket client: enough to prove the server's half.
type wsClient struct {
	conn net.Conn
	r    *bufio.Reader
}

func (c *wsClient) Close() { _ = c.conn.Close() }

// dialWebSocket performs the opening handshake against an httptest server.
func dialWebSocket(t *testing.T, rawURL string) *wsClient {
	t.Helper()
	u, err := url.Parse(rawURL)
	require.NoError(t, err)
	conn, err := net.Dial("tcp", u.Host)
	require.NoError(t, err)
	require.NoError(t, conn.SetDeadline(time.Now().Add(10*time.Second)))

	key := make([]byte, 16)
	_, err = rand.Read(key)
	require.NoError(t, err)
	req := "GET / HTTP/1.1\r\nHost: " + u.Host + "\r\nUpgrade: websocket\r\nConnection: Upgrade\r\n" +
		"Sec-WebSocket-Key: " + base64.StdEncoding.EncodeToString(key) + "\r\nSec-WebSocket-Version: 13\r\n\r\n"
	_, err = conn.Write([]byte(req))
	require.NoError(t, err)

	r := bufio.NewReader(conn)
	resp, err := http.ReadResponse(r, nil)
	require.NoError(t, err)
	require.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode)
	require.NotEmpty(t, resp.Header.Get("Sec-WebSocket-Accept"))
	return &wsClient{conn: conn, r: r}
}

// send writes one masked text frame, as a client must.
func (c *wsClient) send(payload []byte) error {
	head := []byte{0x81}
	n := len(payload)
	switch {
	case n < 126:
		head = append(head, byte(0x80|n))
	default:
		head = append(head, 0x80|126, byte(n>>8), byte(n))
	}
	mask := []byte{1, 2, 3, 4}
	head = append(head, mask...)
	masked := make([]byte, n)
	for i := range payload {
		masked[i] = payload[i] ^ mask[i%4]
	}
	if _, err := c.conn.Write(append(head, masked...)); err != nil {
		return err
	}
	return nil
}

// receive reads one frame's payload, for a caller that does not care which
// opcode carried it.
func (c *wsClient) receive() ([]byte, error) {
	_, payload, err := c.receiveFrame()
	return payload, err
}
