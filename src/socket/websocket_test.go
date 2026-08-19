package socket

import (
	"bufio"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	commonai "github.com/wow-look-at-my/agentic-loop/go/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The websocket half is hand-rolled against RFC 6455, so the frame handling is
// this project's code and gets tested like it: a peer that pings, one that
// closes, one that never asked to upgrade, and an answer too big for a short
// length field.

// wsServer starts a websocket server over the given provider.
func wsServer(t *testing.T, p commonai.Provider) *httptest.Server {
	t.Helper()
	s, err := NewServer(Config{Provider: p})
	require.NoError(t, err)
	srv := httptest.NewServer(s.WebSocketHandler())
	t.Cleanup(srv.Close)
	return srv
}

// sendFrame writes one masked frame with an arbitrary opcode, which the
// client's send() cannot do -- it only ever sends text.
func (c *wsClient) sendFrame(op byte, payload []byte) error {
	head := []byte{0x80 | op, byte(0x80 | len(payload))}
	mask := []byte{9, 8, 7, 6}
	head = append(head, mask...)
	masked := make([]byte, len(payload))
	for i := range payload {
		masked[i] = payload[i] ^ mask[i%4]
	}
	_, err := c.conn.Write(append(head, masked...))
	return err
}

// receiveFrame reads one frame and reports its opcode as well as its payload.
func (c *wsClient) receiveFrame() (byte, []byte, error) {
	var head [2]byte
	if _, err := io.ReadFull(c.r, head[:]); err != nil {
		return 0, nil, err
	}
	op := head[0] & 0x0F
	length := uint64(head[1] & 0x7F)
	switch length {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(c.r, ext[:]); err != nil {
			return 0, nil, err
		}
		length = uint64(ext[0])<<8 | uint64(ext[1])
	case 127:
		var ext [8]byte
		if _, err := io.ReadFull(c.r, ext[:]); err != nil {
			return 0, nil, err
		}
		for _, b := range ext {
			length = length<<8 | uint64(b)
		}
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(c.r, payload); err != nil {
		return 0, nil, err
	}
	return op, payload, nil
}

// A ping is answered with a pong carrying the same payload, and the connection
// carries on -- a keepalive that dropped the conversation would be worse than
// no keepalive.
func TestWebSocketAnswersAPingAndKeepsGoing(t *testing.T) {
	ws := dialWebSocket(t, wsServer(t, &fakeProvider{text: "hello"}).URL)
	defer ws.Close()

	require.NoError(t, ws.sendFrame(opPing, []byte("still there?")))
	op, payload, err := ws.receiveFrame()
	require.NoError(t, err)
	assert.Equal(t, byte(opPong), op)
	assert.Equal(t, "still there?", string(payload))

	require.NoError(t, ws.send(askDoc(t, "one")))
	assert.Contains(t, string(readDocumentFrames(t, ws)), "hello")
}

// A close frame ends the conversation, and the server closes back rather than
// leaving the peer waiting on a handshake that never finishes.
func TestWebSocketClosesBack(t *testing.T) {
	ws := dialWebSocket(t, wsServer(t, &fakeProvider{text: "hi"}).URL)
	defer ws.Close()

	require.NoError(t, ws.sendFrame(opClose, nil))
	op, _, err := ws.receiveFrame()
	require.NoError(t, err)
	assert.Equal(t, byte(opClose), op)
}

// An answer past 64 KiB needs the 8-byte length field. The frame is one
// message either way, so a client that reassembles by concatenation cannot
// tell -- which is the property worth keeping.
func TestWebSocketCarriesAnAnswerTooBigForAShortLength(t *testing.T) {
	big := strings.Repeat("x", 70<<10)
	ws := dialWebSocket(t, wsServer(t, &fakeProvider{text: big}).URL)
	defer ws.Close()

	require.NoError(t, ws.send(askDoc(t, "one")))
	doc := readDocumentFrames(t, ws)
	require.NoError(t, commonai.Validate(doc))
	comp, err := commonai.DecodeResponse(doc)
	require.NoError(t, err)
	assert.Equal(t, big, comp.Message.Content)
}

// The handshake is refused, with a status rather than a hijacked connection,
// when the request is not a websocket upgrade or does not carry the key the
// accept token is derived from.
func TestWebSocketRefusesAHandshakeItCannotComplete(t *testing.T) {
	s, err := NewServer(Config{Provider: &fakeProvider{text: "hi"}})
	require.NoError(t, err)
	h := s.WebSocketHandler()

	for _, tc := range []struct {
		name   string
		header map[string]string
		want   string
	}{
		{"not an upgrade", nil, "not a websocket upgrade"},
		{"no key", map[string]string{"Upgrade": "websocket"}, "no Sec-WebSocket-Key"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodGet, "/", nil)
			for k, v := range tc.header {
				req.Header.Set(k, v)
			}
			w := httptest.NewRecorder()
			h.ServeHTTP(w, req)
			assert.Equal(t, http.StatusBadRequest, w.Code)
			assert.Contains(t, w.Body.String(), tc.want)
		})
	}
}

// A handshake this server cannot take over is refused too. httptest's recorder
// is not a Hijacker, which is exactly the case: the answer says so instead of
// panicking on a type assertion.
func TestWebSocketRefusesWhenItCannotTakeOverTheConnection(t *testing.T) {
	s, err := NewServer(Config{Provider: &fakeProvider{text: "hi"}})
	require.NoError(t, err)
	req := httptest.NewRequest(http.MethodGet, "/", nil)
	req.Header.Set("Upgrade", "websocket")
	req.Header.Set("Sec-WebSocket-Key", "dGhlIHNhbXBsZSBub25jZQ==")
	w := httptest.NewRecorder()

	s.WebSocketHandler().ServeHTTP(w, req)
	assert.Equal(t, http.StatusBadRequest, w.Code)
	assert.Contains(t, w.Body.String(), "cannot hand over the connection")
}

// readDocumentFrames concatenates messages until they make a whole document.
func readDocumentFrames(t *testing.T, ws *wsClient) []byte {
	t.Helper()
	var doc []byte
	for {
		payload, err := ws.receive()
		require.NoError(t, err)
		doc = append(doc, payload...)
		if documentComplete(bufio.NewReader(newByteReader(doc))) {
			return doc
		}
	}
}
