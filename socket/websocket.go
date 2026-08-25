package socket

import (
	"bufio"
	"crypto/sha1"
	"encoding/base64"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
)

// The websocket half is hand-rolled against RFC 6455, using only the standard library.

// wsGUID is the constant RFC 6455 appends to the client's key before hashing.
const wsGUID = "258EAFA5-E914-47DA-95CA-5AB0DC85B11D"

// Frame opcodes.
const (
	opContinuation = 0x0
	opText         = 0x1
	opBinary       = 0x2
	opClose        = 0x8
	opPing         = 0x9
	opPong         = 0xA
)

// maxFrame caps one inbound frame's payload, the same bound the document reader applies.
const maxFrame = 64 << 20

// WebSocketHandler serves the same two operations over a websocket: the client
// sends a document as one or more text messages, and the answer comes back as
// one text message per flush -- which concatenate into exactly the document a
// unix-socket client reads.
func (s *Server) WebSocketHandler() http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, rw, err := acceptWebSocket(w, r)
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		defer conn.Close()
		ws := &wsConn{r: rw.Reader, w: rw.Writer, c: conn}
		s.Handle(r.Context(), ws)
		_ = ws.writeClose()
	})
}

// acceptWebSocket completes the opening handshake and takes the connection off
// net/http.
func acceptWebSocket(w http.ResponseWriter, r *http.Request) (io.Closer, *bufio.ReadWriter, error) {
	if !strings.EqualFold(r.Header.Get("Upgrade"), "websocket") {
		return nil, nil, errors.New("socket: not a websocket upgrade")
	}
	key := r.Header.Get("Sec-WebSocket-Key")
	if key == "" {
		return nil, nil, errors.New("socket: the upgrade carries no Sec-WebSocket-Key")
	}
	h, ok := w.(http.Hijacker)
	if !ok {
		return nil, nil, errors.New("socket: this server cannot hand over the connection")
	}
	conn, rw, err := h.Hijack()
	if err != nil {
		return nil, nil, fmt.Errorf("socket: taking over the connection: %w", err)
	}
	sum := sha1.Sum([]byte(key + wsGUID))
	accept := base64.StdEncoding.EncodeToString(sum[:])
	_, err = rw.WriteString("HTTP/1.1 101 Switching Protocols\r\n" +
		"Upgrade: websocket\r\n" +
		"Connection: Upgrade\r\n" +
		"Sec-WebSocket-Accept: " + accept + "\r\n\r\n")
	if err == nil {
		err = rw.Flush()
	}
	if err != nil {
		conn.Close()
		return nil, nil, fmt.Errorf("socket: completing the handshake: %w", err)
	}
	return conn, rw, nil
}

// wsConn presents a websocket as the plain byte stream the document reader and writer speak.
type wsConn struct {
	r  *bufio.Reader
	w  *bufio.Writer
	c  io.Closer
	mu sync.Mutex
	// rest is the remainder of the frame the last Read did not have room for.
	rest []byte
}

// Read implements io.Reader over inbound text frames.
func (c *wsConn) Read(p []byte) (int, error) {
	for len(c.rest) == 0 {
		op, payload, err := c.readFrame()
		if err != nil {
			return 0, err
		}
		switch op {
		case opText, opBinary, opContinuation:
			c.rest = payload
		case opPing:
			if err := c.writeFrame(opPong, payload); err != nil {
				return 0, err
			}
		case opClose:
			return 0, io.EOF
		}
	}
	n := copy(p, c.rest)
	c.rest = c.rest[n:]
	return n, nil
}

// Write sends one text frame, so a flush of the answer is a message.
func (c *wsConn) Write(p []byte) (int, error) {
	if len(p) == 0 {
		return 0, nil
	}
	if err := c.writeFrame(opText, p); err != nil {
		return 0, err
	}
	return len(p), nil
}

// writeClose sends the closing handshake, best effort: a peer that already left cannot be told.
func (c *wsConn) writeClose() error { return c.writeFrame(opClose, nil) }

// Close closes the underlying connection.
func (c *wsConn) Close() error { return c.c.Close() }

// readFrame reads one frame, unmasking the payload. A client MUST mask, and a
// server MUST NOT.
func (c *wsConn) readFrame() (byte, []byte, error) {
	var head [2]byte
	if _, err := io.ReadFull(c.r, head[:]); err != nil {
		return 0, nil, err
	}
	op := head[0] & 0x0F
	masked := head[1]&0x80 != 0
	length := uint64(head[1] & 0x7F)
	switch length {
	case 126:
		var ext [2]byte
		if _, err := io.ReadFull(c.r, ext[:]); err != nil {
			return 0, nil, err
		}
		length = uint64(binary.BigEndian.Uint16(ext[:]))
	case 127:
		var ext [8]byte
		if _, err := io.ReadFull(c.r, ext[:]); err != nil {
			return 0, nil, err
		}
		length = binary.BigEndian.Uint64(ext[:])
	}
	if length > maxFrame {
		return 0, nil, fmt.Errorf("socket: frame of %d bytes exceeds the %d-byte limit", length, maxFrame)
	}
	var mask [4]byte
	if masked {
		if _, err := io.ReadFull(c.r, mask[:]); err != nil {
			return 0, nil, err
		}
	}
	payload := make([]byte, length)
	if _, err := io.ReadFull(c.r, payload); err != nil {
		return 0, nil, err
	}
	if masked {
		for i := range payload {
			payload[i] ^= mask[i%4]
		}
	}
	return op, payload, nil
}

// writeFrame sends one unmasked frame and flushes it, because an unflushed
// frame is an answer the caller cannot see.
func (c *wsConn) writeFrame(op byte, payload []byte) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	head := []byte{0x80 | op}
	n := len(payload)
	switch {
	case n < 126:
		head = append(head, byte(n))
	case n <= 0xFFFF:
		head = append(head, 126, byte(n>>8), byte(n))
	default:
		head = append(head, 127)
		var ext [8]byte
		binary.BigEndian.PutUint64(ext[:], uint64(n))
		head = append(head, ext[:]...)
	}
	if _, err := c.w.Write(head); err != nil {
		return err
	}
	if _, err := c.w.Write(payload); err != nil {
		return err
	}
	return c.w.Flush()
}
