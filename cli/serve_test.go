package cli

import (
	"bufio"
	"bytes"
	"context"
	"net"
	"net/http"
	"path/filepath"
	"testing"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// serveInBackground starts `cai serve` with the given arguments and returns
// once both listeners answer. The command runs until the returned stop is
// called, which is what a Ctrl-C does to it.
func serveInBackground(t *testing.T, args ...string) (stop func()) {
	t.Helper()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	var out, errOut bytes.Buffer

	resetFlags()
	root.SetOut(&out)
	root.SetErr(&errOut)
	root.SetArgs(args)
	go func() { done <- root.ExecuteContext(ctx) }()

	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			assert.NoError(t, err, "stderr:\n%s", errOut.String())
		case <-time.After(5 * time.Second):
			t.Error("serve did not stop when its context ended")
		}
	})
	return cancel
}

// freeAddr reserves a loopback port and gives it straight back, so serve can
// bind the one address the test knows.
func freeAddr(t *testing.T) string {
	t.Helper()
	l, err := net.Listen("tcp", "127.0.0.1:0")
	require.NoError(t, err)
	addr := l.Addr().String()
	require.NoError(t, l.Close())
	return addr
}

// waitFor retries until the dial succeeds, because a listener started in a
// goroutine is not up the instant the goroutine is.
func waitFor(t *testing.T, network, addr string) {
	t.Helper()
	for range 100 {
		conn, err := net.Dial(network, addr)
		if err == nil {
			_ = conn.Close()
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("nothing listening on %s %s", network, addr)
}

// requestDoc is the document a client of the server sends. It names no
// endpoint and no key: those are the serving process's, which is the point of
// serving at all.
func requestDoc(t *testing.T, text string) []byte {
	t.Helper()
	data, err := commonai.EncodeRequestBytes(commonai.Request{
		Model:    "m",
		Messages: []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: text})},
	})
	require.NoError(t, err)
	return data
}

// One serve carries both surfaces, and both answer the same document -- the
// HTTP body and the unix socket are two ways to the one server, not two
// servers that happen to agree.
func TestServeAnswersOnHTTPAndTheUnixSocket(t *testing.T) {
	up, bodies := upstream(t, "hello there")
	addr := freeAddr(t)
	sock := filepath.Join(t.TempDir(), "cai.sock")
	serveInBackground(t, "serve", "--http", addr, "--socket", sock,
		"--endpoint", up.URL, "--dialect", "openai", "--sessions", t.TempDir())
	waitFor(t, "tcp", addr)
	waitFor(t, "unix", sock)

	resp, err := http.Post("http://"+addr+"/v1/complete", "application/xml", bytes.NewReader(requestDoc(t, "over http")))
	require.NoError(t, err)
	defer resp.Body.Close()
	assert.Equal(t, http.StatusOK, resp.StatusCode)
	comp, err := commonai.DecodeResponse(readAll(t, resp.Body))
	require.NoError(t, err)
	assert.Equal(t, "hello there ", comp.Message.Content)

	conn, err := net.Dial("unix", sock)
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))
	_, err = conn.Write(requestDoc(t, "over the socket"))
	require.NoError(t, err)
	data, err := commonai.ReadDocument(bufio.NewReader(conn))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	comp, err = commonai.DecodeResponse(data)
	require.NoError(t, err)
	assert.Equal(t, "hello there ", comp.Message.Content)

	require.Len(t, *bodies, 2)
	assert.Contains(t, (*bodies)[0], "over http")
	assert.Contains(t, (*bodies)[1], "over the socket")
}

// The websocket route is mounted alongside the HTTP one, on the same server.
func TestServeMountsTheWebSocketRoute(t *testing.T) {
	up, _ := upstream(t, "hi")
	addr := freeAddr(t)
	serveInBackground(t, "serve", "--http", addr,
		"--endpoint", up.URL, "--dialect", "openai", "--sessions", t.TempDir())
	waitFor(t, "tcp", addr)

	conn, err := net.Dial("tcp", addr)
	require.NoError(t, err)
	defer conn.Close()
	require.NoError(t, conn.SetDeadline(time.Now().Add(5*time.Second)))
	_, err = conn.Write([]byte("GET /v1/ws HTTP/1.1\r\nHost: " + addr + "\r\n" +
		"Upgrade: websocket\r\nConnection: Upgrade\r\n" +
		"Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n"))
	require.NoError(t, err)

	resp, err := http.ReadResponse(bufio.NewReader(conn), nil)
	require.NoError(t, err)
	assert.Equal(t, http.StatusSwitchingProtocols, resp.StatusCode)
	// The accept token for that key, computed outside this codebase so it is not the server checking its own arithmetic.
	assert.Equal(t, "7NQHw21/u2y5o3iigl/YosUutlE=", resp.Header.Get("Sec-WebSocket-Accept"))
}

// A socket path that already holds something is refused rather than unlinked:
// whatever is there is either another server or a file somebody wants.
func TestServeRefusesASocketPathInUse(t *testing.T) {
	up, _ := upstream(t, "hi")
	taken := filepath.Join(t.TempDir(), "cai.sock")
	l, err := net.Listen("unix", taken)
	require.NoError(t, err)
	defer l.Close()

	_, err = run(t, "", "serve", "--socket", taken,
		"--endpoint", up.URL, "--dialect", "openai", "--sessions", t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already exists")
}

// Serving needs a provider like every other call does, and says so before it
// binds anything.
func TestServeNeedsAnEndpoint(t *testing.T) {
	_, err := run(t, "", "serve", "--http", "127.0.0.1:0")
	require.ErrorContains(t, err, "no endpoint")
}
