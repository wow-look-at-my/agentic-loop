package cli

import (
	"bytes"
	"context"
	"encoding/base64"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/spf13/cobra"
	commonai "github.com/wow-look-at-my/agentic-loop/core"
	"github.com/wow-look-at-my/agentic-loop/internal/jsontest"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// upstream is an OpenAI-compatible server that streams answer and records
// the bodies it was sent.
func upstream(t *testing.T, answer string) (*httptest.Server, *[]string) {
	t.Helper()
	var bodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		buf := new(bytes.Buffer)
		_, _ = buf.ReadFrom(r.Body)
		bodies = append(bodies, buf.String())
		w.Header().Set("Content-Type", "text/event-stream")
		for _, chunk := range strings.Split(answer, " ") {
			payload := jsontest.Must(jsontest.Obj{"choices": []any{jsontest.Obj{"delta": jsontest.Obj{"content": chunk + " "}}}})
			_, _ = w.Write([]byte("data: " + payload + "\n\n"))
		}
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{},"finish_reason":"stop"}]}` + "\n\ndata: [DONE]\n\n"))
	}))
	t.Cleanup(srv.Close)
	return srv, &bodies
}

// cliMu serializes the tests that drive the command tree, which is package
// state: its flags, its args, and where it writes.
var cliMu sync.Mutex

// run drives the command tree the way a shell does, and returns what a user
// would see on stdout.
func run(t *testing.T, stdin string, args ...string) (string, error) {
	t.Helper()
	cliMu.Lock()
	defer cliMu.Unlock()
	var out, errOut bytes.Buffer
	// Each invocation starts from the defaults; a prior case's flag must not leak.
	resetFlags()
	root.SetOut(&out)
	root.SetErr(&errOut)
	root.SetIn(strings.NewReader(stdin))
	root.SetArgs(args)
	err := root.ExecuteContext(context.Background())
	return out.String(), err
}

// resetFlags puts the per-execution state back: the flag variables, and the context cobra caches on each command.
func resetFlags() {
	flagEndpoint, flagDialect, flagModel, flagSystem = "", "", "", ""
	flagMaxTok, flagTemp, flagImages, flagSessions = 0, -1, nil, ""
	chatSession = "default"
	serveHTTP, serveSocket = "", ""
	forgetContexts(root)
}

// forgetContexts clears the cached context on a command and everything under it.
func forgetContexts(c *cobra.Command) {
	c.SetContext(nil)
	for _, sub := range c.Commands() {
		forgetContexts(sub)
	}
}

// readAll drains a reader, failing the test rather than the assertion after it.
func readAll(t *testing.T, r io.Reader) []byte {
	t.Helper()
	data, err := io.ReadAll(r)
	require.NoError(t, err)
	return data
}

func TestAskPrintsTheAnswer(t *testing.T) {
	srv, bodies := upstream(t, "hello there")
	out, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m", "what is up")
	require.NoError(t, err)
	assert.Equal(t, "hello there \n", out)

	require.Len(t, *bodies, 1)
	assert.Contains(t, (*bodies)[0], "what is up", "the prompt reached the upstream")
	assert.NotContains(t, (*bodies)[0], "<request", "the CLI's surface has no XML in it")
}

// A pipe is a prompt: the shell case that would otherwise push a user into
// building a document by hand.
func TestAskReadsAPipedPrompt(t *testing.T) {
	srv, bodies := upstream(t, "piped")
	out, err := run(t, "  what is this?\n", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m")
	require.NoError(t, err)
	assert.Equal(t, "piped \n", out)
	assert.Contains(t, (*bodies)[0], "what is this?")
}

// An image goes in as a file, never as a base64 blob the user had to make.
func TestAskAttachesAnImage(t *testing.T) {
	srv, bodies := upstream(t, "a cat")
	png := []byte("\x89PNG\r\n\x1a\n" + strings.Repeat("x", 64))
	path := filepath.Join(t.TempDir(), "shot.png")
	require.NoError(t, os.WriteFile(path, png, 0o644))

	_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--image", path, "what is this")
	require.NoError(t, err)
	assert.Contains(t, (*bodies)[0], base64.StdEncoding.EncodeToString(png))
	assert.Contains(t, (*bodies)[0], "image/png", "sniffed from the bytes, not from the name")
}

func TestAskRejectsAFileThatIsNotAnImage(t *testing.T) {
	srv, _ := upstream(t, "never reached")
	path := filepath.Join(t.TempDir(), "notes.png")
	require.NoError(t, os.WriteFile(path, []byte("this is plain text"), 0o644))

	_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--image", path, "what is this")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "not an image")
}

func TestAskNeedsAnEndpointAndAModel(t *testing.T) {
	_, err := run(t, "", "ask", "--model", "m", "hi")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "endpoint")

	srv, _ := upstream(t, "unused")
	_, err = run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "hi")
	require.Error(t, err)
	assert.Contains(t, err.Error(), "model")
}

// The stored conversation is what makes the question land in the same
// conversation as the, and it is kept as the format's own document.
func TestChatKeepsTheConversation(t *testing.T) {
	srv, bodies := upstream(t, "first answer")
	dir := t.TempDir()
	args := []string{"chat", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--sessions", dir, "--session", "work"}

	_, err := run(t, "", append(args, "first question")...)
	require.NoError(t, err)
	_, err = run(t, "", append(args, "second question")...)
	require.NoError(t, err)

	require.Len(t, *bodies, 2)
	assert.NotContains(t, (*bodies)[0], "first answer")
	assert.Contains(t, (*bodies)[1], "first question", "the model sees what was said")
	assert.Contains(t, (*bodies)[1], "first answer")

	// The session on disk is a conversation document, valid against the schema.
	data, err := os.ReadFile(filepath.Join(dir, "work.xml"))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)
	id, conv, err := commonai.DecodeConversation(data)
	require.NoError(t, err)
	assert.Equal(t, "work", id)
	assert.Len(t, conv.Messages, 4)
}

func TestSessionListShowAndRemove(t *testing.T) {
	srv, _ := upstream(t, "hi")
	dir := t.TempDir()
	_, err := run(t, "", "chat", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--sessions", dir, "--session", "work", "hello")
	require.NoError(t, err)

	out, err := run(t, "", "session", "list", "--sessions", dir)
	require.NoError(t, err)
	assert.Contains(t, out, "work")
	assert.Contains(t, out, "2 messages")

	out, err = run(t, "", "session", "show", "work", "--sessions", dir)
	require.NoError(t, err)
	assert.Contains(t, out, "user: hello")
	assert.Contains(t, out, "assistant: hi")

	_, err = run(t, "", "session", "rm", "work", "--sessions", dir)
	require.NoError(t, err)
	out, err = run(t, "", "session", "list", "--sessions", dir)
	require.NoError(t, err)
	assert.Empty(t, out)
}

// A failure after some of the answer arrived keeps the text and still reports
// the failure: output the caller saw is theirs, and a partial answer that
// looks complete is worse than no answer.
func TestAskReportsAFailureAfterOutput(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"half an ans"}}]}` + "\n\n"))
		w.(http.Flusher).Flush()
		// The connection ends mid-stream, with no [DONE].
		if hj, ok := w.(http.Hijacker); ok {
			conn, _, err := hj.Hijack()
			if err == nil {
				_ = conn.Close()
			}
		}
	}))
	defer srv.Close()

	out, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m", "go")
	require.Error(t, err)
	assert.Contains(t, out, "half an ans", "what arrived is still printed")
	assert.Contains(t, err.Error(), "cut off")
}

func TestServeNeedsSomethingToServeOn(t *testing.T) {
	srv, _ := upstream(t, "unused")
	_, err := run(t, "", "serve", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--sessions", t.TempDir())
	require.Error(t, err)
	assert.Contains(t, err.Error(), "nothing to serve on")
}
