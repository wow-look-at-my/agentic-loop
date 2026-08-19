package cli

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	commonai "github.com/wow-look-at-my/agentic-loop/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Each --dialect names a protocol, and the request that reaches the upstream
// is that protocol's -- the flag is not decoration.
func TestDialectFlagPicksTheWireFormat(t *testing.T) {
	for _, tc := range []struct {
		dialect string
		path    string
		body    string
	}{
		{"openai", "/chat/completions", `"messages"`},
		{"anthropic", "/v1/messages", `"max_tokens"`},
		{"responses", "/responses", `"input"`},
	} {
		t.Run(tc.dialect, func(t *testing.T) {
			var gotPath, gotBody string
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				gotPath = r.URL.Path
				gotBody = string(readAll(t, r.Body))
				w.Header().Set("Content-Type", "text/event-stream")
				_, _ = w.Write([]byte("data: [DONE]\n\n"))
			}))
			defer srv.Close()

			_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", tc.dialect,
				"--model", "m", "--max-tokens", "64", "hi")
			require.NoError(t, err)
			assert.Equal(t, tc.path, gotPath)
			assert.Contains(t, gotBody, tc.body)
		})
	}
}

// A dialect nobody implements is refused by name, before a call goes anywhere.
func TestUnknownDialectIsRefused(t *testing.T) {
	_, err := run(t, "", "ask", "--endpoint", "https://example.invalid", "--dialect", "smoke",
		"--model", "m", "hi")
	require.ErrorContains(t, err, `unknown dialect "smoke"`)
}

// Without --dialect, the endpoint is asked what it speaks. When it will not
// say, the error names the flag that settles it rather than guessing.
func TestDialectIsDetectedAndTheFailureNamesTheFlag(t *testing.T) {
	models := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/models") {
			w.Header().Set("Content-Type", "application/json")
			_, _ = w.Write([]byte(`{"data":[{"id":"gpt-x","object":"model"}]}`))
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"choices":[{"delta":{"content":"detected"}}]}` + "\n\ndata: [DONE]\n\n"))
	}))
	defer models.Close()
	out, err := run(t, "", "ask", "--endpoint", models.URL, "--model", "m", "hi")
	require.NoError(t, err)
	assert.Equal(t, "detected\n", out)

	silent := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer silent.Close()
	_, err = run(t, "", "ask", "--endpoint", silent.URL, "--model", "m", "hi")
	require.ErrorContains(t, err, "pass --dialect")
}

// The API key is the process's, never the command line's: a key in argv is a
// key in the shell history and in every process listing on the machine.
func TestTheKeyComesFromTheEnvironment(t *testing.T) {
	var auth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		auth = r.Header.Get("Authorization")
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
	}))
	defer srv.Close()

	t.Setenv("CAI_API_KEY", "sk-from-the-environment")
	_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m", "hi")
	require.NoError(t, err)
	assert.Equal(t, "Bearer sk-from-the-environment", auth)
}

// Endpoint, model and sessions directory each fall back to their environment
// variable, so a shell can set them once.
func TestSettingsFallBackToTheEnvironment(t *testing.T) {
	srv, bodies := upstream(t, "hello")
	dir := t.TempDir()
	t.Setenv("CAI_ENDPOINT", srv.URL)
	t.Setenv("CAI_DIALECT", "openai")
	t.Setenv("CAI_MODEL", "from-the-env")
	t.Setenv("CAI_SESSIONS", dir)

	_, err := run(t, "", "chat", "--session", "work", "hi")
	require.NoError(t, err)
	require.Len(t, *bodies, 1)
	assert.Contains(t, (*bodies)[0], "from-the-env")

	entries, err := os.ReadDir(dir)
	require.NoError(t, err)
	require.Len(t, entries, 1, "the conversation landed in CAI_SESSIONS")
	assert.Equal(t, "work.xml", entries[0].Name())
}

// A prompt is the arguments joined, or stdin when it is piped -- and nothing
// at all is an error that says what was wanted, rather than a command that
// sits waiting on input nobody knew to type.
func TestPromptComesFromArgumentsOrAPipe(t *testing.T) {
	srv, bodies := upstream(t, "ok")
	_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"several", "separate", "words")
	require.NoError(t, err)
	require.Len(t, *bodies, 1)
	assert.Contains(t, (*bodies)[0], "several separate words")

	_, err = run(t, "  piped, and trimmed  \n", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m")
	require.NoError(t, err)
	require.Len(t, *bodies, 2)
	assert.Contains(t, (*bodies)[1], "piped, and trimmed")

	_, err = run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m")
	require.ErrorContains(t, err, "nothing to ask")
}

// An image alone is a question, so a prompt is not required alongside one.
func TestAnImageWithNoPromptIsStillAQuestion(t *testing.T) {
	srv, bodies := upstream(t, "a cat")
	png := filepath.Join(t.TempDir(), "shot.png")
	require.NoError(t, os.WriteFile(png, []byte("\x89PNG\r\n\x1a\n"+strings.Repeat("x", 64)), 0o600))

	_, err := run(t, "", "ask", "--endpoint", srv.URL, "--dialect", "openai", "--model", "m",
		"--image", png)
	require.NoError(t, err)
	require.Len(t, *bodies, 1)
	assert.Contains(t, (*bodies)[0], "image_url")
}

// A stored transcript shows every kind of turn. A tool call that rendered as
// nothing would read as a turn the model never took.
func TestMessageTextRendersEveryPart(t *testing.T) {
	m := commonai.NewMessage(commonai.RoleAssistant,
		commonai.TextPart{Text: "looking"},
		commonai.ThinkingPart{Text: "hmm"},
		commonai.RedactedThinkingPart{Data: "opaque"},
		commonai.ImagePart{MediaType: "image/png", Data: "iVBORw0KGgo="},
		commonai.ToolCallPart{ID: "c1", Name: "grep", Arguments: `{"q":"x"}`})

	got := messageText(m)
	assert.Contains(t, got, "looking")
	assert.Contains(t, got, "[thinking: hmm]")
	assert.Contains(t, got, "[thinking, redacted by the provider]")
	assert.Contains(t, got, "[image image/png]")
	assert.Contains(t, got, `[calls grep {"q":"x"}]`)
}
