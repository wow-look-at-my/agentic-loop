package repo

import (
	"context"
	"encoding/json"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"io"
	"net/http"
	"net/http/httptest"
	"net/url"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
)

// Shared helpers for the repo tool tests (repo_test.go and the per-tool files).

// execRepoTool marshals args (any JSON-serializable shape) and executes the
// named tool.
func execRepoTool(t *testing.T, reg agentic.Tools, name string, args any) agentic.ToolResult {
	t.Helper()
	payload, err := json.Marshal(args)
	require.NoError(t, err)
	tool, ok := reg.Find(name)
	require.True(t, ok, "tool %s must be advertised", name)
	r, err := tool.Execute(context.Background(), payload)
	require.NoError(t, err)
	return r
}

// ghCall is one recorded request against the fake GitHub server.
type ghCall struct {
	Method string
	Path   string
	Auth   string // bearer token, "" when unauthenticated
	Accept string
	Query  url.Values
	Body   string
}

// fakeGitHub is a fake GitHub API that records every request (method, path,
// token, accept, query, body) and answers via a per-test responder.
type fakeGitHub struct {
	mu      sync.Mutex
	calls   []ghCall
	respond func(c ghCall) (status int, body string)
	// headers, when set, adds response headers to each answer. The rate-limit
	// classification reads x-ratelimit-* / retry-after, so a test has to be
	// able to send them.
	headers func(c ghCall) http.Header
}

func (g *fakeGitHub) handler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		c := ghCall{
			Method: r.Method,
			Path:   r.URL.Path,
			Auth:   trimBearer(r.Header.Get("Authorization")),
			Accept: r.Header.Get("Accept"),
			Query:  r.URL.Query(),
			Body:   string(body),
		}
		g.mu.Lock()
		g.calls = append(g.calls, c)
		g.mu.Unlock()
		status, resp := g.respond(c)
		w.Header().Set("Content-Type", "application/json")
		if g.headers != nil {
			for k, vs := range g.headers(c) {
				for _, v := range vs {
					w.Header().Add(k, v)
				}
			}
		}
		w.WriteHeader(status)
		_, _ = w.Write([]byte(resp))
	}
}

func trimBearer(h string) string {
	const p = "Bearer "
	if len(h) >= len(p) && h[:len(p)] == p {
		return h[len(p):]
	}
	return h
}

// Calls returns a copy of the recorded requests.
func (g *fakeGitHub) Calls() []ghCall {
	g.mu.Lock()
	defer g.mu.Unlock()
	return append([]ghCall(nil), g.calls...)
}

// Auths returns the bearer token of every recorded request, in order.
func (g *fakeGitHub) Auths() []string {
	var out []string
	for _, c := range g.Calls() {
		out = append(out, c.Auth)
	}
	return out
}

// newFakeGitHub starts the fake server and returns the repo tools pointed at it.
func newFakeGitHub(t *testing.T, cfg GitHubConfig, respond func(c ghCall) (int, string)) (*fakeGitHub, agentic.Tools) {
	t.Helper()
	g, _, reg := newFakeGitHubFull(t, cfg, respond)
	return g, reg
}

// newFakeGitHubFull also hands back the client behind the tools, which is what
// a host mounting /repos as a folder takes.
func newFakeGitHubFull(t *testing.T, cfg GitHubConfig, respond func(c ghCall) (int, string)) (*fakeGitHub, *GitHub, agentic.Tools) {
	t.Helper()
	g := &fakeGitHub{respond: respond}
	srv := httptest.NewServer(g.handler())
	t.Cleanup(srv.Close)
	cfg.HTTPClient = srv.Client()
	cfg.APIBaseURL = srv.URL
	gh := NewGitHub(cfg)
	return g, gh, NewRepoTools(RepoToolsConfig{GitHub: gh})
}

// repoToolset is the repo tools alone, for the tests that never need the
// client behind them.
func repoToolset(cfg GitHubConfig) agentic.Tools {
	return NewRepoTools(RepoToolsConfig{GitHub: NewGitHub(cfg)})
}
