package repo

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"github.com/wow-look-at-my/agentic-loop/internal/jsontest"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// writableCfg builds the credential config when every configured PAT is allow_model_writes.
func writableCfg(cfg GitHubConfig, tokens ...GitHubToken) GitHubConfig {
	cfg.Tokens = tokens
	cfg.WriteTokens = tokens
	return cfg
}

// fileWriteResponder answers a full repo_file_write flow for the given token:
// repo meta, branch ref, the existence lookup (existingSHA "" = 404, a new
// file; non-empty = the path already exists, which the create-only tool
// refuses), and the contents PUT.
func fileWriteResponder(t *testing.T, goodToken, existingSHA string) func(c ghCall) (int, string) {
	return func(c ghCall) (int, string) {
		if c.Auth != goodToken {
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/git/ref/heads/docs":
			return http.StatusOK, `{"object":{"sha":"refsha"}}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/contents/docs/intro.md":
			if existingSHA == "" {
				return http.StatusNotFound, `{"message":"Not Found"}`
			}
			return http.StatusOK, jsontest.Must(jsontest.Obj{
				"sha":  existingSHA,
				"type": "file",
			})
		case c.Method == http.MethodPut && c.Path == "/repos/octo/hello/contents/docs/intro.md":
			return http.StatusOK, `{"commit":{"sha":"newcommit123","html_url":"https://github.com/octo/hello/commit/newcommit123"}}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	}
}

func TestRepoFileWriteRefusesExistingFile(t *testing.T) {
	cache := newMemCache()
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{Cache: cache}, GitHubToken{ID: "t1", Token: "writer"}), fileWriteResponder(t, "writer", "oldsha"))

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "# Intro\nhello", Message: "update intro",
	})
	require.True(t, res.IsError, "an existing path must be refused, not overwritten")
	assert.Equal(t, `docs/intro.md already exists in octo/hello on branch "docs" — repo_file_write only creates new files, it never overwrites. To change an existing file, ask the user to pull the pull request (or open one for the branch) into a conversation workspace from the files pane, then edit it with workspace_edit's replace.`, res.Content)

	// Nothing was mutated and no winner was cached for the refused call.
	for _, c := range gh.Calls() {
		assert.Equal(t, http.MethodGet, c.Method, "a refused overwrite must not mutate anything")
	}
	_, ok := cache.Get("octo/hello")
	assert.False(t, ok)
}

func TestRepoFileWriteCreatesNewFileWithoutSHA(t *testing.T) {
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), fileWriteResponder(t, "writer", "")) // contents lookup 404s: new file

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "new", Message: "add intro",
	})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "created docs/intro.md on branch docs of octo/hello")
	assert.Contains(t, res.Content, "commit newcommit123")

	// The existence check ran against the target branch; the PUT carried base64, never a blob SHA.
	var put ghCall
	sawCheck := false
	for _, c := range gh.Calls() {
		if c.Method == http.MethodGet && c.Path == "/repos/octo/hello/contents/docs/intro.md" {
			sawCheck = true
			assert.Equal(t, "docs", c.Query.Get("ref"))
		}
		if c.Method == http.MethodPut {
			put = c
		}
	}
	assert.True(t, sawCheck)
	require.Equal(t, http.MethodPut, put.Method)
	var body struct {
		Message string `json:"message"`
		Content string `json:"content"`
		Branch  string `json:"branch"`
		SHA     string `json:"sha"`
	}
	require.NoError(t, json.Unmarshal([]byte(put.Body), &body))
	assert.Equal(t, "add intro", body.Message)
	assert.Equal(t, "docs", body.Branch)
	assert.Empty(t, body.SHA)
	assert.NotContains(t, put.Body, `"sha"`) // no blob SHA, ever
	raw, err := base64.StdEncoding.DecodeString(body.Content)
	require.NoError(t, err)
	assert.Equal(t, "new", string(raw))
}

func TestRepoFileWriteMissingBranchTeachesCreateBranch(t *testing.T) {
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/ghost":
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "ghost", Path: "f.txt",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `branch "ghost" does not exist`)
	assert.Contains(t, res.Content, "create_branch=true")
	assert.Contains(t, res.Content, `"main"`) // names the default branch
	for _, c := range gh.Calls() {
		assert.Equal(t, http.MethodGet, c.Method, "nothing may be mutated without create_branch")
	}
}

func TestRepoFileWriteCreatesBranchFromDefault(t *testing.T) {
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/git/ref/heads/feature":
			return http.StatusNotFound, `{"message":"Not Found"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/commits/main":
			return http.StatusOK, `{"sha":"basesha42"}`
		case c.Method == http.MethodPost && c.Path == "/repos/octo/hello/git/refs":
			return http.StatusCreated, `{"ref":"refs/heads/feature"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/contents/f.txt":
			return http.StatusNotFound, `{"message":"Not Found"}`
		case c.Method == http.MethodPut && c.Path == "/repos/octo/hello/contents/f.txt":
			return http.StatusCreated, `{"commit":{"sha":"c1","html_url":"https://github.com/octo/hello/commit/c1"}}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "feature", Path: "f.txt",
		Content: "x", Message: "m", CreateBranch: true,
	})
	require.False(t, res.IsError, res.Content)
	assert.Contains(t, res.Content, "created f.txt on branch feature of octo/hello")
	assert.Contains(t, res.Content, "branch feature created from main")

	var post ghCall
	for _, c := range gh.Calls() {
		if c.Method == http.MethodPost {
			post = c
		}
		if c.Method == http.MethodGet && c.Path == "/repos/octo/hello/contents/f.txt" {
			// The existence check runs against the SOURCE ref the new branch starts from.
			assert.Equal(t, "main", c.Query.Get("ref"))
		}
	}
	require.Equal(t, http.MethodPost, post.Method)
	assert.JSONEq(t, `{"ref":"refs/heads/feature","sha":"basesha42"}`, post.Body)
}

// The overwrite-via-fresh-branch dodge: the file exists on the branch the new
// branch would be created from, so the create is refused — and NOTHING is
// mutated, the branch included.
func TestRepoFileWriteRefusesExistingFileOnNewBranchSource(t *testing.T) {
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/git/ref/heads/feature":
			return http.StatusNotFound, `{"message":"Not Found"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/commits/main":
			return http.StatusOK, `{"sha":"basesha42"}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/contents/README.md":
			assert.Equal(t, "main", c.Query.Get("ref"), "the check must target the source ref")
			return http.StatusOK, `{"sha":"existing123","type":"file"}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "feature", Path: "README.md",
		Content: "rewritten", Message: "m", CreateBranch: true,
	})
	require.True(t, res.IsError)
	assert.Equal(t, `README.md already exists in octo/hello on "main" (the ref branch "feature" would be created from) — repo_file_write only creates new files, it never overwrites. To change an existing file, ask the user to pull the pull request (or open one for the branch) into a conversation workspace from the files pane, then edit it with workspace_edit's replace.`, res.Content)
	for _, c := range gh.Calls() {
		assert.Equal(t, http.MethodGet, c.Method, "a refused overwrite must not create the branch or commit anything")
	}
}

func TestRepoFileWriteCreateBranchFromExplicitRef(t *testing.T) {
	var resolvedRef string
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/feature":
			return http.StatusNotFound, `{}`
		case c.Method == http.MethodGet && c.Path == "/repos/octo/hello/commits/release-1":
			resolvedRef = "release-1"
			return http.StatusOK, `{"sha":"relsha"}`
		case c.Method == http.MethodPost && c.Path == "/repos/octo/hello/git/refs":
			return http.StatusCreated, `{}`
		case c.Path == "/repos/octo/hello/contents/f.txt" && c.Method == http.MethodGet:
			return http.StatusNotFound, `{}`
		case c.Path == "/repos/octo/hello/contents/f.txt" && c.Method == http.MethodPut:
			return http.StatusCreated, `{"commit":{"sha":"c2"}}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "feature", Path: "f.txt",
		Content: "x", Message: "m", CreateBranch: true, CreateBranchFrom: "release-1",
	})
	require.False(t, res.IsError, res.Content)
	assert.Equal(t, "release-1", resolvedRef)
	assert.Contains(t, res.Content, "branch feature created from release-1")
}

func TestRepoFileWriteMissingBaseRefIsFatal(t *testing.T) {
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "w1"}, GitHubToken{ID: "t2", Token: "w2"}), func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/feature":
			return http.StatusNotFound, `{}`
		case c.Path == "/repos/octo/hello/commits/ghost-base":
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "feature", Path: "f.txt",
		Content: "x", Message: "m", CreateBranch: true, CreateBranchFrom: "ghost-base",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `base ref "ghost-base" does not exist`)
}

func TestRepoFileWriteRetriesPastStaleCachedReadWinner(t *testing.T) {
	// The cache holds t1, a read-only token discovered by the read tools.
	cache := newMemCache()
	cache.Put("octo/hello", "t1")
	gh, ex := newFakeGitHub(t, writableCfg(GitHubConfig{Cache: cache},
		GitHubToken{ID: "t1", Token: "readonly"},
		GitHubToken{ID: "t2", Token: "writer"}), func(c ghCall) (int, string) {
		if c.Auth == "readonly" {
			switch {
			case c.Method == http.MethodPut:
				return http.StatusForbidden, `{"message":"Resource not accessible by personal access token"}`
			case c.Path == "/repos/octo/hello":
				return http.StatusOK, `{"default_branch":"main"}`
			case c.Path == "/repos/octo/hello/git/ref/heads/docs":
				return http.StatusOK, `{"object":{"sha":"refsha"}}`
			case c.Path == "/repos/octo/hello/contents/docs/intro.md":
				return http.StatusNotFound, `{"message":"Not Found"}` // new file
			}
		}
		return fileWriteResponder(t, "writer", "")(c)
	})

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "x", Message: "m",
	})
	require.False(t, res.IsError, res.Content)

	// The cached (read-only) token was tried first, then the writer.
	auths := gh.Auths()
	assert.Equal(t, "readonly", auths[0])
	assert.Contains(t, auths, "writer")
	// Never an unauthenticated attempt on a write.
	for _, a := range auths {
		assert.NotEmpty(t, a, "writes must never fall through to unauthenticated")
	}
	// The cache now records the token that completed the write.
	id, _ := cache.Get("octo/hello")
	assert.Equal(t, "t2", id)
}

func TestRepoFileWriteNoTokensRefusesUnauthenticated(t *testing.T) {
	gh, ex := newFakeGitHub(t, GitHubConfig{}, func(c ghCall) (int, string) {
		t.Errorf("no request may be made without a token, got %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "f.txt",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "never attempted unauthenticated")
	assert.Contains(t, res.Content, "Settings -> github")
	assert.Contains(t, res.Content, `"model can write"`, "a fresh PAT defaults to model writes OFF, so the fix must mention the flag")
	assert.Empty(t, gh.Calls())
}

// The initiator partition: tokens are configured but none is flagged for
// model-initiated writes (an empty write list), so both write tools return
// the recoverable teaching error naming the fix — before any request is made.
// Reads through the same client keep working with the unflagged token.
func TestRepoWriteNoModelWriteTokensTeaches(t *testing.T) {
	gh, client, ex := newFakeGitHubFull(t, GitHubConfig{
		Tokens: []GitHubToken{{ID: "t1", Token: "user-only"}},
	}, func(c ghCall) (int, string) {
		assert.Equal(t, http.MethodGet, c.Method)

		return http.StatusOK, `main file contents`
	})

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "f.txt",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError, "recoverable tool error, not a failed turn")
	assert.Equal(t, noModelWriteTokensMsg, res.Content)

	res = execRepoTool(t, ex, RepoPRCreateToolName, repoPRCreateArgs{
		Org: "octo", Repo: "hello", Title: "T", Head: "h",
	})
	assert.True(t, res.IsError)
	assert.Equal(t, noModelWriteTokensMsg, res.Content)
	assert.Empty(t, gh.Calls(), "neither write touched GitHub")

	// The read path is untouched by the partition.
	read, err := client.Fetch(context.Background(), "octo", "hello", "main.go", "", "application/vnd.github.raw")
	require.NoError(t, err)
	assert.Equal(t, http.StatusOK, read.Status())
	assert.Contains(t, string(read.Body()), "main file contents")
}

// Token A is enabled but unflagged (user-only); token B carries
// allow_model_writes. A model write must never send A's Authorization on any
// write-flow call and completes via B.
func TestRepoFileWriteUsesOnlyFlaggedTokens(t *testing.T) {
	tokens := []GitHubToken{
		{ID: "ta", Token: "user-only"},
		{ID: "tb", Token: "model-ok", AllowModelWrites: true},
	}
	gh, ex := newFakeGitHub(t, GitHubConfig{
		Tokens:      tokens,
		WriteTokens: ModelWriteTokens(tokens),
	}, fileWriteResponder(t, "model-ok", ""))

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "x", Message: "m",
	})
	require.False(t, res.IsError, res.Content)

	auths := gh.Auths()
	require.NotEmpty(t, auths)
	for _, auth := range auths {
		assert.Equal(t, "model-ok", auth, "a model write must only ever send a flagged token")
	}
}

// Winner-cache audit: the cached winner for the repo is the user-only token
// (recorded by a read or a user-initiated push). The model write must skip it
// — its Authorization never appears — rotate the write list normally, and the
// completed write records its own flagged token as the new winner.
func TestRepoFileWriteSkipsCachedUserOnlyWinner(t *testing.T) {
	cache := newMemCache()
	cache.Put("octo/hello", "ta")
	tokens := []GitHubToken{
		{ID: "ta", Token: "user-only"},
		{ID: "tb", Token: "model-ok", AllowModelWrites: true},
	}
	gh, ex := newFakeGitHub(t, GitHubConfig{
		Cache:       cache,
		Tokens:      tokens,
		WriteTokens: ModelWriteTokens(tokens),
	}, fileWriteResponder(t, "model-ok", ""))

	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "x", Message: "m",
	})
	require.False(t, res.IsError, res.Content)
	for _, auth := range gh.Auths() {
		assert.Equal(t, "model-ok", auth, "the cached user-only winner must be skipped, not sent")
	}
	id, ok := cache.Get("octo/hello")
	require.True(t, ok)
	assert.Equal(t, "tb", id, "the completed write caches its own flagged token")
}

// ModelWriteTokens is the construction-time filter chat.go applies.
func TestModelWriteTokens(t *testing.T) {
	assert.Empty(t, ModelWriteTokens(nil))
	assert.Empty(t, ModelWriteTokens([]GitHubToken{{ID: "a"}, {ID: "b"}}))
	got := ModelWriteTokens([]GitHubToken{
		{ID: "a"},
		{ID: "b", AllowModelWrites: true},
		{ID: "c"},
		{ID: "d", AllowModelWrites: true},
	})
	require.Len(t, got, 2)
	assert.Equal(t, "b", got[0].ID)
	assert.Equal(t, "d", got[1].ID)
}

func TestRepoFileWriteFatalErrorDoesNotRetryOtherTokens(t *testing.T) {
	putCalls := 0
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "w1"}, GitHubToken{ID: "t2", Token: "w2"}), func(c ghCall) (int, string) {
		switch {
		case c.Method == http.MethodPut:
			// The file appeared between the existence check and the PUT; GitHub 409s it.
			putCalls++
			return http.StatusConflict, `{"message":"docs/intro.md does not match"}`
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/docs":
			return http.StatusOK, `{"object":{"sha":"refsha"}}`
		case c.Path == "/repos/octo/hello/contents/docs/intro.md":
			return http.StatusNotFound, `{"message":"Not Found"}`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "docs/intro.md",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "409")
	assert.Contains(t, res.Content, "does not match")
	assert.Equal(t, 1, putCalls, "a non-auth failure must not be retried with other tokens")
}

func TestRepoFileWriteOnDirectoryIsFatal(t *testing.T) {
	_, ex := newFakeGitHub(t, writableCfg(GitHubConfig{}, GitHubToken{ID: "t1", Token: "writer"}), func(c ghCall) (int, string) {
		switch {
		case c.Path == "/repos/octo/hello":
			return http.StatusOK, `{"default_branch":"main"}`
		case c.Path == "/repos/octo/hello/git/ref/heads/docs":
			return http.StatusOK, `{"object":{"sha":"refsha"}}`
		case c.Path == "/repos/octo/hello/contents/src":
			return http.StatusOK, `[{"name":"a.go","type":"file"}]`
		}
		t.Errorf("unexpected request %s %s", c.Method, c.Path)
		return http.StatusTeapot, `{}`
	})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{
		Org: "octo", Repo: "hello", Branch: "docs", Path: "src",
		Content: "x", Message: "m",
	})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is a directory")
}

func TestRepoFileWriteArgErrors(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	res := execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{Org: "octo", Repo: "hello", Path: "f", Content: "x", Message: "m"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `"branch"`)

	res = execRepoTool(t, ex, RepoFileWriteToolName, repoFileWriteArgs{Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires both "org" and "repo"`)
}
