package agentic

import (
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"
)

// memCache is an in-memory RepoKeyCache for tests.
type memCache struct {
	mu sync.Mutex
	m  map[string]string
}

func newMemCache() *memCache { return &memCache{m: map[string]string{}} }

func (c *memCache) Get(repo string) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	v, ok := c.m[repo]
	return v, ok
}

func (c *memCache) Put(repo, id string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.m[repo] = id
}

func TestRepoReadWhatValidation(t *testing.T) {
	ex := repoToolset(GitHubConfig{})
	// Missing "what" teaches the valid set.
	res := execRepoTool(t, ex, RepoReadToolName, repoReadArgs{Org: "octo", Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `requires "what"`)
	assert.Contains(t, res.Content, repoReadWhatList)

	// An unknown "what" names the bad value and the valid set.
	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "wat", Org: "octo"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `unknown what "wat"`)
	assert.Contains(t, res.Content, repoReadWhatList)

	// The reads that became filesystem operations redirect by name rather than
	// reporting a merely unknown what.
	for what, want := range map[string]string{
		"tree": "list_dir", "file": "read_file", "filenames": "find_files",
	} {
		res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: what, Org: "octo", Repo: "hello"})
		assert.True(t, res.IsError, what)
		assert.Contains(t, res.Content, "no longer does what="+what)
		assert.Contains(t, res.Content, want)
	}

	// Per-what arg validation teaches the missing pieces.
	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "pr", Org: "octo", Repo: "hello"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `what=pr requires a positive "number"`)
	res = execRepoTool(t, ex, RepoReadToolName, repoReadArgs{What: "commits"})
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, `repo_read what=commits requires both "org" and "repo"`)
}
