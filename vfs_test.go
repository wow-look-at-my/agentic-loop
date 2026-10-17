package agentic

import (
	"context"
	"net/http"
	"testing"

	"github.com/stretchr/testify/require"
)

// staticVFS is a test VFS serving a fixed set of files.
type staticVFS struct {
	files map[string]string // path -> content
	dirs  map[string][]DirEntry
}

func (s *staticVFS) Read(_ context.Context, path string, _ FetchOptions) (GHResponse, error) {
	if content, ok := s.files[path]; ok {
		return GHResponse{status: http.StatusOK, body: []byte(content), ctype: "application/json"}, nil
	}
	return GHResponse{status: http.StatusNotFound}, nil
}

func (s *staticVFS) List(_ context.Context, path string, _ FetchOptions) ([]DirEntry, error) {
	if entries, ok := s.dirs[path]; ok {
		return entries, nil
	}
	return nil, nil
}

func TestVFSMuxRoutesByLongestPrefix(t *testing.T) {
	mux := NewVFSMux()
	diag := &staticVFS{files: map[string]string{"rate_limit": `{"core":{"remaining":18}}`}}
	deep := &staticVFS{files: map[string]string{"x": "deep"}}
	mux.Mount("/diag", diag)
	mux.Mount("/diag/deep", deep)

	// Exact prefix.
	vfs, rest, ok := mux.Resolve("/diag/rate_limit")
	require.True(t, ok)
	require.Equal(t, diag, vfs)
	require.Equal(t, "rate_limit", rest)

	// Longest prefix wins.
	vfs, rest, ok = mux.Resolve("/diag/deep/x")
	require.True(t, ok)
	require.Equal(t, deep, vfs)
	require.Equal(t, "x", rest)

	// Unregistered path falls through.
	_, _, ok = mux.Resolve("/repos/foo/bar")
	require.False(t, ok)
}

func TestVFSMountAnsweredByFetchURL(t *testing.T) {
	vfs := &staticVFS{files: map[string]string{"rate_limit": `{"core":{"remaining":18}}`}}
	mux := NewVFSMux()
	mux.Mount("/diag", vfs)

	gh := NewGitHub(GitHubConfig{VFS: mux})
	res, err := gh.FetchURL(context.Background(), "", "/diag/rate_limit", "application/vnd.github+json")
	require.NoError(t, err)
	require.Equal(t, http.StatusOK, res.Status())
	require.Equal(t, `{"core":{"remaining":18}}`, string(res.Body()))
}

func TestVFSUnregisteredFallsThroughToGitHub(t *testing.T) {
	// A mux with no matching mount must not intercept: the request goes to
	// GitHub (and here fails because there is no server, not because the VFS
	// claimed it).
	mux := NewVFSMux()
	mux.Mount("/diag", &staticVFS{})
	gh := NewGitHub(GitHubConfig{VFS: mux})
	_, err := gh.FetchURL(context.Background(), "", "/repos/foo/bar", "application/vnd.github+json")
	require.Error(t, err, "a non-VFS path must not be answered by the mux")
}
