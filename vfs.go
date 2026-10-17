package agentic

import (
	"context"
	"strings"
)

// VFS is a virtual filesystem that the repo read tools can walk. It is how a
// host exposes state that is not GitHub -- diagnostics, workspace metadata, a
// queue, anything -- as files the model can read with the same read_file /
// repo_read tools it already has.
//
// GHResponse is the shared envelope for both GitHub and virtual mounts: it is
// status + body + content-type + truncated, and nothing in it is GitHub-only.
// A virtual mount returns a synthetic GHResponse exactly like a GitHub
// endpoint would.
//
// A VFS is registered under a path prefix. The prefix is stripped before Read
// or List is called, so a mount at "/diag" serves "/diag/workspaces" as
// path "workspaces".
type VFS interface {
	// Read returns the file at path. A 404 (not found) is a normal answer,
	// returned as a GHResponse with Status()==404, not an error. The caller
	// decides how to render it, the same as a GitHub 404.
	Read(ctx context.Context, path string, opt FetchOptions) (GHResponse, error)
	// List returns the directory listing at path.
	List(ctx context.Context, path string, opt FetchOptions) ([]DirEntry, error)
}

// DirEntry is one entry in a VFS directory listing.
type DirEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
}

// VFSMux routes reads by path prefix to registered virtual filesystems. It is
// the single registration point: any code can Mount a VFS, and the read tools
// consult the mux before touching GitHub.
type VFSMux struct {
	mounts []vfsMount
}

type vfsMount struct {
	prefix string // e.g. "/diag"
	vfs    VFS
}

// NewVFSMux returns an empty mux.
func NewVFSMux() *VFSMux { return &VFSMux{} }

// Mount registers a VFS under a path prefix. The prefix must start with "/"
// and not end with "/" (except the root "/"). A later mount with the same
// prefix replaces the earlier one.
func (m *VFSMux) Mount(prefix string, vfs VFS) {
	prefix = "/" + strings.Trim(prefix, "/")
	if prefix == "/" {
		prefix = ""
	}
	m.mounts = append(m.mounts, vfsMount{prefix: prefix, vfs: vfs})
}

// Resolve finds the VFS that owns path, returning the mount's VFS and the
// path relative to it. ok=false when no mount matches (the caller should
// treat path as a real GitHub URL).
func (m *VFSMux) Resolve(path string) (VFS, string, bool) {
	// Longest prefix wins, so "/diag/workspaces" matches "/diag" not "/di".
	best := -1
	var bestVFS VFS
	var bestRest string
	for i, mount := range m.mounts {
		p := mount.prefix
		if p == "" {
			continue // root mount: handle below
		}
		if path == p || strings.HasPrefix(path, p+"/") {
			rest := strings.TrimPrefix(path, p)
			rest = strings.TrimPrefix(rest, "/")
			if len(p) > best {
				best, bestVFS, bestRest = len(p), mount.vfs, rest
			}
		}
	}
	if best >= 0 {
		return bestVFS, bestRest, true
	}
	// A root mount ("/") owns everything not claimed by a longer prefix.
	for _, mount := range m.mounts {
		if mount.prefix == "" {
			return mount.vfs, strings.TrimPrefix(path, "/"), true
		}
	}
	return nil, "", false
}
