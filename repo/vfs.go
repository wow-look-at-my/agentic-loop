package repo

import (
	"context"
	"strings"
)

// VFS is a virtual filesystem the repo read tools can walk.
type VFS interface {
	// Read returns the file at path.
	Read(ctx context.Context, path string, opt FetchOptions) (GHResponse, error)
	// List returns the directory listing at path.
	List(ctx context.Context, path string, opt FetchOptions) ([]DirEntry, error)
}

// DirEntry is one entry in a VFS directory listing.
type DirEntry struct {
	Name  string `json:"name"`
	IsDir bool   `json:"is_dir"`
}

// VFSMux routes reads by path prefix to registered virtual filesystems.
type VFSMux struct {
	mounts []vfsMount
}

type vfsMount struct {
	prefix string // e.g. "/diag"
	fs     VFS
}

// NewVFSMux returns an empty mux.
func NewVFSMux() *VFSMux { return &VFSMux{} }

// Mount registers a VFS under a path prefix. The prefix must start with "/"
// and not end with "/" (except the root "/"). A later mount with the same
// prefix replaces the earlier one.
func (m *VFSMux) Mount(prefix string, fs VFS) {
	prefix = "/" + strings.Trim(prefix, "/")
	if prefix == "/" {
		prefix = ""
	}
	m.mounts = append(m.mounts, vfsMount{prefix: prefix, fs: fs})
}

// Resolve finds the VFS that owns path, returning the mount's VFS and the path
// relative to it. ok=false when no mount matches (the caller should treat path
// as a real GitHub URL).
func (m *VFSMux) Resolve(path string) (VFS, string, bool) {
	// Longest prefix wins, so "/diag/workspaces" matches "/diag" not "/di".
	best := -1
	var bestVFS VFS
	var bestRest string
	for _, mount := range m.mounts {
		p := mount.prefix
		if p == "" {
			continue // root mount: handled below
		}
		if path == p || strings.HasPrefix(path, p+"/") {
			rest := strings.TrimPrefix(path, p)
			rest = strings.TrimPrefix(rest, "/")
			if len(p) > best {
				best, bestVFS, bestRest = len(p), mount.fs, rest
			}
		}
	}
	if best >= 0 {
		return bestVFS, bestRest, true
	}
	// A root mount ("/") owns everything not claimed by a longer prefix.
	for _, mount := range m.mounts {
		if mount.prefix == "" {
			return mount.fs, strings.TrimPrefix(path, "/"), true
		}
	}
	return nil, "", false
}
