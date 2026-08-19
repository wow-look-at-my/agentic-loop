package vfs

import (
	"context"
	"errors"
	"path"
	"strings"
	"sync"
)

// IFileProvider serves exactly one virtual file at a registered path. Use it
// when a host wants to expose a single document (a generated report, a
// stitched-together brief, a config snapshot) without building a full folder
// hierarchy behind it. The provider is responsible only for the file's
// contents and its display rendering; the tool layer handles everything
// else (listing, finding, grepping) by treating it as a one-file folder.
type IFileProvider interface {
	// Read returns the file's contents. The path argument is the whole virtual
	// path as the model wrote it.
	Read(ctx context.Context, virtualPath string) (File, error)
	// Display is the canonical rendering of the file's path for message text.
	Display(virtualPath string) string
}

// IFolderProvider serves a folder hierarchy under a registered path prefix.
// Every method receives the WHOLE virtual path as the model wrote it, because
// only the folder knows its own grammar -- a repository host's
// `/repos/<org>/<repo>@<ref>/<path>` is its business, not the tool layer's.
//
// Every error is model-facing: the tool layer renders it as a recoverable
// teaching error, never a failed turn.
type IFolderProvider interface {
	Display(path string) string
	List(ctx context.Context, path string) (Listing, error)
	Read(ctx context.Context, path string) (File, error)
	Find(ctx context.Context, path, pattern string, limit int) ([]string, error)
	Grep(ctx context.Context, path string, q GrepQuery) (GrepResult, error)
}

// IWritableFolderProvider is a folder that accepts changes. A folder that does
// not implement it is read-only, and the tool layer says so -- with the
// folder's own words when it implements ReadOnlyExplainer.
type IWritableFolderProvider interface {
	IFolderProvider
	// Writable reports whether THIS path accepts changes, and when it does
	// not, the model-facing reason (which should name what to use instead). A
	// read-only VIEW of a writable folder, and a path that is a directory,
	// both answer false here.
	Writable(path string) (bool, string)
	// Create adds a brand-new file, failing when the path already exists.
	Create(ctx context.Context, path, content string) (string, error)
	// Replace swaps one exact occurrence of oldText for newText.
	Replace(ctx context.Context, path, oldText, newText string) (string, error)
	// Remove deletes an existing file.
	Remove(ctx context.Context, path string) (string, error)
}

// ReadOnlyExplainer lets a read-only folder name the writable route instead of
// being refused with a bare "read-only" -- the difference between a model
// retrying the same call and one that goes where it should have.
type ReadOnlyExplainer interface {
	ReadOnlyReason(path string) string
}

// PathGuard vetoes a path before any provider sees it, with the model-facing
// reason. It is the seam a host uses to redirect one mount to another (reading
// an attached working copy's own repository through the read-only mount would
// show the un-staged remote state). A nil guard allows everything.
type PathGuard func(path string) (blocked bool, reason string)

// DuplicateMountError is returned when a provider is registered at a path
// prefix that already has one. It is loud and obvious: the message names both
// the conflicting path and advises the caller to remove the existing mount
// first.
type DuplicateMountError struct {
	Path string
}

func (e *DuplicateMountError) Error() string {
	return "vfs: a provider is already mounted at " + e.Path + "; remove it first or choose a different path"
}

// mount is one registered provider at a path prefix.
type mount struct {
	prefix    string // lowercased, normalized, with leading slash, no trailing
	displayAs string // original casing the host registered with
	provider  any    // IFolderProvider or IFileProvider
}

// registry holds the set of mounted providers, sorted by prefix depth
// (longest first) so resolve can pick the most specific match.
type registry struct {
	mu     sync.RWMutex
	mounts []*mount
	byKey  map[string]*mount // lowercased prefix -> mount, for duplicate detection
}

func newRegistry() *registry {
	return &registry{byKey: map[string]*mount{}}
}

// normalizePrefix cleans a user-supplied path prefix into a canonical form:
// leading slash, no trailing slash, single slashes between segments. The
// empty path becomes "/".
func normalizePrefix(p string) string {
	p = "/" + strings.Trim(strings.TrimSpace(p), "/")
	if p == "/" {
		return "/"
	}
	// Collapse double slashes.
	for strings.Contains(p, "//") {
		p = strings.ReplaceAll(p, "//", "/")
	}
	return p
}

func (r *registry) add(prefix string, provider any) error {
	key := strings.ToLower(normalizePrefix(prefix))
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.byKey[key]; exists {
		return &DuplicateMountError{Path: key}
	}
	m := &mount{
		prefix:    key,
		displayAs: normalizePrefix(prefix),
		provider:  provider,
	}
	r.byKey[key] = m
	r.mounts = append(r.mounts, m)
	// Sort by segment count descending (longest first), then alphabetically.
	// We re-sort on every add; the list is tiny.
	registrySort(r.mounts)
	return nil
}

func (r *registry) remove(prefix string) {
	key := strings.ToLower(normalizePrefix(prefix))
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.byKey[key]; !exists {
		return
	}
	delete(r.byKey, key)
	for i, m := range r.mounts {
		if m.prefix == key {
			r.mounts = append(r.mounts[:i], r.mounts[i+1:]...)
			break
		}
	}
}

// resolve finds the most specific mount whose prefix is an ancestor of (or
// equal to) the given path. Comparison is case-insensitive. Returns nil when
// nothing matches.
func (r *registry) resolve(raw string) *mount {
	p := strings.ToLower(normalizePrefix(raw))
	r.mu.RLock()
	defer r.mu.RUnlock()
	for _, m := range r.mounts {
		if m.prefix == "/" {
			return m // root catches everything
		}
		if p == m.prefix || strings.HasPrefix(p, m.prefix+"/") {
			return m
		}
	}
	return nil
}

func (r *registry) empty() bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.mounts) == 0
}

// registrySort sorts mounts by prefix depth (longest first), then
// alphabetically for deterministic ordering.
func registrySort(mounts []*mount) {
	// Simple insertion sort — the list is small (a handful of mounts).
	for i := 1; i < len(mounts); i++ {
		for j := i; j > 0; j-- {
			if mountDepth(mounts[j].prefix) > mountDepth(mounts[j-1].prefix) {
				mounts[j], mounts[j-1] = mounts[j-1], mounts[j]
			} else if mountDepth(mounts[j].prefix) == mountDepth(mounts[j-1].prefix) && mounts[j].prefix < mounts[j-1].prefix {
				mounts[j], mounts[j-1] = mounts[j-1], mounts[j]
			} else {
				break
			}
		}
	}
}

func mountDepth(prefix string) int {
	if prefix == "/" {
		return 0
	}
	return strings.Count(prefix, "/") + 1
}

// fileProviderAdapter wraps an IFileProvider to satisfy the IFolderProvider
// interface, so the tool handlers can treat it uniformly.
type fileProviderAdapter struct {
	file IFileProvider
}

func (a *fileProviderAdapter) Display(p string) string {
	return a.file.Display(p)
}

func (a *fileProviderAdapter) List(_ context.Context, p string) (Listing, error) {
	return Listing{Entries: []DirEntry{{Name: path.Base(p), Size: 0}}}, nil
}

func (a *fileProviderAdapter) Read(ctx context.Context, p string) (File, error) {
	return a.file.Read(ctx, p)
}

func (a *fileProviderAdapter) Find(_ context.Context, p, pattern string, _ int) ([]string, error) {
	if MatchesPattern(p, pattern) {
		return []string{p}, nil
	}
	return nil, nil
}

func (a *fileProviderAdapter) Grep(ctx context.Context, p string, q GrepQuery) (GrepResult, error) {
	f, err := a.file.Read(ctx, p)
	if err != nil {
		return GrepResult{}, err
	}
	var res GrepResult
	for i, line := range strings.Split(f.Content, "\n") {
		if !strings.Contains(strings.ToLower(line), strings.ToLower(q.Pattern)) {
			continue
		}
		if len(res.Hits) == q.MaxHits {
			res.Truncated = true
			break
		}
		res.Hits = append(res.Hits, GrepHit{Path: p, Line: i + 1, Text: line})
	}
	res.Files = len(res.Hits)
	return res, nil
}

// asFolderProvider normalizes a registered provider to an IFolderProvider,
// wrapping IFileProvider in an adapter.
func asFolderProvider(provider any) IFolderProvider {
	switch v := provider.(type) {
	case IFolderProvider:
		return v
	case IFileProvider:
		return &fileProviderAdapter{file: v}
	}
	return nil
}

// asWritableProvider returns an IWritableFolderProvider if the provider
// implements it, or nil.
func asWritableProvider(provider any) IWritableFolderProvider {
	if w, ok := provider.(IWritableFolderProvider); ok {
		return w
	}
	return nil
}

// ErrNoProvider is returned internally when no provider matches a path.
var ErrNoProvider = errors.New("no provider for this path")
