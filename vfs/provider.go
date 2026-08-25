package vfs

import (
	"context"
	"errors"
	"path"
	"strings"
	"sync"
)

// IProvider is the common base for every registered provider; it lets callers ask its mount path.
type IProvider interface {
	Path() string
}

// BaseProvider is embedded by providers that want Path() handled for them via the registry's setPath.
type BaseProvider struct {
	path string
}

// Path returns the virtual path this provider was registered at.
func (b *BaseProvider) Path() string { return b.path }

// setPath is the registry's internal hook to inject the mount path; unexported so only the registry calls it.
func (b *BaseProvider) setPath(p string) { b.path = p }

// pathSetter is checked by the registry to inject the path at registration.
type pathSetter interface {
	setPath(string)
}

// IFileProvider serves exactly one virtual file at a registered path. Use it
// when a host wants to expose a single document (a generated report, a
// stitched-together brief, a config snapshot) without building a full folder
// hierarchy behind it.
type IFileProvider interface {
	IProvider
	// Read returns the file's contents; path is the whole virtual path as the model wrote it.
	Read(ctx context.Context, virtualPath string) (File, error)
	// Display is the canonical rendering of the file's path for message text.
	Display(virtualPath string) string
}

// IFolderProvider serves a folder hierarchy under a registered path prefix; every error it returns is model-facing.
type IFolderProvider interface {
	IProvider
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
	// Writable reports whether THIS path accepts changes, else the model-facing reason naming what to use instead.
	Writable(path string) (bool, string)
	// Create adds a brand-new file, failing when the path already exists.
	Create(ctx context.Context, path, content string) (string, error)
	// Replace swaps one exact occurrence of oldText for newText.
	Replace(ctx context.Context, path, oldText, newText string) (string, error)
	// Remove deletes an existing file.
	Remove(ctx context.Context, path string) (string, error)
}

// ReadOnlyExplainer lets a read-only folder name the writable route instead of a bare "read-only" refusal.
type ReadOnlyExplainer interface {
	ReadOnlyReason(path string) string
}

// PathGuard vetoes a path before any provider sees it, with a model-facing reason; nil allows everything.
type PathGuard func(path string) (blocked bool, reason string)

// DuplicateMountError is returned when a provider is registered at a path prefix that already has one.
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
	for strings.Contains(p, "//") {
		p = strings.ReplaceAll(p, "//", "/")
	}
	return p
}

func (r *registry) add(prefix string, provider any) error {
	normalized := normalizePrefix(prefix)
	key := strings.ToLower(normalized)
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, exists := r.byKey[key]; exists {
		return &DuplicateMountError{Path: key}
	}
	// Inject the path into providers that support it.
	if ps, ok := provider.(pathSetter); ok {
		ps.setPath(normalized)
	}
	m := &mount{
		prefix:    key,
		displayAs: normalized,
		provider:  provider,
	}
	r.byKey[key] = m
	r.mounts = append(r.mounts, m)
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
			return m
		}
		if p == m.prefix || strings.HasPrefix(p, m.prefix+"/") {
			return m
		}
	}
	return nil
}

// asFolderProvider normalizes a registered provider to an IFolderProvider,
// wrapping an IFileProvider in an adapter so the tool handlers can treat
// every provider uniformly.
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

// fileProviderAdapter wraps an IFileProvider to satisfy the IFolderProvider
// interface.
type fileProviderAdapter struct {
	file IFileProvider
}

func (a *fileProviderAdapter) Path() string            { return a.file.Path() }
func (a *fileProviderAdapter) Display(p string) string { return a.file.Display(p) }

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

// registrySort sorts mounts by prefix depth (longest first), then
// alphabetically for deterministic ordering.
func registrySort(mounts []*mount) {
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

// ErrNoProvider is returned internally when no provider matches a path.
var ErrNoProvider = errors.New("no provider for this path")
