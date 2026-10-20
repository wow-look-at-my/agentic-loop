package repo

import (
	"context"
	"net/http"
	"path"
	"regexp"
	"strings"

	"github.com/wow-look-at-my/agentic-loop/vfs"
)

// vfsFolder adapts one registered VFS mount to the folder-provider interface
// the file tools use.
type vfsFolder struct {
	name string
	fs   VFS
}

// NewVFSFolder adapts a VFS mounted at name (without the leading slash) to
// the folder-provider interface.
func NewVFSFolder(name string, fs VFS) vfs.IFolderProvider {
	return &vfsFolder{name: strings.Trim(name, "/"), fs: fs}
}

// Path is the virtual path the mount lives at.
func (f *vfsFolder) Path() string { return "/" + f.name }

func (f *vfsFolder) Display(p string) string { return "/" + f.name + strings.TrimPrefix(p, "/"+f.name) }

func (f *vfsFolder) relative(p string) (string, bool) {
	s := strings.TrimPrefix(strings.TrimSpace(p), "/")
	mount, rest, ok := strings.Cut(s, "/")
	if !ok {
		return "", strings.EqualFold(mount, f.name)
	}
	if !strings.EqualFold(mount, f.name) {
		return "", false
	}
	return strings.TrimPrefix(rest, "/"), true
}

func (f *vfsFolder) List(ctx context.Context, p string) (vfs.Listing, error) {
	rel, ok := f.relative(p)
	if !ok {
		return vfs.Listing{}, &vfsPathError{path: p, status: http.StatusNotFound}
	}
	entries, err := f.fs.List(ctx, rel, FetchOptions{})
	if err != nil {
		return vfs.Listing{}, err
	}
	out := vfs.Listing{Entries: make([]vfs.DirEntry, 0, len(entries))}
	for _, e := range entries {
		out.Entries = append(out.Entries, vfs.DirEntry{Name: e.Name, Dir: e.IsDir})
	}
	return out, nil
}

func (f *vfsFolder) Read(ctx context.Context, p string) (vfs.File, error) {
	rel, ok := f.relative(p)
	if !ok {
		return vfs.File{}, &vfsPathError{path: p, status: http.StatusNotFound}
	}
	res, err := f.fs.Read(ctx, rel, FetchOptions{})
	if err != nil {
		return vfs.File{}, err
	}
	if res.Status() < 200 || res.Status() >= 300 {
		return vfs.File{}, &vfsPathError{path: p, status: res.Status()}
	}
	return vfs.File{Content: string(res.Body()), TruncatedNote: truncationNote(res)}, nil
}

func (f *vfsFolder) Find(ctx context.Context, p, pattern string, limit int) ([]string, error) {
	var out []string
	var walk func(string) error
	walk = func(rel string) error {
		if len(out) >= limit {
			return nil
		}
		entries, err := f.fs.List(ctx, rel, FetchOptions{})
		if err != nil {
			return err
		}
		for _, e := range entries {
			child := joinVFSPath(rel, e.Name)
			if e.IsDir {
				if err := walk(child); err != nil {
					return err
				}
				continue
			}
			if matchFileName(child, pattern) {
				out = append(out, "/"+f.name+"/"+child)
				if len(out) >= limit {
					break
				}
			}
		}
		return nil
	}
	if rel, ok := f.relative(p); ok {
		if err := walk(rel); err != nil {
			return nil, err
		}
		return out, nil
	}
	return nil, &vfsPathError{path: p, status: http.StatusNotFound}
}

func (f *vfsFolder) Grep(ctx context.Context, p string, q vfs.GrepQuery) (vfs.GrepResult, error) {
	hits := vfs.GrepResult{}
	found, err := f.Find(ctx, p, "*", vfs.ListMaxEntries)
	if err != nil {
		return hits, err
	}
	var re *regexp.Regexp
	if q.Regexp {
		flags := ""
		if !q.CaseSensitive {
			flags = "(?i)"
		}
		if re, err = regexp.Compile(flags + q.Pattern); err != nil {
			return hits, err
		}
	}
	for _, filePath := range found {
		if !matchGlobs(filePath, q.Globs) {
			continue
		}
		file, rerr := f.Read(ctx, filePath)
		if rerr != nil {
			return hits, rerr
		}
		matchedFile := false
		for n, line := range strings.Split(file.Content, "\n") {
			matched := false
			if re != nil {
				matched = re.MatchString(line)
			} else if q.CaseSensitive {
				matched = strings.Contains(line, q.Pattern)
			} else {
				matched = strings.Contains(strings.ToLower(line), strings.ToLower(q.Pattern))
			}
			if matched {
				hits.Hits = append(hits.Hits, vfs.GrepHit{Path: filePath, Line: n + 1, Text: line})
				matchedFile = true
				if len(hits.Hits) >= q.MaxHits {
					hits.Truncated = true
					return hits, nil
				}
			}
		}
		if matchedFile {
			hits.Files++
		}
	}
	return hits, nil
}

func joinVFSPath(base, name string) string {
	if base == "" {
		return name
	}
	return path.Join(base, name)
}

func matchFileName(p, pattern string) bool {
	name := path.Base(p)
	if ok, _ := path.Match(pattern, name); ok {
		return true
	}
	ok, _ := path.Match(pattern, p)
	return ok
}

func matchGlobs(p string, globs []string) bool {
	if len(globs) == 0 {
		return true
	}
	for _, glob := range globs {
		if ok, _ := path.Match(glob, path.Base(p)); ok {
			return true
		}
		if ok, _ := path.Match(glob, p); ok {
			return true
		}
	}
	return false
}

func truncationNote(res GHResponse) string {
	if res.Truncated() {
		return "the virtual filesystem truncated this file at its response limit"
	}
	return ""
}

type vfsPathError struct {
	path   string
	status int
}

func (e *vfsPathError) Error() string { return "virtual path not found: " + e.path }
