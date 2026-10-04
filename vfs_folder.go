package agentic

import (
	"context"
	"net/http"
	"path"
	"regexp"
	"strings"
)

// vfsFolder adapts one registered VFS mount to the generic Folder interface
// used by the file tools. The VFS itself only needs to implement read and
// directory listing; find_files and grep are provided generically by walking
// the mounted tree.
type vfsFolder struct {
	name string
	vfs  VFS
}

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

func (f *vfsFolder) List(ctx context.Context, p string) (Listing, error) {
	rel, ok := f.relative(p)
	if !ok {
		return Listing{}, &vfsPathError{path: p, status: http.StatusNotFound}
	}
	entries, err := f.vfs.List(ctx, rel, FetchOptions{})
	if err != nil {
		return Listing{}, err
	}
	out := Listing{Entries: make([]DirEntry, 0, len(entries))}
	for _, e := range entries {
		out.Entries = append(out.Entries, DirEntry{Name: e.Name, Dir: e.IsDir})
	}
	return out, nil
}

func (f *vfsFolder) Read(ctx context.Context, p string) (File, error) {
	rel, ok := f.relative(p)
	if !ok {
		return File{}, &vfsPathError{path: p, status: http.StatusNotFound}
	}
	res, err := f.vfs.Read(ctx, rel, FetchOptions{})
	if err != nil {
		return File{}, err
	}
	if res.Status() < 200 || res.Status() >= 300 {
		return File{}, &vfsPathError{path: p, status: res.Status()}
	}
	return File{Content: string(res.Body()), TruncatedNote: truncationNote(res)}, nil
}

func (f *vfsFolder) Find(ctx context.Context, p, pattern string, limit int) ([]string, error) {
	var out []string
	var walk func(string) error
	walk = func(rel string) error {
		if len(out) >= limit {
			return nil
		}
		entries, err := f.vfs.List(ctx, rel, FetchOptions{})
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

func (f *vfsFolder) Grep(ctx context.Context, p string, q GrepQuery) (GrepResult, error) {
	hits := GrepResult{}
	files, err := f.Find(ctx, p, "*", ListMaxEntries)
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
	for _, filePath := range files {
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
				hits.Hits = append(hits.Hits, GrepHit{Path: filePath, Line: n + 1, Text: line})
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
