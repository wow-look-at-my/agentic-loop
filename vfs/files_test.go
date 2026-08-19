package vfs

import (
	"context"
	"encoding/json"
	"errors"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"maps"
	"slices"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// memFolder is an in-memory folder: a flat map of relative path -> content,
// mounted under one prefix. It embeds BaseProvider so the registry injects
// its path automatically.
type memFolder struct {
	*BaseProvider
	files    map[string]string
	readonly bool
	err      error
}

func newMemFolder(files map[string]string) *memFolder {
	return &memFolder{BaseProvider: &BaseProvider{}, files: files}
}

func (f *memFolder) rel(p string) string {
	prefix := strings.ToLower(f.Path())
	lower := strings.ToLower(p)
	if lower == prefix {
		return ""
	}
	if strings.HasPrefix(lower, prefix+"/") {
		return p[len(prefix)+1:]
	}
	return strings.TrimPrefix(p, "/")
}

func (f *memFolder) Display(p string) string {
	rel := f.rel(p)
	if rel == "" {
		return f.Path()
	}
	return f.Path() + "/" + rel
}

func (f *memFolder) List(_ context.Context, p string) (Listing, error) {
	if f.err != nil {
		return Listing{}, f.err
	}
	dir := f.rel(p)
	var out Listing
	seen := map[string]bool{}
	for name, content := range f.files {
		rest, ok := WithinDir(name, dir)
		if !ok {
			continue
		}
		if i := strings.Index(rest, "/"); i >= 0 {
			if child := rest[:i]; !seen[child] {
				seen[child] = true
				out.Entries = append(out.Entries, DirEntry{Name: child, Dir: true})
			}
			continue
		}
		out.Entries = append(out.Entries, DirEntry{Name: rest, Size: int64(len(content))})
	}
	return out, nil
}

func (f *memFolder) Read(_ context.Context, p string) (File, error) {
	if f.err != nil {
		return File{}, f.err
	}
	content, ok := f.files[f.rel(p)]
	if !ok {
		return File{}, errors.New(f.Display(p) + " does not exist")
	}
	return File{Content: content}, nil
}

func (f *memFolder) Find(_ context.Context, p, pattern string, limit int) ([]string, error) {
	if f.err != nil {
		return nil, f.err
	}
	scope := f.rel(p)
	var out []string
	for name := range f.files {
		if _, ok := WithinScope(name, scope); !ok {
			continue
		}
		if MatchesPattern(name, pattern) {
			out = append(out, f.Path()+"/"+name)
		}
		if len(out) == limit {
			break
		}
	}
	return out, nil
}

func (f *memFolder) Grep(_ context.Context, p string, q GrepQuery) (GrepResult, error) {
	if f.err != nil {
		return GrepResult{}, f.err
	}
	scope := f.rel(p)
	var res GrepResult
	files := map[string]bool{}
	for name, content := range f.files {
		if _, ok := WithinScope(name, scope); !ok {
			continue
		}
		for i, line := range strings.Split(content, "\n") {
			if !strings.Contains(strings.ToLower(line), strings.ToLower(q.Pattern)) {
				continue
			}
			if len(res.Hits) == q.MaxHits {
				res.Truncated = true
				return res, nil
			}
			files[name] = true
			res.Hits = append(res.Hits, GrepHit{Path: f.Path() + "/" + name, Line: i + 1, Text: line})
		}
	}
	res.Files = len(files)
	return res, nil
}

// writableFolder adds the write half.
type writableFolder struct{ *memFolder }

func (f *writableFolder) Writable(p string) (bool, string) {
	if f.rel(p) == "" {
		return false, f.Display(p) + " is a directory, not a file."
	}
	return true, ""
}

func (f *writableFolder) Create(_ context.Context, p, content string) (string, error) {
	rel := f.rel(p)
	if _, exists := f.files[rel]; exists {
		return "", errors.New(rel + " already exists")
	}
	f.files[rel] = content
	return "Created " + rel + ".", nil
}

func (f *writableFolder) Replace(_ context.Context, p, oldText, newText string) (string, error) {
	rel := f.rel(p)
	content, ok := f.files[rel]
	if !ok {
		return "", errors.New(rel + " does not exist")
	}
	if strings.Count(content, oldText) != 1 {
		return "", errors.New("old_text must occur exactly once")
	}
	f.files[rel] = strings.Replace(content, oldText, newText, 1)
	return "Edited " + rel + ".", nil
}

func (f *writableFolder) Remove(_ context.Context, p string) (string, error) {
	rel := f.rel(p)
	if _, ok := f.files[rel]; !ok {
		return "", errors.New(rel + " does not exist")
	}
	delete(f.files, rel)
	return "Deleted " + rel + ".", nil
}

// readOnlyRepos names the writable route instead of refusing bare.
type readOnlyRepos struct{ *memFolder }

func (f *readOnlyRepos) ReadOnlyReason(p string) string {
	return f.Display(p) + " is read-only. Edit it under /work instead."
}

func fileRig() (*FileTools, *writableFolder) {
	work := &writableFolder{newMemFolder(map[string]string{
		"main.go":     "package main\n\nfunc main() {}\n",
		"src/util.go": "package src\n\n// TODO: rename\nfunc Util() {}\n",
		"README.md":   "# hello\n",
	})}
	repos := &readOnlyRepos{newMemFolder(map[string]string{"a/b.go": "package b\n"})}
	return NewFileTools(FileToolsConfig{
		Providers: map[string]any{
			"/work": work,
			"/repos": repos,
		},
		MountsBlurb: "/work is editable; /repos is read-only.",
		Notes:       map[string]string{WriteFileToolName: "Writes stage locally."},
		Unavailable: func(m string) string {
			if strings.Contains(m, "/attachments") {
				return "no files are attached to this conversation."
			}
			return ""
		},
	}), work
}

func runFileTool(t *testing.T, ft *FileTools, name, args string) agentic.ToolResult {
	t.Helper()
	tool, ok := ft.Tools().Find(name)
	require.True(t, ok, "tool %s must be advertised", name)
	res, err := tool.Execute(context.Background(), []byte(args))
	require.NoError(t, err)
	return res
}

// The seven tools are advertised together, the reads marked read-only so a
// sub-agent gets them and the writes withheld unless explicitly granted.
func TestFileToolsAdvertisedSurface(t *testing.T) {
	ft, _ := fileRig()
	assert.Equal(t, []string{
		ListDirToolName, ReadFileToolName, FindFilesToolName, GrepToolName,
		WriteFileToolName, EditFileToolName, DeleteFileToolName,
	}, ft.Tools().Names())

	readonly := map[string]bool{}
	descriptions := map[string]string{}
	for _, d := range ft.Tools().Decls() {
		readonly[d.Name] = d.Readonly
		descriptions[d.Name] = d.Description
		assert.Contains(t, d.Description, "/work is editable", "the host's mounts blurb rides every description")
	}
	assert.Contains(t, descriptions[WriteFileToolName], "Writes stage locally.")
	assert.NotContains(t, descriptions[ReadFileToolName], "Writes stage locally.")
	assert.True(t, readonly[GrepToolName])
	assert.False(t, readonly[WriteFileToolName], "a write must never be in a sub-agent's default toolset")

	// Nothing mounted returns non-nil tools that return the unavailable message.
	ft2 := NewFileTools(FileToolsConfig{})
	assert.NotNil(t, ft2, "empty config should return non-nil FileTools")
	res := runFileTool(t, ft2, ListDirToolName, `{"path":"/anything"}`)
	assert.Contains(t, res.Content, "is not available")

	// Nil provider values are skipped.
	ft3 := NewFileTools(FileToolsConfig{Providers: map[string]any{"/x": nil}})
	assert.NotNil(t, ft3, "nil provider value should still return non-nil FileTools")
}

func TestListDirRendersDirectoriesFirst(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, ListDirToolName, `{"path":"/work"}`)
	assert.False(t, res.IsError)
	lines := strings.Split(res.Content, "\n")
	assert.Equal(t, "/work", lines[0])
	assert.True(t, strings.HasPrefix(lines[1], "dir   src/"), "directories sort first, got %q", lines[1])
	assert.Contains(t, res.Content, "file  README.md (8 B)")

	empty := runFileTool(t, reg, ListDirToolName, `{"path":"/work/nothing"}`)
	assert.Contains(t, empty.Content, "(empty directory)")
}

func TestReadFileWindow(t *testing.T) {
	reg, _ := fileRig()
	whole := runFileTool(t, reg, ReadFileToolName, `{"path":"/work/src/util.go"}`)
	assert.Contains(t, whole.Content, "func Util()")
	assert.NotContains(t, whole.Content, "lines 1-")

	win := runFileTool(t, reg, ReadFileToolName, `{"path":"/work/src/util.go","offset":3,"limit":1}`)
	assert.Contains(t, win.Content, "(lines 3-3 of 4; 1 more follow — re-read with offset 4)")
	assert.Contains(t, win.Content, "// TODO: rename")
	assert.NotContains(t, win.Content, "package src")

	past := runFileTool(t, reg, ReadFileToolName, `{"path":"/work/src/util.go","offset":99}`)
	assert.Contains(t, past.Content, "line 99 is past its end")

	missing := runFileTool(t, reg, ReadFileToolName, `{"path":"/work/nope.go"}`)
	assert.True(t, missing.IsError)
	assert.Contains(t, missing.Content, "does not exist")
}

func TestGrepStatesWhatAnEmptyResultProves(t *testing.T) {
	reg, _ := fileRig()
	hit := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"TODO"}`)
	assert.Contains(t, hit.Content, "1 matching line in 1 file")
	assert.Contains(t, hit.Content, "/work/src/util.go")
	assert.Contains(t, hit.Content, "      3: // TODO: rename")

	none := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"absent"}`)
	assert.Contains(t, none.Content, "no matches.")
	assert.Contains(t, none.Content, "genuinely absent from it — this is a real negative")

	one := runFileTool(t, reg, GrepToolName, `{"path":"/work/src/util.go","pattern":"TODO"}`)
	assert.Contains(t, one.Content, "1 matching line")

	blank := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"  "}`)
	assert.True(t, blank.IsError)
	assert.Contains(t, blank.Content, `requires "pattern"`)
}

func TestGrepAndFindAnnounceTheirCaps(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"a","limit":1}`)
	assert.Contains(t, res.Content, "stopped at 1 matching lines — more exist")

	found := runFileTool(t, reg, FindFilesToolName, `{"path":"/work","pattern":"*.go","limit":1}`)
	assert.Contains(t, found.Content, "(first 1; raise limit or narrow the pattern for more)")

	nothing := runFileTool(t, reg, FindFilesToolName, `{"path":"/work","pattern":"*.rs"}`)
	assert.Contains(t, nothing.Content, `No file under /work matches "*.rs".`)
}

func TestWritingToAReadOnlyMountNamesTheAlternative(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, WriteFileToolName, `{"path":"/repos/a/new.go","content":"x"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is read-only. Edit it under /work instead.")

	dir := runFileTool(t, reg, DeleteFileToolName, `{"path":"/work"}`)
	assert.True(t, dir.IsError)
	assert.Contains(t, dir.Content, "is a directory, not a file.")
}

func TestWriteEditDelete(t *testing.T) {
	reg, work := fileRig()

	created := runFileTool(t, reg, WriteFileToolName, `{"path":"/work/new.txt","content":"hello\n"}`)
	assert.False(t, created.IsError)
	assert.Equal(t, "hello\n", work.files["new.txt"])

	again := runFileTool(t, reg, WriteFileToolName, `{"path":"/work/new.txt","content":"x"}`)
	assert.True(t, again.IsError, "write_file never rewrites an existing file")

	edited := runFileTool(t, reg, EditFileToolName, `{"path":"/work/new.txt","old_text":"hello","new_text":"goodbye"}`)
	assert.False(t, edited.IsError)
	assert.Equal(t, "goodbye\n", work.files["new.txt"])

	noOld := runFileTool(t, reg, EditFileToolName, `{"path":"/work/new.txt","old_text":"","new_text":"x"}`)
	assert.True(t, noOld.IsError)
	assert.Contains(t, noOld.Content, `requires "old_text"`)

	removed := runFileTool(t, reg, DeleteFileToolName, `{"path":"/work/new.txt"}`)
	assert.False(t, removed.IsError)
	_, still := work.files["new.txt"]
	assert.False(t, still)
}

func TestUnavailableMountsAndFolderErrors(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, ListDirToolName, `{"path":"/attachments/x"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "no files are attached to this conversation.")

	unknown := runFileTool(t, reg, ListDirToolName, `{"path":"/nope/x"}`)
	assert.True(t, unknown.IsError)
	assert.Contains(t, unknown.Content, "is not available in this conversation.")

	empty := runFileTool(t, reg, ListDirToolName, `{"path":"  "}`)
	assert.True(t, empty.IsError)
	assert.Contains(t, empty.Content, `requires "path"`)

	bad := runFileTool(t, reg, ListDirToolName, `{`)
	assert.True(t, bad.IsError)
	assert.Contains(t, bad.Content, "invalid list_dir arguments")

	broken := NewFileTools(FileToolsConfig{Providers: map[string]any{
		"/work": newMemFolderWithErr(errors.New("disk on fire")),
	}})
	for _, name := range []string{ListDirToolName, ReadFileToolName} {
		r := runFileTool(t, broken, name, `{"path":"/work/x","pattern":"y"}`)
		assert.True(t, r.IsError)
		assert.Contains(t, r.Content, "disk on fire")
	}
	for _, name := range []string{FindFilesToolName, GrepToolName} {
		r := runFileTool(t, broken, name, `{"path":"/work/x","pattern":"y"}`)
		assert.True(t, r.IsError)
		assert.Contains(t, r.Content, "disk on fire")
	}
}

func newMemFolderWithErr(err error) *memFolder {
	f := newMemFolder(map[string]string{"x": "y"})
	f.err = err
	return f
}

func TestPathGuardVetoesBeforeTheFolder(t *testing.T) {
	reg := NewFileTools(FileToolsConfig{
		Providers: map[string]any{"/repos": newMemFolder(map[string]string{"a/b.go": "x"})},
		Guard: func(p string) (bool, string) {
			if strings.HasPrefix(p, "/repos/a") {
				return true, "that repository is open as a working copy; read it at /work instead."
			}
			return false, ""
		},
	})
	res := runFileTool(t, reg, ReadFileToolName, `{"path":"/repos/a/b.go"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "read it at /work instead.")
}

func TestSplitGlobsAndMatching(t *testing.T) {
	assert.Equal(t, []string{"*.c", "*.h"}, SplitGlobs(" *.c , *.h ,, "))
	assert.Nil(t, SplitGlobs(""))

	assert.True(t, MatchesPattern("src/util.go", "*.go"), "a glob matches the base name")
	assert.True(t, MatchesPattern("src/util.go", "src/*.go"), "and the full path")
	assert.True(t, MatchesPattern("src/Util.go", "util"), "a plain substring folds case")
	assert.False(t, MatchesPattern("src/util.go", "*.rs"))
}

func TestFileToolSchemasMatchWhatTheHandlersDecode(t *testing.T) {
	type schema struct {
		Type                 string                     `json:"type"`
		AdditionalProperties bool                       `json:"additionalProperties"`
		Properties           map[string]json.RawMessage `json:"properties"`
		Required             []string                   `json:"required"`
	}
	want := map[string]struct{ properties, required []string }{
		ListDirToolName:    {[]string{"path"}, []string{"path"}},
		DeleteFileToolName: {[]string{"path"}, []string{"path"}},
		ReadFileToolName:   {[]string{"path", "offset", "limit"}, []string{"path"}},
		FindFilesToolName:  {[]string{"path", "pattern", "limit"}, []string{"path", "pattern"}},
		GrepToolName: {
			[]string{"path", "pattern", "glob", "regexp", "case_sensitive", "limit"},
			[]string{"path", "pattern"},
		},
		WriteFileToolName: {[]string{"path", "content"}, []string{"path", "content"}},
		EditFileToolName:  {[]string{"path", "old_text", "new_text"}, []string{"path", "old_text", "new_text"}},
	}
	ft, _ := fileRig()
	for _, d := range ft.Tools().Decls() {
		t.Run(d.Name, func(t *testing.T) {
			var s schema
			require.NoError(t, json.Unmarshal(d.InputSchema, &s))
			assert.Equal(t, "object", s.Type)
			assert.False(t, s.AdditionalProperties, "an argument the handler cannot read must be refused, not ignored")
			assert.ElementsMatch(t, want[d.Name].properties, slices.Collect(maps.Keys(s.Properties)))
			assert.ElementsMatch(t, want[d.Name].required, s.Required)
			for name, prop := range s.Properties {
				assert.Contains(t, string(prop), `"description"`, "%s.%s tells the model nothing", d.Name, name)
			}
		})
	}
}

func TestFileToolSchemasAreInferred(t *testing.T) {
	ft, _ := fileRig()
	tool, ok := ft.Tools().Find(GrepToolName)
	require.True(t, ok)
	assert.JSONEq(t, string(agentic.InferSchema[grepArgs]()), string(tool.Decl().InputSchema))
}

// --- New tests for path-prefix routing ---

// TestLongestMatchRouting verifies that a more specific mount shadows a less
// specific one regardless of registration order.
func TestLongestMatchRouting(t *testing.T) {
	// Register broad first, then narrow.
	ft := NewFileTools(FileToolsConfig{})
	broad := newMemFolder(map[string]string{"other.go": "broad"})
	narrow := newMemFolder(map[string]string{"file.go": "narrow"})

	require.NoError(t, ft.Add("/repos", broad))
	require.NoError(t, ft.Add("/repos/org/repo", narrow))

	// A path under the narrow mount routes to it.
	res := runFileTool(t, ft, ReadFileToolName, `{"path":"/repos/org/repo/file.go"}`)
	assert.Contains(t, res.Content, "narrow")

	// A path under only the broad mount routes to it.
	res = runFileTool(t, ft, ReadFileToolName, `{"path":"/repos/other.go"}`)
	assert.Contains(t, res.Content, "broad")

	// Now test reversed order: narrow first, then broad.
	ft2 := NewFileTools(FileToolsConfig{})
	broad2 := newMemFolder(map[string]string{"other.go": "broad2"})
	narrow2 := newMemFolder(map[string]string{"file.go": "narrow2"})

	require.NoError(t, ft2.Add("/repos/org/repo", narrow2))
	require.NoError(t, ft2.Add("/repos", broad2))

	res = runFileTool(t, ft2, ReadFileToolName, `{"path":"/repos/org/repo/file.go"}`)
	assert.Contains(t, res.Content, "narrow2")

	res = runFileTool(t, ft2, ReadFileToolName, `{"path":"/repos/other.go"}`)
	assert.Contains(t, res.Content, "broad2")
}

// TestDuplicateRegistrationIsLoudError verifies that registering two providers
// at the same path prefix returns a loud error and does NOT silently replace.
func TestDuplicateRegistrationIsLoudError(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{})
	first := newMemFolder(map[string]string{"a": "first"})
	second := newMemFolder(map[string]string{"a": "second"})

	require.NoError(t, ft.Add("/work", first))

	err := ft.Add("/work", second)
	require.Error(t, err)
	assert.Contains(t, err.Error(), "already mounted at")
	assert.Contains(t, err.Error(), "/work")

	// The first provider is still the one routing.
	res := runFileTool(t, ft, ReadFileToolName, `{"path":"/work/a"}`)
	assert.Contains(t, res.Content, "first")

	// Case-insensitive duplicate is also caught.
	err = ft.Add("/Work", second)
	require.Error(t, err)
}

// TestCaseInsensitiveMatchPreservesDisplayCasing verifies that a provider
// registered at mixed case is matched case-insensitively but retains its
// original casing for display.
func TestCaseInsensitiveMatchPreservesDisplayCasing(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{})
	f := newMemFolder(map[string]string{"file.go": "content"})

	require.NoError(t, ft.Add("/Repos", f))

	// Lowercase query resolves to the mixed-case mount.
	res := runFileTool(t, ft, ReadFileToolName, `{"path":"/repos/file.go"}`)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "content")

	// Mixed-case query also routes to the same provider (the error is
	// about the file, not the mount — proving the provider was found).
	res = runFileTool(t, ft, ReadFileToolName, `{"path":"/REPOS/FILE.GO"}`)
	assert.True(t, res.IsError)
	assert.NotContains(t, res.Content, "not available")
}

// TestIFileProvider verifies that an IFileProvider registered at a path makes
// read_file return the file's content and list_dir report a single file.
func TestIFileProvider(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{})
	fp := &memFileProvider{
		BaseProvider: &BaseProvider{},
		content:      "hello world",
	}

	require.NoError(t, ft.AddFile("/docs/readme", fp))

	// read_file returns the file's content.
	res := runFileTool(t, ft, ReadFileToolName, `{"path":"/docs/readme"}`)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "hello world")

	// list_dir reports a single file at that path.
	res = runFileTool(t, ft, ListDirToolName, `{"path":"/docs/readme"}`)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "readme")
}

// memFileProvider is a minimal IFileProvider for testing.
type memFileProvider struct {
	*BaseProvider
	content string
}

func (m *memFileProvider) Read(_ context.Context, p string) (File, error) {
	return File{Content: m.content}, nil
}

func (m *memFileProvider) Display(p string) string {
	return p
}

// TestRuntimeAddRemove verifies runtime provider mutation.
func TestRuntimeAddRemove(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{
		Unavailable: func(m string) string {
			return m + " is not ready yet."
		},
	})
	require.NotNil(t, ft)

	res := runFileTool(t, ft, ListDirToolName, `{"path":"/work"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is not ready yet.")

	work := newMemFolder(map[string]string{"hello.txt": "world"})
	require.NoError(t, ft.Add("/work", work))

	res = runFileTool(t, ft, ListDirToolName, `{"path":"/work"}`)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "hello.txt")

	// Add ignores nil values.
	require.NoError(t, ft.Add("/bad", nil))
	res = runFileTool(t, ft, ListDirToolName, `{"path":"/bad"}`)
	assert.True(t, res.IsError)

	// Add ignores empty prefix.
	require.NoError(t, ft.Add("", work))

	// Remove the provider; tool calls become unavailable again.
	ft.Remove("/work")
	res = runFileTool(t, ft, ListDirToolName, `{"path":"/work"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is not ready yet")

	// Removing a non-existent prefix is harmless.
	ft.Remove("/nope")

	// Re-add after removal.
	require.NoError(t, ft.Add("/work", work))
	res = runFileTool(t, ft, ReadFileToolName, `{"path":"/work/hello.txt"}`)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "world")
}

// TestProviderPathIsInjected verifies that BaseProvider.Path() returns the
// registered path after Add.
func TestProviderPathIsInjected(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{})
	f := newMemFolder(map[string]string{"a": "b"})

	require.NoError(t, ft.Add("/my/mount", f))
	assert.Equal(t, "/my/mount", f.Path())
}

// TestConcurrentAddRemove verifies the mutex protects the registry.
func TestConcurrentAddRemove(t *testing.T) {
	ft := NewFileTools(FileToolsConfig{})
	done := make(chan struct{})
	go func() {
		defer close(done)
		for i := 0; i < 100; i++ {
			_ = ft.Add("/work", newMemFolder(map[string]string{"a": "b"}))
			ft.Remove("/work")
		}
	}()
	for i := 0; i < 100; i++ {
		runFileTool(t, ft, ListDirToolName, `{"path":"/work"}`)
	}
	<-done
}
