package vfs

import (
	"context"
	"encoding/json"
	"errors"
<<<<<<< HEAD:go/files_test.go
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/wow-look-at-my/go-containers/set"
=======
	agentic "github.com/wow-look-at-my/agentic-loop/src"
>>>>>>> 3e4b846 (Move the library to src/ and split the virtual filesystem into vfs.):src/vfs/files_test.go
	"maps"
	"slices"
	"strings"
	"testing"
)

// memFolder is an in-memory folder: a flat map of relative path -> content,
// mounted under one prefix. It is deliberately not a tree -- the tools address
// paths, and a folder decides for itself what "below" means.
type memFolder struct {
	mount    string
	files    map[string]string
	readonly bool
	// listErr, when set, is what every operation fails with.
	err error
}

func (f *memFolder) rel(p string) string {
	return strings.TrimPrefix(strings.TrimPrefix(p, "/"+f.mount), "/")
}

func (f *memFolder) Display(p string) string {
	rel := f.rel(p)
	if rel == "" {
		return "/" + f.mount
	}
	return "/" + f.mount + "/" + rel
}

func (f *memFolder) List(_ context.Context, p string) (Listing, error) {
	if f.err != nil {
		return Listing{}, f.err
	}
	dir := f.rel(p)
	var out Listing
	seen := set.New[string]()
	for name, content := range f.files {
		rest, ok := WithinDir(name, dir)
		if !ok {
			continue
		}
		if i := strings.Index(rest, "/"); i >= 0 {
			if child := rest[:i]; seen.Add(child) {
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
			out = append(out, "/"+f.mount+"/"+name)
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
	files := set.New[string]()
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
			files.Add(name)
			res.Hits = append(res.Hits, GrepHit{Path: "/" + f.mount + "/" + name, Line: i + 1, Text: line})
		}
	}
	res.Files = files.Len()
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

func fileRig() (agentic.Tools, *writableFolder) {
	work := &writableFolder{&memFolder{mount: "work", files: map[string]string{
		"main.go":     "package main\n\nfunc main() {}\n",
		"src/util.go": "package src\n\n// TODO: rename\nfunc Util() {}\n",
		"README.md":   "# hello\n",
	}}}
	repos := &readOnlyRepos{&memFolder{mount: "repos", files: map[string]string{"a/b.go": "package b\n"}}}
	return NewFileTools(FileToolsConfig{
		Folders:     map[string]Folder{"work": work, "repos": repos},
		MountsBlurb: "/work is editable; /repos is read-only.",
		Notes:       map[string]string{WriteFileToolName: "Writes stage locally."},
		Unavailable: func(m string) string {
			if m == "attachments" {
				return "no files are attached to this conversation."
			}
			return ""
		},
	}), work
}

func runFileTool(t *testing.T, reg agentic.Tools, name, args string) agentic.ToolResult {
	t.Helper()
	tool, ok := reg.Find(name)
	require.True(t, ok, "tool %s must be advertised", name)
	res, err := tool.Execute(context.Background(), []byte(args))
	require.NoError(t, err)
	return res
}

// The seven tools are advertised together, the reads marked read-only so a
// sub-agent gets them and the writes withheld unless explicitly granted.
func TestFileToolsAdvertisedSurface(t *testing.T) {
	reg, _ := fileRig()
	assert.Equal(t, []string{
		ListDirToolName, ReadFileToolName, FindFilesToolName, GrepToolName,
		WriteFileToolName, EditFileToolName, DeleteFileToolName,
	}, reg.Names())

	readonly := map[string]bool{}
	descriptions := map[string]string{}
	for _, d := range reg.Decls() {
		readonly[d.Name] = d.Readonly
		descriptions[d.Name] = d.Description
		assert.Contains(t, d.Description, "/work is editable", "the host's mounts blurb rides every description")
	}
	// A per-tool note reaches ITS tool and no other: a write's warning has no
	// business in every read's description.
	assert.Contains(t, descriptions[WriteFileToolName], "Writes stage locally.")
	assert.NotContains(t, descriptions[ReadFileToolName], "Writes stage locally.")
	assert.True(t, readonly[GrepToolName])
	assert.False(t, readonly[WriteFileToolName], "a write must never be in a sub-agent's default toolset")

	// Nothing mounted means no tools at all, rather than tools that can only fail.
	assert.Nil(t, NewFileTools(FileToolsConfig{}))
	assert.Nil(t, NewFileTools(FileToolsConfig{Folders: map[string]Folder{"x": nil}}))
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

// read_file's window is what keeps one function from costing a whole file, and
// it states where to continue.
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

// An empty grep has to state what it PROVES: every line in scope was read, so
// no matches is a real negative rather than a search that gave up.
func TestGrepStatesWhatAnEmptyResultProves(t *testing.T) {
	reg, _ := fileRig()
	hit := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"TODO"}`)
	assert.Contains(t, hit.Content, "1 matching line in 1 file")
	assert.Contains(t, hit.Content, "/work/src/util.go")
	assert.Contains(t, hit.Content, "      3: // TODO: rename")

	none := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"absent"}`)
	assert.Contains(t, none.Content, "no matches.")
	assert.Contains(t, none.Content, "genuinely absent from it — this is a real negative")

	// A single FILE is a scope: rendering one as a directory made every
	// file-scoped search answer "no matches" for text right there.
	one := runFileTool(t, reg, GrepToolName, `{"path":"/work/src/util.go","pattern":"TODO"}`)
	assert.Contains(t, one.Content, "1 matching line")

	blank := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"  "}`)
	assert.True(t, blank.IsError)
	assert.Contains(t, blank.Content, `requires "pattern"`)
}

// A cap that bites is announced, never applied silently.
func TestGrepAndFindAnnounceTheirCaps(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, GrepToolName, `{"path":"/work","pattern":"a","limit":1}`)
	assert.Contains(t, res.Content, "stopped at 1 matching lines — more exist")

	found := runFileTool(t, reg, FindFilesToolName, `{"path":"/work","pattern":"*.go","limit":1}`)
	assert.Contains(t, found.Content, "(first 1; raise limit or narrow the pattern for more)")

	nothing := runFileTool(t, reg, FindFilesToolName, `{"path":"/work","pattern":"*.rs"}`)
	assert.Contains(t, nothing.Content, `No file under /work matches "*.rs".`)
}

// A read-only mount names the writable route: a bare refusal makes a model
// retry the same call.
func TestWritingToAReadOnlyMountNamesTheAlternative(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, WriteFileToolName, `{"path":"/repos/a/new.go","content":"x"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "is read-only. Edit it under /work instead.")

	// A writable folder still refuses a path it says no to.
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

// A mount this run does not serve is explained in the host's words, and a
// folder's own error reaches the model as a teaching error rather than ending
// the turn.
func TestUnavailableMountsAndFolderErrors(t *testing.T) {
	reg, _ := fileRig()
	res := runFileTool(t, reg, ListDirToolName, `{"path":"/attachments/x"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "no files are attached to this conversation.")

	unknown := runFileTool(t, reg, ListDirToolName, `{"path":"/nope/x"}`)
	assert.True(t, unknown.IsError)
	assert.Contains(t, unknown.Content, "/nope is not available in this conversation.")

	empty := runFileTool(t, reg, ListDirToolName, `{"path":"  "}`)
	assert.True(t, empty.IsError)
	assert.Contains(t, empty.Content, `requires "path"`)

	bad := runFileTool(t, reg, ListDirToolName, `{`)
	assert.True(t, bad.IsError)
	assert.Contains(t, bad.Content, "invalid list_dir arguments")

	broken := NewFileTools(FileToolsConfig{Folders: map[string]Folder{
		"work": &memFolder{mount: "work", err: errors.New("disk on fire")},
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

// The guard is how a host redirects one mount to another before the folder
// ever sees the path.
func TestPathGuardVetoesBeforeTheFolder(t *testing.T) {
	reg := NewFileTools(FileToolsConfig{
		Folders: map[string]Folder{"repos": &memFolder{mount: "repos", files: map[string]string{"a/b.go": "x"}}},
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

func TestMountOf(t *testing.T) {
	assert.Equal(t, "repos", MountOf("/repos/octocat/hello/src"))
	assert.Equal(t, "workspace", MountOf("/workspace@base/x"), "a ref rides with the path, not the mount name")
	assert.Equal(t, "workspace", MountOf("/workspace"))
	assert.Equal(t, "", MountOf(""))
}

func TestSplitGlobsAndMatching(t *testing.T) {
	assert.Equal(t, []string{"*.c", "*.h"}, SplitGlobs(" *.c , *.h ,, "))
	assert.Nil(t, SplitGlobs(""))

	assert.True(t, MatchesPattern("src/util.go", "*.go"), "a glob matches the base name")
	assert.True(t, MatchesPattern("src/util.go", "src/*.go"), "and the full path")
	assert.True(t, MatchesPattern("src/Util.go", "util"), "a plain substring folds case")
	assert.False(t, MatchesPattern("src/util.go", "*.rs"))
}

// The seven schemas are hand-written JSON beside the descriptions they belong
// to, so nothing derives them from the structs the handlers decode. This is
// what stands in for that: every advertised property must be a field the
// handler reads, and every required one must be a field it cannot work
// without. An argument the schema omits is one the model never sends; one it
// requires by mistake is a tool the model refuses to call.
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
	reg, _ := fileRig()
	for _, d := range reg.Decls() {
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

// The file tools' schemas come off their argument structs, so this is the one
// check that matters for them: the tool decodes exactly what it advertises.
func TestFileToolSchemasAreInferred(t *testing.T) {
	reg, _ := fileRig()
	tool, ok := reg.Find(GrepToolName)
	require.True(t, ok)
	assert.JSONEq(t, string(agentic.InferSchema[grepArgs]()), string(tool.Decl().InputSchema))
}
