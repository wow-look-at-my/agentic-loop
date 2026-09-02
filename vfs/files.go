package vfs

import (
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
)

// The filesystem tools: vocabulary for reading and writing files behind mounted providers.

// The advertised tool names.
const (
	ListDirToolName    = "list_dir"
	ReadFileToolName   = "read_file"
	FindFilesToolName  = "find_files"
	GrepToolName       = "grep"
	WriteFileToolName  = "write_file"
	EditFileToolName   = "edit_file"
	DeleteFileToolName = "delete_file"
)

// The result caps. They bound what call can put in the model's context;
// each is announced when it bites, never applied silently.
const (
	// FindDefaultLimit and FindMaxLimit bound a find_files result.
	FindDefaultLimit = 20
	FindMaxLimit     = 100
	// GrepDefaultLimit and GrepMaxLimit bound a grep result.
	GrepDefaultLimit = 30
	GrepMaxLimit     = 100
	// ListMaxEntries caps directory listing fed to the model.
	ListMaxEntries = 1000
)

// DirEntry is entry in a directory listing.
type DirEntry struct {
	Name string
	Dir  bool
	Size int64
	// Note annotates the entry, e.g. a repository's visibility or a staged file's state.
	Note string
	// Kind overrides the rendered type column for non-file, non-directory entries (symlink, submodule).
	Kind string
}

// Listing is directory's contents plus an optional header annotation.
type Listing struct {
	Entries []DirEntry
	Note    string
	// Truncated marks a listing the folder itself cut short.
	Truncated bool
}

// File is a file's contents as served by a provider.
type File struct {
	Content string
	// Note annotates the header line, e.g. "staged: modified".
	Note string
	// TruncatedNote, when non-empty, states the folder served less than the whole file and by how much.
	TruncatedNote string
}

// GrepQuery is content search.
type GrepQuery struct {
	// Pattern is matched literally unless Regexp is set.
	Pattern string
	Regexp  bool
	// CaseSensitive matches exactly; the default folds case.
	CaseSensitive bool
	// Globs restrict which filenames are searched, e.g. "*.go".
	Globs []string
	// MaxHits bounds the matching lines returned.
	MaxHits int
}

// GrepHit is matching line.
type GrepHit struct {
	// Path is the full virtual path, ready to hand back to read_file.
	Path string
	Line int
	Text string
}

// GrepResult is a content search's outcome.
type GrepResult struct {
	Hits []GrepHit
	// Files is how many distinct files matched.
	Files int
	// Truncated reports the hit cap was reached, so unlisted matches exist.
	Truncated bool
	// Note carries anything else needed to read the result, e.g. when the search covered only part of the path.
	Note string
}

// FileToolsConfig configures NewFileTools.
type FileToolsConfig struct {
	// Providers are the mounts keyed by virtual path prefix (e.g. "/repos"); an empty map yields unavailable tools.
	Providers map[string]any
	// MountsBlurb is the host's sentence naming its mounts, appended to every tool description.
	MountsBlurb string
	// Notes hold per-tool facts about the host's mounts, appended to that tool's description.
	Notes map[string]string
	// Unavailable explains a mount this run does not serve; it receives the mount name.
	Unavailable func(mount string) string
	// Guard vetoes paths before the provider sees them.
	Guard PathGuard
}

// files is the shared state behind the tools.
type files struct {
	registry    *registry
	unavailable func(string) string
	guard       PathGuard
}

// FileTools is a handle returned by NewFileTools that provides the
type FileTools struct {
	*files
	tools agentic.Tools
}

// NewFileTools builds the filesystem tools over cfg.Providers. Returns a
// non-nil *FileTools even when no providers are mounted — tool calls return
// the unavailable message in that case.
func NewFileTools(cfg FileToolsConfig) *FileTools {
	reg := newRegistry()
	for prefix, p := range cfg.Providers {
		if prefix != "" && p != nil {
			_ = reg.add(prefix, p) // ignore error in constructor; duplicates are unlikely at init
		}
	}
	e := &files{registry: reg, unavailable: cfg.Unavailable, guard: cfg.Guard}
	describe := func(name, base string) string {
		return base + sentence(cfg.MountsBlurb) + sentence(cfg.Notes[name])
	}
	tools := agentic.Tools{
		agentic.NewTool(agentic.ToolDecl{Name: ListDirToolName, Description: describe(ListDirToolName, listDirDescription), InputSchema: pathOnlySchema, Readonly: true, OpenWorld: agentic.Bool(false)}, e.listDir),
		agentic.NewTool(agentic.ToolDecl{Name: ReadFileToolName, Description: describe(ReadFileToolName, readFileDescription), InputSchema: readSchema, Readonly: true, OpenWorld: agentic.Bool(false)}, e.readFile),
		agentic.NewTool(agentic.ToolDecl{Name: FindFilesToolName, Description: describe(FindFilesToolName, findFilesDescription), InputSchema: findSchema, Readonly: true, OpenWorld: agentic.Bool(false)}, e.findFiles),
		agentic.NewTool(agentic.ToolDecl{Name: GrepToolName, Description: describe(GrepToolName, grepDescription), InputSchema: grepSchema, Readonly: true, OpenWorld: agentic.Bool(false)}, e.grep),
		agentic.NewTool(agentic.ToolDecl{Name: WriteFileToolName, Description: describe(WriteFileToolName, writeFileDescription), InputSchema: writeSchema, Destructive: agentic.Bool(true), Idempotent: true, OpenWorld: agentic.Bool(false)}, e.writeFile),
		agentic.NewTool(agentic.ToolDecl{Name: EditFileToolName, Description: describe(EditFileToolName, editFileDescription), InputSchema: editSchema, Destructive: agentic.Bool(true), OpenWorld: agentic.Bool(false)}, e.editFile),
		agentic.NewTool(agentic.ToolDecl{Name: DeleteFileToolName, Description: describe(DeleteFileToolName, deleteFileDescription), InputSchema: pathOnlySchema, Destructive: agentic.Bool(true), Idempotent: true, OpenWorld: agentic.Bool(false)}, e.deleteFile),
	}
	return &FileTools{files: e, tools: tools}
}

// Tools returns the file tools. Safe for use in agentic.Config while
// mutating the provider set concurrently.
func (ft *FileTools) Tools() agentic.Tools {
	return ft.tools
}

// Add registers an IFolderProvider at the given path prefix, erroring loudly on a duplicate mount.
func (ft *FileTools) Add(prefix string, p IFolderProvider) error {
	if prefix == "" || p == nil {
		return nil
	}
	return ft.files.registry.add(prefix, p)
}

// AddFile registers an IFileProvider at the given path. Returns a loud error
// if a provider is already mounted at that path (case-insensitive).
func (ft *FileTools) AddFile(prefix string, p IFileProvider) error {
	if prefix == "" || p == nil {
		return nil
	}
	return ft.files.registry.add(prefix, p)
}

// Remove removes the provider at the given path prefix; later calls there return the unavailable message.
func (ft *FileTools) Remove(prefix string) {
	ft.files.registry.remove(prefix)
}

// sentence prepares a host addendum for appending: nothing for an empty,
// and a separating space otherwise.
func sentence(s string) string {
	if s = strings.TrimSpace(s); s == "" {
		return ""
	}
	return " " + s
}
