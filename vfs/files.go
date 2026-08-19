package vfs

import (
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strings"
)

// The filesystem tools: one vocabulary for reading and writing files, whatever
// is behind them. A host mounts providers under virtual path prefixes --
//
//	/repos/...        a repository host, read-only
//	/workspace/...    an editable working copy
//	/attachments/...  files a user uploaded
//
// -- and the model addresses every one of them with the same seven tools. The
// library owns what a file tool IS: the names, the model-facing descriptions,
// the argument schemas, the caps, and every word of the rendering. A host owns
// what is behind a mount, and nothing about its storage reaches here.
//
// The module is optional: a host that mounts no providers gets non-nil tools
// that return the unavailable message. Providers can be added or removed at
// runtime via the Add/AddFile/Remove methods on the returned *FileTools.
// More specific path prefixes always shadow less specific ones, regardless
// of registration order.

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

// The result caps. They bound what one call can put in the model's context;
// each one is announced when it bites, never applied silently.
const (
	// FindDefaultLimit and FindMaxLimit bound a find_files result.
	FindDefaultLimit = 20
	FindMaxLimit     = 100
	// GrepDefaultLimit and GrepMaxLimit bound a grep result.
	GrepDefaultLimit = 30
	GrepMaxLimit     = 100
	// ListMaxEntries caps one directory listing fed to the model.
	ListMaxEntries = 1000
)

// DirEntry is one entry in a directory listing.
type DirEntry struct {
	Name string
	Dir  bool
	Size int64
	// Note annotates the entry, e.g. a repository's visibility or a staged
	// file's state.
	Note string
	// Kind overrides the rendered type column for entries that are neither a
	// plain file nor a directory (symlink, submodule).
	Kind string
}

// Listing is one directory's contents plus an optional header annotation.
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
	// TruncatedNote, when non-empty, states that the folder served less than
	// the whole file and by how much. It is separate from read_file's own line
	// window, because merging the two would misstate what the line numbers are
	// relative to.
	TruncatedNote string
}

// GrepQuery is one content search.
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

// GrepHit is one matching line.
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
	// Truncated reports that the hit cap was reached, so matches exist that are
	// not listed. Never leave this unsaid.
	Truncated bool
	// Note carries anything else the reader must know to read the result
	// correctly -- most importantly, when the search covered only part of what
	// the path named. A partial search that looks complete is worse than no
	// search, so a folder that could not cover its scope says so here.
	Note string
}

// FileToolsConfig configures NewFileTools.
type FileToolsConfig struct {
	// Providers are the mounts, keyed by their virtual path prefix
	// (e.g. "/repos", "/workspace"). Matching is case-insensitive but
	// preserves the original casing for display. An empty map yields non-nil
	// tools that return the unavailable message. Values are IFolderProvider
	// or IFileProvider.
	Providers map[string]any
	// MountsBlurb is appended to every tool description: the host's own
	// sentence naming what its mounts are, since the library cannot know.
	MountsBlurb string
	// Notes are appended to ONE tool's description, keyed by tool name, for
	// facts about the host's mounts that only that tool needs -- that its grep
	// searches a local copy and so is unmetered, that a write stages locally
	// and must never be reported as pushed. MountsBlurb is what every tool
	// gets; this is what one gets, and the split is what keeps a write's
	// warning out of every read's description.
	Notes map[string]string
	// Unavailable explains a mount the model named that this run does not
	// serve. It receives the mount name; a nil func gets a plain default.
	Unavailable func(mount string) string
	// Guard vetoes paths before the provider sees them.
	Guard PathGuard
}

// files is the shared state behind the seven tools.
type files struct {
	registry    *registry
	unavailable func(string) string
	guard       PathGuard
}

// FileTools is a handle returned by NewFileTools that provides the seven
// file tools and allows runtime mutation of the provider set.
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
		agentic.NewTool(agentic.ToolDecl{Name: ListDirToolName, Description: describe(ListDirToolName, listDirDescription), InputSchema: pathOnlySchema, Readonly: true}, e.listDir),
		agentic.NewTool(agentic.ToolDecl{Name: ReadFileToolName, Description: describe(ReadFileToolName, readFileDescription), InputSchema: readSchema, Readonly: true}, e.readFile),
		agentic.NewTool(agentic.ToolDecl{Name: FindFilesToolName, Description: describe(FindFilesToolName, findFilesDescription), InputSchema: findSchema, Readonly: true}, e.findFiles),
		agentic.NewTool(agentic.ToolDecl{Name: GrepToolName, Description: describe(GrepToolName, grepDescription), InputSchema: grepSchema, Readonly: true}, e.grep),
		agentic.NewTool(agentic.ToolDecl{Name: WriteFileToolName, Description: describe(WriteFileToolName, writeFileDescription), InputSchema: writeSchema}, e.writeFile),
		agentic.NewTool(agentic.ToolDecl{Name: EditFileToolName, Description: describe(EditFileToolName, editFileDescription), InputSchema: editSchema}, e.editFile),
		agentic.NewTool(agentic.ToolDecl{Name: DeleteFileToolName, Description: describe(DeleteFileToolName, deleteFileDescription), InputSchema: pathOnlySchema}, e.deleteFile),
	}
	return &FileTools{files: e, tools: tools}
}

// Tools returns the seven file tools. Safe for use in agentic.Config while
// mutating the provider set concurrently.
func (ft *FileTools) Tools() agentic.Tools {
	return ft.tools
}

// Add registers an IFolderProvider at the given path prefix. Returns a loud
// error if a provider is already mounted at that prefix (case-insensitive).
// More specific prefixes shadow less specific ones regardless of registration
// order.
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

// Remove removes the provider registered at the given path prefix.
// Subsequent tool calls for that path will return the unavailable message.
func (ft *FileTools) Remove(prefix string) {
	ft.files.registry.remove(prefix)
}

// sentence prepares a host addendum for appending: nothing for an empty one,
// and a separating space otherwise.
func sentence(s string) string {
	if s = strings.TrimSpace(s); s == "" {
		return ""
	}
	return " " + s
}
