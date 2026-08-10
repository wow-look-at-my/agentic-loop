package agentic

import (
	"context"
	"strings"
)

// The filesystem tools: one vocabulary for reading and writing files, whatever
// is behind them. A host mounts FOLDERS under virtual path prefixes --
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
// The module is optional: a host that mounts no folders gets no file tools.

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

// File is a file's contents as served by a folder.
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

// Folder serves one virtual path prefix. Every method receives the WHOLE
// virtual path as the model wrote it, because only the folder knows its own
// grammar -- a repository host's `/repos/<org>/<repo>@<ref>/<path>` is its
// business, not the tool layer's.
//
// Every error is model-facing: the tool layer renders it as a recoverable
// teaching error, never a failed turn.
type Folder interface {
	// Display is the canonical rendering of a path, for message text.
	Display(path string) string
	List(ctx context.Context, path string) (Listing, error)
	Read(ctx context.Context, path string) (File, error)
	Find(ctx context.Context, path, pattern string, limit int) ([]string, error)
	// Grep searches file CONTENTS below path. Scope comes from the path,
	// exactly as it does for Find: a directory searches that subtree, and a
	// single FILE is a scope too -- rendering one as a directory makes every
	// file-scoped search answer "no matches" for text right there.
	Grep(ctx context.Context, path string, q GrepQuery) (GrepResult, error)
}

// WritableFolder is a folder that accepts changes. A folder that does not
// implement it is read-only, and the tool layer says so -- with the folder's
// own words when it implements ReadOnlyExplainer.
type WritableFolder interface {
	Folder
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

// PathGuard vetoes a path before any folder sees it, with the model-facing
// reason. It is the seam a host uses to redirect one mount to another (reading
// an attached working copy's own repository through the read-only mount would
// show the un-staged remote state). A nil guard allows everything.
type PathGuard func(path string) (blocked bool, reason string)

// FileToolsConfig configures NewFileTools.
type FileToolsConfig struct {
	// Folders are the mounts, keyed by their leading path segment WITHOUT the
	// slash ("repos", "workspace"). An empty map yields no tools at all.
	Folders map[string]Folder
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
	// Guard vetoes paths before the folder sees them.
	Guard PathGuard
}

// files is the shared state behind the seven tools.
type files struct {
	folders     map[string]Folder
	unavailable func(string) string
	guard       PathGuard
}

// NewFileTools builds the filesystem tools over cfg.Folders, or returns nil
// when nothing is mounted -- a run with no files is never offered a tool that
// could only ever fail.
func NewFileTools(cfg FileToolsConfig) Tools {
	mounted := map[string]Folder{}
	for name, f := range cfg.Folders {
		if name != "" && f != nil {
			mounted[name] = f
		}
	}
	if len(mounted) == 0 {
		return nil
	}
	e := &files{folders: mounted, unavailable: cfg.Unavailable, guard: cfg.Guard}
	describe := func(name, base string) string {
		return base + sentence(cfg.MountsBlurb) + sentence(cfg.Notes[name])
	}
	return Tools{
		NewTool(ToolDecl{Name: ListDirToolName, Description: describe(ListDirToolName, listDirDescription), InputSchema: pathOnlySchema, Readonly: true}, e.listDir),
		NewTool(ToolDecl{Name: ReadFileToolName, Description: describe(ReadFileToolName, readFileDescription), InputSchema: readSchema, Readonly: true}, e.readFile),
		NewTool(ToolDecl{Name: FindFilesToolName, Description: describe(FindFilesToolName, findFilesDescription), InputSchema: findSchema, Readonly: true}, e.findFiles),
		NewTool(ToolDecl{Name: GrepToolName, Description: describe(GrepToolName, grepDescription), InputSchema: grepSchema, Readonly: true}, e.grep),
		NewTool(ToolDecl{Name: WriteFileToolName, Description: describe(WriteFileToolName, writeFileDescription), InputSchema: writeSchema}, e.writeFile),
		NewTool(ToolDecl{Name: EditFileToolName, Description: describe(EditFileToolName, editFileDescription), InputSchema: editSchema}, e.editFile),
		NewTool(ToolDecl{Name: DeleteFileToolName, Description: describe(DeleteFileToolName, deleteFileDescription), InputSchema: pathOnlySchema}, e.deleteFile),
	}
}

// sentence prepares a host addendum for appending: nothing for an empty one,
// and a separating space otherwise.
func sentence(s string) string {
	if s = strings.TrimSpace(s); s == "" {
		return ""
	}
	return " " + s
}
