package vfs

import agentic "github.com/wow-look-at-my/agentic-loop/src"

// What the model is told about each file tool. Every string here is contract:
// they are the whole of what a model knows about the filesystem before it uses
// one, and the host appends only its own MountsBlurb and per-tool Notes.

const (
	listDirDescription  = "Lists a directory."
	readFileDescription = "Reads a file's contents. Binary files are reported as a placeholder, and long files are truncated. " +
		"Give \"offset\" and \"limit\" to read a window of lines rather than the whole file — the line numbers " + GrepToolName + " returns go straight into \"offset\", " +
		"and reading a 200-line window around a hit costs a fraction of what the whole file does."
	findFilesDescription = "Finds files by NAME below a path, recursively — give a filename glob (*.go, test_*) or a plain substring as pattern. " +
		"The path sets the scope exactly as it does for " + GrepToolName + ": a directory, a subdirectory of one — or a single file, which asks whether that one file matches. " +
		"A path the mount does not hold is an error, never an empty result: nothing was listed, so it is not an absence of matching files. " +
		"Searches names and paths only; to search what is INSIDE the files use " + GrepToolName + "."
	grepDescription = "Searches file CONTENTS below a path, recursively — the equivalent of grep -r. " +
		"Works on every path the other file tools do, and the path sets the scope: a directory, a subdirectory of one, or a single file. " +
		"Matching is literal and case-insensitive by default; set \"regexp\" for a pattern or \"case_sensitive\" for an exact match, and narrow with \"glob\". " +
		"Each hit is a path and a line number, ready for " + ReadFileToolName + " and its \"offset\". " +
		"Because every line of every file in scope is read, NO MATCHES really does mean the text is not there — and if coverage was partial the reply says so, so trust the stated scope."
	writeFileDescription  = "Creates a NEW file. Fails when the path already exists — change an existing file with " + EditFileToolName + ", never by rewriting it whole."
	editFileDescription   = "Changes part of an existing file: old_text must match the file's current content exactly and occur exactly once, so read the file first and include enough surrounding context to be unique."
	deleteFileDescription = "Deletes an existing file."
)

// The schemas, each inferred from the struct its handler decodes (schema.go),
// so an argument cannot exist in one and not the other.
var (
	pathOnlySchema = agentic.InferSchema[pathArgs]()
	readSchema     = agentic.InferSchema[readArgs]()
	findSchema     = agentic.InferSchema[findArgs]()
	grepSchema     = agentic.InferSchema[grepArgs]()
	writeSchema    = agentic.InferSchema[writeArgs]()
	editSchema     = agentic.InferSchema[editArgs]()
)
