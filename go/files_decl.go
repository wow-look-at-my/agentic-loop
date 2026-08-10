package agentic

import "encoding/json"

// What the model is told about each file tool. Every string here is contract:
// they are the whole of what a model knows about the filesystem before it uses
// one, and the host appends only its own MountsBlurb naming its mounts.

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

// The argument schemas. Each mirrors the struct its handler decodes; a field
// absent from "required" is optional.
var (
	pathOnlySchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Virtual path." }
  },
  "required": ["path"]
}`)

	readSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Virtual path of the file to read." },
    "offset": { "type": "integer", "description": "First line to return, 1-based. Omit to start at the beginning. A grep hit's line number goes here directly." },
    "limit": { "type": "integer", "description": "How many lines to return from offset. Omit for the rest of the file. The reply says how many lines the file has and where to continue." }
  },
  "required": ["path"]
}`)

	findSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Directory to search below, recursively." },
    "pattern": { "type": "string", "description": "Filename glob (*.go, test_*) or a plain substring. Matched against the file NAME and its path, never its contents." },
    "limit": { "type": "integer", "description": "Maximum results to return, 1-100 (default 20). The reply says when it stopped at the cap." }
  },
  "required": ["path", "pattern"]
}`)

	grepSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Where to search, recursively. A directory, a subdirectory of one, or a single file." },
    "pattern": { "type": "string", "description": "The text to find. Matched literally and case-insensitively anywhere in a line unless regexp or case_sensitive say otherwise. Not tokenized: give the literal text you expect to see in the file." },
    "glob": { "type": "string", "description": "Only search files whose name matches this glob, or a comma-separated list of them, e.g. \"*.usf\" or \"*.c,*.h\"." },
    "regexp": { "type": "boolean", "description": "Treat pattern as an extended regular expression instead of literal text. Defaults to false." },
    "case_sensitive": { "type": "boolean", "description": "Match case exactly. Defaults to false (case-insensitive)." },
    "limit": { "type": "integer", "description": "Maximum matching lines to return, 1-100 (default 30). The reply says when it stopped at the cap." }
  },
  "required": ["path", "pattern"]
}`)

	writeSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Path of the NEW file." },
    "content": { "type": "string", "description": "The file's full contents." }
  },
  "required": ["path", "content"]
}`)

	editSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "path": { "type": "string", "description": "Path of the existing file to change." },
    "old_text": { "type": "string", "description": "The exact text to replace. Must appear in the file exactly once — include enough surrounding lines to be unique." },
    "new_text": { "type": "string", "description": "The replacement text. An empty string deletes the matched text." }
  },
  "required": ["path", "old_text", "new_text"]
}`)
)
