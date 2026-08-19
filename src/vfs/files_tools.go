package vfs

import (
	"context"
	"encoding/json"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
	"strings"
)

// The seven file tools' handlers: decode, route to the folder, render. Every
// failure is a recoverable teaching error -- a bad path is something the model
// can correct, never something that ends a turn.

// One argument struct per tool: each is both what the handler decodes and what
// the advertised schema is inferred from (schema.go), so the two cannot
// disagree. The `jsonschema` tag is the model-facing description of the field,
// and a json tag without `omitempty` marks it required.
type (
	pathArgs struct {
		Path string `json:"path" jsonschema:"Virtual path."`
	}
	readArgs struct {
		Path   string `json:"path" jsonschema:"Virtual path of the file to read."`
		Offset int    `json:"offset,omitempty" jsonschema:"First line to return, 1-based. Omit to start at the beginning. A grep hit's line number goes here directly."`
		Limit  int    `json:"limit,omitempty" jsonschema:"How many lines to return from offset. Omit for the rest of the file. The reply says how many lines the file has and where to continue."`
	}
	findArgs struct {
		Path    string `json:"path" jsonschema:"Directory to search below, recursively."`
		Pattern string `json:"pattern" jsonschema:"Filename glob (*.go, test_*) or a plain substring. Matched against the file NAME and its path, never its contents."`
		Limit   int    `json:"limit,omitempty" jsonschema:"Maximum results to return, 1-100 (default 20). The reply says when it stopped at the cap."`
	}
	grepArgs struct {
		Path          string `json:"path" jsonschema:"Where to search, recursively. A directory, a subdirectory of one, or a single file."`
		Pattern       string `json:"pattern" jsonschema:"The text to find. Matched literally and case-insensitively anywhere in a line unless regexp or case_sensitive say otherwise. Not tokenized: give the literal text you expect to see in the file."`
		Glob          string `json:"glob,omitempty" jsonschema:"Only search files whose name matches this glob, or a comma-separated list of them, e.g. \"*.usf\" or \"*.c,*.h\"."`
		Regexp        bool   `json:"regexp,omitempty" jsonschema:"Treat pattern as an extended regular expression instead of literal text. Defaults to false."`
		CaseSensitive bool   `json:"case_sensitive,omitempty" jsonschema:"Match case exactly. Defaults to false (case-insensitive)."`
		Limit         int    `json:"limit,omitempty" jsonschema:"Maximum matching lines to return, 1-100 (default 30). The reply says when it stopped at the cap."`
	}
	writeArgs struct {
		Path    string `json:"path" jsonschema:"Path of the NEW file."`
		Content string `json:"content" jsonschema:"The file's full contents."`
	}
	editArgs struct {
		Path    string `json:"path" jsonschema:"Path of the existing file to change."`
		OldText string `json:"old_text" jsonschema:"The exact text to replace. Must appear in the file exactly once — include enough surrounding lines to be unique."`
		NewText string `json:"new_text" jsonschema:"The replacement text. An empty string deletes the matched text."`
	}
)

// MountOf is the leading path segment: the folder a virtual path addresses.
// The mount name ends at the first "/" or "@", so a ref suffix rides with the
// rest of the path rather than becoming part of the name.
func MountOf(p string) string {
	s := strings.TrimPrefix(strings.TrimSpace(p), "/")
	if i := strings.IndexAny(s, "/@"); i >= 0 {
		return s[:i]
	}
	return s
}

// resolve routes a path to its folder. Every failure is a recoverable teaching
// error.
func (e *files) resolve(tool, raw string) (Folder, *agentic.ToolResult) {
	if strings.TrimSpace(raw) == "" {
		return nil, &agentic.ToolResult{Content: tool + ` requires "path".`, IsError: true}
	}
	if e.guard != nil {
		if blocked, reason := e.guard(raw); blocked {
			return nil, &agentic.ToolResult{Content: reason, IsError: true}
		}
	}
	mount := strings.ToLower(MountOf(raw))
	f, ok := e.folders[mount]
	if !ok {
		return nil, &agentic.ToolResult{Content: tool + ": " + e.mountUnavailable(mount), IsError: true}
	}
	return f, nil
}

func (e *files) mountUnavailable(mount string) string {
	if e.unavailable != nil {
		if s := e.unavailable(mount); s != "" {
			return s
		}
	}
	return "/" + mount + " is not available in this conversation."
}

// writable resolves a path to a folder that accepts changes, or the teaching
// error naming what to use instead.
func (e *files) writable(tool, raw string) (WritableFolder, *agentic.ToolResult) {
	f, fail := e.resolve(tool, raw)
	if fail != nil {
		return nil, fail
	}
	w, ok := f.(WritableFolder)
	if !ok {
		return nil, &agentic.ToolResult{Content: tool + ": " + readOnlyReason(f, raw), IsError: true}
	}
	if allowed, why := w.Writable(raw); !allowed {
		return nil, &agentic.ToolResult{Content: tool + ": " + why, IsError: true}
	}
	return w, nil
}

// readOnlyReason asks a read-only folder to name the writable route, falling
// back to stating the refusal.
func readOnlyReason(f Folder, p string) string {
	if ex, ok := f.(ReadOnlyExplainer); ok {
		if s := ex.ReadOnlyReason(p); s != "" {
			return s
		}
	}
	return f.Display(p) + " is read-only."
}

// decodeArgs unmarshals a tool's arguments.
func decodeArgs[In any](tool string, args json.RawMessage) (In, *agentic.ToolResult) {
	var in In
	if err := json.Unmarshal(args, &in); err != nil {
		var zero In
		return zero, &agentic.ToolResult{Content: "invalid " + tool + " arguments: " + err.Error(), IsError: true}
	}
	return in, nil
}

func (e *files) listDir(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[pathArgs](ListDirToolName, args)
	if bad != nil {
		return *bad, nil
	}
	f, fail := e.resolve(ListDirToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	listing, err := f.List(ctx, in.Path)
	if err != nil {
		return agentic.ToolResult{Content: ListDirToolName + ": " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: renderListing(f.Display(in.Path), listing)}, nil
}

func (e *files) readFile(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[readArgs](ReadFileToolName, args)
	if bad != nil {
		return *bad, nil
	}
	f, fail := e.resolve(ReadFileToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	file, err := f.Read(ctx, in.Path)
	if err != nil {
		return agentic.ToolResult{Content: ReadFileToolName + ": " + err.Error(), IsError: true}, nil
	}
	header := f.Display(in.Path)
	if file.Note != "" {
		header += "  (" + file.Note + ")"
	}
	body, rangeNote := SliceLines(file.Content, in.Offset, in.Limit)
	if rangeNote != "" {
		header += "\n" + rangeNote
	}
	if file.TruncatedNote != "" {
		// Two different cuts can apply, so they are reported separately: the
		// file was too long to serve whole, and then a window was taken of what
		// survived. Merging them would misstate what the line numbers below are
		// relative to.
		header += "\n" + file.TruncatedNote
	}
	return agentic.ToolResult{Content: header + "\n\n" + body}, nil
}

func (e *files) findFiles(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[findArgs](FindFilesToolName, args)
	if bad != nil {
		return *bad, nil
	}
	pattern := strings.TrimSpace(in.Pattern)
	if pattern == "" {
		return agentic.ToolResult{Content: FindFilesToolName + ` requires "pattern": a filename glob (*.go) or a plain substring of the name or path.`, IsError: true}, nil
	}
	f, fail := e.resolve(FindFilesToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	limit := clampInt(in.Limit, FindDefaultLimit, 1, FindMaxLimit)
	hits, err := f.Find(ctx, in.Path, pattern, limit)
	if err != nil {
		return agentic.ToolResult{Content: FindFilesToolName + ": " + err.Error(), IsError: true}, nil
	}
	where := f.Display(in.Path)
	if len(hits) == 0 {
		return agentic.ToolResult{Content: fmt.Sprintf("No file under %s matches %q.", where, pattern)}, nil
	}
	var b strings.Builder
	fmt.Fprintf(&b, "%d file(s) under %s matching %q", len(hits), where, pattern)
	if len(hits) == limit {
		fmt.Fprintf(&b, " (first %d; raise limit or narrow the pattern for more)", limit)
	}
	b.WriteString(":\n")
	for _, h := range hits {
		b.WriteString(h + "\n")
	}
	return agentic.ToolResult{Content: strings.TrimRight(b.String(), "\n")}, nil
}

// grep searches file contents below a path. Scope is the path and nothing
// else, exactly as for find_files: the same tool searches one subdirectory,
// one repository, or a whole owner, depending only on how much of the path is
// given.
func (e *files) grep(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[grepArgs](GrepToolName, args)
	if bad != nil {
		return *bad, nil
	}
	pattern := strings.TrimSpace(in.Pattern)
	if pattern == "" {
		return agentic.ToolResult{Content: GrepToolName + ` requires "pattern": the text to find inside the files.`, IsError: true}, nil
	}
	f, fail := e.resolve(GrepToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	limit := clampInt(in.Limit, GrepDefaultLimit, 1, GrepMaxLimit)
	globs := SplitGlobs(in.Glob)
	res, err := f.Grep(ctx, in.Path, GrepQuery{
		Pattern:       pattern,
		Regexp:        in.Regexp,
		CaseSensitive: in.CaseSensitive,
		Globs:         globs,
		MaxHits:       limit,
	})
	if err != nil {
		return agentic.ToolResult{Content: GrepToolName + ": " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: renderGrep(f.Display(in.Path), pattern, globs, res)}, nil
}

func (e *files) writeFile(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[writeArgs](WriteFileToolName, args)
	if bad != nil {
		return *bad, nil
	}
	w, fail := e.writable(WriteFileToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	note, err := w.Create(ctx, in.Path, in.Content)
	if err != nil {
		return agentic.ToolResult{Content: WriteFileToolName + ": " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: note}, nil
}

func (e *files) editFile(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[editArgs](EditFileToolName, args)
	if bad != nil {
		return *bad, nil
	}
	if in.OldText == "" {
		return agentic.ToolResult{Content: EditFileToolName + ` requires "old_text": the exact existing text to replace, occurring exactly once in the file. To add a brand-new file use ` + WriteFileToolName + ".", IsError: true}, nil
	}
	w, fail := e.writable(EditFileToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	note, err := w.Replace(ctx, in.Path, in.OldText, in.NewText)
	if err != nil {
		return agentic.ToolResult{Content: EditFileToolName + ": " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: note}, nil
}

func (e *files) deleteFile(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	in, bad := decodeArgs[pathArgs](DeleteFileToolName, args)
	if bad != nil {
		return *bad, nil
	}
	w, fail := e.writable(DeleteFileToolName, in.Path)
	if fail != nil {
		return *fail, nil
	}
	note, err := w.Remove(ctx, in.Path)
	if err != nil {
		return agentic.ToolResult{Content: DeleteFileToolName + ": " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: note}, nil
}
