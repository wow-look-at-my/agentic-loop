package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
)

// ResourceDiffToolName is the advertised name of the tool that resolves a
// change id announced in a resource notice.
const ResourceDiffToolName = "mcp_resource_diff"

const (
	resourceDiffDescription = "Shows exactly what changed in a watched MCP resource. " +
		"The host re-reads this conversation's MCP resources between turns and announces each change with a change_id; " +
		"pass one here to get the unified diff of that specific change, exactly as the host captured it. " +
		"An id keeps resolving to the same before/after for the whole conversation, no matter how many times the " +
		"resource has changed since, so there is no need to fetch a diff before you need it. " +
		"For a resource that was just added, this returns its contents."

	// resourceDiffRecentLimit is how many recent change ids an unknown-id error
	// lists back, so a mistyped or hallucinated id is answered with the real
	// ones rather than a dead end.
	resourceDiffRecentLimit = 20
)

// resourceDiffSchema is the tool's parameter schema.
var resourceDiffSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "change_id": {
      "type": "string",
      "description": "the change_id from the automated resource notice that announced this change"
    },
    "full": {
      "type": "boolean",
      "description": "return the resource's full captured content instead of a diff (default false)"
    }
  },
  "required": ["change_id"]
}`)

// resourceDiffArgs is the tool's argument payload.
type resourceDiffArgs struct {
	ChangeID string `json:"change_id"`
	Full     bool   `json:"full,omitempty"`
}

// ErrNoResourceChange is what a ResourceChanges reader returns for an id it
// does not hold, so the tool can answer with the real ids rather than the
// host's storage error text.
var ErrNoResourceChange = errors.New("agentic: no such resource change")

// StoredResourceChange is one recorded change as the reader hands it back: the
// record the watcher wrote, plus the id and the moment it was captured.
type StoredResourceChange struct {
	ResourceChangeRecord
	ID string
	// CapturedAt is free-form host text (a timestamp), shown to the model.
	CapturedAt string
}

// ResourceChanges is the read side of the change log -- the half
// mcp_resource_diff answers from. Get returns ErrNoResourceChange (wrapped is
// fine) for an id this run does not hold.
type ResourceChanges interface {
	GetChange(ctx context.Context, changeID string) (StoredResourceChange, error)
	RecentChanges(ctx context.Context, limit int) ([]StoredResourceChange, error)
}

// resourceDiffTool implements mcp_resource_diff: it resolves a change id
// announced in a resource notice to the before/after the watcher captured at
// that moment.
//
// It reads the change record's OWN copy of the content, never the current
// snapshot. That is what makes the id honest: a resource that has moved three
// times since still answers the first notice with the first change.
type resourceDiffTool struct {
	changes ResourceChanges
}

// NewResourceDiffTool builds mcp_resource_diff, or returns nil when there is no
// change reader to answer from -- a run with no watched resources is never
// offered a tool that could only ever fail.
func NewResourceDiffTool(changes ResourceChanges) Tool {
	if changes == nil {
		return nil
	}
	return &resourceDiffTool{changes: changes}
}

// Decl advertises mcp_resource_diff. It is read-only: it reads the change log
// and nothing else.
func (e *resourceDiffTool) Decl() ToolDecl {
	return ToolDecl{
		Name:        ResourceDiffToolName,
		Description: resourceDiffDescription,
		InputSchema: resourceDiffSchema,
		Readonly:    true,
	}
}

// Execute resolves one change id. Every failure is a recoverable error result
// that names the ids the model could have used instead.
func (e *resourceDiffTool) Execute(ctx context.Context, raw json.RawMessage) (ToolResult, error) {
	var args resourceDiffArgs
	if len(raw) > 0 {
		if err := json.Unmarshal(raw, &args); err != nil {
			return ToolResult{Content: "could not parse arguments: " + err.Error(), IsError: true}, nil
		}
	}
	id := strings.TrimSpace(args.ChangeID)
	if id == "" {
		return ToolResult{Content: "change_id is required. " + e.recentIDs(ctx), IsError: true}, nil
	}

	change, err := e.changes.GetChange(ctx, id)
	if err != nil {
		if errors.Is(err, ErrNoResourceChange) {
			return ToolResult{
				Content: "no resource change with id " + strconv.Quote(id) + " in this conversation. " + e.recentIDs(ctx),
				IsError: true,
			}, nil
		}
		return ToolResult{Content: "could not read the change: " + err.Error(), IsError: true}, nil
	}
	return ToolResult{Content: RenderResourceChange(change, args.Full)}, nil
}

// RenderResourceChange turns a stored change into the model-facing answer.
func RenderResourceChange(c StoredResourceChange, full bool) string {
	var b strings.Builder
	b.WriteString(c.Kind + ": " + c.URI)
	if c.Label != "" && c.Label != c.URI {
		b.WriteString(" (" + c.Label + ")")
	}
	if c.SourceName != "" {
		b.WriteString(" on server " + strconv.Quote(c.SourceName))
	}
	if c.MimeType != "" {
		b.WriteString("\nmedia type: " + c.MimeType)
	}
	if c.CapturedAt != "" {
		b.WriteString("\ncaptured at: " + c.CapturedAt)
	}

	if c.Binary {
		// Never a diff: base64 bytes are useless as model context, so a binary
		// resource is watched by hash and size and reported as exactly that
		// rather than as a diff that would look empty.
		b.WriteString("\n\nThis resource is binary, so its content is watched by hash and size only.")
		b.WriteString("\nsize: " + HumanSize(c.BeforeBytes) + " -> " + HumanSize(c.AfterBytes))
		b.WriteString("\nsha256: " + shortHash(c.BeforeHash) + " -> " + shortHash(c.AfterHash))
		b.WriteString("\n\nWhat changed inside it is not recoverable from here. Read it with the" +
			" server's own tools if the contents matter.")
		return b.String()
	}
	if c.Truncated {
		b.WriteString("\n\nNOTE: the captured content was cut at the host's per-resource cap, so" +
			" everything below covers only the beginning of this resource. A change past the cut" +
			" is not visible here and was not detected as a change either.")
	}

	switch {
	case full:
		b.WriteString("\n\nFull captured content (" + HumanSize(c.AfterBytes) + "):\n\n")
		b.WriteString(contentOrEmpty(c.AfterContent))
	case c.Kind == ResourceAdded:
		// A unified diff against /dev/null is just the whole file with a '+' on
		// every line — strictly worse to read than the content itself.
		b.WriteString("\n\nThis resource is newly available. Its contents (" + HumanSize(c.AfterBytes) + "):\n\n")
		b.WriteString(contentOrEmpty(c.AfterContent))
	case c.Kind == ResourceRemoved:
		b.WriteString("\n\nThis resource is no longer advertised by its server." +
			" Its last known contents (" + HumanSize(c.BeforeBytes) + "):\n\n")
		b.WriteString(contentOrEmpty(c.BeforeContent))
	default:
		diff := UnifiedDiff("before/"+c.URI, "after/"+c.URI, c.BeforeContent, c.AfterContent)
		if diff == "" {
			// The hash moved but the text did not: a trailing-newline-only edit,
			// or a change entirely past the truncation cut.
			b.WriteString("\n\nThe resource's hash changed but the captured text is identical" +
				" (a whitespace-only edit, or a change past the capture cut).")
			return b.String()
		}
		b.WriteString("\n\nsize: " + HumanSize(c.BeforeBytes) + " -> " + HumanSize(c.AfterBytes))
		b.WriteString("\n\n" + diff)
	}
	return b.String()
}

// contentOrEmpty renders captured content, naming the empty case rather than
// returning a blank tail the model has to guess about.
func contentOrEmpty(s string) string {
	if s == "" {
		return "(the resource is empty)"
	}
	return s
}

// shortHash renders a hash for display, or "(none)" when there was no previous
// state to hash.
func shortHash(h string) string {
	if h == "" {
		return "(none)"
	}
	if len(h) > 12 {
		return h[:12]
	}
	return h
}

// recentIDs lists this run's most recent change ids, so an unknown id is
// answered with the real ones instead of a dead end.
func (e *resourceDiffTool) recentIDs(ctx context.Context) string {
	rows, err := e.changes.RecentChanges(ctx, resourceDiffRecentLimit)
	if err != nil {
		return "The recent change ids could not be listed: " + err.Error()
	}
	if len(rows) == 0 {
		return "No resource changes have been recorded in this conversation yet."
	}
	var b strings.Builder
	b.WriteString("The most recent recorded changes are:")
	for _, r := range rows {
		b.WriteString("\n- " + r.ID + "  " + r.Kind + " " + r.URI)
		if r.SourceName != "" {
			b.WriteString(" (server " + strconv.Quote(r.SourceName) + ")")
		}
	}
	return b.String()
}
