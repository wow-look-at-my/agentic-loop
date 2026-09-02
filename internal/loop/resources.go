package loop

import (
	"context"
	"strconv"
	"strings"
)

// MCP resources are announced as short automated notices of what changed, never content.

// resourceNoticeHeader opens every notice so the model never mistakes it for the user.
const resourceNoticeHeader = "[automated notice -- the host is watching this conversation's MCP resources; this is not a message from the user]"

// Resource change kinds, as reported to the model.
const (
	ResourceAdded    = "added"
	ResourceModified = "modified"
	ResourceRemoved  = "removed"
)

// ResourceChange is detected change, already recorded by the watcher. It
// carries no content: everything here is announced to the model, and the
// before/after bytes stay in storage until mcp_resource_diff asks for them.
type ResourceChange struct {
	// ChangeID is the opaque id the model quotes back to mcp_resource_diff.
	ChangeID string
	// Server is the MCP server's display name.
	Server string
	// URI is the resource's own identifier.
	URI string
	// Label is the resource's human name (title, name, or the URI again).
	Label string
	// Kind is of the Resource* constants above.
	Kind string
	// Summary is a -line shape-of-the-change, e.g. " KB -> KB, + - lines".
	Summary string
	// Note is an accuracy caveat that must travel with the change.
	Note string
}

// ResourcePoll is the outcome of watch pass.
type ResourcePoll struct {
	// Changes are the resources that differ from the last pass.
	Changes []ResourceChange
	// Warnings are servers or resources the pass could NOT account for.
	Warnings []string
	// Baseline marks the pass, where every resource is new; changes only the wording.
	Baseline bool
}

// Empty reports whether the pass found nothing worth telling the model.
func (p ResourcePoll) Empty() bool { return len(p.Changes) == 0 && len(p.Warnings) == 0 }

// ResourceWatcher re-reads the conversation's MCP resources, reporting changes since last pass.
type ResourceWatcher interface {
	// Poll performs pass; remote failures are reported as Warnings, not errors.
	Poll(ctx context.Context) (ResourcePoll, error)
}

// FormatResourceNotice renders watch pass as the delivered message text.
// diffTool is the advertised name of the diff tool, quoted so the model calls
// the name it was actually given rather than the this package assumed.
func FormatResourceNotice(poll ResourcePoll, diffTool string) string {
	var b strings.Builder
	b.WriteString(resourceNoticeHeader)

	if n := len(poll.Changes); n > 0 {
		b.WriteString("\n\n")
		switch {
		case poll.Baseline:
			b.WriteString(plural(n, "MCP resource is", "MCP resources are") +
				" available on the connected servers and now being watched:")
		default:
			b.WriteString(plural(n, "MCP resource changed", "MCP resources changed") + " since the last turn:")
		}
		for _, c := range poll.Changes {
			b.WriteString("\n\n")
			b.WriteString(describeResourceChange(c, poll.Baseline))
		}
	}

	if len(poll.Warnings) > 0 {
		b.WriteString("\n\nThe watch could not account for everything this pass:")
		for _, w := range poll.Warnings {
			b.WriteString("\n- " + w)
		}
		b.WriteString("\nTreat those resources as unknown rather than unchanged.")
	}

	if len(poll.Changes) > 0 && diffTool != "" {
		b.WriteString("\n\nCall " + diffTool + " with a change_id to see exactly what changed" +
			" (or the full contents, for a resource that was just added). Each id stays" +
			" resolvable for the rest of this conversation and always returns the" +
			" before/after captured at that moment, so fetch one when it becomes relevant" +
			" -- there is no need to fetch them now, and no need to fetch any you do not care about.")
	}
	return b.String()
}

// describeResourceChange renders change as its own block: what it is, how it
// moved, and the id that resolves to it.
func describeResourceChange(c ResourceChange, baseline bool) string {
	var b strings.Builder
	verb := c.Kind
	if baseline && c.Kind == ResourceAdded {
		verb = "watching"
	}
	b.WriteString(verb + " -- " + resourceTitle(c))
	if c.Summary != "" {
		b.WriteString(" -- " + c.Summary)
	}
	if c.Note != "" {
		b.WriteString("\n  note: " + c.Note)
	}
	b.WriteString("\n  change_id: " + c.ChangeID)
	return b.String()
}

// resourceTitle renders a change's subject as `"label" (uri, server "name")`,
// collapsing the label when it is just the URI again.
func resourceTitle(c ResourceChange) string {
	var b strings.Builder
	if c.Label != "" && c.Label != c.URI {
		b.WriteString(strconv.Quote(c.Label) + " (" + c.URI)
	} else {
		b.WriteString("(" + c.URI)
	}
	if c.Server != "" {
		b.WriteString(", server " + strconv.Quote(c.Server))
	}
	b.WriteString(")")
	return b.String()
}

// plural renders " <>" or "N <many>".
func plural(n int, one, many string) string {
	if n == 1 {
		return "1 " + one
	}
	return strconv.Itoa(n) + " " + many
}
