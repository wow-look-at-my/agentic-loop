package agentic

import (
	"context"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The recording half of the resource watch: turning a detected difference into
// a durable change record, and the prose the model is shown about it.

// record writes one added/modified change and refreshes the snapshot,
// returning what the model is told about it.
func (w *resourceWatcher) record(ctx context.Context, kind string, c capture, before ResourceSnapshot, existed bool) (ResourceChange, error) {
	beforeContent, beforeHash := "", ""
	var beforeBytes int64
	if existed {
		beforeContent, beforeHash, beforeBytes = before.Content, before.Hash, before.ByteSize
	}
	id, err := w.snapshots.RecordChange(ctx, ResourceChangeRecord{
		SourceID:      c.sourceID,
		SourceName:    c.sourceName,
		URI:           c.uri,
		Label:         c.label,
		MimeType:      c.mimeType,
		Kind:          kind,
		BeforeContent: beforeContent,
		AfterContent:  c.content,
		BeforeHash:    beforeHash,
		AfterHash:     c.hash,
		BeforeBytes:   beforeBytes,
		AfterBytes:    c.bytes,
		Binary:        c.binary,
		Truncated:     c.truncated || (existed && before.Truncated),
	})
	if err != nil {
		return ResourceChange{}, fmt.Errorf("recording a resource change: %w", err)
	}
	if err := w.snapshots.PutSnapshot(ctx, ResourceSnapshot{
		SourceID:  c.sourceID,
		URI:       c.uri,
		Label:     c.label,
		MimeType:  c.mimeType,
		Hash:      c.hash,
		Content:   c.content,
		ByteSize:  c.bytes,
		Binary:    c.binary,
		Truncated: c.truncated,
	}); err != nil {
		return ResourceChange{}, fmt.Errorf("updating a watched resource: %w", err)
	}
	return ResourceChange{
		ChangeID: id,
		Server:   c.sourceName,
		URI:      c.uri,
		Label:    c.label,
		Kind:     kind,
		Summary:  summarizeResourceChange(kind, beforeContent, c.content, beforeBytes, c.bytes, c.binary),
		Note:     captureNote(c.binary, c.truncated, w.maxBytes),
	}, nil
}

// recordRemoval writes a removal change and drops the snapshot. The last known
// content is kept ON the change row, so the model can still diff away a
// resource that no longer exists.
func (w *resourceWatcher) recordRemoval(ctx context.Context, before ResourceSnapshot) (ResourceChange, error) {
	sourceName := w.sourceName(before.SourceID)
	id, err := w.snapshots.RecordChange(ctx, ResourceChangeRecord{
		SourceID:      before.SourceID,
		SourceName:    sourceName,
		URI:           before.URI,
		Label:         before.Label,
		MimeType:      before.MimeType,
		Kind:          ResourceRemoved,
		BeforeContent: before.Content,
		BeforeHash:    before.Hash,
		BeforeBytes:   before.ByteSize,
		Binary:        before.Binary,
		Truncated:     before.Truncated,
	})
	if err != nil {
		return ResourceChange{}, fmt.Errorf("recording a resource removal: %w", err)
	}
	if err := w.snapshots.DeleteSnapshot(ctx, before.SourceID, before.URI); err != nil {
		return ResourceChange{}, fmt.Errorf("dropping a watched resource: %w", err)
	}
	return ResourceChange{
		ChangeID: id,
		Server:   sourceName,
		URI:      before.URI,
		Label:    before.Label,
		Kind:     ResourceRemoved,
		Summary:  "no longer advertised (was " + HumanSize(before.ByteSize) + ")",
	}, nil
}

// newWarnings returns the warning texts not already delivered, and forgets any
// that have stopped recurring so a fault that returns is reported again.
func (w *resourceWatcher) newWarnings(warns []warning) []string {
	w.mu.Lock()
	defer w.mu.Unlock()
	current := make(map[string]bool, len(warns))
	var fresh []string
	for _, warn := range warns {
		current[warn.text] = true
		if !w.warned[warn.text] {
			fresh = append(fresh, warn.text)
		}
	}
	w.warned = current
	sort.Strings(fresh)
	return fresh
}

// sourceFailed reports whether this pass failed to account for a source, which
// makes every one of its resources unknown rather than removed. A source that
// is no longer configured at all has no warning and no listing; its resources
// are genuinely gone from this run's reach.
func (w *resourceWatcher) sourceFailed(sourceID string, warns []warning) bool {
	for _, warn := range warns {
		if warn.sourceID == sourceID {
			return true
		}
	}
	return false
}

// sourceName resolves a source id to its display name, falling back to the id
// for a source that is no longer configured.
func (w *resourceWatcher) sourceName(id string) string {
	for _, s := range w.sources {
		if s.ID() == id {
			return s.Name()
		}
	}
	return id
}

// summarizeResourceChange renders the one-line shape of a change: how the size
// moved and, for text, how many lines were added and removed.
func summarizeResourceChange(kind, before, after string, beforeBytes, afterBytes int64, binary bool) string {
	if binary {
		if kind == ResourceAdded {
			return "binary, " + HumanSize(afterBytes)
		}
		return "binary, " + HumanSize(beforeBytes) + " -> " + HumanSize(afterBytes) + " (content not captured)"
	}
	if kind == ResourceAdded {
		return HumanSize(afterBytes) + ", " + plural(CountLines(after), "line", "lines")
	}
	added, removed := CountLineChanges(before, after)
	return HumanSize(beforeBytes) + " -> " + HumanSize(afterBytes) +
		", +" + strconv.Itoa(added) + " -" + strconv.Itoa(removed) + " lines"
}

// captureNote returns the caveat that must travel with a change whose captured
// content is not the whole resource.
func captureNote(binary, truncated bool, maxBytes int) string {
	switch {
	case binary:
		return "binary content is watched by hash and size only; the diff reports that it changed, not how"
	case truncated:
		return "the captured content was cut at " + HumanSize(int64(maxBytes)) +
			", so the diff covers only the first " + HumanSize(int64(maxBytes)) + " of this resource"
	default:
		return ""
	}
}

// snapshotKey is the (source, uri) identity of a watched resource.
func snapshotKey(sourceID, uri string) string { return sourceID + "\x00" + uri }

// approxBlobBytes estimates a base64 blob's decoded length without decoding it.
func approxBlobBytes(b64 string) int {
	return len(strings.TrimRight(b64, "=")) * 3 / 4
}

// capText cuts content to max bytes, dropping a trailing partial UTF-8 rune so
// what lands in storage stays valid text.
func capText(s string, max int) (string, bool) {
	if len(s) <= max {
		return s, false
	}
	out := s[:max]
	for i := 0; i < 3 && len(out) > 0 && !utf8.ValidString(out); i++ {
		out = out[:len(out)-1]
	}
	return out, true
}
