package agentic

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// UnchangedPrefix labels a collapsed tool result: the tool's output this turn
// was byte-identical to an earlier call in the same conversation, so instead
// of re-dumping the full text the loop feeds back a short marker that points
// the model at the earlier result already in context.
const UnchangedPrefix = "[unchanged]"

// maxDedupEntries bounds the deduper's internal map. FIFO eviction drops the
// oldest key when the cap is reached, so a long-lived deduper (one per Run,
// reused across turns) cannot grow without bound even when a model calls many
// distinct tools with many distinct outputs.
const maxDedupEntries = 512

// OutputDeduper collapses byte-identical, read-only tool results into a short
// marker so the model never re-reads a huge output it already saw this
// conversation. Only read-only tools are eligible, and an IsError result is
// never collapsed -- a marker must never hide a failure.
//
// The deduper is keyed by tool name plus the SHA-256 of the result content:
// two different searches that both return "no matches" are genuinely the same
// information, so they collapse regardless of the differing arguments that
// produced them. The first occurrence of a (tool, content) pair still returns
// the full content -- dedup can only shrink repeats -- and each repeat returns
// the marker with a running occurrence count.
//
// The map is bounded (see maxDedupEntries): when it exceeds the cap the oldest
// entries are evicted first. One OutputDeduper should be created per Run call
// (or per conversation, when reused across turns) and Reset when earlier tool
// outputs leave the model's context, since a marker is only meaningful while
// the full output it references is still in the active thread.
type OutputDeduper struct {
	mu   sync.Mutex
	keys []string
	seen map[string]int
}

// NewOutputDeduper returns an empty, ready-to-use OutputDeduper.
func NewOutputDeduper() *OutputDeduper {
	return &OutputDeduper{seen: map[string]int{}}
}

// Reset clears all remembered outputs, so the next call to each tool counts as
// its first occurrence again. Call it when earlier tool outputs leave the
// model's context (compaction, a rewound thread, a new conversation).
func (d *OutputDeduper) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.keys = nil
	d.seen = make(map[string]int, len(d.seen))
}

// Collapse returns the content to feed back to the model for one tool result.
// The first time a (tool, byte-identical content) pair is seen it returns
// result.Content unchanged with deduped=false; each repeat returns the
// [unchanged] marker with deduped=true and the occurrence count incremented
// (2, 3, ...). An IsError result is always returned unchanged with deduped=false.
func (d *OutputDeduper) Collapse(tool string, result ToolResult) (content string, deduped bool) {
	if result.IsError {
		return result.Content, false
	}
	hash := sha256.Sum256([]byte(result.Content))
	key := tool + "\x00" + hex.EncodeToString(hash[:])

	d.mu.Lock()
	defer d.mu.Unlock()
	if n, ok := d.seen[key]; ok {
		next := n + 1
		d.seen[key] = next
		return unchangedMarker(tool, next), true
	}
	d.seen[key] = 1
	d.keys = append(d.keys, key)
	if len(d.keys) > maxDedupEntries {
		evicted := d.keys[0]
		d.keys = d.keys[1:]
		delete(d.seen, evicted)
	}
	return result.Content, false
}

// unchangedMarker builds the stable marker text for the n-th occurrence
// (n = 2 for the first repeat) of an unchanged read-only tool output.
//
// It states what is known — that the OUTPUT repeated — and never that the
// inputs did. The deduper keys on tool plus content hash, deliberately not on
// arguments, so identical output is also what a tool IGNORING an argument
// produces: three differently-worded queries once collapsed here because the
// tool read none of them, and a marker asserting the caller had repeated
// itself sent the model looking for a fault in itself instead of in the call.
func unchangedMarker(tool string, n int) string {
	return fmt.Sprintf("%s The output of %s is byte-identical to an earlier call in this conversation (repeat #%d). Nothing has changed — reference that earlier result rather than re-reading it. If you did change the arguments, the difference made no difference: either it genuinely does not affect the result, or this tool ignores the field you changed — check its schema before trying a third phrasing.",
		UnchangedPrefix, tool, n)
}
