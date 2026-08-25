package loop

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"sync"
)

// UnchangedPrefix labels a tool result byte-identical to an earlier call in the conversation.
const UnchangedPrefix = "[unchanged]"

// maxDedupEntries bounds the deduper's map; FIFO eviction drops the oldest key at the cap.
const maxDedupEntries = 512

// OutputDeduper collapses byte-identical read-only tool results into a short marker.
type OutputDeduper struct {
	mu   sync.Mutex
	keys []string
	seen map[string]int
}

// NewOutputDeduper returns an empty, ready-to-use OutputDeduper.
func NewOutputDeduper() *OutputDeduper {
	return &OutputDeduper{seen: map[string]int{}}
}

// Reset clears remembered outputs; call it when earlier outputs leave the model's context.
func (d *OutputDeduper) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.keys = nil
	d.seen = make(map[string]int, len(d.seen))
}

// Collapse returns the content for one result; first occurrence unchanged, repeats get the marker.
func (d *OutputDeduper) Collapse(tool ToolDecl, result ToolResult) (content string, deduped bool) {
	if !tool.Readonly || tool.Name == "" || result.IsError {
		return result.Content, false
	}
	hash := sha256.Sum256([]byte(result.Content))
	key := tool.Name + "\x00" + hex.EncodeToString(hash[:])

	d.mu.Lock()
	defer d.mu.Unlock()
	if n, ok := d.seen[key]; ok {
		next := n + 1
		d.seen[key] = next
		return unchangedMarker(tool.Name, next), true
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

// unchangedMarker states the OUTPUT repeated, never that the inputs did.
func unchangedMarker(tool string, n int) string {
	return fmt.Sprintf("%s The output of %s is byte-identical to an earlier call in this conversation (repeat #%d). Nothing has changed — reference that earlier result rather than re-reading it. If you did change the arguments, the difference made no difference: either it genuinely does not affect the result, or this tool ignores the field you changed — check its schema before trying a third phrasing.",
		UnchangedPrefix, tool, n)
}
