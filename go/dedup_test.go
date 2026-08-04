package agentic

import (
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestOutputDeduperFirstOccurrenceFull(t *testing.T) {
	d := NewOutputDeduper()
	content, deduped := d.Collapse("read_file", ToolResult{Content: "full output"})
	assert.False(t, deduped)
	assert.Equal(t, "full output", content, "first occurrence is fed back unchanged")
}

func TestOutputDeduperIdenticalRepeatCollapses(t *testing.T) {
	d := NewOutputDeduper()
	full := ToolResult{Content: "the big diff"}
	first, deduped := d.Collapse("read_file", full)
	assert.False(t, deduped)
	assert.Equal(t, "the big diff", first)

	second, deduped := d.Collapse("read_file", full)
	assert.True(t, deduped)
	assert.Contains(t, second, UnchangedPrefix)
	assert.Contains(t, second, "repeat #2", "the first repeat is counted as #2")
	assert.NotEqual(t, second, "the big diff")

	third, deduped := d.Collapse("read_file", full)
	assert.True(t, deduped)
	assert.Contains(t, third, "repeat #3", "the count increments on each repeat")
}

func TestOutputDeduperMarkerIsInformative(t *testing.T) {
	d := NewOutputDeduper()
	full := ToolResult{Content: "identical"}
	d.Collapse("grep", full)
	marker, deduped := d.Collapse("grep", full)
	require.True(t, deduped)
	assert.Contains(t, marker, "grep", "the marker names the tool")
	assert.Contains(t, marker, "byte-identical")
	assert.Contains(t, marker, "earlier call")
	assert.Contains(t, marker, "Nothing has changed")
	// The marker may only claim what the deduper actually knows. It hashes
	// output, not arguments, so identical output is equally what a tool that
	// IGNORES an argument produces — and telling a caller it repeated itself
	// when it did not points the investigation the wrong way.
	assert.NotContains(t, marker, "with the same inputs")
	assert.Contains(t, marker, "this tool ignores the field you changed")
}

func TestOutputDeduperDifferentContentDoesNotCollapse(t *testing.T) {
	d := NewOutputDeduper()
	d.Collapse("grep", ToolResult{Content: "result one"})
	content, deduped := d.Collapse("grep", ToolResult{Content: "result two"})
	assert.False(t, deduped)
	assert.Equal(t, "result two", content, "a different output is a fresh occurrence")
}

func TestOutputDeduperDifferentToolSameContentDoesNotCollapse(t *testing.T) {
	d := NewOutputDeduper()
	d.Collapse("list_dir", ToolResult{Content: "same bytes"})
	content, deduped := d.Collapse("read_file", ToolResult{Content: "same bytes"})
	assert.False(t, deduped)
	assert.Equal(t, "same bytes", content, "the tool is part of the dedup key")

	// ... and the original tool still collapses on its own repeat.
	_, deduped = d.Collapse("list_dir", ToolResult{Content: "same bytes"})
	assert.True(t, deduped)
}

func TestOutputDeduperErrorResultsNeverCollapse(t *testing.T) {
	d := NewOutputDeduper()
	errResult := ToolResult{Content: "boom", IsError: true}
	content, deduped := d.Collapse("write_file", errResult)
	assert.False(t, deduped)
	assert.Equal(t, "boom", content)

	content, deduped = d.Collapse("write_file", errResult)
	assert.False(t, deduped, "an error result never collapses, even byte-identical")
	assert.Equal(t, "boom", content, "the error text is always fed back verbatim")
}

func TestOutputDeduperResetClears(t *testing.T) {
	d := NewOutputDeduper()
	full := ToolResult{Content: "bytes"}
	d.Collapse("grep", full)
	_, deduped := d.Collapse("grep", full)
	assert.True(t, deduped)

	d.Reset()
	content, deduped := d.Collapse("grep", full)
	assert.False(t, deduped, "after Reset the output is a fresh occurrence again")
	assert.Equal(t, "bytes", content)
}

func TestOutputDeduperResetIsIdempotent(t *testing.T) {
	d := NewOutputDeduper()
	d.Reset()
	d.Reset()
	content, deduped := d.Collapse("grep", ToolResult{Content: "x"})
	assert.False(t, deduped)
	assert.Equal(t, "x", content)
}

func TestOutputDeduperBoundedEviction(t *testing.T) {
	d := NewOutputDeduper()
	// Insert one distinct output per tool name, far past the cap. Names are
	// ASCII and valid; the deduper itself places no naming restriction.
	over := maxDedupEntries + 32
	for i := 0; i < over; i++ {
		name := "tool_" + strconv.Itoa(i)
		_, deduped := d.Collapse(name, ToolResult{Content: "payload " + strconv.Itoa(i)})
		assert.False(t, deduped, "every first occurrence is fed back in full")
	}
	// The map must have stayed at the cap: the oldest keys were evicted.
	assert.Equal(t, maxDedupEntries, len(d.keys))
	assert.Len(t, d.seen, maxDedupEntries)

	// An evicted (oldest) output is treated as new again...
	content, deduped := d.Collapse("tool_0", ToolResult{Content: "payload 0"})
	assert.False(t, deduped, "the evicted oldest entry is a fresh occurrence again")
	assert.Equal(t, "payload 0", content)
	// ...while a key inserted after the eviction window still collapses.
	_, deduped = d.Collapse("tool_"+strconv.Itoa(over-1), ToolResult{Content: "payload " + strconv.Itoa(over-1)})
	assert.True(t, deduped, "a recent key survives eviction and still collapses")
}
