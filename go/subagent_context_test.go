package agentic

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestRenderTranscript(t *testing.T) {
	msgs := []Message{
		{Role: RoleSystem, Content: "be helpful"},
		{Role: RoleUser, Content: "  find X  "},
		{Role: RoleAssistant, Content: "on it", ToolCalls: []ToolCall{
			{ID: "c1", Name: "srch", Arguments: ` {"q":"x"} `},
			{ID: "c2", Name: "bare", Arguments: "   "},
		}},
		{Role: RoleTool, Content: "found at Y"},
		{Role: "moderator", Content: "carry on"},
		{Role: "", Content: "??"},
	}
	got := RenderTranscript(msgs)
	want := "System:\nbe helpful\n\n" +
		"User:\nfind X\n\n" +
		"Assistant:\non it\n[requested tool srch with {\"q\":\"x\"}]\n[requested tool bare]\n\n" +
		"Tool result:\nfound at Y\n\n" +
		"Moderator:\ncarry on\n\n" +
		"Message:\n??"
	assert.Equal(t, want, got)
}

func TestCapRunesTail(t *testing.T) {
	assert.Equal(t, "short", capRunesTail("short", 10))
	assert.Equal(t, "whatever", capRunesTail("whatever", 0), "non-positive cap disables")

	long := strings.Repeat("a", 50) + "TAIL"
	got := capRunesTail(long, 4)
	assert.Equal(t, "[... earlier shared context truncated ...]\n\nTAIL", got,
		"over the cap keeps the most recent tail behind the marker")
}

func TestSelectLastN(t *testing.T) {
	msgs := []Message{{Content: "1"}, {Content: "2"}, {Content: "3"}}
	assert.Nil(t, SelectLastN(msgs, 0))
	assert.Nil(t, SelectLastN(nil, 2))
	assert.Equal(t, msgs[2:], SelectLastN(msgs, 1))
	assert.Equal(t, msgs, SelectLastN(msgs, 3))
	assert.Equal(t, msgs, SelectLastN(msgs, 99), "n past the length returns everything")
}

func TestSelectByEndIndices(t *testing.T) {
	msgs := []Message{{Content: "a"}, {Content: "b"}, {Content: "c"}, {Content: "d"}}
	got := SelectByEndIndices(msgs, []int{1, 3, 3, 0, -2, 99})
	require.Len(t, got, 2, "de-duplicated; out-of-range ignored")
	assert.Equal(t, "b", got[0].Content, "chronological order regardless of index order")
	assert.Equal(t, "d", got[1].Content)
	assert.Empty(t, SelectByEndIndices(msgs, []int{55}))
}

func TestSubagentPreview(t *testing.T) {
	assert.Equal(t, `{"q": "x"}`, subagentPreview("  {\"q\":\n\t \"x\"}  "), "whitespace flattened to single spaces")
	long := strings.Repeat("ab ", 200)
	got := subagentPreview(long)
	assert.Equal(t, subagentPreviewMaxRunes, utf8.RuneCountInString(got))
	assert.True(t, strings.HasSuffix(got, "..."), "capped previews end with an ellipsis")
}

func TestComposeSubagentTask(t *testing.T) {
	assert.Equal(t, "do it", composeSubagentTask("", "do it"))
	assert.Equal(t, "do it", composeSubagentTask("   ", "do it"), "a blank block keeps the isolated default")
	got := composeSubagentTask("the background", "do it")
	want := "Context shared from the parent conversation (background only):\n\n" +
		"the background" +
		"\n\n----------------------------------------\n\nYour task:\n\n" +
		"do it"
	assert.Equal(t, want, got)
}

func TestGenerateContextSummaryEmptyTranscriptSkipsModel(t *testing.T) {
	provider := &scriptProvider{} // any call would fail with "script exhausted"
	got, err := generateContextSummary(context.Background(), provider, "m", nil, 0, nil)
	require.NoError(t, err)
	assert.Empty(t, got)
	assert.Empty(t, provider.reqs, "an empty transcript makes no model call")
}
