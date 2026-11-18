package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// A transcript through the tree and back is the same transcript.
func TestMessagesSurviveTheTree(t *testing.T) {
	for _, d := range []Dialect{DialectOpenAI, DialectAnthropic, DialectResponses} {
		msgs := []Message{
			{Role: RoleUser, ID: "u1", Content: "hi"},
			NewMessage(RoleAssistant,
				ThinkingPart{Text: "plan", Signature: "c2ln", ID: "rs_1"},
				TextPart{Text: "on it"},
				ToolCallPart{ID: "c1", Name: "read", Arguments: `{"p":1}`},
			),
			{Role: RoleTool, ToolCallID: "c1", Content: "result", ToolIsError: true},
			{Role: RoleUser, Kind: "stop_nudge", Content: "note"},
			NewMessage(RoleAssistant, RedactedThinkingPart{Data: "cmVk"}, TextPart{Text: "done"}),
		}
		tr := NewTree()
		var leaf IMessage
		for _, m := range msgs {
			for _, n := range NodesFromMessage(m, d) {
				require.NoError(t, tr.Add(n, leaf))
				leaf = n
			}
		}
		got := MessagesFromPath(tr.PathTo(leaf))
		require.Len(t, got, len(msgs), d)
		for i := range msgs {
			assert.Equal(t, msgs[i].Role, got[i].Role, "%s message %d", d, i)
			assert.Equal(t, msgs[i].Content, got[i].Content, "%s message %d", d, i)
			assert.Equal(t, msgs[i].ToolCallID, got[i].ToolCallID)
			assert.Equal(t, msgs[i].ToolIsError, got[i].ToolIsError)
			assert.Equal(t, msgs[i].Kind, got[i].Kind)
			assert.Equal(t, msgs[i].EffectiveParts(), got[i].EffectiveParts(), "%s message %d", d, i)
		}
	}
}

func TestCompactionIsSentAsAHandoff(t *testing.T) {
	tr := NewTree()
	u := NewUserText("old")
	require.NoError(t, tr.Add(u, nil))
	c := NewCompaction(u)
	WriterOf(c.Summary).Finish("the gist")
	require.NoError(t, tr.Add(c, u))
	got := MessagesFromPath([]IMessage{c})
	require.Len(t, got, 1)
	assert.Equal(t, CompactionHandoffPrefix+"the gist", got[0].Content)
	assert.Equal(t, CompactionKind, got[0].Kind)
}
