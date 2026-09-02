package commonai

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// The behaviour fields are MCP's tool annotations, and of them default to
// TRUE there. A Go struct of bare booleans reads an absent fact as false, which
// is the dangerous answer for both of them: an unannotated tool would come out
// "non-destructive, closed-world". These tests pin the resolution, because the
// value is what a caller gets by writing nothing.
func TestUnstatedBehaviourResolvesToTheCautiousAnswer(t *testing.T) {
	bare := ToolDecl{Name: "anything"}

	assert.True(t, bare.IsDestructive(), "a tool that states nothing may destroy state")
	assert.True(t, bare.IsOpenWorld(), "a tool that states nothing may reach anywhere")
	assert.False(t, bare.IsIdempotent(), "repeating an unknown call is not known to be safe")
	assert.True(t, bare.Vouched(), "the host declared this tool, so it stands behind it")
}

// destructiveHint and idempotentHint are "meaningful only when readOnlyHint ==
// false" per the specification, so read-only wins over whatever they say.
func TestReadOnlyOutranksTheFieldsItMakesMeaningless(t *testing.T) {
	ro := ToolDecl{Name: "grep", Readonly: true, Destructive: Bool(true)}

	assert.False(t, ro.IsDestructive(), "a tool that changes nothing destroys nothing")
	assert.True(t, ro.IsIdempotent(), "reading twice leaves the world where it was")

	// Open-world is NOT of the conditional ones: a read can still leave the machine, as web_fetch does.
	assert.True(t, ToolDecl{Readonly: true}.IsOpenWorld())
	assert.False(t, ToolDecl{Readonly: true, OpenWorld: Bool(false)}.IsOpenWorld())
}

// An explicit false and an absent field are different documents and different
// answers. Collapsing them is what makes a tri-state pointless.
func TestStatedFalseIsNotTheSameAsUnstated(t *testing.T) {
	stated := ToolDecl{Name: "todo_add", Destructive: Bool(false)}
	unstated := ToolDecl{Name: "todo_add"}

	assert.False(t, stated.IsDestructive())
	assert.True(t, unstated.IsDestructive())
}

// A tool source's own claim about itself must stay distinguishable from a fact
// the host compiled in. The specification is blunt about it: a client "should
// never make tool use decisions based on ToolAnnotations received from
func TestAnUnvouchedClaimSaysSo(t *testing.T) {
	claimed := ToolDecl{Name: "srv__delete_everything", Readonly: true, Unvouched: true}

	assert.False(t, claimed.Vouched(), "the server said this, not the host")
	assert.True(t, claimed.Readonly, "the claim is still carried, just marked")
}

// The wire format is the definition, so the tri-state has to survive it: an
// absent attribute means unknown, and a reader that decoded it as false would
// silently promote every unannotated tool to non-destructive.
func TestBehaviourSurvivesTheWireIncludingTheAbsentCase(t *testing.T) {
	req := Request{
		Model: "m",
		Tools: []ToolDecl{
			{Name: "grep", Readonly: true, OpenWorld: Bool(false)},
			{Name: "write_file", Destructive: Bool(true), Idempotent: true, OpenWorld: Bool(false)},
			{Name: "todo_add", Destructive: Bool(false)},
			{Name: "srv__thing", Unvouched: true},
		},
	}

	doc, err := EncodeRequestBytes(req)
	require.NoError(t, err)
	back, err := DecodeRequest(doc)
	require.NoError(t, err)
	require.Len(t, back.Tools, 4)

	assert.True(t, back.Tools[0].Readonly)
	assert.False(t, back.Tools[0].IsOpenWorld())

	assert.True(t, back.Tools[1].IsDestructive())
	assert.True(t, back.Tools[1].IsIdempotent())

	require.NotNil(t, back.Tools[2].Destructive, "a stated false must arrive stated")
	assert.False(t, back.Tools[2].IsDestructive())

	assert.Nil(t, back.Tools[3].Destructive, "nothing was stated, so nothing is claimed")
	assert.True(t, back.Tools[3].IsDestructive(), "and unknown reads as destructive")
	assert.True(t, back.Tools[3].Unvouched)
}
