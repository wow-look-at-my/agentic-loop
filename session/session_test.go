package session

import (
	"os"
	"path/filepath"
	"testing"

	commonai "github.com/wow-look-at-my/agentic-loop/core"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// stores is every backend, so the contract is checked against all of them
// rather than against whichever one a test happened to pick.
func stores(t *testing.T) map[string]Store {
	t.Helper()
	f, err := NewFile(t.TempDir())
	require.NoError(t, err)
	return map[string]Store{"memory": NewMemory(), "file": f}
}

func seed() commonai.Request {
	return commonai.Request{
		Model:     "m",
		System:    "be brief",
		MaxTokens: 128,
		Messages:  []commonai.Message{commonai.NewMessage(commonai.RoleUser, commonai.TextPart{Text: "hello"})},
	}
}

func TestStoreRoundTrip(t *testing.T) {
	for name, s := range stores(t) {
		t.Run(name, func(t *testing.T) {
			id, err := s.Create(seed())
			require.NoError(t, err)

			got, err := s.Get(id)
			require.NoError(t, err)
			assert.Equal(t, "m", got.Model)
			assert.Equal(t, "be brief", got.System)
			assert.Equal(t, 128, got.MaxTokens)
			require.Len(t, got.Messages, 1)
			assert.Equal(t, "hello", got.Messages[0].Content)

			got, err = s.Append(id, commonai.NewMessage(commonai.RoleAssistant, commonai.TextPart{Text: "hi"}))
			require.NoError(t, err)
			require.Len(t, got.Messages, 2)

			got, err = s.Get(id)
			require.NoError(t, err)
			require.Len(t, got.Messages, 2, "the append is what the next turn will see")

			ids, err := s.List()
			require.NoError(t, err)
			assert.Equal(t, []string{id}, ids)

			require.NoError(t, s.Delete(id))
			_, err = s.Get(id)
			assert.ErrorIs(t, err, ErrNotFound)
		})
	}
}

func TestStoreMissingIsNotFound(t *testing.T) {
	for name, s := range stores(t) {
		t.Run(name, func(t *testing.T) {
			_, err := s.Get("nope")
			assert.ErrorIs(t, err, ErrNotFound)
			_, err = s.Append("nope", commonai.NewMessage(commonai.RoleUser))
			assert.ErrorIs(t, err, ErrNotFound)
			assert.ErrorIs(t, s.Delete("nope"), ErrNotFound,
				"deleting what is not there is a belief worth correcting, not a no-op")
		})
	}
}

// A caller holding the returned Request must not be able to edit stored
// history through it.
func TestStoreHandsBackACopy(t *testing.T) {
	for name, s := range stores(t) {
		t.Run(name, func(t *testing.T) {
			id, err := s.Create(seed())
			require.NoError(t, err)
			got, err := s.Get(id)
			require.NoError(t, err)
			got.Messages[0].Content = "tampered"

			again, err := s.Get(id)
			require.NoError(t, err)
			assert.Equal(t, "hello", again.Messages[0].Content)
		})
	}
}

// An id is part of a file path, so a store that takes one has to say no to the
// ones that would leave its directory.
func TestFileStoreRejectsAWanderingID(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFile(dir)
	require.NoError(t, err)
	for _, id := range []string{"../escape", "a/b", "", "with space", "dot.dot"} {
		_, err := s.Get(id)
		require.Error(t, err, id)
		assert.NotErrorIs(t, err, ErrNotFound, "a bad id is refused, not reported missing")
	}
}

// The stored document is the format's own, so it validates and reads back as
// the conversation it names.
func TestFileStoreWritesAValidConversationDocument(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFile(dir)
	require.NoError(t, err)
	id, err := s.Create(seed())
	require.NoError(t, err)

	data, err := os.ReadFile(filepath.Join(dir, id+".xml"))
	require.NoError(t, err)
	require.NoError(t, commonai.Validate(data), "document:\n%s", data)

	storedID, req, err := commonai.DecodeConversation(data)
	require.NoError(t, err)
	assert.Equal(t, id, storedID)
	assert.Equal(t, "m", req.Model)
}

// A document changed on disk into something the schema rejects is a failure to
// read, not a conversation quietly missing half its history.
func TestFileStoreRefusesACorruptedDocument(t *testing.T) {
	dir := t.TempDir()
	s, err := NewFile(dir)
	require.NoError(t, err)
	id, err := s.Create(seed())
	require.NoError(t, err)

	path := filepath.Join(dir, id+".xml")
	require.NoError(t, os.WriteFile(path, []byte(`<?xml version="1.1"?><conversation id="c1"><nope/></conversation>`), 0o644))
	_, err = s.Get(id)
	require.Error(t, err)
	assert.NotErrorIs(t, err, ErrNotFound)

	// An id that does not match the file it sits in is the same class of problem.
	require.NoError(t, os.WriteFile(path, []byte(`<?xml version="1.1"?><conversation id="somethingelse" model="m"/>`), 0o644))
	_, err = s.Get(id)
	require.ErrorContains(t, err, "somethingelse")
}

// A store over a directory that already holds sessions keeps the ones it finds
// rather than writing over them.
func TestFileStoreDoesNotReuseAnExistingID(t *testing.T) {
	dir := t.TempDir()
	first, err := NewFile(dir)
	require.NoError(t, err)
	id, err := first.Create(seed())
	require.NoError(t, err)

	second, err := NewFile(dir)
	require.NoError(t, err)
	next, err := second.Create(seed())
	require.NoError(t, err)
	assert.NotEqual(t, id, next)

	got, err := second.Get(id)
	require.NoError(t, err)
	assert.Equal(t, "hello", got.Messages[0].Content)
}
