package vfs

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentic "github.com/wow-look-at-my/agentic-loop"
)

func writePreflighter(t *testing.T, ft *FileTools) agentic.Preflighter {
	t.Helper()
	tool, ok := ft.Tools().Find(WriteFileToolName)
	require.True(t, ok)
	pf, ok := tool.(agentic.Preflighter)
	require.True(t, ok, "write_file judges a call from its streamed arguments")
	return pf
}

// write_file refuses a taken path the moment the path has streamed, with the words Execute would
// use, and leaves everything else to Execute.
func TestWriteFilePreflightRefusesATakenPathBeforeTheContentArrives(t *testing.T) {
	ft, work := fileRig()
	pf := writePreflighter(t, ft)
	ctx := context.Background()

	// The path is still arriving: no verdict yet.
	refusal, decided := pf.Preflight(ctx, `{"path": "/work/ma`)
	assert.Nil(t, refusal)
	assert.False(t, decided)

	// The path is whole and taken: refused now, in Execute's own words.
	refusal, decided = pf.Preflight(ctx, `{"path": "/work/main.go", "content": "package main\n\nfunc main() {\n`)
	require.True(t, decided)
	require.NotNil(t, refusal)
	assert.True(t, refusal.IsError)
	full := runFileTool(t, ft, WriteFileToolName, `{"path":"/work/main.go","content":"x"}`)
	assert.Equal(t, full.Content, refusal.Content)
	assert.Equal(t, "package main\n\nfunc main() {}\n", work.files["main.go"], "a refusal writes nothing")

	// A free path: decided, and the call runs as usual.
	refusal, decided = pf.Preflight(ctx, `{"path": "/work/fresh.go", "content": "pack`)
	assert.Nil(t, refusal)
	assert.True(t, decided)

	// A read-only mount is refused from the path alone, naming the writable route.
	refusal, decided = pf.Preflight(ctx, `{"path": "/repos/a/b.go", "content": "`)
	require.True(t, decided)
	require.NotNil(t, refusal)
	assert.Equal(t, runFileTool(t, ft, WriteFileToolName, `{"path":"/repos/a/b.go","content":"x"}`).Content, refusal.Content)

	// A mount that is not there is refused from the path alone, with the mount's own words.
	refusal, decided = pf.Preflight(ctx, `{"path": "/attachments/x.txt", "content": "`)
	require.True(t, decided)
	require.NotNil(t, refusal)
	assert.Equal(t, runFileTool(t, ft, WriteFileToolName, `{"path":"/attachments/x.txt","content":"x"}`).Content, refusal.Content)
}

// What the scanner cannot read is Execute's to report: the preflight decides without refusing.
func TestWriteFilePreflightLeavesBadArgumentsToExecute(t *testing.T) {
	ft, _ := fileRig()
	pf := writePreflighter(t, ft)
	ctx := context.Background()
	for _, partial := range []string{`{"content":"x"}`, `{"path": 3,`, `[1,2]`} {
		refusal, decided := pf.Preflight(ctx, partial)
		assert.Nil(t, refusal, partial)
		assert.True(t, decided, partial)
	}
}

// A folder that cannot answer yet, or does not implement CreateRefuser, leaves the verdict to Execute.
func TestWriteFilePreflightLetsAnUndecidedFolderRun(t *testing.T) {
	ft, work := fileRig()
	pf := writePreflighter(t, ft)
	ctx := context.Background()

	work.err = errors.New("storage is busy")
	refusal, decided := pf.Preflight(ctx, `{"path": "/work/main.go", "content": "`)
	assert.Nil(t, refusal)
	assert.True(t, decided)
	work.err = nil

	plain := &plainWritable{w: &writableFolder{newMemFolder(map[string]string{"a.txt": "a"})}}
	plain.memFolder = plain.w.memFolder
	require.NoError(t, ft.Add("/plain", plain))
	refusal, decided = pf.Preflight(ctx, `{"path": "/plain/a.txt", "content": "`)
	assert.Nil(t, refusal)
	assert.True(t, decided)
	res := runFileTool(t, ft, WriteFileToolName, `{"path":"/plain/a.txt","content":"x"}`)
	assert.True(t, res.IsError, "Execute still refuses: "+res.Content)
}

// plainWritable is a writable folder with no RefuseCreate.
type plainWritable struct {
	*memFolder
	w *writableFolder
}

func (f *plainWritable) Writable(p string) (bool, string) { return f.w.Writable(p) }
func (f *plainWritable) Create(ctx context.Context, p, content string) (string, error) {
	return f.w.Create(ctx, p, content)
}

func (f *plainWritable) Replace(ctx context.Context, p, oldText, newText string) (string, error) {
	return f.w.Replace(ctx, p, oldText, newText)
}
func (f *plainWritable) Remove(ctx context.Context, p string) (string, error) { return f.w.Remove(ctx, p) }

var _ IWritableFolderProvider = (*plainWritable)(nil)
