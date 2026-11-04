package vfs

import (
	"context"
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	agentic "github.com/wow-look-at-my/agentic-loop"
)

// writePathHook is write_file's hook on "path", called the way the loop calls it.
func writePathHook(t *testing.T, ft *FileTools) func(raw string) *agentic.ToolResult {
	t.Helper()
	tool, ok := ft.Tools().Find(WriteFileToolName)
	require.True(t, ok)
	w, ok := tool.(agentic.ArgWatcher)
	require.True(t, ok, "write_file watches its path while the content streams")
	hooks := w.ArgHooks()
	require.Len(t, hooks, 1)
	require.Equal(t, "path", hooks[0].Key)
	return func(raw string) *agentic.ToolResult { return hooks[0].On(context.Background(), json.RawMessage(raw)) }
}

func quoted(t *testing.T, s string) string {
	t.Helper()
	b, err := json.Marshal(s)
	require.NoError(t, err)
	return string(b)
}

// write_file refuses a taken path from the path alone, in the words Execute would use, and leaves
// everything else to Execute.
func TestWriteFilePathHookRefusesATakenPath(t *testing.T) {
	ft, work := fileRig()
	hook := writePathHook(t, ft)

	refusal := hook(quoted(t, "/work/main.go"))
	require.NotNil(t, refusal)
	assert.True(t, refusal.IsError)
	assert.Equal(t, runFileTool(t, ft, WriteFileToolName, `{"path":"/work/main.go","content":"x"}`).Content, refusal.Content)
	assert.Equal(t, "package main\n\nfunc main() {}\n", work.files["main.go"], "a refusal writes nothing")

	assert.Nil(t, hook(quoted(t, "/work/fresh.go")), "a free path runs as usual")

	// A read-only mount and a missing mount are refused from the path alone.
	for _, p := range []string{"/repos/a/b.go", "/attachments/x.txt"} {
		refusal = hook(quoted(t, p))
		require.NotNil(t, refusal, p)
		args, err := json.Marshal(map[string]string{"path": p, "content": "x"})
		require.NoError(t, err)
		assert.Equal(t, runFileTool(t, ft, WriteFileToolName, string(args)).Content, refusal.Content)
	}

	// A path that is not a string is Execute's to report.
	assert.Nil(t, hook(`3`))
}

// A folder that cannot answer yet, or has no RefuseCreate, leaves the verdict to Create.
func TestWriteFilePathHookLetsAnUndecidedFolderRun(t *testing.T) {
	ft, work := fileRig()
	hook := writePathHook(t, ft)

	work.err = errors.New("storage is busy")
	assert.Nil(t, hook(quoted(t, "/work/main.go")))
	work.err = nil

	plain := &plainWritable{w: &writableFolder{newMemFolder(map[string]string{"a.txt": "a"})}}
	plain.memFolder = plain.w.memFolder
	require.NoError(t, ft.Add("/plain", plain))
	assert.Nil(t, hook(quoted(t, "/plain/a.txt")))
	res := runFileTool(t, ft, WriteFileToolName, `{"path":"/plain/a.txt","content":"x"}`)
	assert.True(t, res.IsError, "Create still refuses: "+res.Content)
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
func (f *plainWritable) Remove(ctx context.Context, p string) (string, error) {
	return f.w.Remove(ctx, p)
}

var _ IWritableFolderProvider = (*plainWritable)(nil)
