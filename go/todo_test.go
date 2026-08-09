package agentic

import (
	"context"
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// todoCall builds a todo_write ToolCall with the given JSON arguments.
func todoCall(args string) ToolCall {
	return ToolCall{ID: "td-1", Name: TodoWriteToolName, Arguments: args}
}

// recordingTodos captures what the executor hands the host.
type recordingTodos struct {
	got   [][]Todo
	fails error
}

func (r *recordingTodos) write(_ context.Context, todos []Todo) error {
	r.got = append(r.got, todos)
	return r.fails
}

func TestTodoWriteAdvertisement(t *testing.T) {
	exec := NewTodoExecutor(TodoConfig{})
	tools := exec.Tools()
	require.Len(t, tools, 1)
	tool := tools[0]
	assert.Equal(t, TodoWriteToolName, tool.Name)
	assert.False(t, tool.Readonly,
		"todo_write writes host state; a sub-agent inheriting it would overwrite its parent's plan")
	assert.False(t, exec.NeedsApproval(TodoWriteToolName))
	assert.Contains(t, tool.Description, "REPLACES the whole list")

	// The enum is the only thing stopping a model inventing a fourth state, so
	// it is asserted against the states themselves rather than by eye.
	var schema struct {
		Properties struct {
			Todos struct {
				Items struct {
					Properties struct {
						State struct {
							Enum []string `json:"enum"`
						} `json:"state"`
					} `json:"properties"`
					Required []string `json:"required"`
				} `json:"items"`
			} `json:"todos"`
		} `json:"properties"`
	}
	require.NoError(t, json.Unmarshal(tool.InputSchema, &schema))
	assert.Equal(t, []string{string(TodoPending), string(TodoInProgress), string(TodoDone)},
		schema.Properties.Todos.Items.Properties.State.Enum)
	assert.Equal(t, []string{"title"}, schema.Properties.Todos.Items.Required,
		"a task without a state is a task not started, not a rejected call")
}

// The contract the tool offers the model: what you send IS the list. The host
// receives the whole thing every time, so it never has to merge.
func TestTodoWriteHandsTheHostTheWholeList(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(
		`{"todos":[{"title":"write it","state":"done"},{"title":"test it","state":"in_progress"},{"title":"ship it","state":"pending"}]}`))
	require.NoError(t, err)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 1)
	assert.Equal(t, []Todo{
		{Title: "write it", State: TodoDone},
		{Title: "test it", State: TodoInProgress},
		{Title: "ship it", State: TodoPending},
	}, rec.got[0])
	assert.Equal(t, "Task list updated (3 tasks):\n[x] write it\n[~] test it\n[ ] ship it", res.Content)

	// A second call carrying two of the three: the host is told the list is
	// now two, not that one item changed.
	res, err = exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"test it","state":"done"}]}`))
	require.NoError(t, err)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 2)
	assert.Equal(t, []Todo{{Title: "test it", State: TodoDone}}, rec.got[1])
}

// Clearing the list has to reach the host as an empty list, not as nothing at
// all: a finished job stops being shown, and a host that cannot tell the two
// apart leaves a stale plan on screen forever.
func TestClearingTheListReachesTheHostAsAnEmptyList(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[]}`))
	require.NoError(t, err)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 1)
	assert.NotNil(t, rec.got[0])
	assert.Empty(t, rec.got[0])
	assert.Equal(t, "Task list cleared.", res.Content)
}

// A missing state is a task the model has not started. Rejecting the call
// would punish the common shape (write the plan, then set states as you go).
func TestAMissingStateIsPending(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"no state given"}]}`))
	require.NoError(t, err)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 1)
	assert.Equal(t, []Todo{{Title: "no state given", State: TodoPending}}, rec.got[0])
}

// A state nothing renders must never reach the host, and the model must be
// told which item was wrong and what the choices are -- a bare "invalid state"
// leaves it guessing on a list of twenty.
func TestAnUnknownStateIsRefusedWithItsIndexAndTheChoices(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(
		`{"todos":[{"title":"fine","state":"done"},{"title":"bad","state":"blocked"}]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, `todos[1] has state "blocked"; it must be one of pending, in_progress, done`, res.Content)
	assert.Empty(t, rec.got, "a refused call never reaches the host")
}

// An empty title renders as a row nobody can act on; a title long enough to be
// a description does not fit the surface a host gives the list.
func TestAnUnusableTitleIsRefusedWithItsIndex(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"ok"},{"title":"   "}]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "todos[1] has an empty title; every task needs one", res.Content)

	long := strings.Repeat("x", todoMaxTitleRunes+1)
	res, err = exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"`+long+`"}]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "todos[0] has a title of 201 characters; the limit is 200")

	assert.Empty(t, rec.got)
}

func TestATooLongListIsRefused(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	items := make([]string, todoMaxItems+1)
	for i := range items {
		items[i] = `{"title":"t"}`
	}
	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[`+strings.Join(items, ",")+`]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "the list holds at most 100, and you sent 101")
	assert.Empty(t, rec.got)
}

// A list that was not stored must never be reported as stored: the model would
// go on planning against a list that does not exist.
func TestAHostThatCouldNotStoreTheListIsAFailure(t *testing.T) {
	rec := &recordingTodos{fails: errors.New("disk on fire")}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"t"}]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "could not save the task list: disk on fire", res.Content)
}

// With nowhere to keep the list, the tool says so. Accepting a plan nobody
// keeps would have the model believe the user can see it.
func TestNoWriterMeansTheToolRefuses(t *testing.T) {
	exec := NewTodoExecutor(TodoConfig{})
	res, err := exec.Execute(context.Background(), todoCall(`{"todos":[{"title":"t"}]}`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "the task list is unavailable: this run has nowhere to keep it", res.Content)
}

func TestTodoWriteRejectsGarbageAndForeignNames(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoExecutor(TodoConfig{Write: rec.write})

	res, err := exec.Execute(context.Background(), todoCall(`not json`))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "invalid todo_write arguments")

	res, err = exec.Execute(context.Background(), ToolCall{ID: "x", Name: "something_else", Arguments: `{}`})
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "unknown tool: something_else", res.Content)
	assert.Empty(t, rec.got)
}
