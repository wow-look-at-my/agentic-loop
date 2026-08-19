package todo

import (
	"context"
	"encoding/json"
	"errors"
	agentic "github.com/wow-look-at-my/agentic-loop/src"
	"github.com/wow-look-at-my/agentic-loop/src/internal/jsontest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// recordingTodos captures what the executor hands the host. Writing to the
// real shipped tools, each mutation persists the whole post-mutation list.
type recordingTodos struct {
	got   [][]Todo
	fails error
}

func (r *recordingTodos) write(_ context.Context, todos []Todo) error {
	r.got = append(r.got, todos)
	return r.fails
}

// todoTools builds the four real tools round a recording host store, then
// returns the store and a map from advertised name to the actual agentic.Tool.
func todoTools(t *testing.T, rec *recordingTodos) map[string]agentic.Tool {
	t.Helper()
	exec := NewTodoTools(TodoConfig{Write: rec.write})
	byName := map[string]agentic.Tool{}
	for _, tool := range exec {
		byName[tool.Decl().Name] = tool
	}
	require.Len(t, byName, 4, "the four tools are separate agentic.Tools in one flat slice")
	return byName
}

// run executes one tool against the real Execute, asserting it is not a Go error.
func run(t *testing.T, tool agentic.Tool, args string) agentic.ToolResult {
	t.Helper()
	res, err := tool.Execute(context.Background(), json.RawMessage(args))
	require.NoError(t, err, "a refusal is a recoverable tool result, never a Go error")
	return res
}

func TestTodoToolsAreTheFourNamedMutationTools(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	require.Contains(t, byName, TodoAddToolName)
	require.Contains(t, byName, TodoEditToolName)
	require.Contains(t, byName, TodoCancelToolName)
	require.Contains(t, byName, TodoCompleteToolName)

	for _, name := range []string{TodoAddToolName, TodoEditToolName, TodoCancelToolName, TodoCompleteToolName} {
		decl := byName[name].Decl()
		assert.Equalf(t, name, decl.Name, "tool advertises its own name")
		assert.Falsef(t, decl.Readonly,
			"%s writes host state the host shows; a sub-agent inheriting it would overwrite its parent's plan", name)
		assert.Falsef(t, byName[name].NeedsApproval(), "%s is not approval-gated", name)
	}

	// The state enum is the only thing stopping a model inventing a fourth
	// state. It lives on todo_add and todo_edit; the id-only tools have none.
	for _, name := range []string{TodoAddToolName, TodoEditToolName} {
		var schema struct {
			Properties struct {
				State struct {
					Enum []string `json:"enum"`
				} `json:"state"`
			} `json:"properties"`
		}
		require.NoError(t, json.Unmarshal(byName[name].Decl().InputSchema, &schema))
		assert.Equalf(t, []string{string(TodoPending), string(TodoInProgress), string(TodoDone)},
			schema.Properties.State.Enum, "%s advertises the closed state enum", name)
	}
	// edit/cancel/complete all carry a stable "id" argument the model addresses a task by.
	for _, name := range []string{TodoEditToolName, TodoCancelToolName, TodoCompleteToolName} {
		var schema struct {
			Properties struct {
				ID struct {
					Type string `json:"type"`
				} `json:"id"`
			} `json:"properties"`
			Required []string `json:"required"`
		}
		require.NoError(t, json.Unmarshal(byName[name].Decl().InputSchema, &schema))
		assert.Equalf(t, "integer", schema.Properties.ID.Type, "%s addresses a task by integer id", name)
		assert.Containsf(t, schema.Required, "id", "%s requires an id", name)
	}
}

// todo_add appends exactly one task, mints a fresh id, and answers with the new
// full list both as rendered text and as a todo_list part whose JSON equals the
// list the host just stored.
func TestTodoAddAppendsOneTaskAndReturnsTheNewFullList(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	res := run(t, byName[TodoAddToolName], `{"title":"write it","state":"in_progress"}`)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 1)
	require.Len(t, rec.got[0], 1)
	assert.Equal(t, 1, rec.got[0][0].ID, "the new task carries the first minted id")
	assert.Equal(t, "write it", rec.got[0][0].Title)
	assert.Equal(t, TodoInProgress, rec.got[0][0].State)
	assert.Equal(t, "Task list updated (1 tasks):\n[~] #1 write it", res.Content)

	require.Len(t, res.Parts, 1)
	assert.Equal(t, TodoListPartType, res.Parts[0].Type)
	var carried []Todo
	require.NoError(t, json.Unmarshal([]byte(res.Parts[0].Text), &carried))
	assert.Equal(t, rec.got[0], carried, "the todo_list part JSON equals the host's stored list")

	// A second add keeps id 1 and hands the new task id 2.
	res = run(t, byName[TodoAddToolName], `{"title":"ship it"}`)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 2)
	require.Len(t, rec.got[1], 2)
	assert.Equal(t, []Todo{
		{ID: 1, Title: "write it", State: TodoInProgress},
		{ID: 2, Title: "ship it", State: TodoPending},
	}, rec.got[1])
}

// A missing state is a task the model has not started. Rejecting the add would
// punish the common shape (write the plan, then set states as you go).
func TestAMissingAddStateIsPending(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	res := run(t, byName[TodoAddToolName], `{"title":"no state given"}`)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 1)
	assert.Equal(t, []Todo{{ID: 1, Title: "no state given", State: TodoPending}}, rec.got[0])
}

// The heart of the change: edit/cancel/complete address ONE task by its stable
// id and change only that one, leaving every sibling's id, title and state
// byte-for-byte unchanged, across several interleaved mutations. The model
// never resends the parts it did not touch.
func TestInterleavedMutationsTouchOnlyTheNamedTask(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	for _, title := range []string{"one", "two", "three", "four"} {
		res := run(t, byName[TodoAddToolName], jsontest.Must(jsontest.Obj{"title": title}))
		require.False(t, res.IsError, res.Content)
	}
	// Host now holds one..four with ids 1..4, all pending.
	require.Len(t, rec.got, 4)
	require.Equal(t, []int{1, 2, 3, 4}, idsOf(rec.got[3]))

	// Mark #2 in_progress.
	res := run(t, byName[TodoEditToolName], `{"id":2,"state":"in_progress"}`)
	require.False(t, res.IsError, res.Content)
	// Rename #4.
	res = run(t, byName[TodoEditToolName], `{"id":4,"title":"four re"}`)
	require.False(t, res.IsError, res.Content)
	// Complete #1.
	res = run(t, byName[TodoCompleteToolName], `{"id":1}`)
	require.False(t, res.IsError, res.Content)
	// Cancel #3.
	res = run(t, byName[TodoCancelToolName], `{"id":3}`)
	require.False(t, res.IsError, res.Content)

	require.Len(t, rec.got, 8)
	got := rec.got[7]
	// Only the named tasks changed; the remainder is byte-for-byte stable.
	assert.Equal(t, []Todo{
		{ID: 1, Title: "one", State: TodoDone},
		{ID: 2, Title: "two", State: TodoInProgress},
		{ID: 4, Title: "four re", State: TodoPending},
	}, got)
	assert.Equal(t, "Task list updated (3 tasks):\n[x] #1 one\n[~] #2 two\n[ ] #4 four re", res.Content)
}

func idsOf(todos []Todo) []int {
	out := make([]int, len(todos))
	for i, t := range todos {
		out[i] = t.ID
	}
	return out
}

// Every successful mutation must hand the host a todo_list part whose JSON
// equals what it just stored, so a host draws exactly what the model edited.
func TestEverySuccessfulMutationCarriesTheRenderedTextAndTheList(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	steps := []struct {
		name string
		args string
	}{
		{TodoAddToolName, `{"title":"a","state":"done"}`},
		{TodoAddToolName, `{"title":"b"}`},
		{TodoEditToolName, `{"id":2,"state":"in_progress"}`},
		{TodoCompleteToolName, `{"id":1}`},
		{TodoCancelToolName, `{"id":2}`},
	}
	for i, step := range steps {
		res := run(t, byName[step.name], step.args)
		require.Falsef(t, res.IsError, "step %d (%s): %s", i, step.name, res.Content)
		// The rendered text is non-empty and says how many tasks.
		assert.NotEmpty(t, res.Content)
		// Exactly one part: the whole list, ids included, equal to host's store.
		require.Len(t, res.Parts, 1, step.name)
		assert.Equal(t, TodoListPartType, res.Parts[0].Type)
		var carried []Todo
		require.NoError(t, json.Unmarshal([]byte(res.Parts[0].Text), &carried))
		require.Greater(t, len(rec.got), i)
		assert.Equalf(t, rec.got[i], carried, "host stored list equals the todo_list part at step %d (%s)", i, step.name)
	}
}

// Clears the list by canceling its last task: the host must receive an empty
// list, not nothing at all, so a finished plan stops being shown.
func TestCancelingTheLastTaskReachesTheHostAsAnEmptyList(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	run(t, byName[TodoAddToolName], `{"title":"only"}`)
	res := run(t, byName[TodoCancelToolName], `{"id":1}`)
	require.False(t, res.IsError, res.Content)
	require.Len(t, rec.got, 2)
	assert.NotNil(t, rec.got[1])
	assert.Empty(t, rec.got[1])
	assert.Equal(t, "Task list cleared.", res.Content)
	require.Len(t, res.Parts, 1)
	assert.Equal(t, "[]", res.Parts[0].Text)
}

// An address that matches no task is refused with a teaching error naming the
// id, and neither that call nor the store is corrupted.
func TestAnUnknownIdIsRefused(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	run(t, byName[TodoAddToolName], `{"title":"present"}`)
	before := len(rec.got)

	for _, name := range []string{TodoEditToolName, TodoCancelToolName, TodoCompleteToolName} {
		res := run(t, byName[name], `{"id":99}`)
		assert.True(t, res.IsError)
		assert.Equal(t, "no task with id 99; use a task id from the latest reply", res.Content)
		assert.Len(t, rec.got, before, "%s must not reach the host when its target is missing", name)
	}
	// A cancelled id is gone and refused the same way afterwards.
	run(t, byName[TodoCancelToolName], `{"id":1}`)
	res := run(t, byName[TodoEditToolName], `{"id":1,"state":"done"}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "no task with id 1; use a task id from the latest reply", res.Content)
}

// Ids are unique by construction, so an ambiguous address cannot arise through
// the tools; the resolver still refuses one on a damaged store rather than
// silently editing the wrong task. This drives the real resolve on a store
// corrupted to hold two tasks with one id.
func TestAnAmbiguousIdIsRefused(t *testing.T) {
	rec := &recordingTodos{}
	exec := NewTodoTools(TodoConfig{Write: rec.write})
	// The store is the library's internal, shared state; reach it through the
	// tools' Execute indirectly is impossible for a duplicate, so exercise the
	// shipped resolver directly on a hand-built damaged store.
	edit, ok := exec.Find(TodoEditToolName)
	require.True(t, ok)
	tt := edit.(*todoTool)
	tt.store = &todoStore{
		items: []Todo{{ID: 7, Title: "a", State: TodoPending}, {ID: 7, Title: "b", State: TodoPending}},
		next:  8,
	}
	res := run(t, edit, `{"id":7,"state":"done"}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "task id 7 is ambiguous: it names more than one task", res.Content)
	assert.Empty(t, rec.got, "an ambiguous edit changes nothing and reaches the host not at all")
}

// todo_edit needs something to change; a call naming only an id changes nothing.
func TestEditNeedsATitleOrState(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)

	run(t, byName[TodoAddToolName], `{"title":"x"}`)
	before := len(rec.got)
	res := run(t, byName[TodoEditToolName], `{"id":1}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "todo_edit: nothing to change; provide a title and/or a state", res.Content)
	assert.Len(t, rec.got, before, "a refused edit does not reach the host")
}

// A missing/overlong/unknown title and an unknown state are each refused with
// an exact teaching error naming the offending argument; the store is unchanged.
func TestUnusableAddArgumentsAreRefused(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	add := byName[TodoAddToolName]

	res := run(t, add, `{"title":""}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "title is empty; every task needs one", res.Content)

	res = run(t, add, `{"title":"   "}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "title is empty; every task needs one", res.Content)

	long := strings.Repeat("x", todoMaxTitleRunes+1)
	res = run(t, add, jsontest.Must(jsontest.Obj{"title": long}))
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "title has 201 characters; the limit is 200")

	res = run(t, add, `{"title":"fine","state":"blocked"}`)
	assert.True(t, res.IsError)
	assert.Equal(t, `state "blocked"; it must be one of pending, in_progress, done`, res.Content)

	// A title misspells nothing; a failed add reaches the host not at all.
	assert.Empty(t, rec.got, "a refused add never reaches the host")
}

// A state nothing renders must never reach the host when editing either.
func TestAnEditWithAnUnknownStateIsRefused(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	run(t, byName[TodoAddToolName], `{"title":"fine"}`)
	before := len(rec.got)

	res := run(t, byName[TodoEditToolName], `{"id":1,"state":"blocked"}`)
	assert.True(t, res.IsError)
	assert.Equal(t, `state "blocked"; it must be one of pending, in_progress, done`, res.Content)
	assert.Len(t, rec.got, before)
}

// Unparseable payloads are refused per tool with the tool's own name.
func TestUnparseableArgumentsAreRefused(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	for _, name := range []string{TodoAddToolName, TodoEditToolName, TodoCancelToolName, TodoCompleteToolName} {
		res := run(t, byName[name], `not json`)
		assert.True(t, res.IsError)
		assert.Contains(t, res.Content, "invalid "+name+" arguments")
	}
	assert.Empty(t, rec.got)
}

// The list caps at 100 tasks; the 101st add is refused, and the store keeps 100.
func TestATooLongListIsRefused(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	for i := 0; i < todoMaxItems; i++ {
		res := run(t, byName[TodoAddToolName], `{"title":"t"}`)
		require.False(t, res.IsError, res.Content)
	}
	before := len(rec.got)
	res := run(t, byName[TodoAddToolName], `{"title":"overflow"}`)
	assert.True(t, res.IsError)
	assert.Contains(t, res.Content, "the list holds at most 100")
	assert.Len(t, rec.got, before, "the 101st add is refused and never reaches the host")
}

// A list that was not stored must never be reported as stored: the model would
// go on planning against a list that does not exist.
func TestAHostThatCouldNotStoreTheListIsAFailure(t *testing.T) {
	rec := &recordingTodos{fails: errors.New("disk on fire")}
	byName := todoTools(t, rec)
	res := run(t, byName[TodoAddToolName], `{"title":"t"}`)
	assert.True(t, res.IsError)
	assert.Equal(t, "could not save the task list: disk on fire", res.Content)
	// The failed add is not recorded as stored: the host got the error, and the
	// tool answered the failure.
	assert.NotEmpty(t, rec.got)
}

// With nowhere to keep the list, every tool says so. Accepting a plan nobody
// keeps would have the model believe the user can see it.
func TestNoWriterMeansTheToolsRefuse(t *testing.T) {
	exec := NewTodoTools(TodoConfig{})
	for _, tool := range exec {
		res := run(t, tool, `{"title":"t"}`)
		assert.True(t, res.IsError)
		assert.Equal(t, "the task list is unavailable: this run has nowhere to keep it", res.Content)
	}
}

// Two AddCalls in a row hand out distinct ids; ids never collide across adds.
func TestAddMintsMonotonicIds(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	run(t, byName[TodoAddToolName], `{"title":"a"}`)
	run(t, byName[TodoAddToolName], `{"title":"b"}`)
	require.Len(t, rec.got[1], 2)
	assert.Equal(t, 1, rec.got[1][0].ID)
	assert.Equal(t, 2, rec.got[1][1].ID)
}

// Edit may change a title, a state, or both, addressing by id alone.
func TestEditChangesTitleAndOrState(t *testing.T) {
	rec := &recordingTodos{}
	byName := todoTools(t, rec)
	run(t, byName[TodoAddToolName], `{"title":"orig"}`)

	run(t, byName[TodoEditToolName], `{"id":1,"state":"done"}`)
	run(t, byName[TodoEditToolName], `{"id":1,"title":"renamed"}`)
	assert.Equal(t, []Todo{{ID: 1, Title: "renamed", State: TodoDone}}, rec.got[2])

	run(t, byName[TodoEditToolName], `{"id":1,"title":"both","state":"in_progress"}`)
	assert.Equal(t, []Todo{{ID: 1, Title: "both", State: TodoInProgress}}, rec.got[3])
}
