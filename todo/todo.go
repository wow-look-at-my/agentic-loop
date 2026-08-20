package todo

import (
	"context"
	"encoding/json"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"strconv"
	"strings"
)

// The four advertised names of the built-in task-list tools. The model mutates
// one task at a time by its stable id instead of resending the whole list, so
// it never has to reconstruct the plan from memory.
const (
	TodoAddToolName      = "todo_add"
	TodoEditToolName     = "todo_edit"
	TodoCancelToolName   = "todo_cancel"
	TodoCompleteToolName = "todo_complete"
)

// TodoListPartType is the agentic.ToolContentPart type the current list rides back on
// (as a JSON array of Todo, ids included), so a host's display follows a
// running turn instead of waiting for the run to end. Like every structured
// part, it never reaches the model.
const TodoListPartType = "todo_list"

// The task-list caps. A plan longer than this is not a plan, and a title long
// enough to be a description does not fit the narrow surface a host renders
// one in.
const (
	todoMaxItems      = 100
	todoMaxTitleRunes = 200
)

// todoAddDescription is what the model is told about todo_add.
var todoAddDescription = "Adds one task to the task list the host shows the user. " +
	"Give it a title and an optional state. The new task gets an id, shown in the reply, " +
	"which every later todo_edit / todo_cancel / todo_complete call uses to address " +
	"exactly that task. The reply carries the whole current list, ids included, so " +
	"re-read it rather than recalling tasks from memory."

// todoEditDescription is what the model is told about todo_edit.
var todoEditDescription = "Changes ONE existing task on the task list by its id: a new title, a new state, or both. " +
	"Only the id you name is touched; every other task is left as it is. Provide at least one of title or state. " +
	"The reply carries the whole current list, ids included, so re-read it from the reply."

// todoCancelDescription is what the model is told about todo_cancel.
var todoCancelDescription = "Removes ONE existing task from the task list by its id. Only that task is removed; " +
	"every other task keeps its id, title and state. The reply carries the whole current list, ids included, " +
	"so re-read it from the reply."

// todoCompleteDescription is what the model is told about todo_complete.
var todoCompleteDescription = "Marks ONE existing task on the task list done by its id. Only that task changes; " +
	"every other task keeps its id, title and state. The reply carries the whole current list, ids included, " +
	"so re-read it from the reply."

// TodoState is one task's state. The set is closed: a host renders each one,
// and a state it does not know would show as a task with no mark.
type TodoState string

// The task states, in the order a schema advertises them.
const (
	TodoPending    TodoState = "pending"
	TodoInProgress TodoState = "in_progress"
	TodoDone       TodoState = "done"
)

// todoStates is the closed set, for validation and for the teaching error.
var todoStates = []TodoState{TodoPending, TodoInProgress, TodoDone}

// Todo is one task of a run's list. ID is the stable per-task identity the
// model addresses it by; it is minted once and never reused, so it survives
// across turns and unrelated mutations. Title and State are as the model set
// them.
type Todo struct {
	ID    int       `json:"id"`
	Title string    `json:"title"`
	State TodoState `json:"state"`
}

// TodoConfig configures NewTodoTools.
type TodoConfig struct {
	// Write receives the run's whole current task list, ids and all, after
	// every mutation, and persists (or displays) it however the host wants.
	// A non-nil error is reported to the model as a recoverable failure, so a
	// list that was not stored is never reported as stored.
	//
	// A nil Write refuses every call: a tool that silently accepts a plan
	// nobody keeps is worse than one that says it cannot.
	Write func(ctx context.Context, todos []Todo) error

	// Initial is the list this toolset starts holding — what Write persisted
	// on an earlier run, handed back so the model can go on addressing those
	// tasks by the ids it was already given.
	//
	// A host that keeps the list across runs MUST pass it. The list is
	// mutated in memory and Write receives the whole of it, so a toolset that
	// starts empty does not merely forget the earlier tasks: the first
	// mutation of a new run persists a list containing only that one task, and
	// everything the previous run wrote is gone. Nothing in the exchange looks
	// like a failure.
	Initial []Todo
}

// todoStore is the in-memory task list one toolset mutates, plus the minting
// of stable ids. It is created once by NewTodoTools and shared by all four
// tools, so the model edits a single live list that needs no reconciliation.
type todoStore struct {
	items []Todo
	next  int // the next id to hand out, monotonically increasing, never reused
}

// todoTool implements ONE of the four task-list tools.
type todoTool struct {
	kind  todoKind
	cfg   TodoConfig
	store *todoStore
}

// todoKind is which of the four mutations this tool performs.
type todoKind int

const (
	todoKindAdd todoKind = iota
	todoKindEdit
	todoKindCancel
	todoKindComplete
)

// NewTodoTools builds the four task-list tools (todo_add, todo_edit,
// todo_cancel, todo_complete), sharing one in-memory store, as a flat agentic.Tools
// slice. Each is a separate agentic.Tool; none is Readonly and none is approval-gated.
//
// The tools are NOT read-only. They write state the host owns and shows to the
// user, and a sub-agent inheriting them would overwrite its parent's plan;
// granting them to one is the caller's explicit choice (allowed_tools).
func NewTodoTools(cfg TodoConfig) agentic.Tools {
	store := &todoStore{items: append([]Todo(nil), cfg.Initial...), next: 1}
	// Mint above every id already in hand, so a restored list and a task added
	// to it can never share an id — which resolve would then refuse to act on.
	for _, t := range store.items {
		if t.ID >= store.next {
			store.next = t.ID + 1
		}
	}
	return agentic.Tools{
		&todoTool{kind: todoKindAdd, cfg: cfg, store: store},
		&todoTool{kind: todoKindEdit, cfg: cfg, store: store},
		&todoTool{kind: todoKindCancel, cfg: cfg, store: store},
		&todoTool{kind: todoKindComplete, cfg: cfg, store: store},
	}
}

func (e *todoTool) toolName() string {
	switch e.kind {
	case todoKindAdd:
		return TodoAddToolName
	case todoKindEdit:
		return TodoEditToolName
	case todoKindCancel:
		return TodoCancelToolName
	default:
		return TodoCompleteToolName
	}
}

func (e *todoTool) description() string {
	switch e.kind {
	case todoKindAdd:
		return todoAddDescription
	case todoKindEdit:
		return todoEditDescription
	case todoKindCancel:
		return todoCancelDescription
	default:
		return todoCompleteDescription
	}
}

// Decl advertises the one tool.
func (e *todoTool) Decl() agentic.ToolDecl {
	return agentic.ToolDecl{
		Name:        e.toolName(),
		Description: e.description(),
		InputSchema: e.schema(),
		// The task list is this run's own memory: nothing here leaves the
		// process, and none of the four tools throws work away — add appends,
		// and the other three move ONE task to a state it stays in, so
		// repeating any of them lands where the first call did.
		Destructive: agentic.Bool(false),
		Idempotent:  e.kind != todoKindAdd,
		OpenWorld:   agentic.Bool(false),
	}
}

// NeedsApproval always reports false: approval wiring stays the caller's
// concern, as with every built-in tool.
func (e *todoTool) NeedsApproval() bool { return false }

// schema is inferred from the tool's argument struct, per the hard rule that a
// tool's schema is never hand-written. closedState constrains the state field
// to the enum a host can render.
func (e *todoTool) schema() json.RawMessage {
	var closed map[string][]string
	if e.kind == todoKindAdd || e.kind == todoKindEdit {
		closed = map[string][]string{
			"state": {string(TodoPending), string(TodoInProgress), string(TodoDone)},
		}
	}
	switch e.kind {
	case todoKindAdd:
		return agentic.EnumSchema[todoAddArgs](closed)
	case todoKindEdit:
		return agentic.EnumSchema[todoEditArgs](closed)
	case todoKindCancel:
		return agentic.EnumSchema[todoCancelArgs](closed)
	default:
		return agentic.EnumSchema[todoCompleteArgs](closed)
	}
}

// todoAddArgs is the todo_add argument payload.
type todoAddArgs struct {
	Title string    `json:"title" jsonschema:"What the task is, in a few words."`
	State TodoState `json:"state,omitempty" jsonschema:"pending, in_progress or done. Defaults to pending."`
}

// todoEditArgs is the todo_edit argument payload.
type todoEditArgs struct {
	ID    int       `json:"id" jsonschema:"The id of the task to change, as shown in a previous reply."`
	Title string    `json:"title,omitempty" jsonschema:"The task's new title."`
	State TodoState `json:"state,omitempty" jsonschema:"The task's new state: pending, in_progress or done."`
}

// todoCancelArgs is the todo_cancel argument payload.
type todoCancelArgs struct {
	ID int `json:"id" jsonschema:"The id of the task to remove, as shown in a previous reply."`
}

// todoCompleteArgs is the todo_complete argument payload.
type todoCompleteArgs struct {
	ID int `json:"id" jsonschema:"The id of the task to mark done, as shown in a previous reply."`
}

// Execute runs the one mutation and hands the resulting list to the host.
// Every failure — unparseable arguments, an unusable task, a missing target, a
// store that refused — is a recoverable error tool result, never a Go error.
func (e *todoTool) Execute(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	if e.cfg.Write == nil {
		return agentic.ToolResult{Content: "the task list is unavailable: this run has nowhere to keep it", IsError: true}, nil
	}
	switch e.kind {
	case todoKindAdd:
		return e.doAdd(ctx, args)
	case todoKindEdit:
		return e.doEdit(ctx, args)
	case todoKindCancel:
		return e.doCancel(ctx, args)
	default:
		return e.doComplete(ctx, args)
	}
}

func (e *todoTool) doAdd(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	var in todoAddArgs
	if err := unmarshalArgs(e.toolName(), args, &in); err != "" {
		return agentic.ToolResult{Content: err, IsError: true}, nil
	}
	title, msg := e.validTitle(in.Title)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	if title == "" {
		return agentic.ToolResult{Content: "title is empty; every task needs one", IsError: true}, nil
	}
	state, msg := e.validState(in.State)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	if len(e.store.items) >= todoMaxItems {
		return agentic.ToolResult{Content: "too many tasks: the list holds at most " + strconv.Itoa(todoMaxItems) +
			". Track the work at a coarser grain.", IsError: true}, nil
	}
	id := e.store.next
	e.store.next++
	e.store.items = append(e.store.items, Todo{ID: id, Title: title, State: todoDefaultState(state)})
	return e.writeList(ctx)
}

func (e *todoTool) doEdit(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	var in todoEditArgs
	if err := unmarshalArgs(e.toolName(), args, &in); err != "" {
		return agentic.ToolResult{Content: err, IsError: true}, nil
	}
	idx, msg := e.resolve(in.ID)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	if in.Title == "" && in.State == "" {
		return agentic.ToolResult{Content: "todo_edit: nothing to change; provide a title and/or a state", IsError: true}, nil
	}
	title, msg := e.validTitle(in.Title)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	state, msg := e.validState(in.State)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	if title != "" {
		e.store.items[idx].Title = title
	}
	if state != "" {
		e.store.items[idx].State = state
	}
	return e.writeList(ctx)
}

func (e *todoTool) doCancel(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	var in todoCancelArgs
	if err := unmarshalArgs(e.toolName(), args, &in); err != "" {
		return agentic.ToolResult{Content: err, IsError: true}, nil
	}
	idx, msg := e.resolve(in.ID)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	e.store.items = append(e.store.items[:idx], e.store.items[idx+1:]...)
	return e.writeList(ctx)
}

func (e *todoTool) doComplete(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	var in todoCompleteArgs
	if err := unmarshalArgs(e.toolName(), args, &in); err != "" {
		return agentic.ToolResult{Content: err, IsError: true}, nil
	}
	idx, msg := e.resolve(in.ID)
	if msg != "" {
		return agentic.ToolResult{Content: msg, IsError: true}, nil
	}
	e.store.items[idx].State = TodoDone
	return e.writeList(ctx)
}

// resolve finds the single stored task with the given id. Ids are minted
// monotonically and never reused, so an id names at most one task; the check
// is kept so a corrupted store is refused rather than silently edited.
func (e *todoTool) resolve(id int) (int, string) {
	idx := -1
	for i := range e.store.items {
		if e.store.items[i].ID == id {
			if idx != -1 {
				// Ambiguous: more than one task shares the id. This cannot
				// happen through the tools, but a damaged store must not be
				// edited into a lie.
				return -1, "task id " + strconv.Itoa(id) + " is ambiguous: it names more than one task"
			}
			idx = i
		}
	}
	if idx == -1 {
		return -1, "no task with id " + strconv.Itoa(id) + "; use a task id from the latest reply"
	}
	return idx, ""
}

// validTitle normalizes and checks one title, returning a teaching error
// naming the title argument. An empty input means "no title being set": on add
// the caller refuses it as missing; on edit it means "not changing the title".
func (e *todoTool) validTitle(title string) (string, string) {
	if title == "" {
		return "", ""
	}
	trimmed := strings.TrimSpace(title)
	if trimmed == "" {
		return "", "title is empty; every task needs one"
	}
	if n := len([]rune(trimmed)); n > todoMaxTitleRunes {
		return "", "title has " + strconv.Itoa(n) + " characters; the limit is " +
			strconv.Itoa(todoMaxTitleRunes) + ". It is a task name, not a description."
	}
	return trimmed, ""
}

// validState normalizes and checks one state argument. An empty in.State means
// "not being changed" on edit and "pending" on add; the caller decides.
func (e *todoTool) validState(state TodoState) (TodoState, string) {
	if strings.TrimSpace(string(state)) == "" {
		return "", ""
	}
	s := TodoState(strings.TrimSpace(string(state)))
	if !knownTodoState(s) {
		return "", "state " + strconv.Quote(string(s)) + "; it must be one of " + todoStateList()
	}
	return s, ""
}

// todoDefaultState is the pending default a task with no state given starts at.
func todoDefaultState(state TodoState) TodoState {
	if state == "" {
		return TodoPending
	}
	return state
}

// writeList persists the store's whole list, then answers the model with the
// current list rendered in text and carried to the host as a todo_list part.
func (e *todoTool) writeList(ctx context.Context) (agentic.ToolResult, error) {
	list := e.store.items
	if list == nil {
		list = []Todo{}
	}
	if err := e.cfg.Write(ctx, list); err != nil {
		return agentic.ToolResult{Content: "could not save the task list: " + err.Error(), IsError: true}, nil
	}
	return agentic.ToolResult{Content: RenderTodos(list), Parts: []agentic.ToolContentPart{todoListPart(list)}}, nil
}

// todoListPart carries the stored list to the host.
func todoListPart(todos []Todo) agentic.ToolContentPart {
	b, err := json.Marshal(todos)
	if err != nil {
		// Todo is an int and two strings; Marshal cannot fail on it. An empty
		// array is still a valid list, so a host never reads a broken document.
		b = []byte("[]")
	}
	return agentic.ToolContentPart{Type: TodoListPartType, Text: string(b), MimeType: "application/json"}
}

func knownTodoState(s TodoState) bool {
	for _, k := range todoStates {
		if k == s {
			return true
		}
	}
	return false
}

// todoStateList is the states as the teaching error names them.
func todoStateList() string {
	names := make([]string, len(todoStates))
	for i, s := range todoStates {
		names[i] = string(s)
	}
	return strings.Join(names, ", ")
}

// RenderTodos is the model-facing rendering of a stored list: the
// confirmation every mutation answers with, and what a host shows where it has
// only text. Each line carries the task's stable id, so the model re-reads ids
// from the reply instead of recalling them. A call that dropped or renamed a
// task is visible in the reply rather than only on the user's screen.
func RenderTodos(todos []Todo) string {
	if len(todos) == 0 {
		return "Task list cleared."
	}
	var b strings.Builder
	b.WriteString("Task list updated (" + strconv.Itoa(len(todos)) + " tasks):")
	for _, t := range todos {
		b.WriteString("\n" + todoMark(t.State) + " #" + strconv.Itoa(t.ID) + " " + t.Title)
	}
	return b.String()
}

// todoMark is the checkbox one task is rendered with.
func todoMark(state TodoState) string {
	switch state {
	case TodoDone:
		return "[x]"
	case TodoInProgress:
		return "[~]"
	default:
		return "[ ]"
	}
}

// unmarshalArgs decodes a tool payload into out, returning the exact
// recoverable-error text naming the tool, or "" on success.
func unmarshalArgs(tool string, args json.RawMessage, out any) string {
	if len(args) == 0 {
		return "invalid " + tool + " arguments: empty payload"
	}
	if err := json.Unmarshal(args, out); err != nil {
		return "invalid " + tool + " arguments: " + err.Error()
	}
	return ""
}
