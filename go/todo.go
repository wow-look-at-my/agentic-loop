package agentic

import (
	"context"
	"encoding/json"
	"strconv"
	"strings"
)

// TodoWriteToolName is the advertised name of the built-in task-list tool.
const TodoWriteToolName = "todo_write"

// The task-list caps. A plan longer than this is not a plan, and a title long
// enough to be a description does not fit the narrow surface a host renders
// one in.
const (
	todoMaxItems      = 100
	todoMaxTitleRunes = 200

	todoWriteDescription = "Records your task list for this conversation, which the host shows to the user. " +
		"Use it for any job with several steps: write the plan down before starting, then call this again after each " +
		"step to mark what is done and what you are on now. It is how the user follows a long piece of work. " +
		"Every call REPLACES the whole list with what you send, so always send every task, including the finished ones " +
		"- sending only the ones that changed deletes the rest. Exactly one task should be in_progress at a time. " +
		"Send an empty list to clear it."
)

// todoWriteSchema is the tool's parameter schema. The state enum is what stops
// a model inventing a fourth state nothing renders.
var todoWriteSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "todos": {
      "type": "array",
      "description": "The complete task list in order. It replaces the previous one entirely, so include the tasks that have not changed.",
      "items": {
        "type": "object",
        "additionalProperties": false,
        "properties": {
          "title": {
            "type": "string",
            "description": "What the task is, in a few words."
          },
          "state": {
            "type": "string",
            "enum": ["pending", "in_progress", "done"],
            "description": "pending, in_progress or done. Defaults to pending."
          }
        },
        "required": ["title"]
      }
    }
  },
  "required": ["todos"]
}`)

// TodoState is one task's state. The set is closed: a host renders each one,
// and a state it does not know would show as a task with no mark.
type TodoState string

// The task states, in the order the schema advertises them.
const (
	TodoPending    TodoState = "pending"
	TodoInProgress TodoState = "in_progress"
	TodoDone       TodoState = "done"
)

// todoStates is the closed set, for validation and for the teaching error.
var todoStates = []TodoState{TodoPending, TodoInProgress, TodoDone}

// Todo is one task of a run's list, as the model stated it. Order is the
// model's, and the list carries no ids: it is REPLACED whole on every write,
// so nothing has to be tracked across turns.
type Todo struct {
	Title string    `json:"title"`
	State TodoState `json:"state"`
}

// TodoConfig configures NewTodoExecutor.
type TodoConfig struct {
	// Write receives the run's whole new task list, already validated, and
	// persists (or displays) it however the host wants. A non-nil error is
	// reported to the model as a recoverable failure, so a list that was not
	// stored is never reported as stored.
	//
	// A nil Write refuses every call: a tool that silently accepts a plan
	// nobody keeps is worse than one that says it cannot.
	Write func(ctx context.Context, todos []Todo) error
}

// todoExecutor implements todo_write.
type todoExecutor struct {
	cfg TodoConfig
}

// NewTodoExecutor builds the todo_write tool executor: the model's own task
// list, replaced whole on every call and handed to the host to keep.
//
// The tool is NOT read-only. It writes state the host owns and shows to the
// user, and a sub-agent inheriting it would overwrite its parent's plan with
// its own; granting it to one is the caller's explicit choice (allowed_tools).
func NewTodoExecutor(cfg TodoConfig) ToolExecutor { return &todoExecutor{cfg: cfg} }

// Tools advertises todo_write.
func (e *todoExecutor) Tools() []Tool {
	return []Tool{{
		Name:        TodoWriteToolName,
		Description: todoWriteDescription,
		InputSchema: todoWriteSchema,
	}}
}

// NeedsApproval always reports false: approval wiring stays the caller's
// concern, as with every built-in executor.
func (e *todoExecutor) NeedsApproval(string) bool { return false }

// todoWriteArgs is the todo_write argument payload.
type todoWriteArgs struct {
	Todos []Todo `json:"todos"`
}

// Execute validates the list and hands it to the host. Every failure —
// unparseable arguments, an unusable task, a store that refused — is a
// recoverable error tool result, never a Go error.
func (e *todoExecutor) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if call.Name != TodoWriteToolName {
		return ToolResult{Content: "unknown tool: " + call.Name, IsError: true}, nil
	}
	if e.cfg.Write == nil {
		return ToolResult{Content: "the task list is unavailable: this run has nowhere to keep it", IsError: true}, nil
	}
	var in todoWriteArgs
	if len(call.Arguments) > 0 {
		if err := json.Unmarshal([]byte(call.Arguments), &in); err != nil {
			return ToolResult{Content: "invalid todo_write arguments: " + err.Error(), IsError: true}, nil
		}
	}
	todos, msg := validateTodos(in.Todos)
	if msg != "" {
		return ToolResult{Content: msg, IsError: true}, nil
	}
	if err := e.cfg.Write(ctx, todos); err != nil {
		return ToolResult{Content: "could not save the task list: " + err.Error(), IsError: true}, nil
	}
	return ToolResult{Content: RenderTodos(todos)}, nil
}

// validateTodos normalizes and checks the model's list, returning a teaching
// error naming the offending item. It returns a non-nil slice for an empty
// list, so a host's Write can tell "clear it" from "nothing was decoded".
func validateTodos(in []Todo) ([]Todo, string) {
	if len(in) > todoMaxItems {
		return nil, "too many tasks: the list holds at most " + strconv.Itoa(todoMaxItems) +
			", and you sent " + strconv.Itoa(len(in)) + ". Track the work at a coarser grain."
	}
	out := make([]Todo, 0, len(in))
	for i, it := range in {
		where := "todos[" + strconv.Itoa(i) + "]"
		title := strings.TrimSpace(it.Title)
		if title == "" {
			return nil, where + " has an empty title; every task needs one"
		}
		if n := len([]rune(title)); n > todoMaxTitleRunes {
			return nil, where + " has a title of " + strconv.Itoa(n) + " characters; the limit is " +
				strconv.Itoa(todoMaxTitleRunes) + ". It is a task name, not a description."
		}
		state := TodoState(strings.TrimSpace(string(it.State)))
		if state == "" {
			state = TodoPending
		}
		if !knownTodoState(state) {
			return nil, where + " has state " + strconv.Quote(string(state)) + "; it must be one of " + todoStateList()
		}
		out = append(out, Todo{Title: title, State: state})
	}
	return out, ""
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

// RenderTodos is the model-facing rendering of a stored list: the confirmation
// todo_write answers with, and what a host shows where it has only text. A
// call that dropped or renamed a task is visible in the reply rather than only
// on the user's screen.
func RenderTodos(todos []Todo) string {
	if len(todos) == 0 {
		return "Task list cleared."
	}
	var b strings.Builder
	b.WriteString("Task list updated (" + strconv.Itoa(len(todos)) + " tasks):")
	for _, t := range todos {
		b.WriteString("\n" + todoMark(t.State) + " " + t.Title)
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
