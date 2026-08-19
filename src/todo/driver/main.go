// Command todo_driver is the entry-point launch check for the four task-list
// tools. It drives the real NewTodoTools constructor and each tool's Execute
// through a realistic mutation sequence, and verifies that the host's recorded
// store reaches exactly the intended end state. It exits 0 only when the
// shipped code behaved; running it twice must produce identical output.
package main

import (
	"context"
	"fmt"
	"os"

	"github.com/wow-look-at-my/agentic-loop/src/todo"
)

// recorder is a host store: it appends every post-mutation list the executor
// hands it, so the driver can prove the host saw exactly the changes intended.
type recorder struct {
	snapshots [][]todo.Todo
}

func (r *recorder) write(_ context.Context, todos []todo.Todo) error {
	r.snapshots = append(r.snapshots, todos)
	return nil
}

func main() {
	rec := &recorder{}
	tools := todo.NewTodoTools(todo.TodoConfig{Write: rec.write})

	run := func(name string, args string) error {
		tool, ok := tools.Find(name)
		if !ok {
			return fmt.Errorf("tool %s not offered", name)
		}
		result, err := tool.Execute(context.Background(), []byte(args))
		if err != nil {
			return fmt.Errorf("%s returned a Go error: %v", name, err)
		}
		if result.IsError {
			return fmt.Errorf("%s refused: %s", name, result.Content)
		}
		return nil
	}

	// The realistic sequence: add two, edit one, complete one, cancel one.
	for i, args := range []string{
		`{"title":"write it"}`,
		`{"title":"test it","state":"in_progress"}`,
		`{"id":2,"state":"done"}`,
		`{"id":1}`,
		`{"id":2}`,
	} {
		name := [5]string{"todo_add", "todo_add", "todo_edit", "todo_complete", "todo_cancel"}[i]
		if err := run(name, args); err != nil {
			fmt.Fprintln(os.Stderr, "FAIL:", err)
			os.Exit(1)
		}
	}

	// The intended end state after the sequence above:
	//   add "write it"        -> [#1 pending]
	//   add "test it" in_prog -> [#1 pending, #2 in_progress]
	//   edit #2 done          -> [#1 pending, #2 done]
	//   complete #1           -> [#1 done, #2 done]
	//   cancel #2             -> [#1 done]
	want := []todo.Todo{{ID: 1, Title: "write it", State: todo.TodoDone}}
	got := rec.snapshots[len(rec.snapshots)-1]
	if len(got) != len(want) {
		fmt.Fprintln(os.Stderr, "FAIL: final host list has", len(got), "task(s), want", len(want))
		os.Exit(1)
	}
	for i := range want {
		if got[i] != want[i] {
			fmt.Fprintf(os.Stderr, "FAIL: final host task %d mismatch: got %+v, want %+v\n", i, got[i], want[i])
			os.Exit(1)
		}
	}

	// Consistent, non-empty output that a second run reproduces exactly.
	fmt.Println("todo_driver: sequence passed; host stored", len(rec.snapshots), "lists, final list:")
	for _, t := range got {
		fmt.Println("  id", t.ID, t.State, t.Title)
	}
}
