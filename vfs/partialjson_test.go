package vfs

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

// The scanner reads "path" off whatever prefix of the arguments has arrived, and says when it cannot yet.
func TestPartialStringFieldReadsThePathAsItArrives(t *testing.T) {
	cases := []struct {
		name   string
		prefix string
		want   string
		state  fieldState
	}{
		{"nothing yet", "", "", fieldPending},
		{"open brace only", `{`, "", fieldPending},
		{"key cut off", `{"pa`, "", fieldPending},
		{"colon pending", `{"path"`, "", fieldPending},
		{"value cut off", `{"path": "/work/ma`, "", fieldPending},
		{"escape cut off", `{"path": "/work/a\`, "", fieldPending},
		{"value whole, object open", `{"path": "/work/main.go", "content": "pack`, "/work/main.go", fieldFound},
		{"value whole, object closed", `{"path":"/work/main.go","content":"x"}`, "/work/main.go", fieldFound},
		{"escaped value", `{"path":"/work/space \"q\".go",`, `/work/space "q".go`, fieldFound},
		{"path after content", `{"content":"line {with} \"braces\"","path":"/work/new.go"`, "/work/new.go", fieldFound},
		{"path after nested", `{"meta":{"a":[1,{"b":"}"}]},"path":"/work/n.go"`, "/work/n.go", fieldFound},
		{"content arriving first", `{"content":"still writ`, "", fieldPending},
		{"number arriving first", `{"limit": 12`, "", fieldPending},
		{"object closed without path", `{"content":"x"}`, "", fieldAbsent},
		{"path is not a string", `{"path": 3,`, "", fieldAbsent},
		{"not an object", `["path"]`, "", fieldAbsent},
		{"leading space", "  \n{\"path\":\"/w/a\"", "/w/a", fieldFound},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, state := partialStringField(c.prefix, "path")
			assert.Equal(t, c.state, state)
			assert.Equal(t, c.want, got)
		})
	}
}
