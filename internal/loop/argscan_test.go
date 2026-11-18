package loop

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

type member struct{ key, value string }

func scanAll(chunks ...string) []member {
	var s argScanner
	var out []member
	for _, c := range chunks {
		for _, m := range s.feed(c) {
			out = append(out, member{m.key, string(m.value)})
		}
	}
	return out
}

// Every member is reported once its value ends, with the same answer however the text is split.
func TestArgScannerReportsEachMemberAsItCompletes(t *testing.T) {
	cases := []struct {
		name string
		doc  string
		want []member
	}{
		{"strings", `{"path": "/work/main.go", "content": "package main\n"}`,
			[]member{{"path", `"/work/main.go"`}, {"content", `"package main\n"`}}},
		{"escapes", `{"path":"sp\"ace\\"}`, []member{{"path", `"sp\"ace\\"`}}},
		{"nested", `{"meta":{"a":[1,{"b":"}]"}]},"path":"/n.go"}`,
			[]member{{"meta", `{"a":[1,{"b":"}]"}]}`}, {"path", `"/n.go"`}}},
		{"scalars", `{"limit": 12, "ok":true,"x":null}`,
			[]member{{"limit", "12"}, {"ok", "true"}, {"x", "null"}}},
		{"space everywhere", " \n{ \"a\" :\t\"b\" ,\r\n\"c\" : 1 } ", []member{{"a", `"b"`}, {"c", "1"}}},
		{"empty object", `{}`, nil},
		{"not an object", `["path","x"]`, nil},
		{"broken after a member", `{"a":"b" "c":"d"}`, []member{{"a", `"b"`}}},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			assert.Equal(t, c.want, scanAll(c.doc), "whole")
			bytes := make([]string, len(c.doc))
			for i := range c.doc {
				bytes[i] = c.doc[i : i+1]
			}
			assert.Equal(t, c.want, scanAll(bytes...), "a byte at a time")
			for i := 1; i < len(c.doc); i++ {
				assert.Equal(t, c.want, scanAll(c.doc[:i], c.doc[i:]), "split at %d", i)
			}
		})
	}
}

// A member is reported the moment its value ends, not when the object does, and a value still
// arriving is not reported at all.
func TestArgScannerReportsNoMemberEarly(t *testing.T) {
	var s argScanner
	assert.Empty(t, s.feed(`{"path": "/work/ma`))
	got := s.feed(`in.go", "content": "pack`)
	assert.Equal(t, []argMember{{key: "path", value: []byte(`"/work/main.go"`)}}, got)
	assert.False(t, s.ended())

	// A number at the end of what arrived may still grow.
	var n argScanner
	assert.Empty(t, n.feed(`{"limit": 12`))
	assert.Equal(t, []argMember{{key: "limit", value: []byte("123")}}, n.feed(`3}`))
	assert.True(t, n.ended())

	var dead argScanner
	dead.feed(`[`)
	assert.True(t, dead.ended())
}
