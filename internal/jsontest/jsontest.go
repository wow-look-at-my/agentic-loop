// Package jsontest is the place test fixtures build JSON from Go values.
package jsontest

import (
	"encoding/json"
	"fmt"
)

// Must marshals v to compact JSON. It panics: the input is a test literal.
func Must(v any) string {
	raw, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("jsonMust: marshaling %T: %v", v, err))
	}
	return string(raw)
}

// Obj is a JSON object literal.
type Obj = map[string]any

// Arr is a JSON array literal.
type Arr = []any
