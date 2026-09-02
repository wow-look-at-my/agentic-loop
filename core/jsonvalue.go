package commonai

import (
	"bytes"
	"encoding/json"
	"fmt"
)

// marshalValue renders a passthrough parameter value as JSON -- which is what
// it is, since it is handed to a provider verbatim.
func marshalValue(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("commonai: encoding a parameter value: %w", err)
	}
	return raw, nil
}

// unmarshalValue reads a passthrough value back. Numbers stay json.Number so a
// value that came in as does not go out as, and an integer id does not
// come back in scientific notation.
func unmarshalValue(raw []byte) (any, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return nil, fmt.Errorf("commonai: decoding a parameter value: %w", err)
	}
	return v, nil
}
