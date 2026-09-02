package commonai

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strings"
)

// The param tree stores arbitrary JSON data in the format as a validating element: a real tree, not a JSON text node.

// Param value types, matching JSON's own.
const (
	ParamString  = "string"
	ParamNumber  = "number"
	ParamBoolean = "boolean"
	ParamNull    = "null"
	ParamObject  = "object"
	ParamArray   = "array"
)

// Param is node of the value tree: a named object member, an unnamed array item, or the root.
type Param struct {
	Name     string
	Type     string
	Value    string
	Children []Param
}

// ParamFromJSON converts JSON value into a Param named name. Numbers keep
// their literal text and object members keep their order, so the value that
// comes back out of JSON is the that went in.
func ParamFromJSON(name string, raw []byte) (Param, error) {
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	p, err := decodeParam(dec, name)
	if err != nil {
		return Param{}, err
	}
	if _, err := dec.Token(); err == nil {
		return Param{}, fmt.Errorf("commonai: trailing data after JSON value %q", name)
	}
	return p, nil
}

// ParamsFromJSONObject converts a JSON OBJECT into its members, in order. It
// is the shape a request's provider parameters and a usage object take.
func ParamsFromJSONObject(raw []byte) ([]Param, error) {
	if len(bytes.TrimSpace(raw)) == 0 {
		return nil, nil
	}
	p, err := ParamFromJSON("", raw)
	if err != nil {
		return nil, err
	}
	if p.Type != ParamObject {
		return nil, fmt.Errorf("commonai: expected a JSON object, got %s", p.Type)
	}
	return p.Children, nil
}

// decodeParam reads value from dec, which must be positioned at its
// token.
func decodeParam(dec *json.Decoder, name string) (Param, error) {
	tok, err := dec.Token()
	if err != nil {
		return Param{}, fmt.Errorf("commonai: reading JSON value %q: %w", name, err)
	}
	return decodeParamFrom(dec, name, tok)
}

// decodeParamFrom builds a Param from an already-read token.
func decodeParamFrom(dec *json.Decoder, name string, tok json.Token) (Param, error) {
	switch v := tok.(type) {
	case json.Delim:
		switch v {
		case '{':
			return decodeObject(dec, name)
		case '[':
			return decodeArray(dec, name)
		}
		return Param{}, fmt.Errorf("commonai: unexpected %q in JSON value %q", v, name)
	case string:
		return Param{Name: name, Type: ParamString, Value: v}, nil
	case json.Number:
		return Param{Name: name, Type: ParamNumber, Value: v.String()}, nil
	case bool:
		return Param{Name: name, Type: ParamBoolean, Value: boolText(v)}, nil
	case nil:
		return Param{Name: name, Type: ParamNull}, nil
	}
	return Param{}, fmt.Errorf("commonai: unsupported JSON token in value %q", name)
}

// decodeObject reads members until the closing brace, keeping their order.
func decodeObject(dec *json.Decoder, name string) (Param, error) {
	p := Param{Name: name, Type: ParamObject}
	for dec.More() {
		keyTok, err := dec.Token()
		if err != nil {
			return Param{}, fmt.Errorf("commonai: reading a JSON member name in %q: %w", name, err)
		}
		key, ok := keyTok.(string)
		if !ok {
			return Param{}, fmt.Errorf("commonai: a JSON member name in %q is not a string", name)
		}
		child, err := decodeParam(dec, key)
		if err != nil {
			return Param{}, err
		}
		p.Children = append(p.Children, child)
	}
	if _, err := dec.Token(); err != nil {
		return Param{}, fmt.Errorf("commonai: reading the end of JSON object %q: %w", name, err)
	}
	return p, nil
}

// decodeArray reads items until the closing bracket. Items carry no name.
func decodeArray(dec *json.Decoder, name string) (Param, error) {
	p := Param{Name: name, Type: ParamArray}
	for dec.More() {
		child, err := decodeParam(dec, "")
		if err != nil {
			return Param{}, err
		}
		p.Children = append(p.Children, child)
	}
	if _, err := dec.Token(); err != nil {
		return Param{}, fmt.Errorf("commonai: reading the end of JSON array %q: %w", name, err)
	}
	return p, nil
}

// JSON renders the param back to JSON, reproducing member order and scalar
// literals exactly as they arrived.
func (p Param) JSON() ([]byte, error) {
	var b bytes.Buffer
	if err := p.writeJSON(&b); err != nil {
		return nil, err
	}
	return b.Bytes(), nil
}

// ParamsJSON renders a list of named params as JSON object.
func ParamsJSON(params []Param) ([]byte, error) {
	return Param{Type: ParamObject, Children: params}.JSON()
}

// writeJSON renders value.
func (p Param) writeJSON(b *bytes.Buffer) error {
	switch p.Type {
	case ParamObject:
		b.WriteByte('{')
		for i, c := range p.Children {
			if i > 0 {
				b.WriteByte(',')
			}
			key, err := json.Marshal(c.Name)
			if err != nil {
				return fmt.Errorf("commonai: encoding member name %q: %w", c.Name, err)
			}
			b.Write(key)
			b.WriteByte(':')
			if err := c.writeJSON(b); err != nil {
				return err
			}
		}
		b.WriteByte('}')
		return nil
	case ParamArray:
		b.WriteByte('[')
		for i, c := range p.Children {
			if i > 0 {
				b.WriteByte(',')
			}
			if err := c.writeJSON(b); err != nil {
				return err
			}
		}
		b.WriteByte(']')
		return nil
	case ParamString:
		s, err := json.Marshal(p.Value)
		if err != nil {
			return fmt.Errorf("commonai: encoding string value: %w", err)
		}
		b.Write(s)
		return nil
	case ParamNumber:
		if !validNumberLiteral(p.Value) {
			return fmt.Errorf("commonai: %q is not a JSON number", p.Value)
		}
		b.WriteString(p.Value)
		return nil
	case ParamBoolean:
		switch p.Value {
		case "true", "false":
			b.WriteString(p.Value)
			return nil
		}
		return fmt.Errorf("commonai: %q is not a boolean", p.Value)
	case ParamNull:
		b.WriteString("null")
		return nil
	}
	return fmt.Errorf("commonai: unknown param type %q", p.Type)
}

// validNumberLiteral reports whether s is a JSON number, so a hand-written
// document cannot smuggle arbitrary text through a number-typed param.
func validNumberLiteral(s string) bool {
	if s == "" {
		return false
	}
	var n json.Number
	dec := json.NewDecoder(strings.NewReader(s))
	dec.UseNumber()
	if err := dec.Decode(&n); err != nil {
		return false
	}
	if _, err := dec.Token(); err == nil {
		return false
	}
	return n.String() == s
}

// boolText renders a boolean the way both JSON and XSD spell it.
func boolText(v bool) string {
	if v {
		return "true"
	}
	return "false"
}
