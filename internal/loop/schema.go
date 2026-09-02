package loop

import (
	"encoding/json"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// A tool's JSON Schema is INFERRED from the Go struct its handler decodes.

// InferSchema builds the schema for a tool's argument struct.
func InferSchema[In any]() json.RawMessage { return EnumSchema[In](nil) }

// EnumSchema is InferSchema for a struct with closed-set string fields; unknown property names panic.
func EnumSchema[In any](enums map[string][]string) json.RawMessage {
	typ := reflect.TypeFor[In]()
	if typ.Kind() != reflect.Struct {
		panic(fmt.Sprintf("agentic: a tool's arguments must be a struct, got %s", typ))
	}
	props := structProps(typ)

	seen := set.New[string](len(props))
	for _, p := range props {
		seen.Add(p.name)
	}
	for name := range enums {
		if !seen.Contains(name) {
			panic(fmt.Sprintf("agentic: cannot constrain unknown property %q on %s", name, typ))
		}
	}

	var b strings.Builder
	b.WriteString(`{"type":"object","properties":{`)
	var required []string
	for i, p := range props {
		if i > 0 {
			b.WriteByte(',')
		}
		b.Write(mustJSON(p.name))
		b.WriteByte(':')
		p.write(&b, enums[p.name])
		if !p.optional {
			required = append(required, p.name)
		}
	}
	b.WriteString(`},"required":`)
	if required == nil {
		required = []string{}
	}
	b.Write(mustJSON(required))
	b.WriteString(`,"additionalProperties":false}`)
	return json.RawMessage(b.String())
}

// prop is advertised argument.
type prop struct {
	name        string
	jsonType    string
	description string
	optional    bool
	// items is the element type for an array, empty otherwise.
	items string
}

func (p prop) write(b *strings.Builder, enum []string) {
	b.WriteString(`{"type":`)
	b.Write(mustJSON(p.jsonType))
	if p.description != "" {
		b.WriteString(`,"description":`)
		b.Write(mustJSON(p.description))
	}
	if len(enum) > 0 {
		b.WriteString(`,"enum":`)
		b.Write(mustJSON(enum))
	}
	if p.items != "" {
		b.WriteString(`,"items":{"type":`)
		b.Write(mustJSON(p.items))
		b.WriteByte('}')
	}
	b.WriteByte('}')
}

// structProps reads a struct's fields in declaration order for a stable toolset.
func structProps(typ reflect.Type) []prop {
	var out []prop
	for i := range typ.NumField() {
		f := typ.Field(i)
		if !f.IsExported() {
			continue
		}
		if f.Anonymous && f.Type.Kind() == reflect.Struct {
			out = append(out, structProps(f.Type)...)
			continue
		}
		tag, ok := f.Tag.Lookup("json")
		if !ok {
			// encoding/json would fall back to the Go name case-insensitively -- a decoy field name.
			panic(fmt.Sprintf("agentic: %s.%s has no json tag; every argument field must name itself", typ.Name(), f.Name))
		}
		name, opts, _ := strings.Cut(tag, ",")
		if name == "-" && opts == "" {
			continue
		}
		if name == "" {
			panic(fmt.Sprintf("agentic: %s.%s has an empty json name", typ.Name(), f.Name))
		}
		p := prop{
			name:        name,
			description: f.Tag.Get("jsonschema"),
			optional:    slices.Contains(strings.Split(opts, ","), "omitempty"),
		}
		p.jsonType, p.items = schemaType(typ, f)
		out = append(out, p)
	}
	return out
}

// schemaType maps a field's Go type to its JSON type, panicking on anything a
// tool argument has no business being -- an unsupported field advertised as
// the wrong type is worse than a build that stops.
func schemaType(owner reflect.Type, f reflect.StructField) (jsonType, items string) {
	t := f.Type
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.Kind() == reflect.Slice {
		elem := scalarType(t.Elem())
		if elem == "" {
			panic(fmt.Sprintf("agentic: %s.%s is []%s; a tool argument list must hold scalars", owner.Name(), f.Name, t.Elem()))
		}
		return "array", elem
	}
	if s := scalarType(t); s != "" {
		return s, ""
	}
	panic(fmt.Sprintf("agentic: %s.%s is %s, which has no tool-argument schema", owner.Name(), f.Name, f.Type))
}

func scalarType(t reflect.Type) string {
	switch t.Kind() {
	case reflect.String:
		return "string"
	case reflect.Bool:
		return "boolean"
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return "integer"
	case reflect.Float32, reflect.Float64:
		return "number"
	}
	return ""
}

// mustJSON encodes a value that cannot fail to encode.
func mustJSON(v any) []byte {
	b, err := json.Marshal(v)
	if err != nil {
		panic(fmt.Sprintf("agentic: encoding a schema fragment: %v", err))
	}
	return b
}
