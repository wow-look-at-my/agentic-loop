package commonai

import (
	"encoding/json"
	"fmt"
	"iter"
	"reflect"
	"sort"
	"sync"
)

// MetaKey names one typed app field. An app registers each key once, at init.
type MetaKey[T any] struct{ id string }

// ID is the key's GUID.
func (k MetaKey[T]) ID() string { return k.id }

// MetaKeyInfo describes one registered key.
type MetaKeyInfo struct {
	ID, Name, Owner string
	Type            reflect.Type
}

var metaRegistry = struct {
	sync.RWMutex
	keys map[string]MetaKeyInfo
}{keys: map[string]MetaKeyInfo{}}

// RegisterMetaKey registers a key. A second registration of one id panics.
func RegisterMetaKey[T any](id, name, owner string) MetaKey[T] {
	metaRegistry.Lock()
	defer metaRegistry.Unlock()
	if prev, ok := metaRegistry.keys[id]; ok {
		panic(fmt.Sprintf("commonai: meta key %s is already registered as %q by %s", id, prev.Name, prev.Owner))
	}
	metaRegistry.keys[id] = MetaKeyInfo{ID: id, Name: name, Owner: owner, Type: reflect.TypeFor[T]()}
	return MetaKey[T]{id: id}
}

// RegisteredMetaKeys lists every registered key, by id.
func RegisteredMetaKeys() []MetaKeyInfo {
	metaRegistry.RLock()
	defer metaRegistry.RUnlock()
	out := make([]MetaKeyInfo, 0, len(metaRegistry.keys))
	for _, k := range metaRegistry.keys {
		out = append(out, k)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func registeredMeta(id string) (MetaKeyInfo, bool) {
	metaRegistry.RLock()
	defer metaRegistry.RUnlock()
	k, ok := metaRegistry.keys[id]
	return k, ok
}

// metaEntry holds a typed value, or the raw JSON of a key this process did not register.
type metaEntry struct {
	value any
	raw   json.RawMessage
}

// Meta is app data on a session, a node or a content block. The zero value is usable.
type Meta struct {
	mu sync.Mutex
	m  map[string]metaEntry
}

// GetMeta reads a typed entry.
func GetMeta[T any](m *Meta, k MetaKey[T]) (T, bool) {
	var zero T
	if _, ok := registeredMeta(k.id); !ok {
		panic(fmt.Sprintf("commonai: meta key %q is not registered", k.id))
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.m[k.id]
	if !ok {
		return zero, false
	}
	if e.raw != nil {
		var v T
		if err := json.Unmarshal(e.raw, &v); err != nil {
			panic(fmt.Sprintf("commonai: stored meta %q does not decode into %T: %v", k.id, v, err))
		}
		m.m[k.id] = metaEntry{value: v}
		return v, true
	}
	return e.value.(T), true
}

// SetMeta writes a typed entry. A nil value deletes it. A value that cannot be JSON-encoded panics.
func SetMeta[T any](m *Meta, k MetaKey[T], v T) {
	if _, ok := registeredMeta(k.id); !ok {
		panic(fmt.Sprintf("commonai: meta key %q is not registered", k.id))
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	if isNil(v) {
		delete(m.m, k.id)
		return
	}
	if _, err := json.Marshal(v); err != nil {
		panic(fmt.Sprintf("commonai: meta %q cannot be stored: %v", k.id, err))
	}
	if m.m == nil {
		m.m = map[string]metaEntry{}
	}
	m.m[k.id] = metaEntry{value: v}
}

func isNil(v any) bool {
	if v == nil {
		return true
	}
	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer, reflect.Map, reflect.Slice, reflect.Interface, reflect.Func, reflect.Chan:
		return rv.IsNil()
	default:
		return false
	}
}

// All ranges over every entry, by id. A key this process did not register yields its raw JSON.
func (m *Meta) All() iter.Seq2[string, any] {
	return func(yield func(string, any) bool) {
		m.mu.Lock()
		ids := make([]string, 0, len(m.m))
		vals := make(map[string]any, len(m.m))
		for id, e := range m.m {
			ids = append(ids, id)
			if e.raw != nil {
				vals[id] = e.raw
			} else {
				vals[id] = e.value
			}
		}
		m.mu.Unlock()
		sort.Strings(ids)
		for _, id := range ids {
			if !yield(id, vals[id]) {
				return
			}
		}
	}
}

// MarshalEntries is every entry as JSON, for a store.
func (m *Meta) MarshalEntries() (map[string]json.RawMessage, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make(map[string]json.RawMessage, len(m.m))
	for id, e := range m.m {
		if e.raw != nil {
			out[id] = e.raw
			continue
		}
		b, err := json.Marshal(e.value)
		if err != nil {
			return nil, fmt.Errorf("commonai: meta %q: %w", id, err)
		}
		out[id] = b
	}
	return out, nil
}

// LoadEntries restores entries a store wrote. Each decodes when it is first
// read.
func (m *Meta) LoadEntries(entries map[string]json.RawMessage) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if len(entries) == 0 {
		return
	}
	if m.m == nil {
		m.m = make(map[string]metaEntry, len(entries))
	}
	for id, raw := range entries {
		m.m[id] = metaEntry{raw: append(json.RawMessage(nil), raw...)}
	}
}

// Len is the number of entries.
func (m *Meta) Len() uint {
	m.mu.Lock()
	defer m.mu.Unlock()
	return uint(len(m.m))
}
