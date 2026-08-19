package session

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// File is a Store backed by one <conversation> document per session, in a
// directory. The documents are the format's own, so a stored session is
// readable, editable and movable with nothing but a text editor -- and every
// read validates against the schema, because a file on disk is exactly where a
// document can be changed by something that is not this program.
type File struct {
	mu  sync.Mutex
	dir string
	seq int
}

// NewFile returns a store over dir, creating it if it is not there.
func NewFile(dir string) (*File, error) {
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("session: creating %s: %w", dir, err)
	}
	return &File{dir: dir}, nil
}

// path is the document for an id.
func (f *File) path(id string) string { return filepath.Join(f.dir, id+".xml") }

// Create implements Store.
func (f *File) Create(req commonai.Request) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	for {
		f.seq++
		id := fmt.Sprintf("c%d", f.seq)
		if _, err := os.Stat(f.path(id)); err == nil {
			// A directory that outlived the process already holds this id.
			continue
		}
		return id, f.write(id, req)
	}
}

// Put implements Store.
func (f *File) Put(id string, req commonai.Request) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.write(id, req)
}

// Get implements Store.
func (f *File) Get(id string) (commonai.Request, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.read(id)
}

// Append implements Store.
func (f *File) Append(id string, msgs ...commonai.Message) (commonai.Request, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	req, err := f.read(id)
	if err != nil {
		return commonai.Request{}, err
	}
	req.Messages = append(req.Messages, msgs...)
	if err := f.write(id, req); err != nil {
		return commonai.Request{}, err
	}
	return req, nil
}

// Delete implements Store.
func (f *File) Delete(id string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := validID(id); err != nil {
		return err
	}
	if err := os.Remove(f.path(id)); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: %q", ErrNotFound, id)
		}
		return fmt.Errorf("session: deleting %q: %w", id, err)
	}
	return nil
}

// List implements Store.
func (f *File) List() ([]string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	entries, err := os.ReadDir(f.dir)
	if err != nil {
		return nil, fmt.Errorf("session: listing %s: %w", f.dir, err)
	}
	var ids []string
	for _, e := range entries {
		name := e.Name()
		if e.IsDir() || !strings.HasSuffix(name, ".xml") {
			continue
		}
		ids = append(ids, strings.TrimSuffix(name, ".xml"))
	}
	sort.Strings(ids)
	return ids, nil
}

// read loads and validates one conversation document.
func (f *File) read(id string) (commonai.Request, error) {
	if err := validID(id); err != nil {
		return commonai.Request{}, err
	}
	data, err := os.ReadFile(f.path(id))
	if err != nil {
		if os.IsNotExist(err) {
			return commonai.Request{}, fmt.Errorf("%w: %q", ErrNotFound, id)
		}
		return commonai.Request{}, fmt.Errorf("session: reading %q: %w", id, err)
	}
	if err := commonai.Validate(data); err != nil {
		return commonai.Request{}, fmt.Errorf("session: stored conversation %q: %w", id, err)
	}
	storedID, req, err := commonai.DecodeConversation(data)
	if err != nil {
		return commonai.Request{}, fmt.Errorf("session: stored conversation %q: %w", id, err)
	}
	if storedID != id {
		return commonai.Request{}, fmt.Errorf("session: %s.xml holds conversation %q", id, storedID)
	}
	return req, nil
}

// write stores one conversation document, and validates what it is about to
// write: a document this store cannot read back is not worth keeping.
func (f *File) write(id string, req commonai.Request) error {
	if err := validID(id); err != nil {
		return err
	}
	var buf strings.Builder
	if err := commonai.EncodeConversation(&buf, id, req); err != nil {
		return fmt.Errorf("session: encoding %q: %w", id, err)
	}
	data := []byte(buf.String())
	if err := commonai.Validate(data); err != nil {
		return fmt.Errorf("session: encoding %q: %w", id, err)
	}
	tmp := f.path(id) + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return fmt.Errorf("session: writing %q: %w", id, err)
	}
	if err := os.Rename(tmp, f.path(id)); err != nil {
		return fmt.Errorf("session: writing %q: %w", id, err)
	}
	return nil
}
