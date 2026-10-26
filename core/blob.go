package commonai

import (
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
)

// BlobStore keeps binary content as raw bytes in one directory, named by
// hash.
type BlobStore struct {
	dir string
	mu  sync.Mutex
	// maps holds a live mapping per hash and how many readers hold it.
	maps map[string]*blobMap
}

type blobMap struct {
	data  []byte
	unmap func() error
	refs  uint
}

// NewBlobStore stores blobs under dir, which it creates.
func NewBlobStore(dir string) (*BlobStore, error) {
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("commonai: blob store %s: %w", dir, err)
	}
	return &BlobStore{dir: dir, maps: map[string]*blobMap{}}, nil
}

// Dir is the store's directory.
func (s *BlobStore) Dir() string { return s.dir }

func (s *BlobStore) path(hash string) string { return filepath.Join(s.dir, hash) }

// Put copies r to disk and gives its reference. The same bytes twice give one file.
func (s *BlobStore) Put(mediaType string, r io.Reader) (*BlobRef, error) {
	tmp, err := os.CreateTemp(s.dir, ".put-*")
	if err != nil {
		return nil, fmt.Errorf("commonai: blob store: %w", err)
	}
	defer os.Remove(tmp.Name())
	h := sha256.New()
	n, err := io.Copy(io.MultiWriter(tmp, h), r)
	if cerr := tmp.Close(); err == nil {
		err = cerr
	}
	if err != nil {
		return nil, fmt.Errorf("commonai: blob store: %w", err)
	}
	hash := hex.EncodeToString(h.Sum(nil))
	if err := os.Rename(tmp.Name(), s.path(hash)); err != nil {
		return nil, fmt.Errorf("commonai: blob store: %w", err)
	}
	return &BlobRef{Hash: hash, MediaType: mediaType, Size: uint64(n), store: s}, nil
}

// PutBase64 decodes base64 straight to disk. The text is never kept.
func (s *BlobStore) PutBase64(mediaType, b64 string) (*BlobRef, error) {
	return s.Put(mediaType, base64.NewDecoder(base64.StdEncoding, strings.NewReader(b64)))
}

// Ref rebuilds a reference a store wrote, after checking that its file exists.
func (s *BlobStore) Ref(hash, mediaType string) (*BlobRef, error) {
	fi, err := os.Stat(s.path(hash))
	if err != nil {
		return nil, fmt.Errorf("commonai: blob %s: %w", hash, err)
	}
	return &BlobRef{Hash: hash, MediaType: mediaType, Size: uint64(fi.Size()), store: s}, nil
}

// errNoStore is the error of a reference that has no store to read from.
var errNoStore = errors.New("commonai: blob reference has no store")

// Bytes maps the blob and gives its bytes with a release. The bytes are valid until release runs.
func (b *BlobRef) Bytes() (data []byte, release func(), err error) {
	if b == nil || b.store == nil {
		return nil, nil, errNoStore
	}
	s := b.store
	s.mu.Lock()
	defer s.mu.Unlock()
	bm, ok := s.maps[b.Hash]
	if !ok {
		if b.Size == 0 {
			return nil, func() {}, nil
		}
		data, unmap, err := mapBlob(s.path(b.Hash))
		if err != nil {
			return nil, nil, fmt.Errorf("commonai: blob %s: %w", b.Hash, err)
		}
		bm = &blobMap{data: data, unmap: unmap}
		s.maps[b.Hash] = bm
	}
	bm.refs++
	var once sync.Once
	return bm.data, func() { once.Do(func() { s.release(b.Hash) }) }, nil
}

func (s *BlobStore) release(hash string) {
	s.mu.Lock()
	defer s.mu.Unlock()
	bm, ok := s.maps[hash]
	if !ok {
		return
	}
	bm.refs--
	if bm.refs == 0 {
		_ = bm.unmap()
		delete(s.maps, hash)
	}
}

// WriteBase64 writes the blob as standard base64 straight from the mapping.
func (b *BlobRef) WriteBase64(w io.Writer) error {
	data, release, err := b.Bytes()
	if err != nil {
		return err
	}
	defer release()
	enc := base64.NewEncoder(base64.StdEncoding, w)
	if _, err := enc.Write(data); err != nil {
		return err
	}
	return enc.Close()
}

// Base64 is the blob as standard base64, for one request body. The caller must not keep it.
func (b *BlobRef) Base64() (string, error) {
	var sb strings.Builder
	sb.Grow(base64.StdEncoding.EncodedLen(int(b.Size)))
	if err := b.WriteBase64(&sb); err != nil {
		return "", err
	}
	return sb.String(), nil
}
