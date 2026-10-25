//go:build !js && !wasip1

// mapBlob maps the file at path read-only and gives its bytes and the call that unmaps them.

package commonai

import mmap "github.com/wow-look-at-my/go-mmap"

// mapBlob maps the file at path read-only and gives its bytes and the call
// that unmaps them.
func mapBlob(path string) ([]byte, func() error, error) {
	m, err := mmap.MapFile(path)
	if err != nil {
		return nil, nil, err
	}
	return m, m.Unmap, nil
}
