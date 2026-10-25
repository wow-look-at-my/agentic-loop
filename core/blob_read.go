//go:build js || wasip1

// The release call has nothing to free, so the collector reclaims the bytes once no reader holds them.

package commonai

import "os"

// The release call has nothing to free, so the collector reclaims the bytes
// once no reader holds them.
func mapBlob(path string) ([]byte, func() error, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	return data, func() error { return nil }, nil
}
