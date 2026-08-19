package loop

import (
	"bytes"
	"io"
	"unicode/utf8"
)

// ReadCapped reads up to max bytes, reporting truncation. When truncated it
// drops a trailing partial UTF-8 rune (at most a few bytes) so valid text stays
// valid; binary content is left as-is for the caller's binary check.
func ReadCapped(r io.Reader, max int64) (data []byte, truncated bool, err error) {
	var b bytes.Buffer
	n, err := b.ReadFrom(io.LimitReader(r, max+1))
	if err != nil {
		return nil, false, err
	}
	out := b.Bytes()
	if n <= max {
		return out, false, nil
	}
	out = out[:max]
	for i := 0; i < 3 && len(out) > 0 && !utf8.Valid(out); i++ {
		out = out[:len(out)-1]
	}
	return out, true, nil
}
