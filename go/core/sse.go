package commonai

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"net/http"
	"strings"
	"unicode/utf8"
)

// scanSSE parses an SSE stream, calling onData for each "data:" payload until
// "[DONE]" or EOF. Only data lines are processed; comments, event: lines and
// blank separators are skipped (event-named streams like Anthropic's carry the
// discriminator inside the JSON payload as well). The scanner allows long
// lines — 64 KiB initial buffer, 8 MiB max — because tool-call argument
// deltas and base64 payloads can far exceed bufio's default. onData errors
// abort the stream and propagate.
func scanSSE(r io.Reader, onData func(data []byte) error) error {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8*1024*1024)
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data:") {
			continue
		}
		data := strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		if data == "" {
			continue
		}
		if data == "[DONE]" {
			return nil
		}
		if err := onData([]byte(data)); err != nil {
			return err
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("read stream: %w", err)
	}
	return nil
}

// readAPIError turns a non-2xx response into an *APIError, reading at most
// 4 KiB of the body. The bounded body is what downstream matchers (the
// param-strip regexes, the context-overflow detector) run against; an empty
// body falls back to the HTTP status text.
func readAPIError(resp *http.Response) *APIError {
	const maxErrBody = 4 * 1024
	b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
	msg := strings.TrimSpace(string(b))
	if msg == "" {
		msg = resp.Status
	}
	return &APIError{
		Status:          resp.StatusCode,
		Body:            msg,
		ContextOverflow: resp.StatusCode == http.StatusBadRequest && contextOverflowRe.MatchString(msg),
	}
}

// readCapped reads at most max bytes, reporting whether it stopped early. A
// truncated read is trimmed back to a UTF-8 boundary so the caller never has
// to reason about a half-decoded rune.
func readCapped(r io.Reader, max int64) (data []byte, truncated bool, err error) {
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
