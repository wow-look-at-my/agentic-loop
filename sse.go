package agentic

import (
	"bufio"
	"fmt"
	"io"
	"net/http"
	"strings"
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
