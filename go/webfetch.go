package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"
)

// WebFetchToolName is the advertised name of the built-in web-fetch tool.
const WebFetchToolName = "web_fetch"

// The web_fetch caps and timeouts, ported verbatim from the source
// application.
const (
	webFetchMaxBodyBytes   = 5 << 20
	webFetchMaxResultRunes = 200_000
	webFetchRequestTimeout = 45 * time.Second
	webSummaryModelTimeout = 2 * time.Minute
	tikaMaxExtractedBytes  = webFetchMaxResultRunes * 4

	webFetchToolDescription = "Fetches one http/https URL with an unauthenticated, plain HTTP GET and returns cleaned page content. " +
		"Optionally provide summary_prompt to have the same model summarize the cleaned content before it is returned."
)

// webFetchSchema is the tool's parameter schema, ported verbatim.
var webFetchSchema = json.RawMessage(`{
  "type": "object",
  "additionalProperties": false,
  "properties": {
    "url": {
      "type": "string",
      "description": "The http or https URL to fetch. Userinfo credentials are rejected."
    },
    "summary_prompt": {
      "type": "string",
      "description": "Optional instructions for a second call to the same model to summarize the cleaned content."
    }
  },
  "required": ["url"]
}`)

// WebFetchConfig configures NewWebFetchExecutor.
type WebFetchConfig struct {
	// HTTPClient performs the fetch (and Tika) requests; nil defaults to a
	// client with a 45-second timeout.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header on the
	// tool's outbound requests. (The source application injected its UA by
	// wrapping the shared *http.Client; the explicit field mirrors
	// ProviderConfig.)
	UserAgent string
	// TikaURL, when non-empty, is the root URL of an Apache Tika server used
	// to extract text from fetched content (PDFs, office documents, ...).
	// Extraction failures and empty extractions fall back to the built-in
	// HTML cleanup silently.
	TikaURL string
	// Provider, Model, MaxTokens, and Extra serve the optional summary_prompt
	// path: one bounded, tool-less call to the same model summarizes the
	// cleaned content (MaxTokens is required when Provider speaks the
	// Anthropic dialect). With a nil Provider or empty Model, a summary
	// request is a recoverable error result instead.
	Provider  Provider
	Model     string
	MaxTokens int
	Extra     map[string]any
	// BlockURL, when non-nil, is consulted with the validated URL string
	// before fetching. A non-empty return refuses the fetch with exactly that
	// text as a recoverable error result. This is the injectable seam the
	// source application used to block fetches of its workspace repository
	// and redirect the model to other tools — the library ships the hook, not
	// that policy.
	BlockURL func(url string) string
}

// webFetchExecutor implements the web_fetch tool.
type webFetchExecutor struct {
	cfg     WebFetchConfig
	hc      *http.Client
	tikaURL string
}

// NewWebFetchExecutor builds the web_fetch tool executor: an unauthenticated,
// plain HTTP GET returning cleaned page content, with an optional model-backed
// summarize path. Compose it with the rest of the toolset via NewComposite.
// The tool is Readonly (a sub-agent's default toolset includes it) and
// NeedsApproval always reports false — wrap the executor to gate it.
func NewWebFetchExecutor(cfg WebFetchConfig) ToolExecutor {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: webFetchRequestTimeout}
	}
	return &webFetchExecutor{
		cfg:     cfg,
		hc:      hc,
		tikaURL: strings.TrimRight(strings.TrimSpace(cfg.TikaURL), "/"),
	}
}

// Tools advertises web_fetch. It is read-only: the tool only performs a GET,
// so it is safe for sub-agents.
func (e *webFetchExecutor) Tools() []Tool {
	return []Tool{{
		Name:        WebFetchToolName,
		Description: webFetchToolDescription,
		InputSchema: webFetchSchema,
		Readonly:    true,
	}}
}

// NeedsApproval always reports false: approval wiring stays the caller's
// concern (the source application keyed it to a user setting).
func (e *webFetchExecutor) NeedsApproval(string) bool { return false }

// webFetchArgs is the web_fetch argument payload.
type webFetchArgs struct {
	URL           string `json:"url"`
	SummaryPrompt string `json:"summary_prompt"`
}

// Execute fetches the URL and returns cleaned (optionally summarized)
// content. Every failure — validation, a blocked URL, a failed GET, a failed
// summary — is a recoverable error tool result, never a Go error.
func (e *webFetchExecutor) Execute(ctx context.Context, call ToolCall) (ToolResult, error) {
	if call.Name != WebFetchToolName {
		return ToolResult{Content: "unknown tool: " + call.Name, IsError: true}, nil
	}
	var in webFetchArgs
	if err := json.Unmarshal([]byte(call.Arguments), &in); err != nil {
		return ToolResult{Content: "invalid web_fetch arguments: " + err.Error(), IsError: true}, nil
	}
	u, err := validateFetchURL(in.URL)
	if err != nil {
		return ToolResult{Content: err.Error(), IsError: true}, nil
	}
	if e.cfg.BlockURL != nil {
		if msg := e.cfg.BlockURL(u.String()); msg != "" {
			return ToolResult{Content: msg, IsError: true}, nil
		}
	}

	raw, contentType, err := e.fetch(ctx, u)
	if err != nil {
		return ToolResult{Content: err.Error(), IsError: true}, nil
	}
	cleaned := e.clean(ctx, raw, contentType)
	cleaned, truncated := truncateRunes(cleaned, webFetchMaxResultRunes)
	if strings.TrimSpace(cleaned) == "" {
		cleaned = "(no extractable content)"
	}

	prefix := "URL: " + u.String() + "\n"
	if truncated {
		prefix += fmt.Sprintf("Note: cleaned content was truncated to %d runes.\n", webFetchMaxResultRunes)
	}
	if strings.TrimSpace(in.SummaryPrompt) == "" {
		return ToolResult{Content: prefix + "\n" + cleaned}, nil
	}
	if e.cfg.Provider == nil || e.cfg.Model == "" {
		return ToolResult{Content: "web_fetch summary requested, but no model is available for summarization", IsError: true}, nil
	}
	summary, err := generateWebSummary(ctx, e.cfg.Provider, e.cfg.Model, u.String(), cleaned, in.SummaryPrompt, e.cfg.MaxTokens, e.cfg.Extra)
	if err != nil {
		return ToolResult{Content: "web_fetch summary failed: " + err.Error(), IsError: true}, nil
	}
	if strings.TrimSpace(summary) == "" {
		return ToolResult{Content: "web_fetch summary returned empty output", IsError: true}, nil
	}
	return ToolResult{Content: prefix + "\nSummary:\n" + summary}, nil
}

// fetch performs the bounded GET.
func (e *webFetchExecutor) fetch(ctx context.Context, u *url.URL) ([]byte, string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		return nil, "", err
	}
	if e.cfg.UserAgent != "" {
		req.Header.Set("User-Agent", e.cfg.UserAgent)
	}
	resp, err := e.hc.Do(req)
	if err != nil {
		return nil, "", fmt.Errorf("web_fetch GET failed: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, "", fmt.Errorf("web_fetch GET failed: status %s", resp.Status)
	}
	raw, err := readLimited(resp.Body, webFetchMaxBodyBytes)
	if err != nil {
		return nil, "", err
	}
	return raw, resp.Header.Get("Content-Type"), nil
}

// clean extracts readable text: via the configured Tika server when it
// succeeds with non-empty output, else the built-in HTML cleanup. (The source
// logged Tika failures; the library falls back silently.)
func (e *webFetchExecutor) clean(ctx context.Context, raw []byte, contentType string) string {
	if e.tikaURL != "" {
		if text, err := e.extractWithTika(ctx, raw, contentType); err == nil && strings.TrimSpace(text) != "" {
			return normalizeText(text)
		}
	}
	return cleanupHTML(raw)
}

// extractWithTika PUTs the raw bytes to the Tika server's /tika endpoint and
// returns the extracted plain text.
func (e *webFetchExecutor) extractWithTika(ctx context.Context, raw []byte, contentType string) (string, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPut, e.tikaURL+"/tika", bytes.NewReader(raw))
	if err != nil {
		return "", err
	}
	req.Header.Set("Accept", "text/plain")
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	if e.cfg.UserAgent != "" {
		req.Header.Set("User-Agent", e.cfg.UserAgent)
	}
	resp, err := e.hc.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("tika status %s", resp.Status)
	}
	b, err := readLimited(resp.Body, tikaMaxExtractedBytes)
	if err != nil {
		return "", err
	}
	return string(b), nil
}

// webSummarySystemPrompt primes the model for the summary_prompt path.
// Ported verbatim from the source application.
const webSummarySystemPrompt = "You summarize cleaned web content for another assistant in the same conversation. " +
	"Follow the provided summary instructions. If the content is thin, blocked, or unrelated, say so plainly."

// generateWebSummary asks the model to summarize cleaned fetched content:
// one bounded (webSummaryModelTimeout), tool-less call with no retry, via
// OneShot.
func generateWebSummary(ctx context.Context, p Provider, model, url, cleaned, instructions string, maxTokens int, extra map[string]any) (string, error) {
	text, _, err := OneShot(ctx, p, Request{
		Model:  model,
		System: webSummarySystemPrompt,
		Messages: []Message{
			{Role: RoleUser, Content: buildWebSummaryInput(url, cleaned, instructions)},
		},
		MaxTokens: maxTokens,
		Extra:     extra,
	}, webSummaryModelTimeout)
	if err != nil {
		return "", err
	}
	return text, nil
}

// buildWebSummaryInput assembles the summary call's user message, ported
// verbatim.
func buildWebSummaryInput(url, cleaned, instructions string) string {
	var b strings.Builder
	b.WriteString("Fetched URL:\n")
	b.WriteString(strings.TrimSpace(url))
	b.WriteString("\n\nSummary instructions:\n")
	if strings.TrimSpace(instructions) == "" {
		b.WriteString("Produce a concise, faithful summary of the fetched content.")
	} else {
		b.WriteString(strings.TrimSpace(instructions))
	}
	b.WriteString("\n\nCleaned fetched content:\n")
	b.WriteString(cleaned)
	return b.String()
}

// validateFetchURL parses and vets the requested URL: http/https only, a host
// required, userinfo credentials rejected. The error texts are the
// model-facing teaching strings, ported verbatim.
func validateFetchURL(raw string) (*url.URL, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil, errors.New("url is required")
	}
	u, err := url.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid url: %w", err)
	}
	if u.Scheme != "http" && u.Scheme != "https" {
		return nil, errors.New("web_fetch only supports http and https URLs")
	}
	if u.Host == "" {
		return nil, errors.New("web_fetch URL must include a host")
	}
	if u.User != nil {
		return nil, errors.New("web_fetch rejects URLs containing userinfo credentials")
	}
	return u, nil
}

// readLimited reads at most limit bytes, erroring when the source exceeds it.
func readLimited(r io.Reader, limit int64) ([]byte, error) {
	var b bytes.Buffer
	n, err := b.ReadFrom(io.LimitReader(r, limit+1))
	if err != nil {
		return nil, err
	}
	if n > limit {
		return nil, fmt.Errorf("web_fetch response exceeds %d bytes", limit)
	}
	return b.Bytes(), nil
}

// The HTML cleanup pipeline, ported verbatim: strip comments and
// non-content elements, convert breaks and block-element closes to newlines,
// drop the remaining tags, unescape entities, and normalize whitespace.
var (
	htmlCommentRE = regexp.MustCompile(`(?is)<!--.*?-->`)
	htmlDropRE    = regexp.MustCompile(`(?is)<(script|style|noscript|template|svg|canvas|head)[^>]*>.*?</\s*(script|style|noscript|template|svg|canvas|head)\s*>`)
	htmlBreakRE   = regexp.MustCompile(`(?i)<\s*br\s*/?\s*>`)
	htmlBlockRE   = regexp.MustCompile(`(?i)</\s*(address|article|aside|blockquote|dd|details|div|dl|dt|figcaption|figure|footer|form|h[1-6]|header|hr|li|main|nav|ol|p|pre|section|table|tbody|td|tfoot|th|thead|tr|ul)\s*>`)
	htmlTagRE     = regexp.MustCompile(`(?is)<[^>]+>`)
)

// cleanupHTML reduces raw HTML to readable plain text.
func cleanupHTML(raw []byte) string {
	s := string(raw)
	s = htmlCommentRE.ReplaceAllString(s, " ")
	s = htmlDropRE.ReplaceAllString(s, " ")
	s = htmlBreakRE.ReplaceAllString(s, "\n")
	s = htmlBlockRE.ReplaceAllString(s, "\n")
	s = htmlTagRE.ReplaceAllString(s, " ")
	s = html.UnescapeString(s)
	return normalizeText(s)
}

// normalizeText collapses runs of whitespace within lines and runs of blank
// lines between them.
func normalizeText(s string) string {
	s = strings.ReplaceAll(s, "\r\n", "\n")
	s = strings.ReplaceAll(s, "\r", "\n")
	var lines []string
	blank := false
	for _, line := range strings.Split(s, "\n") {
		line = strings.TrimSpace(collapseSpaces(line))
		if line == "" {
			if !blank && len(lines) > 0 {
				lines = append(lines, "")
				blank = true
			}
			continue
		}
		lines = append(lines, line)
		blank = false
	}
	for len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// collapseSpaces folds every whitespace run into one space.
func collapseSpaces(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	space := false
	for _, r := range s {
		if r == '\n' {
			r = ' '
		}
		if unicode.IsSpace(r) {
			if !space {
				b.WriteByte(' ')
				space = true
			}
			continue
		}
		b.WriteRune(r)
		space = false
	}
	return b.String()
}

// truncateRunes caps s at maxRunes runes, reporting whether it truncated.
func truncateRunes(s string, maxRunes int) (string, bool) {
	if maxRunes <= 0 || utf8.RuneCountInString(s) <= maxRunes {
		return s, false
	}
	var b strings.Builder
	count := 0
	for _, r := range s {
		if count >= maxRunes {
			break
		}
		b.WriteRune(r)
		count++
	}
	return strings.TrimSpace(b.String()), true
}
