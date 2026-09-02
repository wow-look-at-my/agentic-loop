package webfetch

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"html"
	"io"
	"net/http"
	"net/url"
	"regexp"
	"strings"
	"time"
	"unicode"
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

// WebFetchConfig configures NewWebFetchTool.
type WebFetchConfig struct {
	// HTTPClient performs the fetch (and Tika) requests; nil defaults to a 45- timeout.
	HTTPClient *http.Client
	// UserAgent, when non-empty, is sent as the User-Agent header on the tool's outbound requests.
	UserAgent string
	// TikaURL, when non-empty, is the root URL of an Apache Tika server used to extract fetched text.
	TikaURL string
	// agentic.Provider, Model, MaxTokens, and Extra serve the optional summary_prompt path.
	Provider  agentic.Provider
	Model     string
	MaxTokens int
	Extra     map[string]any
	// OnCompletion, when non-nil, receives the summary call's *Completion; it is the only route out for its cost.
	OnCompletion func(*agentic.Completion)
	// BlockURL, when non-nil, is consulted with the validated URL string; a non-empty return refuses the fetch.
	BlockURL func(url string) string
}

// webFetchTool implements the web_fetch tool.
type webFetchTool struct {
	cfg     WebFetchConfig
	hc      *http.Client
	tikaURL string
}

// NewWebFetchTool builds the web_fetch tool: an unauthenticated, plain HTTP GET returning cleaned page content.
func NewWebFetchTool(cfg WebFetchConfig) agentic.Tool {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = &http.Client{Timeout: webFetchRequestTimeout}
	}
	return &webFetchTool{
		cfg:     cfg,
		hc:      hc,
		tikaURL: strings.TrimRight(strings.TrimSpace(cfg.TikaURL), "/"),
	}
}

// Decl advertises web_fetch. It is read-only: the tool only performs a GET, so
// it is safe for sub-agents.
func (e *webFetchTool) Decl() agentic.ToolDecl {
	return agentic.ToolDecl{
		Name:        WebFetchToolName,
		Description: webFetchToolDescription,
		InputSchema: webFetchSchema,
		Readonly:    true,
		// Read-only and OPEN-world: fetching a URL changes nothing here but reaches an arbitrary host.
		OpenWorld: agentic.Bool(true),
	}
}

// webFetchArgs is the web_fetch argument payload.
type webFetchArgs struct {
	URL           string `json:"url"`
	SummaryPrompt string `json:"summary_prompt"`
}

// Execute fetches the URL and returns cleaned (optionally summarized)
// content. Every failure — validation, a blocked URL, a failed GET, a failed
// summary — is a recoverable error tool result, never a Go error.
func (e *webFetchTool) Execute(ctx context.Context, args json.RawMessage) (agentic.ToolResult, error) {
	var in webFetchArgs
	if err := json.Unmarshal(args, &in); err != nil {
		return agentic.ToolResult{Content: "invalid web_fetch arguments: " + err.Error(), IsError: true}, nil
	}
	u, err := validateFetchURL(in.URL)
	if err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}, nil
	}
	if e.cfg.BlockURL != nil {
		if msg := e.cfg.BlockURL(u.String()); msg != "" {
			return agentic.ToolResult{Content: msg, IsError: true}, nil
		}
	}

	raw, contentType, err := e.fetch(ctx, u)
	if err != nil {
		return agentic.ToolResult{Content: err.Error(), IsError: true}, nil
	}
	cleaned := e.clean(ctx, raw, contentType)
	cleaned, truncated := agentic.TruncateRunes(cleaned, webFetchMaxResultRunes)
	if strings.TrimSpace(cleaned) == "" {
		cleaned = "(no extractable content)"
	}

	prefix := "URL: " + u.String() + "\n"
	if truncated {
		prefix += fmt.Sprintf("Note: cleaned content was truncated to %d runes.\n", webFetchMaxResultRunes)
	}
	if strings.TrimSpace(in.SummaryPrompt) == "" {
		return agentic.ToolResult{Content: prefix + "\n" + cleaned}, nil
	}
	if e.cfg.Provider == nil || e.cfg.Model == "" {
		return agentic.ToolResult{Content: "web_fetch summary requested, but no model is available for summarization", IsError: true}, nil
	}
	summary, err := generateWebSummary(ctx, e.cfg.Provider, e.cfg.OnCompletion, e.cfg.Model,
		u.String(), cleaned, in.SummaryPrompt, e.cfg.MaxTokens, e.cfg.Extra)
	if err != nil {
		return agentic.ToolResult{Content: "web_fetch summary failed: " + err.Error(), IsError: true}, nil
	}
	if strings.TrimSpace(summary) == "" {
		return agentic.ToolResult{Content: "web_fetch summary returned empty output", IsError: true}, nil
	}
	return agentic.ToolResult{Content: prefix + "\nSummary:\n" + summary}, nil
}

// fetch performs the bounded GET.
func (e *webFetchTool) fetch(ctx context.Context, u *url.URL) ([]byte, string, error) {
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
func (e *webFetchTool) clean(ctx context.Context, raw []byte, contentType string) string {
	if e.tikaURL != "" {
		if text, err := e.extractWithTika(ctx, raw, contentType); err == nil && strings.TrimSpace(text) != "" {
			return normalizeText(text)
		}
	}
	return cleanupHTML(raw)
}

// extractWithTika PUTs the raw bytes to the Tika server's /tika endpoint and
// returns the extracted plain text.
func (e *webFetchTool) extractWithTika(ctx context.Context, raw []byte, contentType string) (string, error) {
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
// bounded (webSummaryModelTimeout), tool-less call with no retry, via
// OneShot. onCompletion, when non-nil, is handed the call's Completion --
// including a partial from a failed call, because those tokens were spent
// too and a host that is not told about them under-counts what the session
// cost.
func generateWebSummary(ctx context.Context, p agentic.Provider, onCompletion func(*agentic.Completion), model, url, cleaned, instructions string, maxTokens int, extra map[string]any) (string, error) {
	comp, err := agentic.OneShot(ctx, p, agentic.Request{
		Model:  model,
		System: webSummarySystemPrompt,
		Messages: []agentic.Message{
			{Role: agentic.RoleUser, Content: buildWebSummaryInput(url, cleaned, instructions)},
		},
		MaxTokens: maxTokens,
		Extra:     extra,
	}, webSummaryModelTimeout)
	if comp != nil && onCompletion != nil {
		onCompletion(comp)
	}
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(comp.Message.Content), nil
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

// collapseSpaces folds every whitespace run into space.
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
