package agentic

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// wfCall builds a web_fetch ToolCall with the given JSON arguments.
func wfCall(args string) json.RawMessage { return json.RawMessage(args) }

func TestWebFetchAdvertisement(t *testing.T) {
	exec := NewWebFetchTool(WebFetchConfig{})
	tools := []ToolDecl{exec.Decl()}
	require.Len(t, tools, 1)
	tool := tools[0]
	assert.Equal(t, WebFetchToolName, tool.Name)
	assert.True(t, tool.Readonly, "web_fetch only performs a GET; safe for subagents")
	assert.Equal(t, "Fetches one http/https URL with an unauthenticated, plain HTTP GET and returns cleaned page content. "+
		"Optionally provide summary_prompt to have the same model summarize the cleaned content before it is returned.",
		tool.Description)
	assert.Contains(t, string(tool.InputSchema), `"summary_prompt"`)
	assert.False(t, exec.NeedsApproval())
}

func TestWebFetchSuccessCleansHTML(t *testing.T) {
	var gotUA string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotUA = r.Header.Get("User-Agent")
		w.Header().Set("Content-Type", "text/html")
		_, _ = w.Write([]byte(`<html><head><title>x</title><style>b{}</style></head>
<body><!-- hidden --><script>evil()</script>
<h1>Big   Title</h1><p>Hello &amp; welcome.<br>Line two.</p></body></html>`))
	}))
	defer srv.Close()

	exec := NewWebFetchTool(WebFetchConfig{UserAgent: "agentic-test/1.0"})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL + "/page"})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, "URL: "+srv.URL+"/page\n\nBig Title\nHello & welcome.\nLine two.", res.Content,
		"scripts/styles/comments dropped, blocks become lines, entities unescaped, whitespace collapsed")
	assert.Equal(t, "agentic-test/1.0", gotUA)
}

func TestWebFetchTruncationNote(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, strings.Repeat("word ", 50_000)) // ~250k runes cleaned
	}))
	defer srv.Close()

	exec := NewWebFetchTool(WebFetchConfig{})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "Note: cleaned content was truncated to 200000 runes.\n")
}

func TestWebFetchBodySizeCap(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write(bytes.Repeat([]byte("a"), webFetchMaxBodyBytes+16))
	}))
	defer srv.Close()

	exec := NewWebFetchTool(WebFetchConfig{})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL})))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "web_fetch response exceeds 5242880 bytes", res.Content)
}

func TestWebFetchNon2xx(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.NotFound(w, r)
	}))
	defer srv.Close()

	exec := NewWebFetchTool(WebFetchConfig{})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL})))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "web_fetch GET failed: status 404 Not Found", res.Content)
}

func TestWebFetchConnectionError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) {}))
	url := srv.URL
	srv.Close()
	exec := NewWebFetchTool(WebFetchConfig{})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": url})))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.True(t, strings.HasPrefix(res.Content, "web_fetch GET failed: "), res.Content)
}

func TestWebFetchURLValidation(t *testing.T) {
	cases := []struct {
		name, url, want string
	}{
		{"empty", "  ", "url is required"},
		{"bad scheme", "ftp://example.invalid/x", "web_fetch only supports http and https URLs"},
		{"no host", "http://", "web_fetch URL must include a host"},
		{"userinfo", "https://user:pw@example.invalid/", "web_fetch rejects URLs containing userinfo credentials"},
	}
	exec := NewWebFetchTool(WebFetchConfig{})
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": tc.url})))
			require.NoError(t, err)
			assert.True(t, res.IsError)
			assert.Equal(t, tc.want, res.Content)
		})
	}

	t.Run("unparseable", func(t *testing.T) {
		res, err := exec.Execute(context.Background(), wfCall(`{"url":"http://[::1"}`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "invalid url: "))
	})
	t.Run("invalid args", func(t *testing.T) {
		res, err := exec.Execute(context.Background(), wfCall(`{`))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "invalid web_fetch arguments: "))
	})
}

func TestWebFetchBlockHook(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Error("a blocked URL must never be fetched")
	}))
	defer srv.Close()

	var sawURL string
	exec := NewWebFetchTool(WebFetchConfig{
		BlockURL: func(u string) string {
			sawURL = u
			return "fetching this repository is disabled; use the workspace tools instead"
		},
	})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL + "/repo"})))
	require.NoError(t, err)
	assert.True(t, res.IsError)
	assert.Equal(t, "fetching this repository is disabled; use the workspace tools instead", res.Content,
		"the hook's text IS the recoverable error result")
	assert.Equal(t, srv.URL+"/repo", sawURL, "the hook sees the validated URL")

	// An empty return allows the fetch.
	allow := NewWebFetchTool(WebFetchConfig{BlockURL: func(string) string { return "" }})
	srv2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "fine")
	}))
	defer srv2.Close()
	res, err = allow.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv2.URL})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Contains(t, res.Content, "fine")
}

func TestWebFetchSummaryPath(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "<p>page body</p>")
	}))
	defer srv.Close()

	provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("  a fine summary  ")}}}
	exec := NewWebFetchTool(WebFetchConfig{
		Provider: provider, Model: "m", MaxTokens: 256, Extra: map[string]any{"temperature": 0.1},
	})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL + "/doc", "summary_prompt": "list the key points"})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, "URL: "+srv.URL+"/doc\n\nSummary:\na fine summary", res.Content)

	require.Len(t, provider.reqs, 1)
	sum := provider.reqs[0]
	assert.Equal(t, webSummarySystemPrompt, sum.System)
	assert.Empty(t, sum.Tools, "the summary call is tool-less")
	assert.Equal(t, 256, sum.MaxTokens)
	assert.Equal(t, map[string]any{"temperature": 0.1}, sum.Extra)
	require.Len(t, sum.Messages, 1)
	input := sum.Messages[0].Content
	assert.True(t, strings.HasPrefix(input, "Fetched URL:\n"+srv.URL+"/doc\n\nSummary instructions:\nlist the key points\n\nCleaned fetched content:\n"))
	assert.Contains(t, input, "page body")
}

func TestWebFetchSummaryErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "content")
	}))
	defer srv.Close()

	t.Run("no model available", func(t *testing.T) {
		exec := NewWebFetchTool(WebFetchConfig{})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL, "summary_prompt": "sum"})))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "web_fetch summary requested, but no model is available for summarization", res.Content)
	})
	t.Run("summary call fails", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{err: &APIError{Status: 500, Body: "down"}}}}
		exec := NewWebFetchTool(WebFetchConfig{Provider: provider, Model: "m"})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL, "summary_prompt": "sum"})))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.True(t, strings.HasPrefix(res.Content, "web_fetch summary failed: "))
	})
	t.Run("summary empty", func(t *testing.T) {
		provider := &scriptProvider{steps: []scriptStep{{comp: assistantComp("   ")}}}
		exec := NewWebFetchTool(WebFetchConfig{Provider: provider, Model: "m"})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL, "summary_prompt": "sum"})))
		require.NoError(t, err)
		assert.True(t, res.IsError)
		assert.Equal(t, "web_fetch summary returned empty output", res.Content)
	})
}

func TestWebFetchTika(t *testing.T) {
	page := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/pdf")
		_, _ = io.WriteString(w, "%PDF-raw-bytes")
	}))
	defer page.Close()

	t.Run("extraction wins", func(t *testing.T) {
		var method, path, accept, ctype, body string
		tika := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			b, _ := io.ReadAll(r.Body)
			method, path, accept, ctype, body = r.Method, r.URL.Path, r.Header.Get("Accept"), r.Header.Get("Content-Type"), string(b)
			_, _ = io.WriteString(w, "Extracted   text\n\n\n\nmore")
		}))
		defer tika.Close()

		exec := NewWebFetchTool(WebFetchConfig{TikaURL: tika.URL + "/"})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": page.URL})))
		require.NoError(t, err)
		assert.False(t, res.IsError)
		assert.Equal(t, "URL: "+page.URL+"\n\nExtracted text\n\nmore", res.Content,
			"tika output is normalized like any other text")
		assert.Equal(t, http.MethodPut, method)
		assert.Equal(t, "/tika", path, "trailing slash on TikaURL is trimmed")
		assert.Equal(t, "text/plain", accept)
		assert.Equal(t, "application/pdf", ctype, "the fetched content type is forwarded")
		assert.Equal(t, "%PDF-raw-bytes", body)
	})

	t.Run("tika failure falls back to HTML cleanup", func(t *testing.T) {
		tika := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			http.Error(w, "boom", http.StatusInternalServerError)
		}))
		defer tika.Close()
		exec := NewWebFetchTool(WebFetchConfig{TikaURL: tika.URL})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": page.URL})))
		require.NoError(t, err)
		assert.False(t, res.IsError)
		assert.Contains(t, res.Content, "%PDF-raw-bytes", "fallback cleans the raw bytes instead")
	})

	t.Run("tika empty output falls back", func(t *testing.T) {
		tika := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			_, _ = io.WriteString(w, "   \n  ")
		}))
		defer tika.Close()
		exec := NewWebFetchTool(WebFetchConfig{TikaURL: tika.URL})
		res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": page.URL})))
		require.NoError(t, err)
		assert.Contains(t, res.Content, "%PDF-raw-bytes")
	})
}

func TestWebFetchNoExtractableContent(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = io.WriteString(w, "<script>only()</script>")
	}))
	defer srv.Close()
	exec := NewWebFetchTool(WebFetchConfig{})
	res, err := exec.Execute(context.Background(), wfCall(jsonMust(jsonObj{"url": srv.URL})))
	require.NoError(t, err)
	assert.False(t, res.IsError)
	assert.Equal(t, "URL: "+srv.URL+"\n\n(no extractable content)", res.Content)
}

func TestWebFetchTextHelpers(t *testing.T) {
	t.Run("normalizeText", func(t *testing.T) {
		assert.Equal(t, "a b\n\nc", normalizeText("  a\t b \r\n\r\n\n\n c \n\n\n"))
	})
	t.Run("collapseSpaces", func(t *testing.T) {
		assert.Equal(t, " a b ", collapseSpaces("  a \t\n b  "))
	})
	t.Run("TruncateRunes", func(t *testing.T) {
		s, cut := TruncateRunes("hello", 10)
		assert.Equal(t, "hello", s)
		assert.False(t, cut)
		s, cut = TruncateRunes("hello world", 5)
		assert.Equal(t, "hello", s)
		assert.True(t, cut)
		s, cut = TruncateRunes("hello", 0)
		assert.Equal(t, "hello", s)
		assert.False(t, cut, "a non-positive cap disables truncation")
	})
	t.Run("buildWebSummaryInput default instructions", func(t *testing.T) {
		got := buildWebSummaryInput(" https://example.invalid ", "body", "  ")
		assert.Equal(t, "Fetched URL:\nhttps://example.invalid\n\n"+
			"Summary instructions:\nProduce a concise, faithful summary of the fetched content.\n\n"+
			"Cleaned fetched content:\nbody", got)
	})
}
