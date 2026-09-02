package search

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
)

// HTTPEmbedder calls an OpenAI-compatible POST /v1/embeddings endpoint.
//
// It ships with the package because an index that needs every host to write
// its own HTTP client before the semantic half does anything is a feature
// delivered half-built. It follows the same rules as the rest of the module:
// the endpoint and key are explicit fields, no environment is read, and all
// I/O goes through an injectable *http.Client.
type HTTPEmbedder struct {
	// BaseURL is everything before "/v1" (e.g. "https://api.openai.com").
	BaseURL string
	// Model is the model id sent with every request.
	Model string
	// APIKey, when non-empty, is sent as a Bearer token.
	APIKey string
	// Headers are extra request headers, set after the bearer token so can override it.
	Headers map[string]string
	// HTTP is the client; nil uses http.DefaultClient.
	HTTP *http.Client

	// DocumentPrefix and QueryPrefix are prepended to inputs; changing them changes the vectors, so it is a re-index.
	DocumentPrefix string
	QueryPrefix    string

	// MaxBatch caps how many inputs go in request; the cap is the endpoint's, and endpoints disagree.
	MaxBatch int
}

// Nomic's text models require a task instruction prefix on every input; these are the that matter for retrieval.
const (
	NomicDocumentPrefix = "search_document: "
	NomicQueryPrefix    = "search_query: "
)

// embeddingsMaxBytes caps response, well above a full batch, to stop an unbounded read.
const embeddingsMaxBytes = 64 << 20

// embeddingsRequest is the POST body; encoding_format is sent explicitly so a base64 gateway can't break it.
type embeddingsRequest struct {
	Model          string   `json:"model"`
	Input          []string `json:"input"`
	EncodingFormat string   `json:"encoding_format"`
}

type embeddingsResponse struct {
	Data []struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	} `json:"data"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// EmbedDocuments implements Embedder, prefixing each input with
// DocumentPrefix and splitting the call to respect MaxBatch.
func (e HTTPEmbedder) EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	prefixed := make([]string, len(texts))
	for i, t := range texts {
		prefixed[i] = e.DocumentPrefix + t
	}

	size := e.MaxBatch
	if size <= 0 {
		size = len(prefixed)
	}
	out := make([][]float32, 0, len(prefixed))
	for start := 0; start < len(prefixed); start += size {
		end := min(start+size, len(prefixed))
		vecs, err := e.post(ctx, prefixed[start:end])
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// EmbedQuery implements Embedder, prefixing the query with QueryPrefix.
func (e HTTPEmbedder) EmbedQuery(ctx context.Context, text string) ([]float32, error) {
	vecs, err := e.post(ctx, []string{e.QueryPrefix + text})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// post makes embeddings request.
func (e HTTPEmbedder) post(ctx context.Context, texts []string) ([][]float32, error) {
	body, err := json.Marshal(embeddingsRequest{Model: e.Model, Input: texts, EncodingFormat: "float"})
	if err != nil {
		return nil, fmt.Errorf("search: encode embeddings request: %w", err)
	}
	url := strings.TrimRight(e.BaseURL, "/") + "/v1/embeddings"
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("search: build embeddings request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if e.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+e.APIKey)
	}
	for k, v := range e.Headers {
		req.Header.Set(k, v)
	}

	hc := e.HTTP
	if hc == nil {
		hc = http.DefaultClient
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: embeddings: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode/100 != 2 {
		const maxErrBody = 4 << 10
		b, _ := io.ReadAll(io.LimitReader(resp.Body, maxErrBody))
		msg := strings.TrimSpace(string(b))
		if msg == "" {
			msg = resp.Status
		}
		return nil, &HTTPError{Status: resp.StatusCode, Body: msg}
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, embeddingsMaxBytes))
	if err != nil {
		return nil, fmt.Errorf("search: read embeddings: %w", err)
	}
	var parsed embeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("search: decode embeddings: %w", err)
	}
	// Some gateways answer with an error object; reading it as empty data would lose the real reason.
	if parsed.Error != nil && parsed.Error.Message != "" {
		return nil, fmt.Errorf("search: embeddings: %s", parsed.Error.Message)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("search: embeddings: %d vectors for %d inputs", len(parsed.Data), len(texts))
	}

	// The response carries an explicit index per vector, so we sort by it to keep each vector attached to its text.
	sort.Slice(parsed.Data, func(a, b int) bool { return parsed.Data[a].Index < parsed.Data[b].Index })

	out := make([][]float32, len(texts))
	for i, d := range parsed.Data {
		if d.Index != i {
			return nil, fmt.Errorf("search: embeddings: response covers index %d where %d was expected", d.Index, i)
		}
		if len(d.Embedding) == 0 {
			return nil, fmt.Errorf("search: embeddings: empty vector at index %d", i)
		}
		out[i] = d.Embedding
	}
	return out, nil
}

// HTTPError is a non-2xx answer from the embeddings endpoint, keeping the status code separate from the body.
type HTTPError struct {
	// Status is the HTTP status code the endpoint answered with.
	Status int
	// Body is the response body, trimmed and capped, or the status line when empty.
	Body string
}

func (e *HTTPError) Error() string {
	return fmt.Sprintf("search: embeddings: status %d: %s", e.Status, e.Body)
}
