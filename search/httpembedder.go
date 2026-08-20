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
	// Headers are extra request headers, set after the bearer token so one can
	// override it.
	Headers map[string]string
	// HTTP is the client; nil uses http.DefaultClient.
	HTTP *http.Client
}

// embeddingsMaxBytes caps one response. A batch of 64 vectors at 3072
// dimensions is about 4 MB of JSON floats, so the cap sits well above what a
// full batch from the widest model in use produces and exists only to stop an
// unbounded read.
const embeddingsMaxBytes = 64 << 20

// embeddingsRequest is the POST body. encoding_format is sent explicitly: it
// defaults to "float", but a gateway defaulting it to "base64" would answer
// with strings where this expects numbers, and that failure would read as a
// malformed response rather than as a setting.
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

// Embed implements Embedder.
func (e HTTPEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
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
		return nil, fmt.Errorf("search: embeddings: status %d: %s", resp.StatusCode, msg)
	}
	raw, err := io.ReadAll(io.LimitReader(resp.Body, embeddingsMaxBytes))
	if err != nil {
		return nil, fmt.Errorf("search: read embeddings: %w", err)
	}
	var parsed embeddingsResponse
	if err := json.Unmarshal(raw, &parsed); err != nil {
		return nil, fmt.Errorf("search: decode embeddings: %w", err)
	}
	// Some gateways answer 200 with an error object instead of a status code.
	// Reading that body as an empty data list would report "0 vectors" and
	// lose the reason the provider actually gave.
	if parsed.Error != nil && parsed.Error.Message != "" {
		return nil, fmt.Errorf("search: embeddings: %s", parsed.Error.Message)
	}
	if len(parsed.Data) != len(texts) {
		return nil, fmt.Errorf("search: embeddings: %d vectors for %d inputs", len(parsed.Data), len(texts))
	}

	// The response carries an explicit index per vector, so array order is not
	// the contract -- and a provider that batches concurrently can return them
	// out of order. Sorting by the index the provider stated is what keeps
	// each vector attached to the text it describes.
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
