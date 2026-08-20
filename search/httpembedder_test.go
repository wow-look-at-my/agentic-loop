package search

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// embeddingsServer serves /v1/embeddings, answering with reply if it is
// non-nil and otherwise with one unit vector per input, recording the request.
func embeddingsServer(t *testing.T, status int, reply any) (*httptest.Server, *[]embeddingsRequest) {
	t.Helper()
	var got []embeddingsRequest
	mux := http.NewServeMux()
	mux.HandleFunc("/v1/embeddings", func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		var req embeddingsRequest
		_ = json.Unmarshal(body, &req)
		got = append(got, req)

		w.Header().Set("Content-Type", "application/json")
		if status != http.StatusOK {
			w.WriteHeader(status)
		}
		if reply != nil {
			_ = json.NewEncoder(w).Encode(reply)
			return
		}
		data := make([]map[string]any, len(req.Input))
		for i := range req.Input {
			data[i] = map[string]any{"index": i, "embedding": []float32{float32(i + 1), 1}}
		}
		_ = json.NewEncoder(w).Encode(map[string]any{"object": "list", "data": data})
	})
	srv := httptest.NewServer(mux)
	t.Cleanup(srv.Close)
	return srv, &got
}

func TestHTTPEmbedderSendsTheModelAndReturnsAVectorPerInput(t *testing.T) {
	srv, got := embeddingsServer(t, http.StatusOK, nil)
	e := HTTPEmbedder{BaseURL: srv.URL, Model: "text-embed", APIKey: "k", HTTP: srv.Client()}

	vecs, err := e.Embed(context.Background(), []string{"one", "two"})
	require.NoError(t, err)
	require.Len(t, vecs, 2)
	assert.Equal(t, []float32{1, 1}, vecs[0])
	assert.Equal(t, []float32{2, 1}, vecs[1])

	require.Len(t, *got, 1)
	assert.Equal(t, "text-embed", (*got)[0].Model)
	assert.Equal(t, []string{"one", "two"}, (*got)[0].Input)
	// Sent explicitly: a gateway defaulting this to base64 would answer with
	// strings where the decoder expects numbers.
	assert.Equal(t, "float", (*got)[0].EncodingFormat)
}

// The response carries an explicit index per vector, so array order is not the
// contract and a provider batching concurrently can answer out of order.
func TestHTTPEmbedderRestoresTheProvidersStatedOrder(t *testing.T) {
	srv, _ := embeddingsServer(t, http.StatusOK, map[string]any{
		"object": "list",
		"data": []map[string]any{
			{"index": 1, "embedding": []float32{9, 9}},
			{"index": 0, "embedding": []float32{1, 1}},
		},
	})
	e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}

	vecs, err := e.Embed(context.Background(), []string{"first", "second"})
	require.NoError(t, err)
	assert.Equal(t, []float32{1, 1}, vecs[0], "index 0's vector must land on the first input")
	assert.Equal(t, []float32{9, 9}, vecs[1])
}

func TestHTTPEmbedderReportsWhatTheProviderSaid(t *testing.T) {
	t.Run("http status", func(t *testing.T) {
		srv, _ := embeddingsServer(t, http.StatusTooManyRequests, map[string]any{"error": "slow down"})
		e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}
		_, err := e.Embed(context.Background(), []string{"x"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "429")
		assert.Contains(t, err.Error(), "slow down")
	})

	// Some gateways answer 200 with an error object. Reading that as an empty
	// data list would report "0 vectors" and lose the reason given.
	t.Run("error object under a 200", func(t *testing.T) {
		srv, _ := embeddingsServer(t, http.StatusOK, map[string]any{
			"error": map[string]any{"message": "model not found"},
		})
		e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}
		_, err := e.Embed(context.Background(), []string{"x"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "model not found")
	})

	t.Run("wrong number of vectors", func(t *testing.T) {
		srv, _ := embeddingsServer(t, http.StatusOK, map[string]any{
			"object": "list",
			"data":   []map[string]any{{"index": 0, "embedding": []float32{1}}},
		})
		e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}
		_, err := e.Embed(context.Background(), []string{"one", "two"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "1 vectors for 2 inputs")
	})

	t.Run("an empty vector", func(t *testing.T) {
		srv, _ := embeddingsServer(t, http.StatusOK, map[string]any{
			"object": "list",
			"data":   []map[string]any{{"index": 0, "embedding": []float32{}}},
		})
		e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}
		_, err := e.Embed(context.Background(), []string{"x"})
		require.Error(t, err)
		assert.Contains(t, err.Error(), "empty vector")
	})
}

func TestHTTPEmbedderSendsNoRequestForNoInput(t *testing.T) {
	srv, got := embeddingsServer(t, http.StatusOK, nil)
	e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}
	vecs, err := e.Embed(context.Background(), nil)
	require.NoError(t, err)
	assert.Empty(t, vecs)
	assert.Empty(t, *got)
}

// The index and a real endpoint, end to end: the embedder the package ships is
// the one it is tested against.
func TestIndexSearchesThroughTheShippedEmbedder(t *testing.T) {
	ctx := context.Background()
	srv, _ := embeddingsServer(t, http.StatusOK, nil)
	e := HTTPEmbedder{BaseURL: srv.URL, Model: "m", HTTP: srv.Client()}

	idx := testIndex(t)
	src := newFakeSource()
	src.put("c1", "u1", msg("m1", "user", "the only message"))
	_, err := idx.Ingest(ctx, src)
	require.NoError(t, err)

	n, err := idx.EmbedPending(ctx, "u1", "m", e, 10)
	require.NoError(t, err)
	assert.Equal(t, 1, n)

	hits, _, err := idx.Search(ctx, Query{Owner: "u1", Text: "message", Limit: 10, Model: "m", Embedder: e})
	require.NoError(t, err)
	require.Len(t, hits, 1)
	assert.Equal(t, "m1", hits[0].MessageID)
}
