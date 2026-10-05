package commonai

import (
	"compress/gzip"
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// gzipSSEHandler answers every request with a gzipped SSE stream, recording the
// Accept-Encoding the client advertised.
type gzipSSEHandler struct {
	body   string
	accept string
}

func (h *gzipSSEHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.accept = r.Header.Get("Accept-Encoding")
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Content-Encoding", "gzip")
	gz := gzip.NewWriter(w)
	_, _ = gz.Write([]byte(h.body))
	_ = gz.Close()
	w.(http.Flusher).Flush()
}

// The transport advertises gzip when nothing set the header, and decodes a
// gzipped answer transparently, so a compressed upstream is read as if it had
// sent plaintext. Setting Accept-Encoding by hand would suppress that decoding
// and hand the parser raw gzip bytes shaped like SSE, so no dialect sets it.
func TestProvidersAdvertiseAndDecodeGzip(t *testing.T) {
	for _, tc := range []struct {
		name  string
		build func(t *testing.T, baseURL string) Provider
		body  string
	}{
		{
			name:  "openai",
			build: func(t *testing.T, u string) Provider { return oaProvider(t, u) },
			body:  "data: {\"choices\":[{\"delta\":{\"content\":\"ok\"},\"finish_reason\":\"stop\"}]}\n\ndata: [DONE]\n\n",
		},
		{
			name:  "anthropic",
			build: func(t *testing.T, u string) Provider { return anProvider(t, u) },
			body: "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":3,\"output_tokens\":1}}}\n\n" +
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"ok\"}}\n\n" +
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"output_tokens\":2}}\n\n" +
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			h := &gzipSSEHandler{body: tc.body}
			srv := httptest.NewServer(h)
			defer srv.Close()

			comp, err := tc.build(t, srv.URL).Complete(context.Background(), Request{Model: "m", MaxTokens: 16}, nil)
			require.NoError(t, err)

			assert.Equal(t, "gzip", h.accept,
				"the transport's injected Accept-Encoding reaches the upstream")
			assert.Equal(t, "ok", comp.Message.Content,
				"a gzipped stream decoded transparently, no raw gzip bytes reached the parser")
		})
	}
}
