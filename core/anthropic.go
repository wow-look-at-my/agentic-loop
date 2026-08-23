package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// anthropicProvider is the Provider for the Anthropic Messages API, built by
// NewAnthropicProvider. baseURL is the API root; requests POST
// to baseURL + "/v1/messages" with x-api-key and anthropic-version headers
// (version empty defaults to "2023-06-01"). headers are applied after the
// defaults so a caller-supplied header can override them.
//
// Unless disableCaching is set, every request carries exactly two ephemeral
// prompt-cache breakpoints: a static one on the system block (the cache
// hierarchy is tools → system → messages, so it covers the tools array too)
// and a moving one on the last content block of the last message, so each
// turn cache-hits everything through the previous turn's tail. Both markers
// are applied to the per-request wire structures only — the caller's Messages
// are never mutated, so the stored transcript stays marker-free.
//
// The fields are read-only during Complete, so a value is safe for concurrent
// use.
type anthropicProvider struct {
	baseURL        string
	apiKey         string
	version        string
	httpClient     *http.Client
	userAgent      string
	disableCaching bool
	headers        map[string]string
}

const defaultAnthropicVersion = "2023-06-01"

// anReserved are the Extra keys the typed core always overrides.
var anReserved = set.Of("model", "max_tokens", "stream", "system", "messages", "tools")

// cacheEphemeral is the prompt-cache breakpoint marker. The Messages API
// allows at most 4 breakpoints per request; this provider uses exactly 2.
var cacheEphemeral = map[string]string{"type": "ephemeral"}

// Complete implements Provider over a streaming Messages API call. The
// Messages API requires max_tokens on every request, so a Request without a
// positive MaxTokens fails fast before any I/O.
func (a *anthropicProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	if req.MaxTokens <= 0 {
		return nil, badRequestErr("anthropic: Request.MaxTokens must be positive (the Messages API requires max_tokens)")
	}
	body, err := a.buildBody(req)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(a.baseURL, "/") + "/v1/messages"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("anthropic: build request: " + err.Error())
	}
	version := a.version
	if version == "" {
		version = defaultAnthropicVersion
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	hreq.Header.Set("anthropic-version", version)
	if a.apiKey != "" {
		hreq.Header.Set("x-api-key", a.apiKey)
	}
	if a.userAgent != "" {
		hreq.Header.Set("User-Agent", a.userAgent)
	}
	for k, v := range a.headers {
		hreq.Header.Set(k, v)
	}
	client := a.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("anthropic: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	// A 200 that is NOT an SSE stream is a plain JSON response -- the server
	// ignored stream:true and answered with the non-streaming Messages shape.
	// It is accepted transparently and reassembled into a Completion with
	// Streamed false, like openai.
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		body, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, badRequestErr("anthropic: read non-streaming response: " + rerr.Error())
		}
		return parseAnthropicNonStream(body)
	}

	st := &anStream{ev: ev, blocks: map[int]*anBlock{}}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("anthropic: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	return st.completion(), nil
}

// buildBody assembles the Messages API request. Extra passthrough params are
// merged FIRST so the typed core fields always win; reserved keys in Extra
// (model, max_tokens, stream, system, messages, tools) are silently ignored.
// The library does not gate thinking/temperature by model — deciding what to
// send is the caller's job via Extra.
func (a *anthropicProvider) buildBody(req Request) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.ParamsFor(DialectAnthropic) {
		if anReserved.Contains(k) {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	body["max_tokens"] = req.MaxTokens
	body["stream"] = true
	if req.System != "" {
		// system is a content-block array because cache_control lives on
		// blocks, not string bodies. The static breakpoint on the (last)
		// system block covers the tools array too via the cache hierarchy. No
		// system prompt → no system field and no static breakpoint (the
		// moving one still covers the whole prefix).
		sys := map[string]any{"type": "text", "text": req.System}
		if !a.disableCaching {
			sys["cache_control"] = cacheEphemeral
		}
		body["system"] = []map[string]any{sys}
	}
	msgs, err := anWireMessages(req.Messages)
	if err != nil {
		return nil, err
	}
	if !a.disableCaching {
		markTranscriptTail(msgs)
	}
	body["messages"] = msgs
	if len(req.Tools) > 0 {
		tools := make([]map[string]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, map[string]any{
				"name":         t.Name,
				"description":  t.Description,
				"input_schema": t.schema(),
			})
		}
		body["tools"] = tools
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, badRequestErr("anthropic: marshal request: " + err.Error())
	}
	return b, nil
}
