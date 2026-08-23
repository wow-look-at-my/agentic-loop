package commonai

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/wow-look-at-my/go-containers/set"
)

// openaiProvider is the Provider for OpenAI-compatible chat-completions APIs,
// built by NewOpenAIProvider. baseURL is the API root including
// the version segment (e.g. "https://api.openai.com/v1"); requests POST to
// baseURL + "/chat/completions". apiKey, when non-empty, is sent as a Bearer
// token, and headers are applied after the defaults so a caller-supplied
// header can override them. selfHosted adds cache_prompt:true to every
// request -- the KV-cache prefix-reuse opt-in llama.cpp-style servers honor --
// and must stay false for hosted OpenAI/Azure, which reject unknown body
// fields with a 400. promptCache adds the two Anthropic-style ephemeral
// cache_control breakpoints in openai shape for Anthropic-fronting gateways
// that pass them through; replayReasoning echoes each assistant message's
// accumulated reasoning back as message.reasoning (the gateway extension a
// model needs to keep seeing its chain-of-thought). A nil httpClient uses
// http.DefaultClient.
//
// The fields are read-only during Complete, so a value is safe for concurrent
// use.
type openaiProvider struct {
	baseURL         string
	apiKey          string
	httpClient      *http.Client
	userAgent       string
	selfHosted      bool
	promptCache     bool
	replayReasoning bool
	headers         map[string]string
}

// oaReserved are the Extra keys the typed core always overrides.
var oaReserved = set.Of("messages", "model", "stream", "tools")

// Complete implements Provider over a streaming chat completion. When the
// first attempt fails before anything streamed with a 400 that names NO
// recoverable parameter at all (some servers answer "Invalid API parameter,
// please check the documentation" -- no name whatsoever, so NewParamStripper's
// regexes have nothing to match and a caller wrapping this Provider in it
// gets no help), and the request carried the AUTO-added default
// stream_options (never a caller-requested field, only a usage-in-stream
// convenience), Complete retries once with that default left off. A 400 that
// DOES name a parameter is left untouched here -- that is NewParamStripper's
// job, and guessing "it must be stream_options" over a name that points
// somewhere else would just burn an extra round trip while the real culprit
// (e.g. a caller-supplied reasoning_effort) survives untouched into the
// retry. Dropping stream_options is always safe when it does fire: the
// caller never asked for it, and its absence only means no usage figures on
// this call, the same degradation an upstream with no stream_options support
// already produces. A context-overflow 400 is excluded too: it is permanent
// regardless of stream_options, and IsContextOverflow's callers expect it
// unretried.
func (o *openaiProvider) Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error) {
	comp, err := o.complete(ctx, req, ev, true)
	if comp != nil || err == nil || !o.shouldRetryWithoutStreamOptions(req, err) {
		return comp, err
	}
	return o.complete(ctx, req, ev, false)
}

// shouldRetryWithoutStreamOptions reports whether a failed first attempt is
// worth retrying with the default stream_options left off: the failure must
// be a pre-stream 400 whose text names no recoverable parameter (a named one
// is NewParamStripper's job, not a guess made here), and the request must
// actually have carried the AUTO-added default -- a caller-supplied
// stream_options (via Extra) is never touched.
func (o *openaiProvider) shouldRetryWithoutStreamOptions(req Request, err error) bool {
	var ae *APIError
	if !errors.As(err, &ae) || ae.Status != 400 || ae.ContextOverflow {
		return false
	}
	if _, named := rejectedParamName(ae.Body); named {
		return false
	}
	_, hasOwn := req.ParamsFor(DialectOpenAI)["stream_options"]
	return !hasOwn
}

// complete runs one attempt of the streaming chat completion.
// includeDefaultStreamOptions gates the {"include_usage":true} default (see
// buildBody); Complete calls this twice only when a first attempt with it set
// is rejected outright.
func (o *openaiProvider) complete(ctx context.Context, req Request, ev *StreamEvents, includeDefaultStreamOptions bool) (*Completion, error) {
	body, err := o.buildBody(req, includeDefaultStreamOptions)
	if err != nil {
		return nil, err
	}
	url := strings.TrimRight(o.baseURL, "/") + "/chat/completions"
	hreq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(body))
	if err != nil {
		return nil, badRequestErr("openai: build request: " + err.Error())
	}
	hreq.Header.Set("Content-Type", "application/json")
	hreq.Header.Set("Accept", "text/event-stream")
	if o.apiKey != "" {
		hreq.Header.Set("Authorization", "Bearer "+o.apiKey)
	}
	if o.userAgent != "" {
		hreq.Header.Set("User-Agent", o.userAgent)
	}
	for k, v := range o.headers {
		hreq.Header.Set(k, v)
	}
	client := o.httpClient
	if client == nil {
		client = http.DefaultClient
	}
	resp, err := client.Do(hreq)
	if err != nil {
		return nil, fmt.Errorf("openai: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode/100 != 2 {
		return nil, readAPIError(resp)
	}

	// A 200 that is NOT an SSE stream is a plain JSON response -- the server
	// ignored stream:true (or a proxy buffered it) and answered with the
	// non-streaming shape. It is accepted transparently and reassembled into a
	// Completion with Streamed false, so the caller keeps a truthful record of
	// how the call was actually transported.
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		body, rerr := io.ReadAll(resp.Body)
		if rerr != nil {
			return nil, badRequestErr("openai: read non-streaming response: " + rerr.Error())
		}
		return o.parseNonStream(body)
	}

	st := &oaStream{ev: ev, acc: newToolCallAccumulator()}
	if scanErr := scanSSE(resp.Body, st.onData); scanErr != nil {
		wrapped := fmt.Errorf("openai: %w", scanErr)
		if st.sawData {
			return st.completion(), wrapped
		}
		return nil, wrapped
	}
	comp := st.completion()
	if err := st.emitRemaining(comp); err != nil {
		return comp, err
	}
	return comp, nil
}

// buildBody assembles the JSON request body. Extra passthrough params are
// merged FIRST so the typed core fields always win; reserved keys in Extra
// (messages, model, stream, tools) are silently ignored so they cannot break
// routing. stream is always forced true, tools are sent only when non-empty
// (no tool_choice is ever sent), and stream_options defaults to
// {"include_usage":true} only when the caller has not supplied a
// stream_options key via Extra AND includeDefaultStreamOptions is true --
// without it OpenAI and most compatibles omit usage from streamed responses
// entirely, but a few reject the field outright, which is what
// includeDefaultStreamOptions=false is for (see Complete's retry). MaxTokens
// > 0 sets max_tokens (overriding an Extra value); 0 leaves the field to
// Extra or the provider default. CacheKey, when set, rides as
// prompt_cache_key, and selfHosted adds cache_prompt:true. promptCache marks
// the per-request wire copy with the two ephemeral cache breakpoints;
// replayReasoning echoes assistant reasoning back as message.reasoning.
func (o *openaiProvider) buildBody(req Request, includeDefaultStreamOptions bool) ([]byte, error) {
	body := map[string]any{}
	for k, v := range req.ParamsFor(DialectOpenAI) {
		if oaReserved.Contains(k) {
			continue
		}
		body[k] = v
	}
	body["model"] = req.Model
	msgs, err := oaWireMessages(req.System, req.Messages, o.replayReasoning)
	if err != nil {
		return nil, err
	}
	if o.promptCache {
		oaMarkPromptCache(msgs)
	}
	body["messages"] = msgs
	body["stream"] = true
	if len(req.Tools) > 0 {
		tools := make([]oaTool, 0, len(req.Tools))
		for _, t := range req.Tools {
			tools = append(tools, oaTool{Type: "function", Function: oaToolFunction{
				Name:        t.Name,
				Description: t.Description,
				Parameters:  t.schema(),
			}})
		}
		body["tools"] = tools
	}
	if req.MaxTokens > 0 {
		body["max_tokens"] = req.MaxTokens
	}
	if _, ok := body["stream_options"]; !ok && includeDefaultStreamOptions {
		body["stream_options"] = map[string]any{"include_usage": true}
	}
	if req.CacheKey != "" {
		body["prompt_cache_key"] = req.CacheKey
	}
	if o.selfHosted {
		body["cache_prompt"] = true
	}
	b, err := json.Marshal(body)
	if err != nil {
		return nil, badRequestErr("openai: marshal request: " + err.Error())
	}
	return b, nil
}
