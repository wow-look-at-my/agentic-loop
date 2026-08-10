package agentic

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// Which wire protocol an endpoint speaks, established by asking it rather than
// by asking the user.
//
// The two dialects answer the same question -- "what models do you have?" --
// with structurally different documents, and that difference is a fact about
// the server rather than a guess about its hostname:
//
//	OpenAI:    {"object":"list","data":[{"id":"gpt-x","object":"model",...}]}
//	Anthropic: {"data":[{"type":"model","id":"claude-x",...}],"has_more":false}
//
// So one GET identifies most endpoints, including a self-hosted Anthropic
// gateway on a domain no hostname rule would recognize.
//
// What it cannot see is DialectResponses: an OpenAI endpoint answers the same
// model list whether or not it also serves /v1/responses, so nothing in the
// document distinguishes them. Detection reports the chat-completions dialect
// there, and using the Responses API is an explicit choice a caller makes.
//
// What it cannot settle, and why callers keep an override: this reads the
// MODELS endpoint and infers the CHAT endpoint from it. A gateway is free to
// serve those two independently -- an OpenAI-shaped model list in front of a
// /v1/messages chat endpoint is a legal thing to build -- and the only way to
// prove the chat dialect is to post to it, which either spends tokens or
// deliberately sends a malformed request to read the error shape back. Neither
// is worth doing on every save, so detection answers with what it saw and the
// host may overrule it.

// Dialect names a wire protocol.
type Dialect string

const (
	// DialectAuto asks the endpoint (DetectDialect). It is the zero value, so
	// a config that predates the field detects rather than assuming.
	DialectAuto Dialect = ""
	// DialectOpenAI is the OpenAI-compatible chat-completions API.
	DialectOpenAI Dialect = "openai"
	// DialectAnthropic is the native Anthropic Messages API.
	DialectAnthropic Dialect = "anthropic"
	// DialectResponses is the OpenAI Responses API. Detection NEVER returns
	// it: an endpoint serving /v1/responses serves the same /v1/models
	// document as one serving /v1/chat/completions, so nothing about the
	// model list distinguishes them. It is a deliberate choice -- reasoning
	// carried across tool calls, at the cost of an API only some servers
	// implement -- and a choice is exactly what an explicit setting is for.
	DialectResponses Dialect = "responses"
)

// Valid reports whether d is a dialect this library can speak.
func (d Dialect) Valid() bool {
	switch d {
	case DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses:
		return true
	}
	return false
}

// Label is how a dialect reads on a screen. It lives here so a host's UI does
// not have to carry its own copy of the vocabulary -- a second list that goes
// stale the moment a dialect is added.
func (d Dialect) Label() string {
	switch d {
	case DialectAnthropic:
		return "anthropic messages"
	case DialectOpenAI:
		return "openai-compatible"
	case DialectResponses:
		return "openai responses"
	case DialectAuto:
		return "detect"
	}
	return string(d)
}

// Dialects is every dialect a host may offer, in the order it should present
// them: the default first.
func Dialects() []Dialect {
	return []Dialect{DialectAuto, DialectOpenAI, DialectAnthropic, DialectResponses}
}

// DetectDialect asks an endpoint which protocol it speaks by reading its model
// list. It returns DialectAuto with an error when the answer is not
// established -- never a guess dressed as a finding, because a wrong dialect
// does not degrade, it breaks chat outright.
func DetectDialect(ctx context.Context, cfg ProviderConfig) (Dialect, error) {
	base := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	if base == "" {
		return DialectAuto, fmt.Errorf("detecting the dialect: no base URL")
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, base+"/v1/models", nil)
	if err != nil {
		return DialectAuto, fmt.Errorf("detecting the dialect: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if cfg.APIKey != "" {
		// Both credential forms, because the point of the request is that we
		// do not yet know which server is answering. Each dialect ignores the
		// other's header.
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		req.Header.Set("x-api-key", cfg.APIKey)
	}
	req.Header.Set("anthropic-version", defaultAnthropicVersion)
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := hc.Do(req)
	if err != nil {
		return DialectAuto, fmt.Errorf("detecting the dialect: %w", err)
	}
	defer resp.Body.Close()
	body, _, err := readCapped(resp.Body, detectMaxBytes)
	if err != nil {
		return DialectAuto, fmt.Errorf("detecting the dialect: reading the model list: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return DialectAuto, fmt.Errorf("detecting the dialect: the model list answered %d", resp.StatusCode)
	}
	d := DialectOfModelList(body)
	if d == DialectAuto {
		return DialectAuto, fmt.Errorf("detecting the dialect: the model list matches neither dialect")
	}
	return d, nil
}

// detectMaxBytes caps the model list read. A list is small; anything larger is
// not the document this is trying to recognize.
const detectMaxBytes = 1 << 20

// DialectOfModelList reads the structural tell out of a model-list document,
// answering DialectAuto when it matches neither. The ENVELOPE is checked before
// the items, because a list with no models at all still identifies its server.
//
// It is exported for the host that fetches model lists of its own: reading the
// tell off a response it was already going to make costs no request, and going
// through this function is what stops the rule from being declared a second
// time somewhere it can drift.
func DialectOfModelList(body []byte) Dialect {
	var doc struct {
		Object  string `json:"object"`
		HasMore *bool  `json:"has_more"`
		Data    []struct {
			Object string `json:"object"`
			Type   string `json:"type"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return DialectAuto
	}
	switch {
	case doc.Object == "list":
		return DialectOpenAI
	case doc.HasMore != nil:
		return DialectAnthropic
	}
	for _, m := range doc.Data {
		switch {
		case m.Type == "model":
			return DialectAnthropic
		case m.Object == "model":
			return DialectOpenAI
		}
	}
	return DialectAuto
}
