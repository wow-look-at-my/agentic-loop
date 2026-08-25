package commonai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ModelList is what an endpoint's model list says: one fetch answers protocol and prices.
type ModelList struct {
	// Dialect is the protocol the document identifies, or DialectAuto; never DialectResponses (identical document).
	Dialect Dialect

	// Prices is per-model rates, keyed by id; a model with no pricing block is ABSENT, not zero.
	Prices map[string]Rates
}

// modelListMaxBytes caps the read; a model list is small, so anything larger is not the document being read.
const modelListMaxBytes = 1 << 20

// FetchModelList reads an endpoint's model list.
//
// It sends both credential forms, because which server is answering is exactly
// what is not yet known, and each dialect ignores the other's header.
func FetchModelList(ctx context.Context, cfg ProviderConfig) (*ModelList, error) {
	url, err := modelListURL(cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("reading the model list: %w", err)
	}
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("reading the model list: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if cfg.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		req.Header.Set("x-api-key", cfg.APIKey)
	}
	req.Header.Set("anthropic-version", defaultAnthropicVersion)
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}

	resp, err := hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("reading the model list: %w", err)
	}
	defer resp.Body.Close()
	body, _, err := readCapped(resp.Body, modelListMaxBytes)
	if err != nil {
		return nil, fmt.Errorf("reading the model list: %w", err)
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("reading the model list: %s answered %d", url, resp.StatusCode)
	}
	return DecodeModelList(body)
}

// modelListURL is where an endpoint's model list lives; the trailing /v1 is trimmed because the chat dialects disagree
func modelListURL(baseURL string) (string, error) {
	base := strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if base == "" {
		return "", fmt.Errorf("no base URL")
	}
	return strings.TrimSuffix(base, "/v1") + "/v1/models", nil
}

// DecodeModelList reads a model-list document.
//
// A document that will not parse is an ERROR, not an empty result. The two are
// different facts a host has to be able to tell apart: an endpoint that
// publishes no prices renders an em dash and is working correctly, and an
// endpoint answering with an HTML error page is not — reporting both as "no
// prices" is how a misconfigured URL looks exactly like a cheap provider.
func DecodeModelList(body []byte) (*ModelList, error) {
	var doc struct {
		Object  string `json:"object"`
		HasMore *bool  `json:"has_more"`
		Data    []struct {
			ID      string            `json:"id"`
			Object  string            `json:"object"`
			Type    string            `json:"type"`
			Pricing *modelListPricing `json:"pricing"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("the model list is not JSON: %w", err)
	}

	out := &ModelList{Prices: make(map[string]Rates, len(doc.Data))}

	// The ENVELOPE decides first, because a list with no models at all still
	// identifies its server.
	switch {
	case doc.Object == "list":
		out.Dialect = DialectOpenAI
	case doc.HasMore != nil:
		out.Dialect = DialectAnthropic
	}

	for _, m := range doc.Data {
		if out.Dialect == DialectAuto {
			switch {
			case m.Type == "model":
				out.Dialect = DialectAnthropic
			case m.Object == "model":
				out.Dialect = DialectOpenAI
			}
		}
		id := strings.TrimSpace(m.ID)
		if id == "" || m.Pricing == nil {
			continue
		}
		if r, ok := ratesOf(*m.Pricing); ok {
			out.Prices[id] = r
		}
	}
	return out, nil
}
