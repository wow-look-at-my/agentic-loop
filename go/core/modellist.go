package commonai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ModelList is what an endpoint's model list says. One request answers both
// questions a host has before it can talk to an endpoint at all — which
// protocol it speaks, and what its models charge — so there is one fetch, one
// decode, and one value rather than a pair of functions racing to parse the
// same bytes twice.
type ModelList struct {
	// Dialect is the protocol the document identifies, or DialectAuto when it
	// matches neither. It is never DialectResponses: that endpoint serves an
	// identical document, so naming it would be a guess.
	Dialect Dialect

	// Prices is per-model rates, keyed by model id. A model that published no
	// pricing block is ABSENT rather than present with zeros — a host has to be
	// able to tell a free model from an unpriced one, because the second
	// renders an em dash and the first renders nothing owed.
	Prices map[string]Rates
}

// modelListMaxBytes caps the read. A model list is small; anything larger is
// not the document this is trying to read.
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

// modelListURL is where an endpoint's model list lives.
//
// The trailing /v1 is trimmed first because the two chat dialects disagree
// about what a base URL contains: the OpenAI request is baseURL +
// "/chat/completions", so its base ENDS in /v1, while the Anthropic request is
// baseURL + "/v1/messages", so its base does not. A host writes whichever its
// endpoint wants, and appending "/v1/models" to the first spelling asks for
// /v1/v1/models — a 404 that reads as an endpoint publishing neither a dialect
// nor a price.
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
