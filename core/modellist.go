// modellist.go holds the modellist code.

package commonai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	neturl "net/url"
	"slices"
	"strings"
)

// ModelList is what an endpoint's model list says: fetch answers protocol and prices.
type ModelList struct {
	// Dialect is the protocol the document identifies, or DialectAuto; never DialectResponses (identical document).
	Dialect Dialect

	// Prices is per-model rates, keyed by id; a model with no pricing block is ABSENT, not.
	Prices map[string]Rates

	// Limits is per-model token limits, keyed by id; a model that published neither limit is ABSENT.
	Limits map[string]Limits

	// Endpoints is the protocols each model ADVERTISES (OpenRouter's "endpoints" array), keyed by id.
	Endpoints map[string][]Dialect
}

// EndpointsFor answers the protocols the list advertises for model, with the
// same suffix rule as LimitsFor.
func (l *ModelList) EndpointsFor(model string) ([]Dialect, bool) {
	if l == nil || model == "" {
		return nil, false
	}
	if d, ok := l.Endpoints[model]; ok {
		return d, true
	}
	var found []Dialect
	matches := 0
	for id, d := range l.Endpoints {
		if strings.HasSuffix(id, "/"+model) {
			found = d
			matches++
		}
	}
	if matches != 1 {
		return nil, false
	}
	return found, true
}

// Limits is what a model list says a model can hold and produce, in tokens. Zero means unpublished.
type Limits struct {
	// ContextWindow is the most tokens one request may carry, prompt and output together.
	ContextWindow int

	// MaxOutput is the most tokens one response may produce.
	MaxOutput int
}

// LimitsFor answers the limits the list publishes for model: the entry with
// that id, else the entry whose id ends in "/"+model. A gateway names the
// model it served without its provider prefix (deepseek/x for the list's
// openrouter/deepseek/x), and that suffix is the same model. Entries with
// that suffix name nothing.
func (l *ModelList) LimitsFor(model string) (Limits, bool) {
	if l == nil || model == "" {
		return Limits{}, false
	}
	if lim, ok := l.Limits[model]; ok {
		return lim, true
	}
	var found Limits
	matches := 0
	for id, lim := range l.Limits {
		if strings.HasSuffix(id, "/"+model) {
			found = lim
			matches++
		}
	}
	if matches != 1 {
		return Limits{}, false
	}
	return found, true
}

// modelListMaxBytes caps one page.
const modelListMaxBytes = 16 << 20

// FetchModelList reads an endpoint's model list, every page of it, decoded.
func FetchModelList(ctx context.Context, cfg ProviderConfig) (*ModelList, error) {
	pages, err := FetchModelListPages(ctx, cfg)
	if err != nil {
		return nil, err
	}
	var out *ModelList
	for _, p := range pages {
		l, err := DecodeModelList(p)
		if err != nil {
			return nil, err
		}
		out = mergeModelLists(out, l)
	}
	return out, nil
}

// modelListMaxPages bounds the Anthropic page walk. A list longer than this is an ERROR, never a silent cut.
const modelListMaxPages = 50

// FetchModelListPages reads an endpoint's model list and returns each page's
// raw document, for a host that decodes fields of its own.
//
// The first request is the OpenAI form: a Bearer token and NO
// anthropic-version. A gateway that speaks both reads that header as a choice,
// and answers a different, shorter list in Anthropic's shape. Only a refusal of
// the OpenAI form re-asks in the Anthropic one. An Anthropic list is paged
// (has_more, last_id), so every page is read.
func FetchModelListPages(ctx context.Context, cfg ProviderConfig) ([][]byte, error) {
	url, err := modelListURL(cfg.BaseURL)
	if err != nil {
		return nil, fmt.Errorf("reading the model list: %w", err)
	}
	body, status, err := getModelList(ctx, cfg, url, false)
	if err != nil {
		return nil, err
	}
	anthropic := false
	if refusesCredentialForm(status) {
		openaiStatus, openaiBody := status, body
		body, status, err = getModelList(ctx, cfg, url+"?limit=1000", true)
		if err != nil {
			return nil, err
		}
		if status < 200 || status >= 300 {
			return nil, fmt.Errorf("reading the model list: %s answered %d to a Bearer token (%s) and %d to x-api-key (%s)",
				url, openaiStatus, bodySnippet(openaiBody), status, bodySnippet(body))
		}
		anthropic = true
	} else if status < 200 || status >= 300 {
		return nil, fmt.Errorf("reading the model list: %s answered %d (%s)", url, status, bodySnippet(body))
	}
	pages := [][]byte{body}
	for {
		var page struct {
			HasMore bool   `json:"has_more"`
			LastID  string `json:"last_id"`
		}
		if json.Unmarshal(body, &page) != nil || !page.HasMore || page.LastID == "" {
			return pages, nil
		}
		if len(pages) >= modelListMaxPages {
			return nil, fmt.Errorf("reading the model list: %s still has more after %d pages", url, len(pages))
		}
		next := url + "?limit=1000&after_id=" + neturl.QueryEscape(page.LastID)
		body, status, err = getModelList(ctx, cfg, next, anthropic)
		if err != nil {
			return nil, err
		}
		if status < 200 || status >= 300 {
			return nil, fmt.Errorf("reading the model list: %s answered %d (%s)", next, status, bodySnippet(body))
		}
		pages = append(pages, body)
	}
}

// refusesCredentialForm reports a status that may mean "not this credential form". A server failure says nothing about one.
func refusesCredentialForm(status int) bool {
	return status == 400 || status == 401 || status == 403
}

// bodySnippet is an error body, trimmed and bounded, so a refusal says why.
func bodySnippet(b []byte) string {
	s := strings.TrimSpace(string(b))
	if s == "" {
		return "no body"
	}
	if r := []rune(s); len(r) > 300 {
		return string(r[:300]) + "..."
	}
	return s
}

// getModelList makes one model-list request, in the Anthropic form or the OpenAI one.
func getModelList(ctx context.Context, cfg ProviderConfig, url string, anthropic bool) ([]byte, int, error) {
	hc := cfg.HTTPClient
	if hc == nil {
		hc = http.DefaultClient
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, 0, fmt.Errorf("reading the model list: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	if cfg.APIKey != "" {
		if anthropic {
			req.Header.Set("x-api-key", cfg.APIKey)
		} else {
			req.Header.Set("Authorization", "Bearer "+cfg.APIKey)
		}
	}
	if anthropic {
		req.Header.Set("anthropic-version", defaultAnthropicVersion)
	}
	for k, v := range cfg.Headers {
		req.Header.Set(k, v)
	}
	resp, err := hc.Do(req)
	if err != nil {
		return nil, 0, fmt.Errorf("reading the model list: %w", err)
	}
	defer resp.Body.Close()
	body, truncated, err := readCapped(resp.Body, modelListMaxBytes)
	if err != nil {
		return nil, 0, fmt.Errorf("reading the model list: %w", err)
	}
	if truncated {
		return nil, 0, fmt.Errorf("reading the model list: %s is larger than %d bytes", url, modelListMaxBytes)
	}
	return body, resp.StatusCode, nil
}

// advertisedDialects maps advertised API paths onto the dialects that speak
// them, in the order given. A path no dialect speaks is left out, so a
// batch-only model comes back empty rather than absent.
func advertisedDialects(paths []string) []Dialect {
	out := []Dialect{}
	for _, p := range paths {
		d := dialectOfPath(p)
		if d == DialectAuto || slices.Contains(out, d) {
			continue
		}
		out = append(out, d)
	}
	return out
}

// mergeModelLists folds a later page into the list so far; the first page names the dialect.
func mergeModelLists(into, page *ModelList) *ModelList {
	if into == nil {
		return page
	}
	if into.Dialect == DialectAuto {
		into.Dialect = page.Dialect
	}
	for k, v := range page.Prices {
		into.Prices[k] = v
	}
	for k, v := range page.Limits {
		into.Limits[k] = v
	}
	for k, v := range page.Endpoints {
		into.Endpoints[k] = v
	}
	return into
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
// A document that will not parse is an ERROR, not an empty result. The are
// different facts a host has to be able to tell apart. An endpoint that
// publishes no prices renders an em dash and is working correctly, and an
// endpoint answering with an HTML error page is not. Reporting both as "no
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
			// Endpoints is OpenRouter's list of the API paths that serve the model.
			Endpoints []string `json:"endpoints"`
			modelListLimits
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("the model list is not JSON: %w", err)
	}

	out := &ModelList{
		Prices:    make(map[string]Rates, len(doc.Data)),
		Limits:    make(map[string]Limits, len(doc.Data)),
		Endpoints: map[string][]Dialect{},
	}

	// The ENVELOPE decides, because a list with no models at all still
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
		if id == "" {
			continue
		}
		if m.Endpoints != nil {
			out.Endpoints[id] = advertisedDialects(m.Endpoints)
		}
		if l := m.limits(); l != (Limits{}) {
			out.Limits[id] = l
		}
		if m.Pricing == nil {
			continue
		}
		if r, ok := ratesOf(*m.Pricing); ok {
			out.Prices[id] = r
		}
	}
	return out, nil
}
