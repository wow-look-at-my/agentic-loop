// Package modelinfo answers what a modelinfo catalogue (https://modelinfo.pazer.ai) publishes about a model: its rates
// and its token limits. It is the source a host falls back to when the endpoint's own model list does not say.
//
// The catalogue folds a model to an undated canonical id ("anthropic/claude-sonnet-4-5"), while a provider names a dated
// snapshot ("claude-sonnet-4-5-20250929") that the catalogue carries only as an alias. So a lookup answers by every name
// a model answers to: its id, each alias, and the id without its "provider/" prefix. A name models both claim answers
// for neither, because one model priced at another's rates is worse than no price.
package modelinfo

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// DefaultURL is the public catalogue.
const DefaultURL = "https://modelinfo.pazer.ai"

// maxBytes caps the read. The whole catalogue, every mode, is a few megabytes.
const maxBytes = 16 << 20

// Model is what the catalogue publishes about one model.
type Model struct {
	// ID is the catalogue's canonical id for the model the lookup resolved to.
	ID string
	// Rates is meaningful only when Priced is true.
	Rates  commonai.Rates
	Priced bool
	// Limits is zero where the catalogue publishes no limit.
	Limits commonai.Limits
}

// RetryAfter is how long a Catalogue answers a failed read's error before it asks the catalogue again.
const RetryAfter = time.Minute

// Catalogue reads a modelinfo catalogue once and answers lookups from it.
type Catalogue struct {
	base string
	hc   *http.Client
	now  func() time.Time

	mu       sync.Mutex
	byName   map[string]*Model
	err      error
	failedAt time.Time
}

// New is a Catalogue over the instance at base; a nil client uses http.DefaultClient.
func New(base string, hc *http.Client) *Catalogue {
	if hc == nil {
		hc = http.DefaultClient
	}
	return &Catalogue{base: strings.TrimRight(strings.TrimSpace(base), "/"), hc: hc, now: time.Now}
}

// Lookup answers what the catalogue publishes about model, by any name the model answers to. ok is false for a name
// the catalogue does not know, or one that some of its models both claim.
func (c *Catalogue) Lookup(ctx context.Context, model string) (Model, bool, error) {
	name := strings.ToLower(strings.TrimSpace(model))
	if name == "" {
		return Model{}, false, fmt.Errorf("modelinfo: cannot look up a model with no id")
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.byName == nil {
		if c.err != nil && c.now().Sub(c.failedAt) < RetryAfter {
			return Model{}, false, c.err
		}
		byName, err := c.fetch(ctx)
		if err != nil {
			c.err, c.failedAt = err, c.now()
			return Model{}, false, err
		}
		c.byName, c.err = byName, nil
	}
	m := c.byName[name]
	if m == nil {
		return Model{}, false, nil
	}
	return *m, true, nil
}

func (c *Catalogue) fetch(ctx context.Context) (map[string]*Model, error) {
	if c.base == "" {
		return nil, fmt.Errorf("modelinfo: the catalogue has no URL")
	}
	// mode=all: the bare endpoint answers only the chat modes, and drops every embedding model a host may also price.
	url := c.base + "/v1/models?mode=all"
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", url, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("modelinfo: %s answered %s", url, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", url, err)
	}
	if len(body) > maxBytes {
		return nil, fmt.Errorf("modelinfo: %s answered more than %d bytes", url, maxBytes)
	}
	byName, err := Decode(body)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", url, err)
	}
	return byName, nil
}

// Decode reads a catalogue document into a table keyed by every lower-cased name a model answers to. A name models
// claim maps to nil. A document that will not parse is an error, never an empty table.
func Decode(body []byte) (map[string]*Model, error) {
	list, err := commonai.DecodeModelList(body)
	if err != nil {
		return nil, err
	}
	var doc struct {
		Data []struct {
			ID      string   `json:"id"`
			Aliases []string `json:"aliases"`
		} `json:"data"`
	}
	if err := json.Unmarshal(body, &doc); err != nil {
		return nil, fmt.Errorf("the catalogue is not JSON: %w", err)
	}

	byName := make(map[string]*Model, len(doc.Data)*2)
	claim := func(name string, m *Model) {
		name = strings.ToLower(strings.TrimSpace(name))
		if name == "" {
			return
		}
		if prior, taken := byName[name]; taken && prior != m {
			byName[name] = nil
			return
		}
		byName[name] = m
	}
	for _, d := range doc.Data {
		id := strings.TrimSpace(d.ID)
		if id == "" {
			continue
		}
		rates, priced := list.Prices[id]
		m := &Model{ID: id, Rates: rates, Priced: priced, Limits: list.Limits[id]}
		claim(id, m)
		for _, a := range d.Aliases {
			claim(a, m)
		}
		if prefix, rest, ok := strings.Cut(id, "/"); ok && prefix != "" && rest != "" {
			claim(rest, m)
		}
	}
	return byName, nil
}
