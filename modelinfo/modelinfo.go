// Package modelinfo answers what a modelinfo catalogue (https://modelinfo.pazer.ai) publishes about a model: its rates
// and its token limits. It is the source a host falls back to when the endpoint's own model list does not say.
//
// A lookup asks the catalogue's /v1/models/{name} for the one model it names. The whole catalogue is tens of
// megabytes, so it is never read in full.
package modelinfo

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	commonai "github.com/wow-look-at-my/agentic-loop/core"
)

// DefaultURL is the public catalogue.
const DefaultURL = "https://modelinfo.pazer.ai"

// maxBytes caps one model's record, which is a few kilobytes.
const maxBytes = 1 << 20

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

// Catalogue asks a modelinfo catalogue about each model once and keeps the
// answer, a model it does not know included.
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

// Lookup answers what the catalogue publishes about model, by any name the catalogue resolves. ok is false for a name
// it does not know, or one that several of its models claim.
func (c *Catalogue) Lookup(ctx context.Context, model string) (Model, bool, error) {
	name := strings.ToLower(strings.TrimSpace(model))
	if name == "" {
		return Model{}, false, fmt.Errorf("modelinfo: cannot look up a model with no id")
	}
	if c.base == "" {
		return Model{}, false, fmt.Errorf("modelinfo: the catalogue has no URL")
	}
	c.mu.Lock()
	m, done := c.byName[name]
	err := c.err
	if err != nil && c.now().Sub(c.failedAt) >= RetryAfter {
		err = nil
	}
	c.mu.Unlock()
	switch {
	case done && m == nil:
		return Model{}, false, nil
	case done:
		return *m, true, nil
	case err != nil:
		return Model{}, false, err
	}

	// No lock across the request, so lookups of different models run at once.
	m, err = c.fetch(ctx, name)
	c.mu.Lock()
	defer c.mu.Unlock()
	if err != nil {
		c.err, c.failedAt = err, c.now()
		return Model{}, false, err
	}
	c.err = nil
	if c.byName == nil {
		c.byName = make(map[string]*Model)
	}
	c.byName[name] = m
	if m == nil {
		return Model{}, false, nil
	}
	return *m, true, nil
}

// lookupParallelism bounds the requests LookupMany has out at once.
const lookupParallelism = 16

// LookupMany looks up every name, several at a time, and answers the ones the catalogue resolved, keyed by the name as
// given. The error joins every failed lookup; the names that did resolve are answered alongside it.
func (c *Catalogue) LookupMany(ctx context.Context, names []string) (map[string]Model, error) {
	out := make(map[string]Model, len(names))
	var mu sync.Mutex
	var errs []error
	sem := make(chan struct{}, lookupParallelism)
	var wg sync.WaitGroup
	for _, name := range names {
		wg.Add(1)
		sem <- struct{}{}
		go func() {
			defer wg.Done()
			defer func() { <-sem }()
			m, ok, err := c.Lookup(ctx, name)
			mu.Lock()
			defer mu.Unlock()
			if err != nil {
				errs = append(errs, err)
				return
			}
			if ok {
				out[name] = m
			}
		}()
	}
	wg.Wait()
	return out, errors.Join(errs...)
}

func (c *Catalogue) fetch(ctx context.Context, name string) (*Model, error) {
	endpoint := c.base + "/v1/models/" + url.PathEscape(name)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, endpoint, nil)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %w", err)
	}
	req.Header.Set("Accept", "application/json")
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", endpoint, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode == http.StatusNotFound {
		return nil, nil
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("modelinfo: %s answered %s", endpoint, resp.Status)
	}
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxBytes+1))
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", endpoint, err)
	}
	if len(body) > maxBytes {
		return nil, fmt.Errorf("modelinfo: %s answered more than %d bytes", endpoint, maxBytes)
	}
	m, err := Decode(body)
	if err != nil {
		return nil, fmt.Errorf("modelinfo: %s: %w", endpoint, err)
	}
	return &m, nil
}

// Decode reads one model's record. A record that will not parse is an error, never an empty model.
func Decode(body []byte) (Model, error) {
	var rec struct {
		ID string `json:"id"`
	}
	if err := json.Unmarshal(body, &rec); err != nil {
		return Model{}, fmt.Errorf("the answer is not a model record: %w", err)
	}
	id := strings.TrimSpace(rec.ID)
	if id == "" {
		return Model{}, fmt.Errorf("the model record names no id")
	}
	// The record is a model-list item, so the model list's decoder reads its rates and limits.
	wrapped, err := json.Marshal(struct {
		Object string            `json:"object"`
		Data   []json.RawMessage `json:"data"`
	}{"list", []json.RawMessage{body}})
	if err != nil {
		return Model{}, err
	}
	list, err := commonai.DecodeModelList(wrapped)
	if err != nil {
		return Model{}, err
	}
	rates, priced := list.Prices[id]
	return Model{ID: id, Rates: rates, Priced: priced, Limits: list.Limits[id]}, nil
}
