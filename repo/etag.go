// etagCache replays a GET whose resource has not changed.

package repo

import (
	"container/list"
	"net/http"
	"sync"
)

// etagCacheEntries bounds the replay cache. The oldest entry goes first, which costs one paid request, not a wrong answer.
const etagCacheEntries = 512

type etagKey struct {
	target string
	accept string
	token  string
}

type etagEntry struct {
	key  etagKey
	etag string
	res  GHResponse
}

type etagCache struct {
	mu    sync.Mutex
	order *list.List
	items map[etagKey]*list.Element
}

func newETagCache() *etagCache {
	return &etagCache{order: list.New(), items: map[etagKey]*list.Element{}}
}

// lookup returns the stored ETag and response for a key.
func (c *etagCache) lookup(k etagKey) (string, GHResponse, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	el, ok := c.items[k]
	if !ok {
		return "", GHResponse{}, false
	}
	c.order.MoveToFront(el)
	e := el.Value.(*etagEntry)
	return e.etag, e.res, true
}

// A truncated body is never replayed.
func (c *etagCache) store(k etagKey, res GHResponse) {
	etag := res.header.Get("ETag")
	if res.status != http.StatusOK || etag == "" || res.truncated {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	if el, ok := c.items[k]; ok {
		el.Value = &etagEntry{key: k, etag: etag, res: res}
		c.order.MoveToFront(el)
		return
	}
	c.items[k] = c.order.PushFront(&etagEntry{key: k, etag: etag, res: res})
	for c.order.Len() > etagCacheEntries {
		old := c.order.Back()
		c.order.Remove(old)
		delete(c.items, old.Value.(*etagEntry).key)
	}
}
