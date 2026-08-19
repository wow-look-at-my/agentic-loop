package resources

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	agentic "github.com/wow-look-at-my/agentic-loop"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/wow-look-at-my/go-containers/set"
)

// The polling half of resource watching: once per turn it re-reads every
// watched resource, hashes it, and records what moved.
//
// Polling rather than a subscription is a deliberate choice, and the reason is
// the diff. A subscription notification carries a URI and nothing else, so even
// with one the host would have to read the resource and compare it against what
// it held before -- the read and the comparison are the work, and the
// notification only saves the wait.
//
// Nothing here knows what MCP is. A host hands the watcher SOURCES (whatever
// publishes resources) and a SNAPSHOT store (wherever it keeps what it saw).

const (
	// DefaultResourceMax is the per-source resource cap.
	DefaultResourceMax = 64
	// DefaultResourceMaxBytes is the per-resource capture cap (256 KiB).
	DefaultResourceMaxBytes = 256 << 10
	// resourceReadConcurrency bounds simultaneous reads, so a source
	// advertising sixty resources gets a bounded burst rather than sixty
	// parallel requests every turn.
	resourceReadConcurrency = 8
)

// Resource is one watchable resource as its source advertises it.
type Resource struct {
	URI      string
	Name     string
	Title    string
	MimeType string
}

// Label is the resource's human name: its title, else its name, else the URI.
func (r Resource) Label() string {
	switch {
	case strings.TrimSpace(r.Title) != "":
		return r.Title
	case strings.TrimSpace(r.Name) != "":
		return r.Name
	default:
		return r.URI
	}
}

// ResourceContent is one block of a resource's content. Exactly one of Text and
// Blob (base64) is set.
type ResourceContent struct {
	Text     string
	Blob     string
	MimeType string
}

// ResourceSource is one thing that publishes watchable resources -- an MCP
// server, a directory, anything a host can list and read.
type ResourceSource interface {
	// ID is stable across renames: it keys the stored snapshots.
	ID() string
	// Name is what the model is shown.
	Name() string
	// List advertises up to max resources; truncated reports that there were
	// more, which is announced rather than silently dropped.
	List(ctx context.Context, max int) (resources []Resource, truncated bool, err error)
	// Read returns one resource's current content, in one or more blocks.
	Read(ctx context.Context, uri string) ([]ResourceContent, error)
}

// ResourceSnapshot is one watched resource as of the last pass.
type ResourceSnapshot struct {
	SourceID  string
	URI       string
	Label     string
	MimeType  string
	Hash      string
	Content   string
	ByteSize  int64
	Binary    bool
	Truncated bool
}

// ResourceChangeRecord is the durable record of one detected change. It carries
// its OWN copy of the before/after, which is what makes a change id honest: a
// resource that has moved three times since still answers the first notice with
// the first change.
type ResourceChangeRecord struct {
	SourceID      string
	SourceName    string
	URI           string
	Label         string
	MimeType      string
	Kind          string
	BeforeContent string
	AfterContent  string
	BeforeHash    string
	AfterHash     string
	BeforeBytes   int64
	AfterBytes    int64
	Binary        bool
	Truncated     bool
}

// ResourceSnapshots is where the watcher keeps what it has seen. RecordChange
// returns the id the model is given, so the host owns id generation (it is the
// side that has to resolve one later).
type ResourceSnapshots interface {
	ListSnapshots(ctx context.Context) ([]ResourceSnapshot, error)
	PutSnapshot(ctx context.Context, snap ResourceSnapshot) error
	DeleteSnapshot(ctx context.Context, sourceID, uri string) error
	RecordChange(ctx context.Context, rec ResourceChangeRecord) (changeID string, err error)
}

// ResourceWatchConfig configures a watcher.
type ResourceWatchConfig struct {
	Sources   []ResourceSource
	Snapshots ResourceSnapshots
	// MaxResources bounds how many resources ONE source contributes. A listing
	// longer than this is cut and the cut is announced to the model, never
	// silently dropped. 0 uses DefaultResourceMax.
	MaxResources int
	// MaxBytes bounds the content captured per resource. Text past it is stored
	// prefix-only and every reader says so. 0 uses DefaultResourceMaxBytes.
	MaxBytes int
}

// resourceWatcher implements agentic.ResourceWatcher over a set of sources.
type resourceWatcher struct {
	sources      []ResourceSource
	snapshots    ResourceSnapshots
	maxResources int
	maxBytes     int

	// warned holds warnings already delivered, so a persistently broken source
	// is reported once rather than re-announced every single turn. A warning
	// that stops recurring and recurs again is delivered again.
	mu     sync.Mutex
	warned set.Set[string]
}

// NewResourceWatcher returns a watcher over cfg.Sources, or nil when there is
// nothing to watch or nowhere to keep it -- which is the common case, and the
// one where this whole subsystem must cost nothing.
func NewResourceWatcher(cfg ResourceWatchConfig) agentic.ResourceWatcher {
	var sources []ResourceSource
	for _, s := range cfg.Sources {
		if s != nil {
			sources = append(sources, s)
		}
	}
	if len(sources) == 0 || cfg.Snapshots == nil {
		return nil
	}
	w := &resourceWatcher{
		sources:      sources,
		snapshots:    cfg.Snapshots,
		maxResources: cfg.MaxResources,
		maxBytes:     cfg.MaxBytes,
		warned:       set.New[string](),
	}
	if w.maxResources <= 0 {
		w.maxResources = DefaultResourceMax
	}
	if w.maxBytes <= 0 {
		w.maxBytes = DefaultResourceMaxBytes
	}
	return w
}

// capture is one resource as this pass read it.
type capture struct {
	sourceID   string
	sourceName string
	uri        string
	label      string
	mimeType   string
	content    string
	hash       string
	bytes      int64
	binary     bool
	truncated  bool
}

// warning is one thing this pass could not account for, tagged with the source
// it concerns so a removal is never inferred from an unreachable source.
type warning struct {
	sourceID string
	text     string
}

// Poll performs one watch pass: list and read every source's resources, compare
// against the stored snapshot, and record each difference as its own change
// row. Remote failures become warnings; only a storage failure is an error.
func (w *resourceWatcher) Poll(ctx context.Context) (agentic.ResourcePoll, error) {
	var poll agentic.ResourcePoll

	stored, err := w.snapshots.ListSnapshots(ctx)
	if err != nil {
		return poll, fmt.Errorf("reading watched resources: %w", err)
	}
	// A run with nothing recorded has never been polled, so every resource is
	// new because the watch is new -- not because anything moved.
	poll.Baseline = len(stored) == 0

	prev := make(map[string]ResourceSnapshot, len(stored))
	for _, s := range stored {
		prev[snapshotKey(s.SourceID, s.URI)] = s
	}

	captures, warnings := w.readAll(ctx)
	poll.Warnings = w.newWarnings(warnings)

	seen := set.New[string](len(captures))
	for _, c := range captures {
		key := snapshotKey(c.sourceID, c.uri)
		seen.Add(key)
		before, existed := prev[key]
		if existed && before.Hash == c.hash {
			continue // unchanged; nothing to record and nothing to say
		}
		kind := agentic.ResourceAdded
		if existed {
			kind = agentic.ResourceModified
		}
		change, err := w.record(ctx, kind, c, before, existed)
		if err != nil {
			return poll, err
		}
		poll.Changes = append(poll.Changes, change)
	}

	// A resource that vanished from the listing is a change too: the model was
	// told it existed, and must be told it no longer does.
	for key, before := range prev {
		if seen.Contains(key) || w.sourceFailed(before.SourceID, warnings) {
			// Never report a removal on a source whose listing failed this
			// pass: absent-because-unreachable is not absent.
			continue
		}
		change, err := w.recordRemoval(ctx, before)
		if err != nil {
			return poll, err
		}
		poll.Changes = append(poll.Changes, change)
	}

	sort.Slice(poll.Changes, func(i, j int) bool {
		if poll.Changes[i].Server != poll.Changes[j].Server {
			return poll.Changes[i].Server < poll.Changes[j].Server
		}
		return poll.Changes[i].URI < poll.Changes[j].URI
	})
	return poll, nil
}

// readAll lists and reads every source's resources, bounded per source and with
// bounded read concurrency. Every failure and every cap that bit becomes a
// warning sentence rather than a silent omission.
func (w *resourceWatcher) readAll(ctx context.Context) ([]capture, []warning) {
	var (
		mu       sync.Mutex
		captures []capture
		warns    []warning
	)
	for _, src := range w.sources {
		list, cut, err := src.List(ctx, w.maxResources)
		if err != nil {
			warns = append(warns, warning{
				sourceID: src.ID(),
				text: "server " + strconv.Quote(src.Name()) + " could not be listed (" +
					err.Error() + "), so its resources were not checked",
			})
			continue
		}
		if cut {
			warns = append(warns, warning{
				sourceID: src.ID(),
				text: "server " + strconv.Quote(src.Name()) + " advertises more than " +
					strconv.Itoa(w.maxResources) + " resources; only the first " +
					strconv.Itoa(w.maxResources) + " are watched and the rest are unmonitored",
			})
		}

		sem := make(chan struct{}, resourceReadConcurrency)
		var wg sync.WaitGroup
		for _, res := range list {
			wg.Add(1)
			go func(res Resource) {
				defer wg.Done()
				sem <- struct{}{}
				defer func() { <-sem }()

				contents, err := src.Read(ctx, res.URI)
				if err != nil {
					mu.Lock()
					warns = append(warns, warning{
						sourceID: src.ID(),
						text: "resource " + res.URI + " on server " + strconv.Quote(src.Name()) +
							" could not be read (" + err.Error() + "), so its state is unknown",
					})
					mu.Unlock()
					return
				}
				c := w.capture(src, res, contents)
				mu.Lock()
				captures = append(captures, c)
				mu.Unlock()
			}(res)
		}
		wg.Wait()
	}
	sort.Slice(captures, func(i, j int) bool {
		if captures[i].sourceID != captures[j].sourceID {
			return captures[i].sourceID < captures[j].sourceID
		}
		return captures[i].uri < captures[j].uri
	})
	return captures, warns
}

// capture folds one read result into the watched shape. A read may return
// several content blocks (a directory-shaped URI yields one per file); they are
// joined so the whole read is one hashable unit.
//
// Binary blobs are hashed but NOT stored: base64 bytes are useless as model
// context and ruinous to prompt caching, so a binary resource is watched by
// size and hash and reported as such.
func (w *resourceWatcher) capture(src ResourceSource, res Resource, contents []ResourceContent) capture {
	c := capture{
		sourceID:   src.ID(),
		sourceName: src.Name(),
		uri:        res.URI,
		label:      res.Label(),
		mimeType:   res.MimeType,
	}
	var text strings.Builder
	sum := sha256.New()
	for _, part := range contents {
		if part.MimeType != "" && c.mimeType == "" {
			c.mimeType = part.MimeType
		}
		switch {
		case part.Blob != "":
			c.binary = true
			sum.Write([]byte(part.Blob))
			c.bytes += int64(approxBlobBytes(part.Blob))
		default:
			sum.Write([]byte(part.Text))
			c.bytes += int64(len(part.Text))
			if text.Len() > 0 {
				text.WriteString("\n")
			}
			text.WriteString(part.Text)
		}
	}
	c.hash = hex.EncodeToString(sum.Sum(nil))
	if !c.binary {
		c.content, c.truncated = capText(text.String(), w.maxBytes)
	}
	return c
}
