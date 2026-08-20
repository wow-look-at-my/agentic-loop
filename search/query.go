package search

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

// Mode names which half of the index answered a query. It is reported on every
// search: a semantic result and a literal one are different claims about why a
// message is here, and a caller who has not finished (or has not started)
// embedding must not be handed a text-only list as though it were the whole
// thing.
type Mode string

const (
	// ModeText is the FTS5 index alone.
	ModeText Mode = "text"
	// ModeSemantic is the vector scan alone.
	ModeSemantic Mode = "semantic"
	// ModeHybrid is both, fused.
	ModeHybrid Mode = "hybrid"
	// ModeSubstring is the literal fallback. It answers the queries FTS5
	// structurally cannot: a fragment inside a word, or a query made entirely
	// of punctuation, neither of which produces a token to match.
	ModeSubstring Mode = "substring"
)

// Hit is one message the index matched.
type Hit struct {
	MessageID      string
	ConversationID string
	Role           string
	Content        string
	CreatedAt      string
	// Position is the message's index in its conversation's transcript, so a
	// caller can jump to it without searching the transcript again.
	Position int
	// Score is the rank score, comparable only against other hits from the
	// same query.
	Score float64
	// Text and Semantic record which halves matched this message, so a caller
	// can tell a literal quotation from a merely related passage.
	Text     bool
	Semantic bool
}

// candidate is one half's ranked output before fusion.
type candidate struct {
	messageID string
	score     float64
}

// rrfK is the constant in reciprocal rank fusion: score = sum over halves of
// 1/(rrfK + rank). 60 is the value from the original formulation and the one
// every implementation since has used.
//
// Fusion is by RANK rather than by score because the two halves do not produce
// comparable numbers: bm25 is an unbounded relevance score whose scale depends
// on the corpus, and cosine similarity is a bounded [-1,1] geometric one.
// Normalizing them onto a common axis means inventing an exchange rate between
// them; using positions means never claiming one exists.
const rrfK = 60.0

// Query is one search.
type Query struct {
	// Owner scopes the search. It must be the same value the Source reported
	// for the conversations this search may reach: the match is exact, and
	// there is deliberately no wildcard, so a host that gets this wrong
	// returns nothing rather than everything.
	Owner string
	// Text is what the user typed. It is never parsed as a query language.
	Text string
	// Limit caps the hits returned. Zero or negative returns nothing.
	Limit int
	// Model and Embedder turn on the semantic half. Leave Embedder nil for a
	// text-only search -- that is the state of any caller without an embedding
	// endpoint, and it is not an error.
	Model    string
	Embedder Embedder
}

// Search runs q against the index and returns at most q.Limit hits, with the
// mode that produced them.
func (i *Index) Search(ctx context.Context, q Query) ([]Hit, Mode, error) {
	text := strings.TrimSpace(q.Text)
	if text == "" || q.Limit <= 0 {
		return []Hit{}, ModeText, nil
	}

	// Each half is asked for more than the caller wants, because fusion
	// reorders them: a result that is second in both lists outranks one that
	// leads only the first, and it cannot do that if it was cut before fusing.
	textHits, err := i.searchText(ctx, q.Owner, text, q.Limit*2)
	if err != nil {
		return nil, ModeText, err
	}

	var semHits []candidate
	if q.Embedder != nil && q.Model != "" {
		semHits, err = i.searchSemantic(ctx, q, text, q.Limit*2)
		if err != nil {
			return nil, ModeText, err
		}
	}

	// Neither half found anything by word. Fall back to the literal substring
	// search rather than reporting an absence the index cannot establish --
	// "rep" does not tokenize to anything inside "grep", and a query of pure
	// punctuation tokenizes to nothing at all.
	if len(textHits) == 0 && len(semHits) == 0 {
		hits, err := i.searchSubstring(ctx, q.Owner, text, q.Limit)
		return hits, ModeSubstring, err
	}

	mode := ModeText
	switch {
	case len(textHits) > 0 && semHits != nil:
		mode = ModeHybrid
	case len(textHits) == 0:
		mode = ModeSemantic
	}

	ranked := fuse(textHits, semHits)
	if len(ranked) > q.Limit {
		ranked = ranked[:q.Limit]
	}
	hits, err := i.hydrate(ctx, q.Owner, ranked)
	return hits, mode, err
}

// fuse merges two ranked lists by reciprocal rank fusion, tagging each result
// with which halves contributed to it.
func fuse(text, semantic []candidate) []Hit {
	scores := map[string]*Hit{}
	add := func(list []candidate, isText bool) {
		for rank, c := range list {
			h, ok := scores[c.messageID]
			if !ok {
				h = &Hit{MessageID: c.messageID}
				scores[c.messageID] = h
			}
			h.Score += 1.0 / (rrfK + float64(rank+1))
			if isText {
				h.Text = true
			} else {
				h.Semantic = true
			}
		}
	}
	add(text, true)
	add(semantic, false)

	out := make([]Hit, 0, len(scores))
	for _, h := range scores {
		out = append(out, *h)
	}
	// Ties break on message id so the order is total and a repeated query
	// returns the same list in the same order.
	sort.Slice(out, func(a, b int) bool {
		if out[a].Score != out[b].Score {
			return out[a].Score > out[b].Score
		}
		return out[a].MessageID < out[b].MessageID
	})
	return out
}

// searchText runs the FTS5 half, best match first.
func (i *Index) searchText(ctx context.Context, owner, query string, limit int) ([]candidate, error) {
	match := ftsQuery(query)
	if match == "" {
		return nil, nil
	}
	// Ties break on recency. bm25 gives identical scores to messages that use
	// a term the same way -- which, in a chat history, is most of them -- and
	// the newer of two equally relevant messages is the one being looked for.
	rows, err := i.sql.QueryContext(ctx, `
		SELECT m.message_id, bm25(messages_fts)
		FROM messages_fts
		JOIN indexed_messages m ON m.rowid = messages_fts.rowid
		WHERE messages_fts MATCH ? AND m.owner = ?
		ORDER BY bm25(messages_fts), m.created_at DESC
		LIMIT ?`, match, owner, limit)
	if err != nil {
		return nil, fmt.Errorf("search: text query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []candidate
	for rows.Next() {
		var c candidate
		if err := rows.Scan(&c.messageID, &c.score); err != nil {
			return nil, fmt.Errorf("search: scan text hit: %w", err)
		}
		out = append(out, c)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate text hits: %w", err)
	}
	return out, nil
}

// ftsQuery turns arbitrary user input into an FTS5 MATCH expression, or ""
// when the input holds nothing to match on.
//
// Every term is wrapped in double quotes, which is what makes this safe for
// input nobody wrote for a query parser: quoted, FTS5's own operators (AND,
// OR, NOT, NEAR, -, ^, :, parentheses) are literal text rather than syntax, so
// a search for "NOT" or "c++" is a search and not a parse error.
//
// Terms are split on the same boundary the unicode61 tokenizer uses, so a
// query can never contain a term the index had no chance to store. That split
// is also why no term needs escaping inside its quotes: a double quote is not
// a letter or a digit, so it ends a term rather than appearing in one.
//
// The last term gets a prefix match ONLY when the query ends mid-word. This
// backs an as-you-type search box, where the final word is usually half-typed
// and requiring it whole empties the list exactly while the user is looking at
// it. But a query ending in a separator has said where that word stops: "100%"
// asks for the token "100", and prefix-matching it there would return every
// "100x" and "1000" too, turning the character the user typed to be literal
// into the wildcard it resembles.
func ftsQuery(query string) string {
	terms := strings.FieldsFunc(query, isSeparator)
	if len(terms) == 0 {
		return ""
	}
	quoted := make([]string, len(terms))
	for i, t := range terms {
		quoted[i] = `"` + t + `"`
	}
	if r := []rune(query); !isSeparator(r[len(r)-1]) {
		quoted[len(quoted)-1] += "*"
	}
	return strings.Join(quoted, " ")
}

// isSeparator reports whether a rune ends a token, matching what the unicode61
// tokenizer does to the indexed side.
func isSeparator(r rune) bool { return !unicode.IsLetter(r) && !unicode.IsDigit(r) }

// searchSemantic embeds the query and scans every vector the owner has under
// the model, returning the best chunk score per message.
//
// The scan is exhaustive: there is no vector index, approximate or otherwise.
// see docs/search.md for what that costs at real corpus sizes and why the
// answer is "less than the embedding call that precedes it".
func (i *Index) searchSemantic(ctx context.Context, q Query, text string, limit int) ([]candidate, error) {
	vecs, err := q.Embedder.Embed(ctx, []string{text})
	if err != nil {
		return nil, fmt.Errorf("search: embed query: %w", err)
	}
	if len(vecs) != 1 {
		return nil, fmt.Errorf("search: %q returned %d vectors for one query", q.Model, len(vecs))
	}
	unit, ok := normalize(vecs[0])
	if !ok {
		return nil, fmt.Errorf("search: %q returned a vector with no direction for the query", q.Model)
	}

	rows, err := i.sql.QueryContext(ctx, `
		SELECT e.message_id, e.vector
		FROM embeddings e
		JOIN indexed_messages m ON m.message_id = e.message_id
		WHERE m.owner = ? AND e.model = ?`, q.Owner, q.Model)
	if err != nil {
		return nil, fmt.Errorf("search: vector scan: %w", err)
	}
	defer func() { _ = rows.Close() }()

	// A message contributes its BEST chunk. Summing or averaging its chunks
	// would rank a long message above a short exact match for having more
	// chances to be slightly relevant.
	best := map[string]float64{}
	var mismatched int
	for rows.Next() {
		var id string
		var blob []byte
		if err := rows.Scan(&id, &blob); err != nil {
			return nil, fmt.Errorf("search: scan vector: %w", err)
		}
		score, ok := dotBlob(unit, blob)
		if !ok {
			mismatched++
			continue
		}
		if cur, seen := best[id]; !seen || score > cur {
			best[id] = score
		}
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate vectors: %w", err)
	}
	// Vectors of another width under this model's own name mean the model
	// changed dimensions underneath the index. Skipping them silently would
	// shrink the searchable corpus with no sign anywhere.
	if mismatched > 0 {
		return nil, fmt.Errorf("search: %d vectors stored under %q have a different dimension than it returns now; re-index that model", mismatched, q.Model)
	}

	out := make([]candidate, 0, len(best))
	for id, score := range best {
		out = append(out, candidate{messageID: id, score: score})
	}
	sort.Slice(out, func(a, b int) bool {
		if out[a].score != out[b].score {
			return out[a].score > out[b].score
		}
		return out[a].messageID < out[b].messageID
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// searchSubstring is the literal fallback: a LIKE match against the index's own
// copy of the content.
//
// The needle is escaped at the binding site so LIKE's %, _ and \ match
// literally -- searching "100%" finds "100%", not every "100...".
func (i *Index) searchSubstring(ctx context.Context, owner, query string, limit int) ([]Hit, error) {
	rows, err := i.sql.QueryContext(ctx, `
		SELECT message_id, conversation_id, role, content, position, created_at
		FROM indexed_messages
		WHERE owner = ? AND content LIKE '%' || ? || '%' ESCAPE '\'
		ORDER BY created_at DESC
		LIMIT ?`, owner, escapeLike(query), limit)
	if err != nil {
		return nil, fmt.Errorf("search: substring query: %w", err)
	}
	defer func() { _ = rows.Close() }()

	hits := []Hit{}
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.MessageID, &h.ConversationID, &h.Role, &h.Content, &h.Position, &h.CreatedAt); err != nil {
			return nil, fmt.Errorf("search: scan substring hit: %w", err)
		}
		h.Text = true
		hits = append(hits, h)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate substring hits: %w", err)
	}
	return hits, nil
}

// escapeLike escapes the LIKE metacharacters (%, _) and the escape character
// itself so the needle matches literally under the ESCAPE '\' clause.
// Backslash is replaced first or it would re-escape the escapes it just
// introduced.
func escapeLike(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `%`, `\%`)
	s = strings.ReplaceAll(s, `_`, `\_`)
	return s
}

// hydrate fills in the stored fields for a ranked list of message ids, keeping
// the ranking order.
//
// owner is re-checked here even though every half already scoped by it. The
// halves fuse into a list of bare ids, and a list of ids is exactly the shape
// that loses its scope on the next edit to this file.
func (i *Index) hydrate(ctx context.Context, owner string, ranked []Hit) ([]Hit, error) {
	if len(ranked) == 0 {
		return []Hit{}, nil
	}
	args := make([]any, 0, len(ranked)+1)
	args = append(args, owner)
	placeholders := make([]string, len(ranked))
	for n, h := range ranked {
		args = append(args, h.MessageID)
		placeholders[n] = "?"
	}
	rows, err := i.sql.QueryContext(ctx, `
		SELECT message_id, conversation_id, role, content, position, created_at
		FROM indexed_messages
		WHERE owner = ? AND message_id IN (`+strings.Join(placeholders, ",")+`)`, args...)
	if err != nil {
		return nil, fmt.Errorf("search: hydrate hits: %w", err)
	}
	defer func() { _ = rows.Close() }()

	byID := map[string]Hit{}
	for rows.Next() {
		var h Hit
		if err := rows.Scan(&h.MessageID, &h.ConversationID, &h.Role, &h.Content, &h.Position, &h.CreatedAt); err != nil {
			return nil, fmt.Errorf("search: scan hydrated hit: %w", err)
		}
		byID[h.MessageID] = h
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("search: iterate hydrated hits: %w", err)
	}

	out := make([]Hit, 0, len(ranked))
	for _, r := range ranked {
		full, ok := byID[r.MessageID]
		if !ok {
			// The message left the index between the ranking and this read. It
			// is gone, so it is dropped rather than rendered as a blank row.
			continue
		}
		full.Score, full.Text, full.Semantic = r.Score, r.Text, r.Semantic
		out = append(out, full)
	}
	return out, nil
}
