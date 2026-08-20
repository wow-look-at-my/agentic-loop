# Searching stored conversations

`search` indexes a host's conversations so they can be found by word and, when
an embedding model is available, by meaning. It owns one SQLite file: an FTS5
index over message text, plus the vectors for a semantic search over the same
messages.

The conversations stay wherever the host keeps them. The index reads them
through `Source`, and `SessionSource` adapts a `session.Store`, so `cai` and the
http/socket servers get searchable history without changing how any of them
store anything.

## Why the vectors are scanned in Go

`sqlite-vec` and its siblings are C loadable extensions. The driver here is
`modernc.org/sqlite`, which is SQLite transpiled to Go: there is no C ABI to
load an extension into, so those extensions cannot work regardless of how they
are configured. Reaching for one means switching to a cgo driver, which costs
the static cross-compiled binary this module's consumers build.

So the vectors are stored as normalized little-endian float32 BLOBs and scanned
in Go. Normalizing at write time makes the similarity a plain dot product, so
the read path does one multiply-add per dimension and no square roots.

### What that costs

`BenchmarkVectorScan`, on the sandbox this was developed on (4 cores). To
reproduce, run `go-toolchain` — it runs the benchmarks after the tests.

| vectors × dimensions | scan | allocated |
|---|---|---|
| 1,000 × 768 | 7.4 ms | 3.1 MB |
| 10,000 × 768 | 96.7 ms | 31.0 MB |
| 10,000 × 1536 | 119.2 ms | 60.3 MB |
| 10,000 × 3072 | 181.6 ms | 118.9 MB |
| 50,000 × 768 | 447.8 ms | 154.0 MB |

Two things the shape of that table says:

**Rows cost more than dimensions.** Doubling the width of a vector adds about
25%, while doubling the number of vectors roughly doubles the time. The
arithmetic is not the expense — fetching a row is. That is why the scan reads
each blob through `sql.RawBytes`, borrowing the driver's buffer rather than
copying the whole corpus into fresh slices on every query; that change alone
took the 10,000 × 1536 case from 160 ms to 119 ms and halved the allocation.

**The ceiling is around 50,000 vectors.** Below roughly 10,000 the scan is
smaller than the embedding round trip that precedes it, so it is free in any
sense a user perceives. At 50,000 it is about half a second and has become the
dominant cost. Past that it stops being interactive, and the honest answer is a
real approximate-nearest-neighbour index or a narrower embedding model — not a
bigger constant factor here.

Most messages are one chunk, so "vectors" is close to "messages" for a typical
history. A host expecting far more than 50,000 should know this is where the
design runs out.

## The Source seam

```go
type Source interface {
    Conversations(ctx context.Context) ([]Conversation, error)
    Messages(ctx context.Context, conversationID string) ([]Message, error)
}
```

`Conversation.Revision` is opaque to the index: anything that changes when the
transcript changes. `Ingest` compares it against the revision recorded when
that conversation was last read, and re-reads only what moved.

A conversation that moved is re-indexed **whole**, because the revision says
only that something changed. Re-indexing whole does not re-embed whole: the
vectors are keyed by message id and are never touched by `Ingest`, so a
conversation that gained one message costs one embedding.

**A revision that fails to move when the transcript does is the one way to make
this index quietly wrong.** A store that cannot answer honestly should return
something that always differs and pay for the re-read.

### Message ids and what they cost

`Message.ID` must be stable across re-reads: it is what the embeddings are keyed
by. An id that changes re-embeds the message; one reused for different text
attaches the wrong vector to it.

A host with real message ids passes them. `SessionSource` has none — a
`session.Store` transcript is a list — so it derives `conversationID:position`.
That is stable for the append-only transcript a turn produces: appending message
N+1 does not move the first N. Inserting or removing a message mid-transcript
does shift every position after it, and each of those is then embedded again.
`session.Put` replacing a transcript wholesale is how that happens.

The alternative, hashing the content, is worse: it silently merges two identical
messages in one conversation into a single searchable row.

### The stat optimization

Without help, `SessionSource` reads every conversation on every pass just to
learn which ones moved. A store can implement `Revisions() (map[string]string,
error)` to answer that cheaply; `session.File` does, from each document's size
and modification time, which turns a full read of the store into a stat per
conversation.

## Two schema versions

`ftsSchemaVersion` covers the text tables, `embedSchemaVersion` the vectors.
They are separate because their rebuild costs are not comparable: re-indexing
text is a re-read of what the host already has, and re-embedding a history costs
money at the caller's own endpoint.

So a tokenizer or column change bumps the text version, the text tables are
dropped and rebuilt, and every vector survives — `Ingest` re-indexes the same
message ids and the embeddings are re-adopted with nothing re-sent.

An edit, rather than an append, retires a message id whose vector no conversation
delete reaches. `dropOrphanedVectors` sweeps those at the end of every `Ingest`;
without it they accumulate for the life of the index as paid-for storage no
query can read.

## Ranking

The two halves are fused by **reciprocal rank fusion**: `score = Σ 1/(60 +
rank)`. Fusion is by position rather than by score because bm25 is an unbounded
relevance score whose scale depends on the corpus and cosine similarity is a
bounded geometric one. Putting them on a common axis means inventing an exchange
rate between them; using positions means never claiming one exists.

Each half is asked for twice the caller's limit, because fusion reorders: a
result placed second in both lists outranks one that leads only the first, and
it cannot do that if it was cut before fusing.

Within the text half, bm25 ties break on **recency**. Identical scores are the
common case in a chat history — most messages use a term the same way — and the
newer of two equally relevant messages is the one being looked for.

A message contributes its **best** chunk to the semantic half. Summing or
averaging its chunks would rank a long message above a short exact match for
having more chances to be slightly relevant.

## The query is never a query language

`ftsQuery` wraps every term in double quotes, so FTS5's own operators — `AND`,
`OR`, `NOT`, `NEAR`, `-`, `^`, `:`, parentheses — are literal text. A search for
`NOT` or `c++` is a search, not a parse error. Terms are split on the same
boundary the `unicode61` tokenizer uses, which is also why no term needs escaping
inside its quotes: a double quote is not a letter or a digit, so it ends a term
rather than appearing in one.

The last term gets a prefix match **only when the query ends mid-word**. This
backs an as-you-type search box, where the final word is usually half-typed and
requiring it whole empties the list exactly while the user is looking at it. A
query that ends in a separator has said where that word stops: `100%` asks for
the token `100`, and prefix-matching there would return every `100x` and `1000`
too — turning the character the user typed to be literal into the wildcard it
resembles.

### The substring fallback

When neither half matches, the search falls back to a literal `LIKE` scan and
reports `ModeSubstring`. FTS5 structurally cannot answer two shapes of query: a
fragment inside a word (`rep` is inside `grep` but is not a token of it) and a
query made entirely of punctuation, which tokenizes to nothing. Without the
fallback the index would be strictly worse than a plain substring search for
both.

## Lag is reported, never hidden

The index is always slightly behind. Embedding needs a network call, so it could
never be part of a write path.

`Status` is what makes that honest: `StaleConversations` is how many
conversations sit at a revision the index has not read, `PendingEmbeddings` how
many messages still need a vector, `TruncatedMessages` how many are longer than
`maxChunksPerMessage` and so covered only in part by the semantic half, and
`LastError` the last failure verbatim. Without those, "no results" and "not
indexed yet" are indistinguishable.

`Search` reports its `Mode` for the same reason. A semantic hit may share no word
with the query at all, and a result list that does not say which kind it is
reads as a bug.

Nothing is ever marked permanently failed. A message that cannot be embedded
today is picked up by the next pass, however many passes that takes, and the
reason is in `Status.LastError` rather than buried in a counter that ran out.

## The embedder seam is asymmetric

```go
type Embedder interface {
    EmbedDocuments(ctx context.Context, texts []string) ([][]float32, error)
    EmbedQuery(ctx context.Context, text string) ([]float32, error)
}
```

Two methods, not one, because retrieval is asymmetric in most modern embedding
models: a stored passage and the question that should find it are embedded
differently, and the model is told which it is being given.

Nomic's text models are the plain example — the card for
`nomic-embed-text-v1.5` says the prompt *"must include a task instruction
prefix"*, and the two that matter are `search_document: ` and `search_query: `.
The E5 and BGE families have their own conventions. Nothing about this shows up
at runtime: with the wrong prefix every call succeeds, every vector is
well-formed, and the results are quietly worse. A single symmetric `Embed`
method makes that mistake both the default and invisible, which is why there
isn't one. For a symmetric model, `EmbedQuery` is a one-line call through to
`EmbedDocuments`.

`HTTPEmbedder` takes the prefixes as configuration:

```go
search.HTTPEmbedder{
    BaseURL: "http://localhost:8080", Model: "nomic-embed-text-v1.5",
    DocumentPrefix: search.NomicDocumentPrefix, // "search_document: "
    QueryPrefix:    search.NomicQueryPrefix,    // "search_query: "
    MaxBatch:       32,
}
```

They are empty by default, which is correct for a symmetric model, and are
never inferred from a model's name: applying a prefix to a model that was not
trained on it is the same silent damage in the other direction. The two Nomic
constants exist so nobody has to guess the exact literal, not as a default.

Changing a prefix changes the vectors it produces, so it is a re-index:
`DropModel` that model and let the backfill run again.

`MaxBatch` splits one call into several requests. The cap belongs to the
endpoint and endpoints disagree — a self-hosted inference server commonly caps
a batch far below what a hosted API accepts — and without it an index batching
more than the endpoint allows fails every pass forever, which reads as a broken
index rather than a setting.

## Chunking

A message longer than `chunkRunes` (1200) is split into overlapping windows,
because every embedding model flattens its whole input to a single vector and a
window that is too wide averages unrelated topics into a direction that matches
none of them. `chunkOverlap` (120) means a passage straddling a boundary is
whole inside one of the two.

`maxChunksPerMessage` (16) caps what one message can cost — a 40 KB tool result
would otherwise be 30-odd embeddings of build output. The cap is recorded rather
than applied silently: `embed_status.chunks_total` is what the content needed and
`chunks` what was embedded, so `Status.TruncatedMessages` can report it. The text
index still covers every word, so the cap narrows the semantic half alone.

A message's chunks never span two requests, which is what lets `embed_status` be
trusted as "this message is done": a batch that lands is a whole number of
finished messages.

## Scoping

`Conversation.Owner` is carried onto every indexed row, and `Query.Owner` must
match it exactly. There is deliberately no wildcard — a host that gets the value
wrong returns nothing rather than everything. A single-user host leaves it empty
on both sides.

`hydrate` re-checks the owner even though both halves already scoped by it: the
halves fuse into a list of bare ids, and a list of ids is exactly the shape that
loses its scope on the next edit to that file.
