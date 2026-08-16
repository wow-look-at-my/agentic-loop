# agentic-loop

An agentic tool loop and the wire half it runs on, in one Go module: three
provider dialects behind one message model, streaming, tool calling with
approval gating, retry, compaction, built-in tools — plus one XML format for a
model call and its answer, and the transports that carry it.

## Layout

One module, `github.com/wow-look-at-my/agentic-loop/go`.

| Package | What it is |
| --- | --- |
| [`go/`](go/) | the loop: `Run`, tools, approval, sub-agents, compaction (package `agentic`) |
| [`go/core`](go/core/) | the format, its schema, and the three dialects |
| [`go/client`](go/client/) | the Go API: `Provider`, `Completion`, folded usage |
| [`go/extras`](go/extras/) | retry and a fixed-rate request gate |
| [`go/session`](go/session/) | conversation storage: memory and one document per file |
| [`go/http`](go/http/) | HTTP, stateless and stateful |
| [`go/socket`](go/socket/) | unix socket and websocket |
| [`go/cli`](go/cli/) | the `cai` commands (`go/cmd/cai` is its `main`) |

`ts/` is a planned TypeScript port. The Go source and
[go/README.md](go/README.md) are its specification.

## Two ways in

```go
comp, err := agentic.Run(ctx, cfg, agentic.Request{Model: "claude-x", ...})
```

```sh
cai ask "what is in this image?" --image shot.png   # a CLI, with no XML in sight
cai serve --http :8080 --socket /run/cai.sock       # HTTP, websocket, unix socket
```

## The document

```xml
<?xml version="1.1" encoding="UTF-8"?>
<request xmlns="https://github.com/wow-look-at-my/common-ai-api/schema/v1"
         xmlns:anthropic="https://github.com/wow-look-at-my/common-ai-api/schema/v1/anthropic"
         model="claude-x" max-tokens="4096" anthropic:top-k="40">
  <system><text>You are ...</text></system>
  <messages>
    <message role="user">
      <text>what is in this image?</text>
      <image media-type="image/png">iVBORw0...</image>
    </message>
  </messages>
</request>
```

A message's content is an ordered list of parts, so a reply whose text brackets
a thinking block survives intact. Anything provider-specific is namespaced and
declared: a scalar rides as a qualified attribute, anything object-shaped as a
namespaced element. Nothing is a wildcard, so a misspelled provider parameter
is a validation error here rather than a 400 from an upstream.

Streaming is the same document, written as it arrives — one vocabulary, and a
stream that cannot disagree with its own result.

## Design points

- **Three dialects, one model.** A neutral `Message`/`Tool`/`Usage` model maps
  onto OpenAI chat completions, the OpenAI Responses API and the Anthropic
  Messages API; the loop and every seam is dialect-agnostic.
- **Faithful passthrough.** Provider-specific parameters ride in
  `Request.Extra` verbatim, and a "rejected parameter" 400 is recovered by
  stripping exactly the named key and retrying once.
- **Caching is built in.** Anthropic requests carry exactly two ephemeral cache
  breakpoints applied to per-request copies, so the stored transcript stays
  marker-free.
- **Honest accounting.** Cache counts are tri-state, and streamed usage
  snapshots are kept in order rather than summed.

## Build

```sh
cd go && go-toolchain
```

Depth: [`CLAUDE.md`](CLAUDE.md), [`go/README.md`](go/README.md) and
[`docs/`](docs/).

## License

MIT — see [LICENSE](LICENSE).
