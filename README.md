# agentic-loop

An agentic tool loop and the wire half it runs on, in one Go module: three
provider dialects behind one message model, streaming, tool calling with
approval gating, retry, compaction, built-in tools — plus one XML format for a
model call and its answer, and the transports that carry it.

## Layout

Package `agentic` is the module root:

```go
import agentic "github.com/wow-look-at-my/agentic-loop"
```

Optional families are sibling packages: [`vfs/`](vfs/), [`repo/`](repo/),
[`subagent/`](subagent/), [`webfetch/`](webfetch/), [`todo/`](todo/),
[`resources/`](resources/). See [USAGE.md](USAGE.md) for the API tour.

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

Depth: [`CLAUDE.md`](CLAUDE.md), [`USAGE.md`](USAGE.md) and
[`docs/`](docs/).

## License

MIT — see [LICENSE](LICENSE).
