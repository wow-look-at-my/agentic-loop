# agentic-loop

Reusable agentic-loop libraries for OpenAI-compatible and Anthropic APIs:
streaming (with abortable callbacks and provider-reported timings), the
tool-calling loop with approval gating, transient-failure retry with
context-overflow classification, rejected-parameter recovery, prompt
caching on both dialects, conversation compaction, and optional built-in
tools (`run_subagent`, `web_fetch`) that compose like any other tool
executor — extracted from an internal chat application so the same loop can
be embedded in other hosts.

## Layout

- [`src/`](src/) — the Go library. Package `agentic` is the loop and
  providers; [`src/vfs/`](src/vfs/) is the optional virtual-filesystem
  tools (`vfs.NewFileTools`). See [src/README.md](src/README.md) for the
  API tour and examples.

## Design points

- **Two dialects, one model.** A neutral `Message`/`Tool`/`Usage` model maps
  onto OpenAI chat completions and the Anthropic Messages API; the loop and
  every seam (tool execution, approval, retry, compaction) are
  dialect-agnostic.
- **Faithful passthrough.** Provider-specific parameters ride in
  `Request.Extra` verbatim; the library never interprets or gates them, and
  a "rejected parameter" 400 is recovered by stripping exactly the named
  key and retrying once.
- **Caching is built in.** Anthropic requests carry exactly two ephemeral
  cache breakpoints (system + transcript tail) applied to per-request
  copies — the stored transcript stays marker-free; OpenAI requests default
  `stream_options.include_usage` and support `prompt_cache_key` /
  self-hosted `cache_prompt`.
- **Honest accounting.** Usage is normalized to the full prompt on both
  dialects, cache counts are tri-state (reported-number-including-zero vs
  absent), streamed snapshots are merged newest-wins and never summed.

## License

MIT — see [LICENSE](LICENSE).
