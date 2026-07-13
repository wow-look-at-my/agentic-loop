# agentic-loop

Reusable agentic-loop libraries for OpenAI-compatible and Anthropic APIs:
streaming, the tool-calling loop, tool approvals, retries, prompt caching,
and conversation compaction — extracted from an internal chat application
so the same loop can be embedded in other hosts.

## Layout

- `go/` — the Go library (current).
- `ts/` — a planned TypeScript port with full behavioral parity (not
  started; the layout reserves its place).

## License

MIT — see [LICENSE](LICENSE).
