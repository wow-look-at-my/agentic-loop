# CLAUDE.md

Notes for Claude working in this repository.

## What this is

`agentic-loop` — reusable agentic-loop libraries for OpenAI-compatible and
Anthropic chat APIs. `go/` holds the Go library (package `agentic`, a single
package); `ts/` is a planned TypeScript port whose specification is
[PARITY.md](PARITY.md) — keep that file in sync with any behavioral change
to the Go code.

Where the semantics came from (the Go library is an extraction, not a
redesign — check these when a behavior question comes up):

- `simple-llm-ui` `internal/chat` + `internal/upstream` + `internal/tools`:
  the OpenAI dialect (wire shapes, the Message content presence rule, the
  tool-call accumulator, usage merging, the llama.cpp `timings` decode,
  param-strip retry, the SSE scanner and its buffer sizes), the loop
  semantics (RunSubagent's final-turn tools-withheld + wrap-up fallback;
  Run's approval flow and cancel/approval finalization), the executor
  combinators, and the two built-in tool executors — `run_subagent`
  (`internal/tools/subagent.go` + `internal/chat/subagent.go` +
  `context.go`/`summary.go`: schema, share_context modes, allowed_tools
  grants, Gate, activity telemetry) and `web_fetch`
  (`internal/tools/webfetch.go`: caps, URL validation, HTML cleanup, Tika,
  the model-backed summary).
- `ai-shadertoy` `src/ai` (`providers.ts`, `compact.ts`): the Anthropic
  dialect reference (message/thinking/tool_result mapping, cache
  breakpoints, stream events) and compaction.
- `model-benchmark` `src` (`env.ts`, `agent.ts`): retry constants and
  classification, context-overflow detection, the transcript-tail cache
  marker discipline.

## Build & test

ALWAYS build and test with `go-toolchain` (no args) from `go/`. NEVER run
bare `go build` / `go test` / `go mod tidy` — the toolchain does mod tidy,
vet, lint, tests with an **80% coverage gate**, and the build.

```sh
cd go && go-toolchain
```

- go-toolchain refuses a dirty tree: commit first, run it, then commit its
  auto-fixes as a follow-up commit. Its rewrites (formatting, import order,
  testify in tests, go.mod/go.sum) are canonical — never revert them.
- Tests use `testify` (`assert`/`require`); the toolchain enforces it.
- Test against `httptest` fake servers and the in-process `scriptProvider`
  stub (`run_test.go`) — no network, no credentials.

## Hard rules

- **Runtime is standard library only.** testify is test-only. No new
  runtime dependencies (the web-fetch HTML cleanup and the subagent
  machinery are deliberately hand-rolled stdlib, like the source).
- **No environment reads.** The library never calls `os.Getenv`; all I/O
  goes through the injectable `*http.Client`; endpoints/keys are explicit
  fields.
- **No secrets or org-internal URLs** in code, tests, examples, or docs —
  placeholder endpoints (`https://api.openai.com/v1`,
  `https://api.anthropic.com`) and placeholder keys only.
- **Providers are built ONLY via the per-dialect constructors**
  (`NewOpenAIProvider(OpenAIConfig)` / `NewAnthropicProvider(AnthropicConfig)`,
  each embedding the shared `ProviderConfig` connection base). The dialect
  implementations (`openaiProvider`, `anthropicProvider`) stay unexported;
  do not re-export them or add construction side doors.
- Exact strings are contract: `DeniedMessage`, the executor refusal texts,
  `tool execution failed: ...`, the wrap-up instruction, the compaction
  request text, the param-strip regexes, the overflow regex, and the two
  built-in tools' prompts/schemas/teaching errors (the subagent
  description + schema, `DefaultSubagentSystemPrompt`, the share_context
  and allowed_tools error texts, the context-summary and web-summary
  prompts, the web_fetch validation/cap/result texts) are pinned by tests
  and by PARITY.md. Do not "improve" them.
- Callback errors (`StreamEvents.On*`, `Events.OnToolCall/OnToolResult`
  returning non-nil) must keep their contract: abort + partial
  result/completion, `errors.Is`-reachable, never `*APIError`, never
  transient, delivery marked before the callback fires.

## CI

`.github/workflows/ci.yml` runs the org go-toolchain action with
`working-directory: go`. Org constraints:

- The workflow trigger stays `on: push:` only.
- The required status check is named exactly **`all-builds`**, but it is
  posted by the org's required-builds-manager app (which aggregates this
  repo's builds) — never by a job. Never name a workflow job `all-builds`:
  the guard step inside go-toolchain@v1 hard-fails any run whose workflow
  defines a job by that name. The CI job here is `build`.
- The `permissions` block (`id-token: write`, `contents: write`,
  `actions: read`, `checks: read`, `deployments: write`,
  `artifact-metadata: write`) is required by the toolchain action; don't trim
  it. Everything past `id-token` backs a step of the default-on autorelease —
  dependency-graph submission, the GitHub Deployment, the artifact storage
  record — none of which can be opted out of, so a missing grant fails the
  build *after* tests and vet have passed. A red run whose test output looks
  clean is usually this, not the code; `action.yml`'s `autorelease` input
  description is the authoritative list.

## Git workflow

- One branch and one PR per session; branch names follow `claude/<name>`.
- Commit and push frequently — the working VM is ephemeral; unpushed work
  is lost.
- PRs are squash-merged: add follow-up commits, never rebase or force-push
  a pushed branch.

## Documentation upkeep

When changing the API surface or any behavior: update `go/README.md`,
`PARITY.md` (the ts/ port contract), and this file in the same commit.
