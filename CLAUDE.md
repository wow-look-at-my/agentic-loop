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
  semantics (RunSubagent's wrap-up fallback — but NOT its turn cap, which
  this library deliberately dropped, see Hard rules;
  Run's approval flow and cancel/approval finalization), and the built-in
  tools — `run_subagent`
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

## Layering — read this before moving anything between layers

Two layers, and most design arguments in this repo are really this question
asked sideways. `go/README.md` has the full statement; the short form:

- **The loop (`Run`) is high-level.** It asks the model, runs the tools it
  asks for, feeds results back, repeats. It knows nothing about HTTP,
  status codes, or backoff. Its whole contract with the layer below is
  "complete this request".
- **An error reaching the loop is REAL and PERMANENT, so the loop stops.**
  It is entitled to assume that: the layer whose job was to make the call
  happen has already given up. The loop never retries and has no retry knob.
- **The provider carries the loop's instructions out.** When the loop says
  "complete this request", making that true — across a 429, a 502, a dropped
  connection, a rejected parameter — is the provider's job. Those are
  implementation details of doing the thing, not outcomes to propagate. It
  errors only when the operation genuinely cannot be completed.

Provider owns HOW a call gets made and everything transient in the attempt;
loop owns WHAT calls to make and what to do with results. That is why
retry and `NewParamStripper` are provider-side, and why `Config` /
`SubagentConfig` carry no retry policy. When adding something new, ask which
side of that line it falls on — do not put it on both.

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
- **Retry belongs to the Provider and is ON by default.** Both constructors
  end at `newProvider`, which wraps what they build (`ProviderConfig.Retry`,
  nil = `DefaultRetry` = 10 attempts; a one-attempt policy disables it and
  returns the dialect provider unwrapped). `ProviderConfig.Retry` is the
  library's ONE retry knob — do NOT add another to `Config` or
  `SubagentConfig`: two layers multiply (10 x 10), and an opt-in retry is
  one callers forget to enable. The provider is also the only layer that
  knows whether a call streamed anything, which is what makes re-sending
  safe.
- **A tool is an individual thing, and nothing groups them.** `Tool` is
  `Decl`/`Execute`/`NeedsApproval`, and a run's toolset is a flat `Tools`
  slice `Run` indexes by advertised name. There is no `ToolExecutor`, no
  composite, and no view wrapper: concatenating toolsets is `append`, and
  restricting one is `Readonly()`/`Subset()` returning a shorter slice. A
  wrapper whose only job is to hide part of another wrapper is the design
  this replaced -- do not reintroduce it. The call id being answered is not
  an `Execute` argument: it rides the context (`WithToolCallID`/`ToolCallID`),
  which `Run` sets around every call, because almost no tool wants it.
- **There is NO turn cap, and adding one back is a regression.** No
  `MaxTurns` on `Config` or `SubagentConfig`, no `DefaultMaxTurns`, no
  tools-withheld final turn. A counted cap cannot tell a model looping
  uselessly from one deep in a hard task, so it fires at the worst possible
  moment: after the run has spent every call gathering context and right
  before the model writes any of it down — the most expensive failure mode
  available, since the whole investigation is paid for and then discarded.
  What bounds a run is `ErrStuck` (repetition is the only mechanically
  detectable form of not-progressing) and the caller's `ctx`.
  `TestRunHasNoTurnCap` guards this.
- **Retrying must stay observable.** 10 attempts of uncapped backoff is
  ~255s; `StreamEvents.OnRetry` fires before each one so the host can show
  the failure and the wait. A retry notification is not a stream event and
  never withholds a retry.
- Exact strings are contract: `DeniedMessage`, the sub-agent refusal texts,
  `tool execution failed: ...`, the wrap-up instruction, the stuck nudge
  (`stuckNudgeInstruction`; `StuckNudgeAt`/`StuckFailAt` are constants, not
  knobs), the compaction
  request text, the param-strip regexes, the overflow regex, and the three
  built-in tools' prompts/schemas/teaching errors (the subagent
  description + schema, `DefaultSubagentSystemPrompt`, the share_context
  and allowed_tools error texts, `SubagentCutOffNote`,
  `SubagentNoReportText`, the context-summary and web-summary
  prompts, the web_fetch validation/cap/result texts, the todo_write
  description/schema/teaching errors and `RenderTodos`) are pinned by tests
  and by PARITY.md. Do not "improve" them.
- **A sub-agent's report is what it ANSWERED, never what it was mid-way
  through saying.** A backend that fails to parse a model's tool-call
  template makes the model emit the call as text, so the run ends on an
  envelope; `subagentReport` (`subagent_leak.go`) cuts it and flags the
  result rather than handing working notes up as findings. Matching is
  line-start only — prose quoting `<tool_call>` mid-line is a legitimate
  report and must stay untouched.
- **`SubagentActivity.Content` is the whole text; `Detail` is the preview.**
  Hosts render the former; do not cap it, and do not drop it back to a
  preview — a 160-rune hint of a file listing tells a reader nothing.
- Callback errors (`StreamEvents.On*`, `Events.OnToolCall/OnToolResult`
  returning non-nil) must keep their contract: abort + partial
  result/completion, `errors.Is`-reachable, never `*APIError`, never
  transient, delivery marked before the callback fires.

## Fix the bug. Never build around it.

When something is broken — here or in anything this library touches — the
only acceptable response is to fix the broken thing. Not a wrapper that
avoids it, not a sentinel that satisfies it, not a comment explaining how to
live with it. Fixed bugs are the deliverable. Workarounds are a bill sent to
everyone who comes later.

The turn cap is the case study. The loop capped every run at 10 model calls
and offered no way to say "no cap". A consumer needed uncapped runs, so
instead of removing the cap it wrote this:

```go
// subagentNoCapTurns is the effectively-unbounded turn count the wrapper maps
// the app's "0 = no cap" convention onto: the library treats MaxTurns <= 0 as
// its own default (10), so the documented SUBAGENT_MAX_TURNS=0 behavior needs
// an explicit huge cap instead.
const subagentNoCapTurns = 1 << 30
```

Read what that comment is: a correct, well-written diagnosis of a library
bug, checked in NEXT TO the bug instead of on top of it. Whoever wrote it
understood the defect completely — they had to, in order to route around it.
The fix was deleting a field. They wrote a paragraph instead.

What the workaround actually cost:

- **The bug stayed.** Every other consumer still got a silent hard 10.
- **It hid the bug.** The symptom was gone in one place, so nothing pushed
  anyone toward the real fix. Bugs get fixed because they hurt; a workaround
  is anaesthetic.
- **It cost days.** Agents were cut off mid-investigation, produced truncated
  reports, and were relaunched to redo work they had already done. Every one
  of those hours went on re-deriving something the workaround's own author
  already knew.
- **It made the codebase lie.** `MaxTurns` looked like a supported knob.
  `1 << 30` looked like a considered value. Neither was true.

**Why the time cost is the part that matters.** A workaround is not a neutral
trade of elegance for speed. It converts a bounded, one-time cost — fix it
now, while the diagnosis is in your head and the file is open — into an
unbounded, recurring one, paid by other people who do NOT have that context,
at unpredictable moments, usually under deadline. A bug you fix is gone
forever. A bug you route around is rediscovered from scratch by everyone who
trips on it, and each rediscovery costs more than the original fix, because
they must reverse-engineer the workaround before they can even see the bug.
Days spent that way buy nothing: no feature ships, no question is answered,
no code improves. It is the purest waste available in software, and twenty
minutes on the real fix avoids all of it.

Concretely:

- **Diagnosing a bug obligates you to fix it.** If you understood it well
  enough to avoid it, you understood it well enough to fix it.
- **Never encode a bug's shape into a constant, sentinel, retry, sleep, or
  wrapper.** `1 << 30`, a magic default, a layer that exists only to undo a
  lower layer — all the same move.
- **A comment explaining a workaround is the alarm, not the solution.** If
  you are writing prose about why the code is shaped wrong, stop and reshape
  it.
- **Another layer, repo, or author is not an exemption.** Fix it there and
  get it merged. If you genuinely cannot (no access), say so plainly, name
  the exact fix, and make the workaround loud and temporary — never quiet
  and permanent.
- **"It works now" is not done.** It works for you, once. Done means the
  next person cannot hit it.

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
