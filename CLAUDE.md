# CLAUDE.md

Notes for Claude working in this repository.

## What this is

`agentic-loop` — a reusable agentic-loop library for OpenAI-compatible and
Anthropic chat APIs. Package `agentic` is the module root
(`github.com/wow-look-at-my/agentic-loop`). Sibling packages are `vfs`,
`repo`, `subagent`, `webfetch`, `todo`, and `resources`. The loop package
does not import the optional packages. There is no TypeScript port and none
is planned.

Where the semantics came from (the Go library is an extraction, not a
redesign — check these when a behavior question comes up):

- `simple-llm-ui` `internal/chat` + `internal/upstream` + `internal/tools`:
  the loop semantics (RunSubagent's wrap-up fallback — but NOT its turn cap,
  which this library deliberately dropped, see Hard rules; Run's approval flow
  and cancel/approval finalization), the filesystem
  tools (`internal/tools/fs*.go`: the seven tools' names, descriptions,
  schemas, caps and rendering), the GitHub client and repo tools
  (`internal/tools/repo*.go` + `token_probe.go`: the credential rotation,
  the winning-token cache, repo_read's what-dispatch, the two gated writes,
  and every failure explanation), and the built-in
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

The Responses dialect has NO source repo — it was written here, against the
API's own shapes. Do not go looking for the semantics somewhere else; the
answers are `USAGE.md` and `responses_test.go`.

## Build & test

ALWAYS build and test with `go-toolchain` (no args) from the repo root.
NEVER run bare `go build` / `go test` / `go mod tidy` — the toolchain does
mod tidy, vet, lint, tests with an **80% coverage gate**, and the build.
Build output goes in `/build` at the repo root.

```sh
go-toolchain
```

- go-toolchain refuses a dirty tree: commit first, run it, then commit its
  auto-fixes as a follow-up commit. Its rewrites (formatting, import order,
  testify in tests, go.mod/go.sum) are canonical — never revert them.
- Tests use `testify` (`assert`/`require`); the toolchain enforces it.
- Test against `httptest` fake servers (`upstream_test.go`) and the in-process
  `scriptProvider` stub (`run_test.go`) — no network, no credentials. A test
  whose subject is a DIALECT lives beside that dialect in `core/`; a test of
  the loop.s behavior when a call streams, retries, or breaks lives at the repo root.
- No `go.work` and nothing to check out alongside: the only org dependencies
  are `xml-validator/validator` and `go-containers`, both resolved from the
  proxy.

## Layering — read this before moving anything between layers

Two layers, and most design arguments in this repo are really this question
asked sideways. `USAGE.md` has the full statement; the short form:

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

- **`go-containers/set` is not optional.** go-toolchain's `mapset` analyzer
  hard-fails an org module that uses a `map[K]bool` as a set, so a set is
  `set.Set[K]` here. testify is test-only. A dependency reaches only the
  packages that import it and a binary links only what it imports, so weight
  is an argument about `go.sum` -- `docs/module-layout.md` weighs it and says
  what that is worth.
- **A search index that is behind SAYS SO.** `search` is asynchronous by
  construction -- embedding is a network call -- so `Status` reports the stale
  conversations, the pending embeddings, the truncated messages and the last
  error verbatim, and `Search` reports which half answered. Nothing is ever
  marked permanently failed: a message that cannot be embedded today is picked
  up by the next pass. **`Embedder` is asymmetric on purpose**
  (`EmbedDocuments`/`EmbedQuery`): most embedding models want a task prefix and
  a different one per side, and a wrong prefix is invisible -- every call
  succeeds and the results are merely worse. Do not collapse it back to one
  `Embed`. Its other invariants -- the two schema versions and the shape check
  that catches a version describing another index's tables, why the vectors are
  scanned in Go and what that measures at, and why a message id must be stable
  -- are in `docs/search.md`.
- **No environment reads.** The library never calls `os.Getenv`; all I/O
  goes through the injectable `*http.Client`; endpoints/keys are explicit
  fields.
- **No secrets or org-internal URLs** in code, tests, examples, or docs —
  placeholder endpoints (`https://api.openai.com/v1`,
  `https://api.anthropic.com`) and placeholder keys only.
- **`aliases.go` is aliases and thin calls, never a copy.** Every wire type is
  `= client.X`, so the two packages share one declaration. Never redeclare a
  type as a struct of its own, and never widen the re-export past what the loop
  uses: a name added here is a name this package now owns forever. Anything the
  loop needs and cannot reach is a gap in `client`'s surface — fix it there.
- **Providers are built ONLY via the per-dialect constructors**
  (`NewOpenAIProvider` / `NewResponsesProvider` / `NewAnthropicProvider`, each
  embedding the shared `ProviderConfig` connection base). The dialect
  implementations (`openaiProvider`, `responsesProvider`, `anthropicProvider`)
  stay unexported; do not re-export them or add construction side doors.
  `Dialect` (dialect.go) NAMES a protocol for a host's settings — it never
  constructs one.
- **The Responses dialect exists for exactly one thing** (`responses.go` +
  `responses_wire.go`): a reasoning model's chain of thought surviving a tool
  call, which chat-completions has no field for. So the reasoning ITEM with its
  `encrypted_content` is what gets replayed, never a summary alone — a summary
  is prose about the reasoning. `Store` is FALSE by default against the API's
  own default (third-party retention is the caller's decision to make out
  loud), `previous_response_id` is never sent (the transcript is the caller's),
  and detection can never name this dialect, since the model list looks
  identical. Depth: `USAGE.md`.
- **Retry belongs to the Provider and is ON by default.** Both constructors
  end at `newProvider`, which wraps what they build (`ProviderConfig.Retry`,
  nil = `DefaultRetry` = 10 attempts; a one-attempt policy disables it and
  returns the dialect provider unwrapped). `ProviderConfig.Retry` is the
  library's ONE retry knob — do NOT add another to `Config` or
  `SubagentConfig`: two layers multiply (10 x 10), and an opt-in retry is
  one callers forget to enable. The provider is also the only layer that
  knows whether a call streamed anything, which is what makes re-sending
  safe. That is why it is a knob the loop passes along and never reads.
- **A tool is an individual thing, and nothing groups them.** `Tool` is
  `Decl`/`Execute`, and a run's toolset is a flat `Tools`
  slice `Run` indexes by advertised name. There is no `ToolExecutor`, no
  composite, and no view wrapper: concatenating toolsets is `append`, and
  restricting one is `Readonly()`/`Subset()` returning a shorter slice. A
  wrapper whose only job is to hide part of another wrapper is the design
  this replaced -- do not reintroduce it. The call id being answered is not
  an `Execute` argument: it rides the context (`WithToolCallID`/`ToolCallID`),
  which `Run` sets around every call, because almost no tool wants it.
- **A tool does not decide whether it is asked about.** `Config.Approver` is
  consulted for EVERY call, read-only included — a deny rule that cannot fire
  on part of the toolset is a lie about what it protects — and `ToolDecl.Readonly`
  is the ONE declaration of what a tool does to state. Do not put a
  `NeedsApproval` back on `Tool`, and do not reintroduce a gating wrapper: a nil
  `Approver` allows a `Readonly` call and denies the rest, which is the same
  fail-closed default expressed once. A denial carries `Approval.Reason`, and
  only an empty one falls back to `DeniedMessage` — an optional reason is one
  that goes missing exactly when a rule was written in a hurry.
- **A tool states FACTS, and two of them default to true.** `ToolDecl` carries
  MCP's four annotations (`Readonly`, `Destructive`, `Idempotent`,
  `OpenWorld`) plus `Unvouched`. `Destructive` and `OpenWorld` are POINTERS
  because their MCP defaults are true: a bare bool reads an unstated fact as
  the dangerous answer, and absent must stay distinguishable from `false` on
  the wire too. Read them only through `IsDestructive`/`IsIdempotent`/
  `IsOpenWorld`/`Vouched`, which also apply the spec's rule that the first two
  are meaningless when `Readonly`. `Unvouched` marks an MCP server's claim
  about its own tool: the spec forbids deciding from an untrusted server's
  annotations, and a nil `Approver` plus `Tools.Readonly()` would otherwise
  auto-run a lying server's tool and hand it to sub-agents. Depth: `USAGE.md`.
- **Nothing caps a run by default, and adding a default back is a
  regression.** `Config.MaxTurns` is the HOST's cap and is off at zero; there
  is no `DefaultMaxTurns`. A counted cap cannot tell a model looping uselessly
  from one deep in a hard task, so a default fires at the worst possible
  moment: after the run has spent every call gathering context and right
  before the model writes any of it down — the most expensive failure mode
  available, since the whole investigation is paid for and then discarded. An
  uncapped run is bounded by `ErrStuck` (repetition is the only mechanically
  detectable form of not-progressing) and the caller's `ctx`. A capped run
  makes its last call tool-less, so the model answers instead of asking for a
  tool nothing will run. `TestRunHasNoTurnCap` guards the default.
- **Auto-compaction is caller-armed and loop-fired.** `Request.AutoCompact`
  is the fraction (0..1) of `Config.ContextWindow` at which `Run` compacts
  the transcript mid-run; `DefaultAutoCompact` is 0.8. The fraction lives on
  `Request` (it is a session property that persists with the conversation
  document); the window lives on `Config` (it is a model property the host
  supplies). After a turn whose `PromptTokens` reaches the threshold, the
  loop calls `Compact`, replaces the transcript, resets the deduper, and
  fires `Events.OnCompaction`. Zero disables; a server reporting no usage
  never triggers; a compaction failure is non-fatal. Depth: `USAGE.md`.
- **A message a queue ACCEPTS reaches the model.** `SystemMessages` and
  `UserMessages` are drained at the top of every turn, system first, and a
  message queued when the model would otherwise finish starts another turn —
  every time. `Events.OnStop` is asked at every stop boundary for the same
  reason a turn cap is a regression: a count cannot tell a host re-arming
  with a reason from one spinning, and the cap fired at the worst moment.
  `Queue` returns whether the queue took it, `Run`
  closes both queues as it returns (so a racing producer starts a new run
  instead), and whatever a failed, cancelled or capped run never delivered
  comes back in `Result.Undelivered`. Depth: `USAGE.md`.
- **Every entry point that makes a model call surfaces its `*Completion`.**
  Never a `Usage`, never a bare string: only `Completion.UsageReported`
  separates "reported zeros" from "reported nothing", and a projection drops
  `CostUsd`, `ReasoningTokens`, `RawUsage`, `Timings` and `Streamed`. So
  `OneShot` returns `(*Completion, error)`, `CompactResult` carries
  `Completion`, `SubagentActivity` has the `turn_end` kind, `WebFetchConfig`
  has `OnCompletion`, and `SubagentReport`/`SubagentUpdate` carry `Usages`
  (the sub-run's turns plus its `share_context=summary` briefing, in order,
  never summed). Under-counted money cannot be recovered later — the numbers
  were never emitted.
- Exact strings are contract: `DeniedMessage`, the sub-agent refusal texts,
  `tool execution failed: ...`, the wrap-up instruction, the stuck nudge
  (`stuckNudgeInstruction`; `StuckNudgeAt`/`StuckFailAt` are constants, not
  knobs), the compaction
  request text, and the
  built-in tools' prompts/schemas/teaching errors (the subagent
  description + schema, `DefaultSubagentSystemPrompt`, the share_context
  and allowed_tools error texts, `SubagentCutOffNote`,
  `SubagentNoReportText`, `SubagentLaunchReceipt` and the
  `FormatSubagentDelivery` text, the context-summary and web-summary
  prompts, the web_fetch validation/cap/result texts, the task-list tools'
  descriptions/schemas/teaching errors and `RenderTodos`, and every word the
  seven file tools render — descriptions, schemas, the cap announcements,
  and grep's real-negative sentence) are pinned by tests.
  Do not "improve" them.
- **A tool's schema is INFERRED from the struct its handler decodes**
  (`InferSchema`/`EnumSchema`, hand-rolled reflection in `schema.go`, since
  what a tool argument needs is small). Never hand-write one: that is a second
  declaration of the argument list, and nothing keeps it true. Field prose is
  the `jsonschema` tag; `omitempty` is what makes an argument optional; a
  field with no json tag panics at construction.
- **A GitHub failure ESTABLISHES its cause, and reports the most informative
  attempt.** The anonymous read runs LAST and its 401 says only that no
  credential was sent, so returning it made a spent rate limit read as a
  permanent auth problem (`MoreInformativeAuthFailure`). A write that
  exhausts its credentials re-reads the repository before blaming them: a
  404 on a named object with the repository readable means the OBJECT is
  gone. Writes use `WriteTokens` only and never fall through to anonymous.
- **A file tool's rendering IS the tool.** A cap that bites is announced
  (truncated listing, `find_files` at its limit, `grep` at `MaxHits`) and an
  empty `grep` states that every line in scope was read, because a partial
  result that reads as complete is worse than no result. That is also why a
  scope the mount does not hold is an ERROR rather than an empty result, and
  why a single FILE is a scope (`WithinScope`) — treating one as a directory
  made every file-scoped search answer "no matches" for text right there.
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
- **The tool-call seam is one ordered pass, and the transcript is not part of
  it.** `OnToolCall(*ToolCall)` may rewrite the call, `Approver.Ask` then judges
  the REWRITTEN call, and `Execute` receives it — reordering those puts a hook's
  rewrite past the decision that was supposed to cover it. The assistant message
  keeps the model's own arguments (what was asked and what ran are two facts),
  and the tool result answers the model's own call id, so a rewritten `ID` is
  ignored rather than allowed to orphan the result. `OnToolResult` carries the
  RECORDED `Message` as a third argument, because dedup may have replaced the
  content with an `UnchangedPrefix` marker and only the loop knows that.
- Callback errors (`StreamEvents.On*`, `Events.OnToolCall/OnToolResult`
  returning non-nil) must keep their contract: abort + partial
  result/completion, `errors.Is`-reachable, never `*APIError`, never
  transient, delivery marked before the callback fires.

## Hard rules — the wire half (`core/`, `client/`, and the transports)

- **The format is the definition.** The XSD in `core/schema/` is normative, so
  an implementation in another language can speak it from the schema alone. The
  Go types are a conforming projection of it, not the other way round: when
  they disagree, the Go code is what is wrong.
- **core translates and does not decide.** It maps a document onto a dialect
  and back, faithfully. Anything that folds, merges, retries, or applies a
  policy is `client`'s business.
- **A usage report is what the provider said.** `core` keeps every report, in
  order (`Completion.Usages`); `client` folds them into the one figure a caller
  bills against. A document reporting an invented number would be the library's
  arithmetic wearing the provider's name.
- **Validate at every boundary.** Anything arriving from outside is checked
  against the schema before anything acts on it.
- **Nothing is a wildcard.** No `xs:any`, no `xs:anyAttribute`. An undeclared
  attribute is an error, which is the point.
- **A dialect that cannot express something FAILS.** Never drop it: a request
  quietly stripped of what the caller asked about is a wrong answer that looks
  like a right one.
- **A turn that says NOTHING is dropped**, which strips no content and so is
  not the rule above. Anthropic and Responses omit an assistant turn with no
  text, no tool call and no replayable thinking: an empty text block fails the
  WHOLE request, and one stored in a transcript then kills every later turn in
  that conversation. A tool-call-only turn is not empty.
- **A zero-argument tool call is `{}`, never a missing field** (`toolArgs`).
  A model calling such a tool sends no argument bytes, so every read path
  normalizes the empty string, and both wire writers apply it again for a
  transcript that came from the host's storage. `omitempty` on `arguments`
  is what made Z.AI 400 the turn — forever, since the call is persisted.
- **The Responses dialect exists for exactly one thing** (`core/responses.go` +
  `core/responses_wire.go`): a reasoning model's chain of thought surviving a
  tool call, which chat-completions has no field for. So the reasoning ITEM
  with its `encrypted_content` is replayed, never a summary alone — a summary
  is prose ABOUT the reasoning. `Store` is FALSE by default against the API's
  own default (third-party retention is the caller's decision to make out
  loud), `previous_response_id` is never sent (the transcript is the
  caller's), and detection can never name this dialect, since the model list
  looks identical.
- **`&#0;` is how NUL travels.** Our writer emits it and our validator accepts
  it — see `docs/nul-char.md`.
- **No environment reads outside `cli/`.** Endpoints and keys are explicit
  fields; all I/O goes through an injectable `*http.Client`.

## Where the depth lives

- `docs/format.md` — the document vocabulary, the param tree, provider
  namespaces, and what makes it lossless.
- `docs/streaming.md` — the progressive document, `OnPart`, and why there is no
  event vocabulary.
- `docs/module-layout.md` — why this is one module, and what the seven-module
  split cost before it was collapsed.
- `docs/search.md` — the conversation index: the Source seam, why the vectors
  are scanned in Go, what that costs measured, and how lag is reported.
- `docs/nul-char.md` — `&#0;`, and the one deviation from XML 1.1's `Char`.

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

`.github/workflows/ci.yml` runs the org go-toolchain action at the
repository root. Org constraints:

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

When changing the API surface or any behavior: update `USAGE.md` and
this file in the same commit.
