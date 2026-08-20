# agentic (Go)

A reusable agentic loop for chat-model APIs, with three provider dialects —
OpenAI-compatible chat completions, the OpenAI Responses API, and the
Anthropic Messages API — plus the
machinery a production tool loop needs: streaming callbacks that can abort
the call, tool execution with a per-call approval seam, transient-failure retry,
rejected-parameter recovery, prompt caching on both dialects, conversation
compaction, and two optional built-in tools (a sub-agent and a web fetcher).

The runtime is **standard library only** (plus `xml-validator/validator`,
`go-containers/set`, and cobra in `cli/`). All I/O goes through an injectable
`*http.Client`, and the package reads **no environment variables** — every
endpoint, key, and knob is explicit configuration.

The three dialects themselves live in [`core/`](core/), which speaks one XML
format and translates it to and from each provider, behind the Go API in
[`client/`](client/). This package re-exports what it uses as type aliases, so
`agentic.Message` **is** `client.Message`: a value built here is one those
packages take without a conversion step.

## Install

```sh
go get github.com/wow-look-at-my/agentic-loop
```

```go
import (
	agentic "github.com/wow-look-at-my/agentic-loop"
	"github.com/wow-look-at-my/agentic-loop/vfs"
	"github.com/wow-look-at-my/agentic-loop/repo"
	"github.com/wow-look-at-my/agentic-loop/subagent"
	"github.com/wow-look-at-my/agentic-loop/webfetch"
	"github.com/wow-look-at-my/agentic-loop/todo"
	"github.com/wow-look-at-my/agentic-loop/resources"
)
```

The loop is `agentic.Run`. Optional tools are sibling packages: `vfs.NewFileTools`,
`repo.NewRepoTools`, `subagent.NewSubagentTool`, `webfetch.NewWebFetchTool`,
`todo.NewTodoTools`, `resources.NewResourceWatcher`.

## Quick start — OpenAI-compatible

```go
provider, err := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
	ProviderConfig: agentic.ProviderConfig{
		BaseURL: "https://api.openai.com/v1", // any OpenAI-compatible endpoint
		APIKey:  "YOUR_API_KEY",
	},
})
if err != nil { /* misconfiguration */ }

res, err := agentic.Run(ctx, agentic.Config{
	Provider: agentic.NewParamStripper(provider),
	Tools:    myTools,   // an agentic.Tools slice; empty for tool-less
	Events: agentic.Events{
		StreamEvents: agentic.StreamEvents{
			OnText: func(s string) error { fmt.Print(s); return nil },
		},
	},
}, agentic.Request{
	Model:  "your-model-id",
	System: "You are a helpful assistant.",
	Messages: []agentic.Message{
		{Role: agentic.RoleUser, Content: "What changed in the last release?"},
	},
	Extra: map[string]any{"reasoning_effort": "high"}, // verbatim passthrough
})
if err != nil { /* res may still carry the partial transcript */ }
fmt.Println(res.Final.Content)
```

## Quick start — OpenAI Responses

Reach for this dialect over chat completions when the model reasons and uses
tools in the same turn: it is the only one of the three that can hand a
model back its own chain of thought after a tool call.

```go
provider, err := agentic.NewResponsesProvider(agentic.ResponsesConfig{
	ProviderConfig: agentic.ProviderConfig{
		BaseURL: "https://api.openai.com/v1", // same root as chat completions
		APIKey:  "YOUR_API_KEY",
	},
	// Store defaults to FALSE: server-side retention of your conversations
	// is opt-in here even though the API's own default is on.
})
if err != nil { /* misconfiguration */ }

res, err := agentic.Run(ctx, agentic.Config{Provider: provider, Tools: myTools},
	agentic.Request{
		Model:    "your-model-id",
		System:   "You are a helpful assistant.", // sent as `instructions`
		Messages: []agentic.Message{{Role: agentic.RoleUser, Content: "Hi!"}},
		Extra:    map[string]any{"reasoning": map[string]any{"effort": "high"}},
	})
```

## Quick start — Anthropic

```go
provider, err := agentic.NewAnthropicProvider(agentic.AnthropicConfig{
	ProviderConfig: agentic.ProviderConfig{
		BaseURL: "https://api.anthropic.com",
		APIKey:  "YOUR_API_KEY",
	},
	// Version (the anthropic-version header) defaults to "2023-06-01".
})
if err != nil { /* misconfiguration */ }

res, err := agentic.Run(ctx, agentic.Config{Provider: provider, Tools: myTools},
	agentic.Request{
		Model:     "your-model-id",
		System:    "You are a helpful assistant.",
		Messages:  []agentic.Message{{Role: agentic.RoleUser, Content: "Hi!"}},
		MaxTokens: 4096, // REQUIRED on this dialect — the call fails fast without it
	})
```

## Layering — which layer owns what

Two layers, and the split decides where anything new belongs. Get this
backwards and you end up with the same concern implemented twice, fighting
each other. The two layers are two packages — the lower one is `client/` over
`core/` — so the compiler enforces the direction: the provider cannot reach for
anything here, because the import would be a cycle.

**The loop (`Run`) is a high-level construct.** It asks the model something,
runs the tools the model asks for, feeds the results back, and repeats until
there is an answer. It knows nothing about HTTP, connections, status codes,
or backoff — those words do not appear in it. Its contract with the layer
below is simply *"complete this request."*

Consequently **an error that reaches the loop is treated as real and
permanent, and the loop stops.** It is entitled to that assumption: by the
time a failure has surfaced this far, the layer whose job was to make the
call happen has already given up. The loop does not second-guess it, does
not retry, and has no retry knob to configure — deliberately.

**The provider is where the loop's instructions are actually carried out.**
When the loop says "complete this request", making that true is the
provider's responsibility, including everything that can go transiently
wrong in the attempt: a 429, a 502, a dropped connection, a rejected
parameter. Those are implementation details of *doing the thing*, not
outcomes worth propagating. The provider surfaces an error only when the
operation genuinely cannot be completed — at which point it is, by
construction, permanent.

So: the provider owns **how** a model call gets made and everything that
can transiently go wrong doing it; the loop owns **what** calls to make and
what to do with the results.

That is why `ProviderConfig.Retry` is the library's one retry knob, why
`NewParamStripper` is a provider decorator rather than a loop feature, and
why `Config`/`SubagentConfig` carry no retry policy. It is also why retry
lives where it can see whether a call streamed anything — the condition
that decides whether re-sending is even safe — which the loop cannot see.

The one thing the provider must NOT do is hide the wait: see
`StreamEvents.OnRetry` under [Retry](#retry-and-error-classification).

## The pieces

### Provider

```go
type Provider interface {
	Complete(ctx context.Context, req Request, ev *StreamEvents) (*Completion, error)
}

func NewOpenAIProvider(cfg OpenAIConfig) (Provider, error)
func NewResponsesProvider(cfg ResponsesConfig) (Provider, error)
func NewAnthropicProvider(cfg AnthropicConfig) (Provider, error)
```

The per-dialect constructors are the only way to build the three dialect
providers — the implementations are unexported, so consumers hold nothing
but the `Provider` interface. `ProviderConfig` is the shared connection
base — required `BaseURL`, plus `APIKey`, `HTTPClient`, `UserAgent`,
`Headers` — embedded in each dialect's config: `OpenAIConfig` adds
`SelfHosted` (the `cache_prompt` opt-in for llama.cpp-style servers),
`ResponsesConfig` adds `Store`, and
`AnthropicConfig` adds `Version` (the `anthropic-version` header) and
`DisableCaching`. An empty `BaseURL` fails fast with a permanent error.

One streaming model call. `StreamEvents` carries optional callbacks —
`OnText`, `OnReasoning`, `OnUsage`, `OnProgress`, `OnTimings`, `OnRetry` —
all nil-safe,
and **every callback returns an error**: a non-nil return aborts the stream
read immediately, and `Complete` returns the partial `*Completion` (content,
reasoning, tool calls, usage so far) together with that error. The error
surfaces unwrapped or `%w`-wrapped, so `errors.Is` against your sentinel
holds; it is never converted into an `*APIError` and never classified
transient, so neither the retry policy nor the param stripper re-sends a call
whose sink failed. On any other mid-stream failure or cancellation **after
data arrived**, `Complete` likewise returns the partial `*Completion`
alongside the error. Both dialects are safe for concurrent use.

A stored transcript can hold an assistant turn that carries nothing — a run
cancelled before the first token, a model that answered with an empty
message. The Anthropic and Responses dialects drop such a turn. An empty
text block fails the whole request on both, so one turn like this in the
history would fail every later turn in that conversation, permanently. Both
APIs combine consecutive same-role turns, so the drop loses nothing. A turn
whose only output was a tool call is not empty, and is always replayed.

`Request.Extra` is a verbatim top-level passthrough (`temperature`,
`reasoning_effort`, `num_ctx`, `thinking`, ...). It is merged first, so the
typed core fields always win; the library never interprets, gates, or filters
what you put there.

### The Responses dialect, and when it is worth it

Two of the three dialects talk to the same OpenAI endpoints. They are not
interchangeable, and exactly one thing separates them:

**On chat completions, a reasoning model's chain of thought does not survive
a tool call.** The reasoning arrives as deltas, and the request format has
nowhere to put it back — so on the next turn the model re-derives what it
already worked out, pays for those tokens again, and loses the prompt cache
at every reasoning boundary. In a long tool-using investigation that is the
dominant cost.

**The Responses API models a turn as an ordered list of items**, and a
reasoning item can be sent back verbatim. This provider asks for
`include: ["reasoning.encrypted_content"]`, keeps what comes back in
`ThinkingBlock` (summary in `Text`, encrypted payload in `Signature`, item
id in `ID`), and replays it — reasoning first, then the text, then the tool
calls, in the order the model emitted them. A block with no payload is
dropped rather than half-replayed: a summary is prose *about* the reasoning,
and sending it as though it were the reasoning hands the model a paraphrase
of its own thinking.

Two positions this provider takes deliberately:

- **`Store` defaults to false**, unlike the API. Server-side retention of
  every prompt and response is a decision for the caller to make out loud,
  not one to inherit from a default. Reasoning still survives with it off —
  that is what the encrypted payload is for.
- **`previous_response_id` is never sent.** This library's contract is a flat
  transcript the caller owns and can edit, fork, compact or persist; a
  server-side conversation id would make that transcript a partial lie about
  what the model actually sees. Every call sends its full input.

Everything else behind the seam is unchanged: the same `Request`, the same
callbacks, the same retry and error classification, the same `Run`. What
differs is the wire — `instructions` rather than a system message,
`max_output_tokens` rather than `max_tokens`, `input_tokens`/`output_tokens`
rather than `prompt`/`completion`, no `finish_reason` (the shape of the turn
says it), and a failure that can arrive inside a 200 as a `response.failed`
event. That last one surfaces as a permanent error, never a transient one:
re-sending a request the server accepted and then rejected just gets billed
twice.

### Which dialect an endpoint speaks

```go
type Dialect string // DialectAuto (""), DialectOpenAI, DialectAnthropic

func (d Dialect) Valid() bool
func (d Dialect) Label() string          // "detect" / "openai-compatible" / "anthropic messages"
func Dialects() []Dialect                // every dialect, default first
```

Picking a provider constructor means knowing which protocol an endpoint
speaks, and a hostname is a guess about a server that is available to ask.
The two dialects answer the same question — "what models do you have?" —
with structurally different documents:

```
OpenAI:    {"object":"list","data":[{"id":"gpt-x","object":"model",...}]}
Anthropic: {"data":[{"type":"model","id":"claude-x",...}],"has_more":false}
```

`FetchModelList` reads that shape off the document. It answers `DialectAuto`
when the document matches neither — never a guess dressed as a finding,
because a wrong dialect does not degrade, it breaks chat outright, so a caller
that needs one refuses rather than picking.

The envelope is checked before the items, so a list with no models at all
still identifies its server.

### The model list

```go
type ModelList struct {
    Dialect Dialect
    Prices  map[string]Rates // absent = published no pricing
}

func FetchModelList(ctx context.Context, cfg ProviderConfig) (*ModelList, error)
func DecodeModelList(body []byte) (*ModelList, error)
```

One document answers both questions a host has before it can talk to an
endpoint at all — which protocol it speaks, and what its models charge — so
there is one request and one decode. `FetchModelList` does a
`GET {base}/v1/models`, sending both credential forms, since which server is
answering is the very thing in question.

The trailing `/v1` is trimmed off the base first. The two chat dialects
disagree about what a base URL contains — the OpenAI request is
`baseURL + "/chat/completions"`, so its base **ends** in `/v1`, while the
Anthropic one is `baseURL + "/v1/messages"`, so its base does not — and
appending to the first spelling asks for `/v1/v1/models`, a 404 that reads as
an endpoint publishing neither a dialect nor a price.

**A document that will not parse is an error, never an empty result.** An
endpoint that publishes no prices is working correctly and renders an em dash;
an endpoint answering with an HTML error page is not. Reporting both as "no
prices" makes a misconfigured URL look exactly like a cheap provider.

```go
type Rates struct{ Prompt, Completion, CacheRead, CacheWrite, CacheWrite1h float64 } // USD per TOKEN

func (r Rates) Cost(u Usage) float64
func Anomalous(u Usage) bool
```

**A model that published no pricing is ABSENT from `Prices`, never present with
zeros.** A host has to be able to tell a free model from an unpriced one,
because the first owes nothing and the second has to render an em dash. An
endpoint that publishes no prices at all is not an error — most do not.

`Cost` is where the money is, and three things the obvious formula gets wrong:

- `PromptTokens` **already contains** the cached tokens on every dialect, so
  the uncached term is `Prompt - CacheRead - CacheWrite`. Pricing the whole
  prompt and then adding the cache terms bills them twice, and cache reads are
  routinely 60-90% of a long session's prompt — about 5.7× too high on a
  cache-warm turn, wrong in the direction that makes caching look useless.
- Cache-write tokens are prompt tokens charged at the **write** rate, not at
  the input rate **plus** the write rate.
- Reasoning tokens are already inside `CompletionTokens`, so there is no
  reasoning term. Adding one roughly doubles a thinking-heavy turn. `Cost`
  takes a `Usage` and reads only the four billable counts, so adding that term
  means changing the signature.

A provider reporting more cached tokens than prompt tokens is inconsistent:
the uncached term clamps at zero and `Anomalous` reports the same condition, so
a host says so out loud instead of quietly billing under the invoice.

`CacheWrite1h` is carried and **not used by `Cost`**: `CacheWriteTokens` is one
integer with no tier in it, and the library places only five-minute
breakpoints, so `CacheWrite` is the right rate for every call it makes.

What this cannot settle, and why a host should keep an override: it reads the
MODELS endpoint and infers the CHAT endpoint from it. A gateway may serve those
independently, and proving the chat dialect means posting to it — spending
tokens, or deliberately sending a malformed request to read the error shape
back. `Dialects()` and `Label()` exist so that override's UI does not carry its
own copy of the vocabulary.

### Timings

llama.cpp-style upstreams (llama.cpp, ollama) attach a `timings` object to
streamed chunks. The OpenAI dialect decodes it wire-faithfully into

```go
type Timings struct {
	PromptN     int     `json:"prompt_n,omitempty"`
	PromptMS    float64 `json:"prompt_ms,omitempty"`
	PredictedN  int     `json:"predicted_n,omitempty"`
	PredictedMS float64 `json:"predicted_ms,omitempty"`
}
```

Each reported snapshot fires `StreamEvents.OnTimings`, and the **last** one
lands on `Completion.Timings` — a pointer that stays nil when the provider
never reported timings (tri-state, like the usage cache fields). The library
only surfaces what the provider said: it never synthesizes timings from
wall-clock time (tok/s fallbacks stay the caller's concern). The Anthropic
dialect has no equivalent and never fires it.

### Tools and approval

```go
type Tool interface {
	Decl() ToolDecl
	Execute(ctx context.Context, args json.RawMessage) (ToolResult, error)
}
type Tools []Tool // the flat set one run offers
type Approval struct {
	OK     bool
	Reason string // when !OK, recorded instead of DeniedMessage
}
type Approver interface {
	Ask(ctx context.Context, call ToolCall) (Approval, error)
}
```

**A tool has no say in whether it is asked about.** `Config.Approver` is
consulted for EVERY call. A tool declaring itself unremarkable used to skip the
approver entirely, which meant a host's deny rules could not fire on read-only
tools at all -- a permission engine that silently does not apply to most calls
is a lie about what it protects. It also could not express the decision being
made: approval is per-CALL (`bash git status` and `bash rm -rf` are one tool),
and the old `NeedsApproval() bool` took no arguments. `ToolDecl.Readonly` is
now the single declaration of what a tool does to state, and a nil `Approver`
reads it: a `Readonly` call runs, anything else is denied. That is the old
fail-closed default, expressed once instead of maintained by hand in a wrapper.

**A denial says why.** `Approval.Reason`, when a refusal carries one, is
recorded as the tool result in place of `DeniedMessage`. A model refused
because the write was outside the workspace should retry inside it; a model
refused because the program is banned should stop and ask. Told the same
sentence about a user either way, it cannot tell them apart, and in plan mode
that sentence is also false -- no user was asked. An empty (or whitespace-only)
`Reason` keeps `DeniedMessage`, which is still the right sentence for the case
it was written for: a user pressing deny.

A tool is an individual thing, and nothing groups them. Every tool a run
offers -- a built-in below, or one a host discovered on an MCP server -- is one
`Tool` in a flat `Tools` slice that `Config.Tools` takes; the loop resolves a
requested name against it and can no more tell the kinds apart than the model
can. There is no executor and no routing table. Concatenating two sources is
`append`, and the first tool to claim a name answers it.

The id of the call being answered is NOT an argument -- almost no tool wants
it. `Run` puts it on the context, and the one built-in that needs it (the
sub-agent tool, to stamp its telemetry) reads it back with `ToolCallID(ctx)`.

A `ToolResult` is `Content` (the text the MODEL is fed), `IsError`, and
`Parts []ToolContentPart` — structured content for the HOST: images, audio,
embedded files, or a block a tool and its host agree on. Parts never reach the
model, so a result can hand a front end a megabyte of image while costing the
context only what `Content` says. `Run` passes the whole result to
`Events.OnToolResult`, alongside the message it recorded for it; nothing else in
the loop reads them.

Restricting a toolset is filtering a slice, so `Tools` carries the four
helpers the loop and the sub-agent tool need:

- `Decls()` / `Names()` — the advertised declarations and names, in order.
  An unnamed tool is skipped rather than sent to a provider that rejects it.
- `Find(name)` — resolve a requested name; a miss is the loop's recoverable
  `unknown tool: <name>` result.
- `Readonly()` — the tools with `ToolDecl.Readonly`. The toolset to hand a
  sub-agent: a mutating tool is ABSENT from it, not merely refused by it.
- `Subset(names)` — the named tools, in the registry's order (the advertised
  list is part of the prompt-cache prefix, so it must not depend on the
  caller's argument order).

`NewTool(decl, run)` is the shorthand for a plain function tool. Gating it is
not part of building it: the host's `Approver` sees the call either way.

### Built-in tools (the plugin pattern)

The `Tool` seam **is** the plugin interface: a "plugin" is nothing more than a
value implementing it, appended to your toolset. The library ships the optional
tools below; callers who don't append them are unaffected.

```go
tools := append(agentic.Tools{},
	myTools...,
)
tools = append(tools,
	webfetch.NewWebFetchTool(webfetch.WebFetchConfig{Provider: provider, Model: model}),
)
tools = append(tools, subagent.NewSubagentTool(subagent.SubagentConfig{
	Provider: provider, Model: model,
	Tools: tools,                 // the FULL parent toolset — grants select from it
	Gate:  agentic.NewGate(1),    // serialize sub-agents (the source app's choice)
	OnActivity: func(a subagent.SubagentActivity) { /* live telemetry */ },
}))
```

- **`NewSubagentTool(SubagentConfig)`** — the `run_subagent` tool: the
  model offloads a focused task to a sub-agent running its own in-memory
  loop (this package's `Run`) and gets back only the distilled final report.
  By default the sub-agent sees only the read-only subset
  (`cfg.Tools.Readonly()`); the orchestrator can pin it to — and thereby
  explicitly grant, including non-read-only tools — an exact set via the
  `allowed_tools` argument (exact advertised names, with an unambiguous
  bare-name fallback for `server__tool` prefixes; `run_subagent` itself is
  never grantable, so no recursion). The advertised schema embeds the
  grantable names as an enum, flagging non-read-only ones
  `(modifies state)`. `share_context` optionally shares parent history —
  `none` (default) / `full` / `last_n` / `messages` / `summary` (one bounded
  briefing call) / `custom` — rendered as plain text
  (`RenderTranscript`/`SelectLastN`/`SelectByEndIndices`) and folded into
  one delimited task message. Misuse (a bad mode, an unknown grant) is a
  recoverable teaching error listing the valid options. A shared `*Gate`
  (`NewGate(n)`, a context-cancellable cap-n semaphore; nil = unlimited)
  bounds concurrency, and `OnActivity` streams live
  `SubagentActivity{CallID, Kind, Turn, Tool, Detail, Content, IsError, Completion}`
  steps (kinds `turn`/`tool_call`/`tool_result`/`text`/`thinking`/`turn_end`).
  `turn_end` carries the finished turn's whole `*Completion` — a sub-agent
  answers its parent in text, so this and the report's `Usages` are the only
  routes out for what it spent, and a host without them charges every
  sub-agent, briefing and web summary as free. It fires once per model call
  that produced a completion. `Detail`
  is the one-line preview — whitespace-flattened, capped at 160 runes —
  and `Content` is the same thing WHOLE: the full arguments, full tool
  output, full answer or full reasoning, so a host can show what the
  sub-agent actually read and said rather than a truncated hint. The
  sub-run's streaming never leaks into the parent's `StreamEvents` — the
  parent sees only the final report and the activity feed.
  **`Runs *SubagentRuns` makes the tool ASYNCHRONOUS**: the call returns
  `SubagentLaunchReceipt(description)` the moment the sub-agent starts, so a
  model can fan several out in one turn and keep working, and each report is
  delivered between turns instead of blocking the call. `Config.Subagents`
  (the same registry) is what makes `Run` keep that promise: a turn that
  would otherwise END while sub-agents are out waits for the next report and
  appends it as a user message (`FormatSubagentDelivery`). `SubagentReport` and
  the terminal `SubagentUpdate` carry `Usages []Usage` — one entry per model
  call the sub-run made, plus the `share_context=summary` briefing when there
  was one, in order and never summed — so a host that watches only lifecycle
  still gets the total. Every failure past
  the JSON parse — an unconfigured model, a misused argument — reaches the
  model as that report rather than as the launch's result. A nil `Runs` keeps
  the call synchronous: it blocks until the sub-agent answers. A final message
  that is really a leaked tool-call envelope (a backend that did not parse
  the model's tool-call template) is never passed off as findings: it is cut
  at the envelope and returned as an error result carrying
  `SubagentCutOffNote`, or `SubagentNoReportText` when nothing usable
  survives.
- **`NewWebFetchTool(WebFetchConfig)`** — the `web_fetch` tool: one
  unauthenticated, plain HTTP GET (http/https only, userinfo rejected, 5 MiB
  body cap, 45 s default client timeout), cleaned to readable text (built-in
  HTML cleanup, or an optional Apache Tika server via `TikaURL` with silent
  fallback) and rune-capped at 200 000 with an explicit truncation note. An
  optional `summary_prompt` argument runs one bounded, tool-less call to the
  configured `Provider`/`Model` to summarize the cleaned content
  (`OneShot`, 2 min timeout); `OnCompletion func(*Completion)` is handed that
  call's completion — including a partial one from a failed call, because those
  tokens were spent too — since the tool itself answers only with text.
  `BlockURL func(url string) string` is the
  injectable refusal seam: return a non-empty teaching message to refuse a
  fetch (the source application used it to redirect fetches of its
  workspace repository); the library ships the hook, not the policy. The
  tool is `Readonly`, so a sub-agent's default toolset includes it.
- **`NewTodoTools(TodoConfig)`** — four task-list tools, `todo_add`,
  `todo_edit`, `todo_cancel` and `todo_complete`, returned as one flat
  `Tools` slice and sharing a single in-memory store: the model's own task
  list, so a long job stays legible to the user while it runs. Each tool
  mutates ONE task by its stable per-task `id` (`Todo.ID`), which is minted
  once, never reused, and re-read in every reply — the model never resends or
  reconstructs the rest of the list from memory. `TodoConfig.Write(ctx,
  []Todo)` is where the host keeps and displays the list; it receives the
  whole post-mutation list, ids and all, after every call, and an empty list
  arrives as an empty (non-nil) one because clearing the list is a real
  instruction. A host that KEEPS that list between runs must hand it back as
  `TodoConfig.Initial`: the store lives in memory, so a toolset built without
  it starts empty, and the new run's first mutation persists a list holding
  only that one task — the earlier tasks are gone, and nothing in the exchange
  looks like a failure. Each successful mutation's result also carries the list itself
  as a `TodoListPartType` content part whose JSON equals what `Write` just
  stored, so a host can draw the plan mid-turn instead of parsing it back out
  of the text. The library owns the tools' semantics — the schemas'
  `pending`/`in_progress`/`done` enum, the caps (100 tasks, 200-rune titles),
  the exact teaching errors naming the offending `title`/`state`/`id`, and
  `RenderTodos` (exported, so a host renders the same text the model is
  answered with, each task line carrying its `#id`). A `Write` that returns
  an error is reported to the model as a failure: a list that was not stored
  is never reported as stored. NOT `Readonly` — they write state the host
  shows, and a sub-agent inheriting them would overwrite its parent's plan.

No built-in tool gates itself — each one only declares `Readonly` truthfully,
and `Config.Approver` decides every call. Which means a run with a nil
`Approver` executes the read-only built-ins (`repo_read`, the four read file
tools, `web_fetch`, `mcp_resource_diff`) and refuses the rest
(`repo_file_write`, `repo_pr_create`, `write_file`/`edit_file`/`delete_file`,
the four task-list tools, and `run_subagent`) with `DeniedMessage`. A host that
wants any of those runnable wires an `Approver`; the sub-agent tool's own nested
run already carries an approve-everything one, because a tool the sub-agent
holds was authorized when it was granted.

The two model-calling built-ins use the `Provider` you hand them exactly as given — the
library never wraps it. In the source application every one of these model
calls (the sub-agent's nested loop, the context-summary briefing, and the
web summary) went through its rejected-parameter recovery; to reproduce
that, pass a `NewParamStripper`-wrapped provider in
`SubagentConfig`/`WebFetchConfig` — the same wrapped value you give
`Config.Provider`.

### The loop

`Run(ctx, cfg, req)` drives the turn loop: call the model, execute the
requested tools, feed the results back, repeat. Key behaviors:

- **Nothing caps the run unless you ask for it.** The loop runs until the
  model stops asking for tools. A counted cap cannot tell a model looping
  uselessly from one deep in a hard task, so a default one fires at the worst
  possible moment: after the run has spent every call gathering context and
  just before the model writes any of it down. Bound a run with the two
  mechanisms that judge the right thing — `ErrStuck` below (evidence the model
  stopped progressing) and your own `ctx` (wall-clock and spend, without
  discarding work in flight). `Config.MaxTurns` is there for a host that must
  bound a turn anyway (an interactive UI answering one request); it counts
  model calls, and the last permitted call is made WITHOUT tools so the model
  answers instead of asking for a tool nothing will run.
- **A stuck model is caught, not waited out.** A turn whose tool calls are
  byte-identical to the previous turn's cannot learn anything new — the same
  calls return the same results, which produce the same turn again. The
  third identical turn in a row (`StuckNudgeAt`) gets one nudge appended
  after its tool results; the sixth (`StuckFailAt`) ends the run with
  `ErrStuck` (match it with `errors.Is`) rather than letting it spin
  — the failing batch is never executed and its tool calls are
  cleared, so the partial transcript stays replayable. Any change in what
  the model asks for clears the count; call IDs are excluded from the
  comparison, since providers mint a fresh one per call.
- **Tool failures never abort the loop.** An `Execute` error becomes a
  `tool execution failed: ...` error result; a refused call records the
  `Approval.Reason`, or exactly `DeniedMessage` ("The user denied permission to
  run this tool.") when the refusal gave none; a call to a name the run does
  not offer gets `unknown tool: ...`. In every case the loop continues so the
  model can react.
- **One pass over every call, in a fixed order.** `OnToolCall(call *ToolCall)`
  observes and may REWRITE the call (its `Arguments`, in practice), then
  `Approver.Ask` decides on the rewritten call, then the tool executes it. The
  order is load-bearing: an approver judging the original while a hook rewrote
  the command would be a hole with a process around it. The transcript keeps
  recording what the MODEL asked for — what was requested and what ran are two
  different facts, and a mutation that silently rewrote history would be worse
  than no mutation at all, so a host that needs the executed version records it
  alongside. The tool result answers the model's own call id either way; a
  rewritten id is ignored, since a mismatch there is an orphan no upstream will
  replay.
- **`OnToolResult(call, result, recorded)` reports both halves of the event.**
  `recorded` is the `RoleTool` message the loop actually appended — after output
  dedup has possibly replaced the content with an `[unchanged]` marker, carrying
  the `ToolCallID` and `ToolIsError` as recorded, and equal to the corresponding
  entry in `Result.Messages`. A host that persists the transcript must store
  what the model saw, and the loop is the only thing that knows it: deriving it
  by diffing `Result.Messages` against the events afterwards is a second copy of
  dedup's rules, in another repository, wrong the day they change.
- **Callbacks can abort.** `Events.OnToolCall`/`OnToolResult` (like the
  stream callbacks) return an error; a non-nil return ends the run the way a
  cancellation does — the pending batch is cleared (the assistant message
  keeps its content and reasoning but loses its tool calls, and the batch's
  already-appended results are dropped) so the transcript stays replayable
  with no orphan tool calls, and the partial `Result` is returned together
  with your error.
- **Per-turn hooks.** `Events.OnTurnBegin(turn, req)` fires before each
  numbered model call (turns are numbered from 1; the stall wrap-up call
  fires one past the turn that stalled) with the 1-based turn number and a
  pointer to the per-call `Request`: mutate it (append a wind-down message,
  tweak `System`, add `Extra`) and the change applies to that one call only,
  never the persistent transcript. `Events.OnTurnEnd(turn, comp, err)` fires
  after each call with the `Completion` (nil when the call produced none) and
  the call's error. A non-nil return from either aborts the run like any
  other callback error: `OnTurnBegin` aborts before the call, `OnTurnEnd`
  after it with the completed data kept. The internal subagent telemetry
  hook is unaffected.
- **Approval**: every call pauses for `Approver.Ask` — read-only ones
  included, so a host's deny rules cover the whole toolset. A nil `Approver`
  fails closed on anything that is not `Readonly`. A refusal records
  `Approval.Reason`, or `DeniedMessage` when it carried none. If `Ask` returns
  an error (the decision never arrived), the run ends with the pending batch
  cleared exactly as above.
- **Stall fallback**: if the loop ends with no written answer (a
  thinking-only turn, or a run its ctx cut short) and tools were in play,
  one extra tool-less wrap-up turn forces the model to synthesize its answer
  from what it gathered; failing that, the final content falls back to the
  accumulated reasoning, then to a clear placeholder.
- **Result.Usages** holds one entry per model call, in order — deliberately
  **not summed**: successive prompts overlap (each turn re-sends the growing
  transcript), so summing would double-count the shared prefix.

#### Delivering a message into a running loop

A run is not a closed box. Two queues carry a message from anywhere in your
program into the transcript the model is working on: `Config.SystemMessages`
for automated notices (a CI status change, a stop-hook nudge, a sub-agent
report) and `Config.UserMessages` for what the user typed while the model was
busy. Both are `*MessageQueue`, safe for concurrent producers.

```go
sys := &agentic.MessageQueue{}
go func() {
    if !sys.Queue(agentic.Message{Role: agentic.RoleUser, Kind: "ci_status_change",
        Content: "[CI status changed] checks went red on abc1234 ..."}) {
        startANewRun(notice) // the run ended; nothing else will deliver it
    }
}()
res, err := agentic.Run(ctx, agentic.Config{Provider: p, SystemMessages: sys, ...}, req)
```

- **A queued message always reaches the model.** Both queues are drained at
  the top of every turn, system first, and a message queued when the model
  would otherwise have finished **starts another turn** — every time, not once
  per run. There is nothing to poll and no window in which a notice is quietly
  dropped: a turn boundary is either coming or is created.
- **`Queue` reports whether the queue took the message.** Run closes both
  queues as it returns, so `false` means the run has ended and no other run
  will show that message to anyone. That is the signal to start a new run
  with it — the alternative, a hopeful `true`, is how a CI notice ends up
  sitting in a queue nothing reads. A nil queue is closed by the same logic.
- **`Result.Undelivered` is what the run never delivered**, system first.
  It is empty for a run that ended normally, and non-empty when the run ended
  first for another reason — a cancelled `ctx`, a model-call error, or a host
  `MaxTurns` cap that left no turn to deliver into. Re-deliver those into the
  next run rather than dropping them.
- **The host persists them**, through `Events.OnSystemMessage`: the loop hands
  over each message it is about to append and takes back the durable id the
  host minted for it, so the stored thread and the transcript agree.
- The `Kind` you set rides along untouched (`SubagentReportKind` is the loop's
  own), which is what lets a host render a CI notice differently from a nudge.

### Retry and error classification

`RetryPolicy` (default `DefaultRetry`: **10 attempts**, 500ms base, delay =
base × 2^(n−1), no jitter, no cap) retries only what `IsTransient`
allows: HTTP 408,
429, any 5xx, and network/transport errors. Context cancellation, other
4xx, and **errors your own callbacks returned** are permanent. A 400 whose
body says the prompt exceeded the context window is flagged — check with
`IsContextOverflow(err)` — and never retried. Anthropic in-stream `error`
events (an HTTP 200 whose stream carries the error — how `overloaded_error`
typically arrives) are mapped onto the same `APIError` using Anthropic's
documented error-type → status table, so an in-stream overload (529) or rate
limit (429) retries exactly like its non-2xx counterpart.

**Retry is on by default and belongs to the Provider**, not to `Run`: every
provider a dialect constructor builds already retries, so there is nothing
to opt into and nothing to remember. Tune or disable it per provider:

```go
// The default — retries transient failures.
p, _ := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{BaseURL: url},
})

// Explicitly off.
p, _ = agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{
        BaseURL: url,
        Retry:   &agentic.RetryPolicy{MaxAttempts: 1},
    },
})
```

The provider is the right home because it is the layer that knows whether a
call streamed anything — the condition that decides whether re-sending is
safe. A call is re-attempted **only when the failed attempt streamed
nothing**; once data arrived the error surfaces with the partial completion
attached. A provider marks data as seen **before** each emit, so even a
callback that fails on the very first delta yields a partial completion —
the call is never re-sent into a dead sink. That single signal, the non-nil
completion, is the whole mechanism: nothing watches the callbacks to
second-guess it.

**Retrying is not silent.** Ten attempts of uncapped backoff is minutes of
wall-clock, so every retry fires `StreamEvents.OnRetry` *before* its
backoff, carrying which attempt failed, of how many, the delay about to be
waited, and the error — show it, don't leave the user staring at nothing:

```go
ev := &agentic.StreamEvents{
    OnRetry: func(a agentic.RetryAttempt) error {
        log.Printf("attempt %d/%d failed (%v), retrying in %s", a.Attempt, a.Of, a.Err, a.Delay)
        return nil // returning an error stops the retrying
    },
}
```

`OnRetry` fires from the retry layer, not from a dialect provider, and does
**not** count as "streamed something" — a notification about a failed
attempt cannot make the next one unsafe.

`Run` therefore has no retry knob, and a retried call is one turn: `Run`
only ever sees the outcome. A custom `Provider` implementation is
responsible for its own retry.

### Rate limiting

`ProviderConfig.RateLimiter`, when set, throttles the provider's outgoing
request starts to the limiter's fixed rate — a provider's per-minute request
cap, spaced evenly:

```go
limiter := agentic.NewRateLimiter(30) // at most 30 requests/minute
p, _ := agentic.NewOpenAIProvider(agentic.OpenAIConfig{
    ProviderConfig: agentic.ProviderConfig{
        BaseURL:     url,
        RateLimiter: limiter,
    },
})
```

Like retry, the throttle belongs to the Provider: staying under an upstream's
rate limit is part of making the call happen. The limiter is wired in as an
`http.RoundTripper` on the provider's client, so **retries ride the same
gate** — a re-sent attempt is a request too. Only request **starts** are
counted (a slow call never pushes the average over the limit), and because
consecutive starts are at least one interval apart, no 60-second window can
contain more than the configured number of started requests.

**Share one limiter to throttle many providers together.** `RateLimiter` is
safe for concurrent use; a single instance passed to every provider of a run
(e.g. the concurrent jobs of a benchmark) keeps them under one per-endpoint
cap — per-provider limiters would multiply it by the number of callers.

### Rejected-parameter recovery

```go
provider := agentic.NewParamStripper(inner)
```

Some upstreams 400 on a parameter others accept. The stripper parses the
offending parameter name out of the error body (four provider phrasings,
matched case-insensitively and by normalized name, so a camelCase report
matches your snake_case key), removes that one key from `Extra`, and retries
once. The strip is remembered for later calls through the same stripper. It
never fires on context cancellation or after data streamed.

The OpenAI provider handles one narrower case itself, below the stripper:
some upstreams (Z.AI, for one) 400 on the auto-added `stream_options` default
with an error that names no parameter at all, so the stripper's regexes have
nothing to match. When a pre-stream 400 names no recoverable parameter and
the request carried that default (never one you set via `Extra`), `Complete`
retries once with it left off — safe because you never asked for it, and its
absence only costs the usage figures on that one call. A 400 that DOES name a
parameter is left for `NewParamStripper` to handle, and a context-overflow
400 is never retried by either mechanism.

### Prompt caching

- **Anthropic**: always on unless `DisableCaching` — exactly two ephemeral
  `cache_control` breakpoints per request: a static one on the system block
  (which also covers the tools array via the tools → system → messages
  hierarchy) and a moving one on the last content block of the last message.
  Both are applied to per-request wire structures only; **your `Messages` are
  never mutated**, so the stored transcript stays marker-free.
- **OpenAI**: hosted caching is automatic server-side; the provider defaults
  `stream_options: {"include_usage": true}` (unless your `Extra` already sets
  `stream_options`) so usage — including cached-token counts — arrives on the
  stream. Set `CacheKey` to send `prompt_cache_key` (a routing hint), and set
  `SelfHosted` to add `cache_prompt: true` for llama.cpp-style servers
  (never send it to hosted OpenAI/Azure -- they 400 on unknown fields). Set
  `PromptCache` to add the two Anthropic-style ephemeral `cache_control`
  breakpoints in openai shape (a static one on the leading system message
  and a moving one on the tail content block) for Anthropic-fronting
  gateways that pass them through; keep it off for strict OpenAI-compatible
  servers, which 400 on the unknown marker.

### Usage accounting

`Usage.PromptTokens` is always the **full prompt** including cached tokens
(the Anthropic provider adds `input_tokens + cache_read + cache_creation`,
because Anthropic's `input_tokens` excludes cached tokens; OpenAI's
`prompt_tokens` already includes them). `CacheReadTokens`/`CacheWriteTokens`
are tri-state pointers: nil means the provider reported no cache info; a
value — including an explicit 0 — is a real report. `CachedTokens()` gives
the normalized cache-hit count. Streamed usage snapshots are merged
newest-wins (xAI attaches a cumulative snapshot to every chunk; they are
never summed), and the finalized total is floored at prompt+completion while
preserving a genuine surplus (reasoning tokens).

`Completion.UsageReported` is true iff at least one usage snapshot was
merged during the call: `Usage` is a value type, so this is how a caller
reading only the returned `Completion` distinguishes an upstream that
reported all-zero usage from one that reported none at all (common on local
servers) — check it before persisting or displaying `Usage`.

`Completion.RawUsage` carries the provider's usage object verbatim (the raw
wire JSON on the openai dialect, the merged wire-shaped object on Anthropic)
for logging and for extracting provider extras the normalized `Usage` drops:
`Completion.ReasoningTokens` (openai
`usage.completion_tokens_details.reasoning_tokens`) and
`Completion.CostUsd` (`usage.cost`, falling back to `usage.estimated_cost`),
each a tri-state pointer present only when the upstream reported it.

`Completion.Streamed` records whether the response actually arrived as an SSE
stream. A 200 that is NOT `text/event-stream` is read as a plain JSON body
and reassembled into a Completion with `Streamed` false -- a server that
ignores `stream: true` is accepted transparently, and the flag preserves the
truth of how the call was transported.

### Compaction and one-off calls

```go
cr, err := agentic.Compact(ctx, provider, agentic.Request{
	Model:    "your-model-id",
	System:   "You are compacting a conversation ...", // your summarizer brief
	Messages: history,
})
// cr.Messages is the replacement round:
//   user(agentic.CompactRequestText), assistant(cr.Summary)
```

`Compact` sends the whole history with the summarize instruction as the
trailing user message and no tools; you replace your history with the
returned two-message round. `CompactResult.Completion` is that call's whole
completion — compaction reads the entire history, so it is one of the most
expensive calls a session makes and charging it needs more than a `Usage`
value could say. `OneShot(ctx, p, req, timeout) (*Completion, error)` is the
bounded tool-less single call (titles, micro-summaries); the answer is
`strings.TrimSpace(comp.Message.Content)`. Compose it with
`context.WithoutCancel(parent)` for fire-and-forget work that must survive
the parent request ending.

**Every entry point that makes a model call surfaces its `*Completion`, not a
projection of it.** A `Usage` return cannot distinguish an upstream that
reported all-zero usage from one that reported none — that is exactly what
`Completion.UsageReported` is for — and it silently drops `CostUsd`,
`ReasoningTokens`, `RawUsage`, `Timings` and `Streamed`. Under-counting money
is not a rounding error, and it cannot be repaired afterwards, because the
numbers were never emitted. That principle is why `OneShot` returns a
completion, why `CompactResult` carries one, why `SubagentActivity` has a
`turn_end` kind, why `WebFetchConfig` has `OnCompletion`, and why
`SubagentReport`/`SubagentUpdate` carry `Usages`.

## Concurrency

Providers built by the dialect constructors are read-only during `Complete`
and safe to share across goroutines. The `NewParamStripper` wrapper is also safe for
concurrent use (its strip memory is mutex-guarded). One `Run` drives one
conversation; run concurrent conversations by calling `Run` concurrently,
sharing the provider. A shared `Gate` bounds concurrent sub-agents.

## Testing

The test suite runs entirely against `httptest` fake servers and an
in-process scripted provider — no network, no credentials.

### Resource watching (optional)

A resource is not advertised in the model's context the way a tool is, so a
model never learns one exists. `NewResourceWatcher(ResourceWatchConfig)` closes
that gap: at each turn boundary it re-reads every watched resource, hashes it,
records what moved, and hands back a `ResourcePoll` a host renders with
`FormatResourceNotice` — names and change ids, never content, because dumping a
changed resource into the thread would cost its full size on every later turn.

Nothing in it knows what MCP is. A host supplies two seams:

- **`ResourceSource`** — `ID`/`Name`/`List`/`Read`. One thing that publishes
  resources (an MCP server, a directory, anything listable).
- **`ResourceSnapshots`** — where the watcher keeps what it saw, and where it
  records each change. `RecordChange` returns the id, so the host owns id
  generation (it is the side that resolves one later).

`NewResourceDiffTool(ResourceChanges)` is the matching tool: it resolves a
change id to the before/after captured AT THAT MOMENT, from the change record's
own copy — a resource that has moved three times since still answers the first
notice with the first change. Failures are recoverable results that list the
real ids; a source that could not be listed becomes a WARNING, never a removal
(absent-because-unreachable is not absent). Binary content is watched by hash
and size and reported as such, never rendered as a diff.

`UnifiedDiff`/`CountLineChanges`/`HumanSize` are exported for hosts rendering
their own changes with the same words.

### Filesystem tools (optional, package `vfs`)

`vfs.NewFileTools(vfs.FileToolsConfig)` returns a `*vfs.FileTools` handle
providing the seven-tool file vocabulary — `list_dir`, `read_file`,
`find_files`, `grep`, `write_file`, `edit_file`, `delete_file` — over
whatever a host mounts. The library owns what a file tool IS: the names,
the model-facing descriptions, the argument schemas, the caps and every
word of the rendering. A host owns what is behind a mount, and nothing
about its storage reaches here. Mount nothing and you get non-nil tools
that return the unavailable message, rather than nil tools.

A host mounts `IFolderProvider`s (folder hierarchies) and `IFileProvider`s
(single files) under virtual path prefixes. More specific prefixes always
shadow less specific ones, regardless of registration order:

```go
import "github.com/wow-look-at-my/agentic-loop/vfs"

ft := vfs.NewFileTools(vfs.FileToolsConfig{
	Providers: map[string]any{
		"/repos":             repoFolder,       // a vfs.IFolderProvider
		"/repos/org/repo":    repoSubFolder,    // shadows /repos for this repo
		"/workspace":         workspaceFolder,  // a vfs.IWritableFolderProvider
	},
	MountsBlurb: "/repos is read-only; /workspace is editable.",
	Notes: map[string]string{
		vfs.WriteFileToolName: "Writes stage locally until the user pushes.",
	},
})

// Use ft.Tools() in agentic.Config to pass the seven tools to the loop.
// Add or remove providers at runtime:
err := ft.Add("/tmp", tmpFolder)       // returns error on duplicate path
ft.AddFile("/docs/readme", singleFile) // register a single virtual file
ft.Remove("/repos")
```

- **`IProvider`** — the common base: `Path() string`. Every provider knows the
  virtual path it was registered at. Embed `*BaseProvider` to get this for free;
  the registry injects the path automatically.
- **`IFolderProvider`** — embeds `IProvider`; adds `Display`/`List`/`Read`/
  `Find`/`Grep`. Every method receives the WHOLE virtual path as the model
  wrote it, because only the folder knows its own grammar.
- **`IFileProvider`** — embeds `IProvider`; adds `Read`/`Display`. Serves
  exactly one virtual file at its registered path. Use it when you only need
  to expose a single document without a full folder hierarchy.
- **`IWritableFolderProvider`** adds `Writable`/`Create`/`Replace`/`Remove`. A
  folder that is not one gets the three write tools' refusals for free, and a
  `ReadOnlyExplainer` lets it say *why* a particular path is read-only.
- **`PathGuard`** blocks a path before any provider sees it, with the reason
  the model is shown.
- **`DuplicateMountError`** — returned by `Add`/`AddFile` when a provider is
  already registered at that path (case-insensitive). Never a silent overwrite.

Two properties are the module's whole point, and a host cannot opt out of
either. **A cap that bites is announced**: a truncated listing, a `find_files`
that stopped at its limit, a `grep` that hit `MaxHits` all say so, because a
partial result that reads as complete is worse than no result. And **an empty
`grep` is a real negative**: every line in scope was read, so no matches means
the text is not there — which is why a scope the mount does not hold is an
error rather than an empty result, and why `GrepResult.Note` exists for a
folder that could only cover part of what the path named.

### The GitHub module (optional)

`NewGitHub(GitHubConfig)` is a credential-rotating GitHub REST client, and
`NewRepoTools(RepoToolsConfig{GitHub: gh})` are the three tools over it:
`repo_read` (commits, one commit's diff, pull requests, issues, CI status,
one check run, one Actions job's log), plus the two non-`Readonly` writes
`repo_file_write` and `repo_pr_create` (so a run without an `Approver` refuses
them).
A nil client yields no tools — a run with no GitHub access is never offered one
that could only fail.

```go
gh := repo.NewGitHub(repo.GitHubConfig{
	Tokens:      readTokens,                            // rotated, then anonymous
	WriteTokens: repo.ModelWriteTokens(readTokens),  // only what the user flagged
	Cache:       myRepoKeyCache,                        // which token works, per repo
})
tools := repo.NewRepoTools(repo.RepoToolsConfig{GitHub: gh})
```

Two properties run through all of it:

- **A read tries the cached winner, then every token, then anonymously** (so a
  public repository works with no credential at all), and **when every attempt
  fails it reports the MOST INFORMATIVE one** — never the anonymous attempt's
  401, whose only content is that no credential was sent. That is how a spent
  rate limit stopped reading as a permanent auth problem.
- **`what=status` answers "why is CI red" from whichever API the credential can
  read.** The Checks API (`/commits/{sha}/check-runs`) needs the `checks`
  permission, which a GitHub App installation can hold and a fine-grained
  personal access token cannot — its repository-permission list has no
  "Checks" entry — so a token-backed host is refused there routinely. When
  that read fails, the report falls back to the Actions API
  (`/actions/runs?head_sha=`, then each run's jobs) and renders the workflow
  runs, their jobs, and the failed steps inside each failed job; `actions` IS
  a permission a token can hold. Which endpoint answered is plumbing and the
  report never mentions it: the reader wanted the verdict, not a permissions
  lecture they may be unable to act on. The fallback runs only after a
  failure, so a readable Checks API costs no extra requests, and only when
  NEITHER answers — the one case where there is genuinely no verdict — are
  both failures reported.
- **`what=job_log` is where that chain ends.** Naming the step that failed only
  renames the question; the compiler error and the failing assertion are in the
  job's log, so every failed job in the report carries the `job_log` call that
  fetches it. `/actions/jobs/{id}/logs` is served under `actions` too, so a
  token-backed host can reach it. It answers 302 with a signed URL on storage,
  and that hop is made WITHOUT the token — the signature is the credential
  there, and forwarding a PAT to a third party is not a tidiness question. The
  whole log is returned when it fits; past that the tail comes back (where a
  failure is) and `offset`/`limit` address any other window, so nothing in the
  log is out of reach. GitHub also 404s this endpoint for two reasons that
  have nothing to do with a token's access, and both look identical to "no
  token can see this": while the job has not finished — undocumented, but
  reported: the log is not archived to storage until it is — and when GitHub
  marks the job "skipped", which never ran a step and so never produced one.
  A 404 here re-reads the job's own status first, and reports whichever of
  those it PROVES. When the job really is completed and not skipped, that
  read just confirmed the job exists and these tokens can see it, so the
  generic "none of your tokens can see it" wording would directly contradict
  what was just read — the message instead points at the log itself as what
  is missing (most likely expired log retention), not the job.
- **A write never falls through to anonymous** and uses ONLY `WriteTokens`.
  Filtering that list by initiator is the host's job: a model-facing toolset
  gets `ModelWriteTokens(...)`, a user-initiated action gets everything.
  `RepoToolsConfig.Blocked` vetoes a write per repository (a working copy whose
  staged state a direct commit would bypass); reads are never asked, because a
  working copy holds no version of history or CI.

The client is a separate value from the tools because a host needs it directly.
A `/repos` filesystem folder and a "test this token" button take the same
`*GitHub`, so all three share one credential order and one winning-token cache
— `Fetch`, `FetchURL`, `OwnerRepos`, `DefaultBranch`, `CIStatusReport`,
`TestToken`, and, for a host running its own git push, `WriteCredentials`,
`Do`, `ClassifyWriteStatus` and `MoreInformativeAuthFailure`.
