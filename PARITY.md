# PARITY.md — the specification for the TypeScript port

This document is the contract for `ts/`: a TypeScript implementation with
**full behavioral parity** with `go/`. It captures every rule the Go library
implements — including the exact strings, regexes, and edge-case orderings a
reimplementation gets wrong first. Where the Go code and this document ever
disagree, the Go code (and its tests) wins; update this file.

The semantics originate from three applications (an OpenAI-compatible chat
server in Go, a browser agent with a native Anthropic dialect, and a
benchmark harness with retry/caching plumbing); the Go library is the
distilled, dependency-free extraction.

---

## 1. Data model

| Type | Fields | Notes |
|---|---|---|
| `Role` | `"system" \| "user" \| "assistant" \| "tool"` | |
| `ThinkingBlock` | `text`, `signature`, `redacted` | `redacted` holds an opaque `redacted_thinking` payload; when set, the other two are empty. |
| `ToolCall` | `id`, `name`, `arguments` | `arguments` is the RAW JSON object **text** (OpenAI: concatenated streamed fragments; Anthropic: accumulated `input_json_delta.partial_json`). |
| `Message` | `role`, `content`, `thinking[]`, `toolCalls[]`, `toolCallID`, `toolIsError` | thinking/toolCalls assistant-only; toolCallID/toolIsError tool-only. |
| `Tool` | `name`, `description`, `inputSchema`, `readonly` | nil/absent schema marshals as `{"type":"object"}`. `readonly` is never sent to the upstream. |
| `ToolResult` | `content`, `isError` | Text-only by design (see cuts). |
| `Usage` | `promptTokens`, `completionTokens`, `totalTokens`, `cacheReadTokens?`, `cacheWriteTokens?` | The two cache fields are **tri-state**: absent/undefined = provider reported no cache info; a number (including 0) = a real report. Never zero-fill, never estimate. |
| `PromptProgress` | `processed`, `total`, `cache`, `timeMS` | Wire-faithful to the upstream `prompt_progress` object (`{total, cache, processed, time_ms}`). |
| `Request` | `model`, `system`, `messages`, `tools`, `maxTokens`, `extra`, `cacheKey` | See §3/§4 for per-dialect handling. |
| `Completion` | `message`, `usage`, `stopReason` | |
| `Result` | `messages`, `final`, `usages[]`, `turns` | `usages` = one entry per model call IN ORDER, never summed (successive prompts overlap; summing double-counts the shared prefix). |

Seams (interfaces): `Provider.complete(req, events) -> Completion`
(streaming under the hood), `ToolExecutor {tools(), execute(call),
needsApproval(name)}`, `Approver {ask(call) -> boolean}` (throw = decision
never arrived).

Normalized stop reasons: `end_turn`, `tool_use`, `max_tokens`. OpenAI
mapping: `stop → end_turn`, `tool_calls → tool_use`, `length → max_tokens`,
anything else raw. Post-stream inference (both dialects): missing stop
reason becomes `tool_use` if calls were assembled, else `end_turn`.

## 2. Executor combinators

- **Composite**: skip nil executors and empty tool names; **first
  registration of a name wins**; advertised order = iteration order across
  executors; zero tools ⇒ the combinator itself is nil/absent ("no tools").
  Unknown call ⇒ recoverable error result `unknown tool: <name>` (never a
  thrown error). `needsApproval` routes to the owning executor; unknown ⇒
  false.
- **ReadonlyView**: only `readonly` tools; nil inner or zero survivors ⇒
  nil. Refusal text: `tool not available to subagent (read-only tools
  only): <name>`. `needsApproval` = allowed && inner's flag.
- **SubsetView(names)**: only named tools; nil inner / empty names / no
  matches ⇒ nil. Refusal text: `tool not in the sub-agent's allowed set:
  <name>`.

## 3. OpenAI dialect

**Endpoint**: POST `BaseURL + "/chat/completions"` (BaseURL includes `/v1`;
default `https://api.openai.com/v1`; trim trailing slashes before joining).
Headers: `Content-Type: application/json`, `Accept: text/event-stream`,
`Authorization: Bearer <key>` when a key is set, `User-Agent` when set,
then caller headers (which may override).

**Body build order (precedence is load-bearing)**:
1. Merge `extra` first, skipping the reserved keys `messages`, `model`,
   `stream`, `tools` (an "evil" extra cannot break routing).
2. Set `model`, `messages`, `stream: true` (always forced true), and
   `tools` only when non-empty. **No `tool_choice` is ever sent.**
3. `max_tokens` only when `maxTokens > 0` (overriding an extra value);
   0 leaves the field to extra or the provider default.
4. `stream_options: {"include_usage": true}` **only when extra did not
   supply a `stream_options` key** (a caller's `{"include_usage": false}`
   must survive verbatim).
5. `prompt_cache_key` when `cacheKey` non-empty.
6. `cache_prompt: true` when `SelfHosted` — NEVER otherwise (hosted
   OpenAI/Azure 400 on unknown fields). SelfHosted is explicit
   configuration in this library (the TS reference derived it from the URL;
   don't).

**Message wire mapping**:
- `system` (when non-empty) is prepended as `{role:"system", content}`.
- **The content presence rule** (the single most-regressed detail): the
  `content` field is ALWAYS emitted — even when empty — EXCEPT for an
  assistant message with empty content that carries `tool_calls` (there the
  spec makes content optional and the model produced none, so it is
  omitted). An empty tool result must serialize as
  `{"role":"tool","content":"","tool_call_id":"..."}` exactly; dropping the
  empty `content` makes upstreams reject the request with
  `invalid message content type: <nil>` / 400, failing the whole turn.
- Assistant `toolCalls` replay as
  `{id, type:"function", function:{name, arguments}}` with `arguments` the
  raw string.
- `thinking` is NOT replayed on this dialect (no wire field);
  `toolIsError` has no wire equivalent.
- Tools: `{type:"function", function:{name, description?, parameters}}`
  with `parameters` = schema or `{"type":"object"}`.

**SSE decode**:
- Scanner processes only `data:` lines (comments/`event:`/blank skipped);
  trim after the prefix; empty payloads skipped; `[DONE]` ends the stream;
  buffer sizes **64 KiB initial / 8 MiB max line** (large tool-call
  argument deltas WILL exceed small defaults).
- Unparseable payload JSON is silently ignored (keep-alive tolerance).
- `choices[].delta.content` → OnText + accumulate.
- Reasoning arrives under TWO field names: `reasoning_content`
  (OpenAI/DeepSeek) and `reasoning` (Ollama); `reasoning_content` wins when
  both are present on one delta. Accumulated reasoning becomes ONE
  `ThinkingBlock{text}` on the final message.
- `choices[].delta.tool_calls` accumulate **by `index`**: first delta for
  an index creates the slot (type defaults `"function"`) and records
  first-appearance order; `id`/`type`/`function.name` overwrite when
  non-empty; `function.arguments` CONCATENATES. Output in first-appearance
  order.
- `choices[].finish_reason` (last non-empty wins).
- `prompt_progress` rides a choices-less chunk → OnProgress, and nothing
  else is read from that chunk.
- `usage` may arrive once on a final chunk (OpenAI) or as a cumulative
  snapshot on EVERY chunk (xAI). See §5 for the merge.

## 4. Anthropic dialect

**Endpoint**: POST `BaseURL + "/v1/messages"` (default
`https://api.anthropic.com`). Headers: `Content-Type`, `Accept`,
`anthropic-version` (default `2023-06-01`), `x-api-key` when set,
`User-Agent` when set, then caller headers. Do NOT send the browser-only
`anthropic-dangerous-direct-browser-access` header from a server-side
library.

**Validation**: `maxTokens` must be positive — the Messages API requires
`max_tokens` on every request; fail fast before any I/O, and classify the
failure as permanent (never retried).

**Body build order**: merge `extra` first, skipping reserved keys `model`,
`max_tokens`, `stream`, `system`, `messages`, `tools`; then set the core
fields. The library does NO model gating of `thinking`/`temperature` — what
to send is the caller's job via `extra`.

**System**: emitted as a content-block ARRAY
`[{type:"text", text: system}]` (cache_control lives on blocks, not string
bodies); omitted entirely when empty (and then there is no static cache
breakpoint — the moving one still covers the prefix).

**Message mapping** (build fresh wire structures per request — NEVER mutate
the caller's messages):
- Role: assistant stays assistant; everything else rides as `user`
  (`system` in the transcript is a degenerate case — `Request.system` is
  the system channel on this dialect).
- User messages: plain string `content`.
- Assistant messages: block array in this exact order —
  1. thinking blocks FIRST (required, or tool-use continuations 400):
     `{type:"thinking", thinking, signature}` with signatures replayed
     unchanged, or `{type:"redacted_thinking", data}` for redacted blocks
     (NOTE: redacted_thinking replay is an addition beyond the TS reference
     app, which never handled it; the Go library replays it verbatim per
     the API docs — port this),
  2. `{type:"text", text}` when content non-empty,
  3. `{type:"tool_use", id, name, input}` per call, where `input` is the
     **PARSED JSON object** (invalid, empty, or non-object arguments → `{}`).
  A degenerate assistant message with no blocks at all degrades to string
  content.
- Tool messages: **consecutive** RoleTool messages fold into ONE user
  message whose content is a list of
  `{type:"tool_result", tool_use_id, is_error, content}` blocks (string
  content; `is_error` always present, false by default).
- Tools: `{name, description, input_schema}` (NOT the function wrapper;
  `input_schema`, not `parameters`); nil schema → `{"type":"object"}`. No
  tool_choice.

**Prompt-cache breakpoints** (unless disabled): EXACTLY two ephemeral
markers per request (API max 4):
1. STATIC on the (last) system block — the cache hierarchy is tools →
   system → messages, so this one marker covers the tools array too.
2. MOVING on the **last content block of the last message**: a string
   content becomes a one-block text array carrying the marker; a non-empty
   block array gets the marker on a copy of its last block; empty arrays
   and unrecognized shapes pass through UNMARKED (caching is an
   optimization, never a correctness requirement).
Markers exist only in the per-request wire copy; the persistent transcript
must remain marker-free (pin with a test that calls twice and deep-equals
the caller's messages before/after). `DisableCaching` removes both markers.

**SSE decode** (the payload's `type` field discriminates; the SSE `event:`
name is redundant):
- `message_start`: `message.usage` → `input_tokens`, `output_tokens`,
  tri-state `cache_read_input_tokens` / `cache_creation_input_tokens`.
- `content_block_start`: seed block state by `index` — `id`/`name` for
  tool_use, `thinking`/`signature` for thinking, `data` for
  redacted_thinking.
- `content_block_delta` by `delta.type`: `text_delta.text` → OnText +
  accumulate; `thinking_delta.thinking` → OnReasoning + accumulate on the
  block; `signature_delta.signature` accumulates; `input_json_delta.
  partial_json` accumulates the raw JSON string.
- `content_block_stop`: finalize the block — tool_use →
  `ToolCall{id, name, arguments: accumulatedJSON}` (arguments stays the raw
  string); thinking → `ThinkingBlock{text, signature}`; redacted_thinking →
  `ThinkingBlock{redacted: data}`. Blocks that never see stop (a cut-off
  turn) are dropped from the result.
- `message_delta`: `delta.stop_reason` (values `end_turn`, `tool_use`,
  `max_tokens`, `stop_sequence` pass through); `usage.output_tokens` is a
  cumulative snapshot — OVERWRITE, never sum.
- `message_stop`: done. `ping`: ignored. Unparseable payloads: ignored.
- `error`: `{error:{message}}` → abort the stream with that message.

**Usage normalization**: Anthropic `input_tokens` EXCLUDES cached tokens —
`promptTokens = input + cache_read + cache_creation` (so both dialects
report the FULL prompt); `completionTokens = output_tokens`;
`totalTokens = prompt + completion`. Tri-state cache fields pass through as
reported.

## 5. Usage merging, flooring, cache normalization

- **Merge rule** (per model call): streamed usage objects are monotonic
  snapshots. `merge(prev, next)`: next absent → prev; prev absent OR
  `evidence(next) >= evidence(prev)` → next; else prev. `evidence(u) =
  max(totalTokens, promptTokens + completionTokens)`. Consequences to pin:
  never summed; a regressing final snapshot (zeros) is discarded; **equal
  evidence lets the LATER snapshot win** (it may carry richer cache
  detail); evidence uses max(total, parts) so a total that omits reasoning
  tokens still compares correctly.
- **Finalize-time floor**: `totalTokens = max(totalTokens, prompt +
  completion)`; a genuine surplus (xAI: total = prompt + completion +
  reasoning) is PRESERVED. Applied only when the call finalizes.
- **OpenAI cache decode**: the cache-read count is the LARGEST signal
  present among `prompt_tokens_details.cached_tokens` (OpenAI/vLLM/
  OpenRouter), `prompt_cache_hit_tokens` (DeepSeek), and
  `cache_read_input_tokens` (compat layers) — tri-state (absent when none
  of the fields appeared; an explicit 0 is a report). When cache info was
  reported, `cacheWriteTokens` is an explicit 0 (OpenAI has no separate
  write class); when none was, both stay absent. (The TS reference set
  write=0 unconditionally on any usage object; this library keys it to a
  cache report to honor the tri-state contract — match the library.)
- **CachedTokens()** accessor: cacheRead or 0; clamped to promptTokens when
  promptTokens > 0; floored at 0.

## 6. Errors, retry, overflow

- `APIError {status, body, contextOverflow}` — body is the first 4 KiB of
  the non-2xx response (falling back to the HTTP status text when empty)
  and MUST be embedded in the error's message text: the param-strip
  regexes run against it.
- **Transient** = status 408, 429, or ≥ 500; or any network/transport
  error. NEVER context cancellation/deadline, NEVER any other 4xx
  (including overflow), NEVER the library's own validation errors.
- **Context overflow**: a 400 whose body matches (case-insensitive)
  `prompt (is )?too long|context (length|window)|maximum context|too many tokens|exceeds?.{0,20}(context|token)`
  is flagged `contextOverflow` — surfaced explicitly, never retried.
- **RetryPolicy**: defaults 4 attempts (1 try + 3 retries), 500ms base,
  `delay = base × 2^(attempt−1)`, no jitter, no cap. Retry only transient
  failures; the final attempt's error surfaces regardless; a sleep
  interrupted by cancellation stops retrying and surfaces the last fn
  error.

## 7. Rejected-parameter strip middleware

Wraps a Provider. On a failure that (a) is not a context cancellation,
(b) delivered NO stream events and no partial completion, and (c) whose
error text matches one of four phrasings, the named parameter — matched
against `extra` keys by normalized form — is removed and the call retried
ONCE. The strip is REMEMBERED: subsequent calls through the same wrapper
drop the key up front (the source mutated the shared map in place; the
library keeps its own memory and never mutates the caller's map — either
way the key stays gone for later turns). No match, or the named param not
in extra ⇒ surface the original error, no retry.

The four patterns (case-insensitive; each captures the name):

```
does not support parameter\s+["'`]?([^"'`\s,.;:)]+)
unsupported parameter:?\s+["'`]?([^"'`\s,.;:)]+)
unknown field\s+["'`]?([^"'`\s,.;:)]+)
unrecognized request argument:?\s+["'`]?([^"'`\s,.;:)]+)
```

Captured token cleanup: trim the characters `` space \t " ' ` . , : ; ) ``
from both ends. Normalization: lowercase + remove underscores (so
`reasoningEffort` ≡ `reasoning_effort`).

## 8. The loop (Run)

Shape: the **subagent-style** loop (see the asymmetry note below).

- `maxTurns <= 0` → default **10**.
- Each turn advertises `tools.tools()`; `request.tools` is ignored and
  overwritten (nil executor ⇒ no tools advertised).
- `lastTurn = (turn == maxTurns − 1)`: the request carries **no tools** on
  the last permitted turn.
  - **Asymmetry vs the parent app's loop**: the source chat server's
    parent `Run` still ADVERTISES tools on its capped last turn (its
    lastTurn flag only suppresses persisting the never-executed calls and
    ends the loop); its `RunSubagent` — and this library — WITHHOLD them so
    the model must answer. Port the library's (subagent) behavior.
- Continue looping while the executor is non-nil… actually: while the turn
  produced tool calls AND !lastTurn. Per call, in order:
  1. fire OnToolCall;
  2. approval gate: only if `needsApproval(name)` — nil Approver ⇒ deny
     with `DeniedMessage`; `ask` false ⇒ deny with `DeniedMessage` (exact
     string: `The user denied permission to run this tool.`), tool NOT
     executed, loop continues; `ask` THROWS ⇒ **approval-cancel
     finalization**: truncate the transcript back to the assistant
     message (dropping this batch's already-appended tool results), clear
     that assistant message's toolCalls (content/thinking preserved),
     return the partial Result together with the error — the transcript
     stays replayable with no orphans;
  3. execute; an executor throw/error becomes
     `tool execution failed: <message>` with isError (the loop NEVER
     aborts on tool failure);
  4. fire OnToolResult; append
     `{role:"tool", content, toolCallID, toolIsError}`.
- With a nil executor, a hallucinated call gets the teaching result
  `unknown tool: <name>` and the loop continues (deviation from the source
  subagent, which ended the loop; the teaching behavior is deliberate).
- **Retry** wraps each model call (policy from config) but re-attempts ONLY
  when the failed attempt streamed nothing (no partial completion, no
  delivered events). A retried call counts as ONE turn.
- **Mid-stream break/cancel**: the provider returns a partial completion +
  error; the loop appends the partial assistant message with its
  **toolCalls cleared** (they never executed) and returns the partial
  Result + error. The partial call's usage snapshot IS recorded.
- **Ending with text**: trimmed non-empty content ends the loop; the final
  assistant message is appended with any dangling (capped last turn)
  toolCalls cleared.
- **Stall fallback** (content empty at loop end):
  - with an executor: one extra TOOL-LESS wrap-up call on transcript +
    user(wrapUpInstruction). The stalling turn's assistant message is NOT
    in that request (it was never appended — only the tool branch
    appends), so the wrap-up cannot be rejected for an unanswered tool
    call. Non-empty wrap-up content becomes the final (transcript gains
    the wrap-up user message + the answer); a wrap-up error is swallowed
    and falls through. Note the wrap-up is an EXTRA call: turns can reach
    maxTurns + 1.
  - last resort: final content = trimmed content, else the concatenated
    thinking text, else the placeholder `(subagent produced no output)`.
  - wrapUpInstruction, verbatim:
    `Stop researching and write your final answer now, using only the information already gathered above. Do not call any tools and do not keep thinking -- output the complete, self-contained report that directly answers the task.`
- `usages`: one entry per model call that produced a completion (success or
  partial), in order; a fully-failed call contributes none. `turns` counts
  logical model calls (retries collapse).
- On error, Run returns the Result-so-far (non-nil) alongside the error,
  except for the nil-Provider misconfiguration.

## 9. Compaction and OneShot

- `CompactRequestText`, verbatim:
  `Summarize this entire conversation in detail for a future instance of yourself to pick up. Output only the summary.`
- `Compact`: append `user(CompactRequestText)` to the history (per-call
  copy), send NO tools, ONE call, `request.system` passed through (the
  summarizer brief is the caller's — the reference app's system prompt was
  app-specific: identity paragraph + "capture the conversational context"
  bullets: goals/requests in order, stylistic intent, decisions and
  dead-ends, unresolved items + next step, "output ONLY the summary").
  Trimmed empty summary ⇒ error (`the model returned an empty summary;
  nothing was compacted` — the Go error string is lowercased per Go
  convention; the TS reference threw the same sentence capitalized).
  Result messages = exactly
  `[user(CompactRequestText), assistant(summary)]` — a valid round ending
  on assistant so the next real prompt continues clean alternation.
- `OneShot(p, req, timeout)`: strip tools, apply the timeout when > 0, ONE
  attempt (never retried), return the trimmed final text + usage. Callers
  compose with a detached context for fire-and-forget.

## 10. Deliberate cuts (do NOT implement in ts/ either)

- DB/persistence (message trees, leaf advancement, statuses) — the library
  works on a flat in-memory transcript.
- Server-side SSE fan-out (the source's meta/token/done/error event
  protocol and sinks).
- Subagent registry/gating/telemetry (`run_subagent` tool, the process-wide
  gate, `subagent_activity` progress events, shared-context selection/
  rendering).
- Title sanitization (`sanitizeTitle`, `<think>`-stripping, length caps) —
  `OneShot` returns raw trimmed text.
- Structured tool-result content parts (multimodal blocks); `ToolResult` is
  text-only.
- Model-gated thinking/temperature/effort tables (the reference app's
  per-model gates) — callers own `extra`.
- Pricing/cost computation.
- Steering (mid-turn user-message injection).
- Model-id namespacing (`<upstream>/<model>` stripping) — callers pass the
  bare model id.
- llama.cpp `timings` decode and tok/s synthesis (PromptProgress IS kept).
- The `isHostedOpenAI` URL heuristic — replaced by the explicit
  `SelfHosted` flag.

## 11. Test pins the ts/ suite must replicate

- The tool message JSON pin: `{"role":"tool","content":"","tool_call_id":"call_1"}`.
- Merge table incl. equal-evidence-later-wins and regression-discard;
  floor-preserves-surplus.
- Anthropic: two-breakpoint placement (system + tail), transcript
  unchanged after two calls, thinking replayed first with signature,
  redacted_thinking round-trip, tool_result folding with is_error,
  max_tokens fail-fast, tri-state cache fields, full-prompt normalization,
  DisableCaching strips all markers.
- OpenAI: both reasoning field names (reasoning_content precedence),
  split tool-call deltas by index, cumulative usage on every chunk,
  prompt_progress, stream_options default + caller override,
  cache_prompt/SelfHosted, prompt_cache_key, reserved-extra protection.
- Loop: denied string, Ask-error batch clearing, final-turn withheld
  tools, stall wrap-up request shape (tool-less; instruction as the last
  user message; stalled assistant absent), hallucinated-call teaching
  error, no-retry-after-partial, retried-call-is-one-turn.
- Param stripper: all four phrasings, camelCase↔snake_case normalization,
  retry-once, memory across calls, never on cancel/after delivery.
