# Postmortem: transcript decay

Transcript decay is the loss of part of a model's own output between the response that carried it and the next request that replays it. The model then works from a transcript that no longer holds its reasoning. Each turn it re-derives what it already knew, or contradicts a plan it can no longer see. Every call succeeds. Nothing logs an error. The only symptom is worse output.

Use the `/transcript-decay` skill (`.claude/skills/transcript-decay/SKILL.md`) to check a host for it and to fix what the check finds.

## Summary

A grok-build session that ran through bifrost to OpenRouter lost the thinking of every non-OpenAI model on each tool-loop turn. Output quality fell off sharply over a long task. First fixes covered the hop that was seen:

- [wow-look-at-my/bifrost#6](https://github.com/wow-look-at-my/bifrost/pull/6): the OpenAI Responses request converter dropped a reasoning item that carried only `reasoning_text` content.
- [wow-look-at-my/grok-build#257](https://github.com/wow-look-at-my/grok-build/pull/257): the streaming fallback duplicated such an item as a summary on replay.

A follow-up sweep asked whether the same loss happens. On Ollama endpoints, and in the agentic-loop clients simple-llm-ui and simple-llm-harness. It did, in places:

| Hop | What was lost | Fix |
| --- | --- | --- |
| grok-build to Ollama, chat completions | Replayed thinking went out only as `reasoning_content`. Ollama reads `reasoning`. | [wow-look-at-my/grok-build#263](https://github.com/wow-look-at-my/grok-build/pull/263) sends both names, except to xAI's API |
| bifrost to Ollama, chat completions | The same rename, on bifrost's outbound wire. | [wow-look-at-my/bifrost#10](https://github.com/wow-look-at-my/bifrost/pull/10) |
| bifrost to OpenRouter, chat completions | `reasoning_details` was never forwarded. OpenRouter reads encrypted reasoning and signatures only from that field. | [wow-look-at-my/bifrost#10](https://github.com/wow-look-at-my/bifrost/pull/10) |
| agentic-loop Responses dialect | A reasoning item with `reasoning_text` content, an empty summary and no `encrypted_content` was dropped on read. | [wow-look-at-my/agentic-loop#82](https://github.com/wow-look-at-my/agentic-loop/pull/82) adds `ThinkingBlock.Raw` and replays the item as content |
| simple-llm-harness storage and projection | A thinking block with no signature and no redacted payload was dropped before a Responses request. | [wow-look-at-my/simple-llm-harness#66](https://github.com/wow-look-at-my/simple-llm-harness/pull/66) stores and projects `Raw` |
| simple-llm-ui storage | Nothing. It stores each whole `agentic.ThinkingBlock` as JSON, so `Raw` survives with no change. | [wow-look-at-my/simple-llm-ui#323](https://github.com/wow-look-at-my/simple-llm-ui/pull/323) adds the test that proves it |

The agentic-loop chat-completions dialect was already correct. It writes `reasoning_content`, `reasoning` and `reasoning_details` on every assistant message, and `core/conservation_test.go` fails when a dialect stops sending a transcript field.

## Impact

- Every tool-loop turn of a reasoning model through an affected hop lost the turn's thinking. A long task lost all of it.
- The loss was silent. The provider accepted each request, because a request without reasoning is valid.
- The cost rose too. The model re-did reasoning it had already paid for, on every turn.

## Root cause

Each provider names the same thing differently, and reads back only its own name:

| Provider and API | Where thinking arrives | What it reads on replay |
| --- | --- | --- |
| Ollama, OpenAI-compatible chat (local and ollama.com) | `reasoning` | `reasoning` only. `reasoning_content` is ignored. |
| Ollama, Responses | the summary and `encrypted_content` | the summary only |
| OpenRouter, chat completions | `reasoning`, plus `reasoning_details` | `reasoning_details` for encrypted reasoning and signatures |
| OpenRouter, Responses, a non-OpenAI model | a reasoning item with `content:[{type:"reasoning_text"}]`, an empty summary, no `encrypted_content` | the item's `content` |
| DeepSeek-style chat completions | `reasoning_content` | `reasoning_content` |

A translator that knows one spelling writes that spelling, and a translator that checks for a payload it expects drops an item. That item carries a different one. A store that keeps the fields of one dialect loses the field that only another dialect sets. Each of these is correct for the provider its author tested against. In a chain of hops, the field has to survive every one of them, and one hop that drops it is enough.

## Why it was not caught

- No hop had a test that replayed a response through the next request and asserted the reasoning came back. Unit tests checked the request shape against the author's provider.
- The failure has no error. A conservation check has to ask for the field by name, or nothing notices.
- Each repository owned one hop. Nobody read the chain end to end.

## What went well

- agentic-loop already had `core/conservation_test.go`. It made the chat-completions dialect provably correct and gave the Responses fix a place for its test.
- `ErrStateLost` (`internal/loop/statecheck.go`) ends a run when a turn's output does not reach the next call. It catches a dropped turn, though not a dropped field inside one.

## What went poorly

- The first fixes covered the hop that was observed. The same defect sat in other hops for the same reason.
- Live verification was not possible. The OpenRouter key available returned 401, so every fix was tested against fake servers built from the documented wire shapes.

## Lessons

- **A transcript field must survive every hop.** Test it at each hop: response in, next request out, the field present.
- **Write every name a provider can read.** On chat completions, send `reasoning_content` and `reasoning` and `reasoning_details` when the response carried them, unless a provider rejects one. Narrow the set only on a rejection that names the field.
- **Never drop a reasoning item for the shape of its payload.** An item with content and no summary is still reasoning.
- **A store keeps every field of every dialect.** A field that only one dialect sets is still part of the transcript.
