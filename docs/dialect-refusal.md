# A refusal that names an endpoint

## `DialectRefused(err) (Dialect, bool)`

A model list identifies its SERVER (`DecodeModelList`), and for a long time that was the whole answer to "which protocol do I speak here". It is not, because a host serves one model over one protocol and refuses the same request for another model. OpenAI answers a chat-completions call for a Responses-only model with:

```json
{"error":{"message":"This model is not supported in v1/chat/completions. Use v1/responses instead.",
          "type":"invalid_request_error","code":"unsupported_model"}}
```

That is the endpoint naming the protocol for that model, in its own words. It is the same kind of fact as the model list's envelope, arriving on a different request, so it is read the same way. This is established, not guessed. Nothing here looks at a model NAME. A caller that mapped `o1-*` or `claude-*` onto a dialect will be encoding today's model lineup into code. It will be wrong for the next model and for every gateway that renames one.

## What it reads

The body must DIRECT the caller somewhere. The pattern takes a path only after a directive phrase -- `use`, `switch to`, `call`, `post to`, `belongs on`, `must be called through`, `only supported in`, `served via`, and their neighbours. And a phrase carrying a NEGATION (`not`, `n't`, `never`, `no longer`) is skipped, because a negated phrase names the path that was refused rather than the one to use. Both rules are needed for one sentence: the refusal above says `not supported in v1/chat/completions` and then `Use v1/responses`. As a result, it must answer `DialectResponses` while reading the same paths. A body that names the refused path alone ("this model is not supported in v1/chat/completions.") reports nothing: it says the request failed, not where the request belongs.

The path decides the dialect by its tail: `/responses` is `DialectResponses`, `/messages` is `DialectAnthropic`, `/chat/completions` is `DialectOpenAI`. The version segment is matched but not read, so a `v2/responses` still answers.

## The statuses it accepts

400, 404, 405 and 422 -- a refusal of THIS request. Everything else is excluded because it is about something other than the protocol: a 5xx is the server failing, a 401 or 403 is the credential. A 429 is the quota. A body that happens to quote a path under one of those statuses must not be read. This is as a direction, or a transient blip will repoint a model at a protocol nobody chose.

It reaches through wrapping (`errors.As`), because a host puts its own text on the error before anything classifies it.

## What it is not

It is not a retry. It does not act. `DialectRefused` answers a question. The caller decides what to do with the answer. Typically re-running the same request against the other dialect's provider, and remembering the answer so the next turn goes straight there. That decision needs to know which providers exist and which credentials reach them, which is host knowledge. This is the same split as `IsTransient`: classification here, policy above.

Safety of that re-run comes free from the status list. Every accepted status is a non-2xx response, so nothing streamed and no output reached the caller before the refusal.
