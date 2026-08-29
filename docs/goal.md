# Goal mode

A stopping condition the user states in words, and a stop policy that refuses to
end a run until it holds. `goal` is the whole policy; the host owns three seams
and nothing else.

Every string the package emits is contract and is pinned by a test. A host shows
these words to its user and sends the directive to its model, so two hosts
running goal mode behave the same way. Do not "improve" them.

## The three seams

- **State persistence.** `State` is a plain struct of exported scalars and one
  `time.Time`, and this package ships no encoding for it: how it is stored — a
  key-value row, one column per field, a file — is entirely the host's. A
  library that shipped an `Encode`/`Decode` pair would be deciding what a row in
  somebody else's database holds, and nothing there could query into it or
  notice a field this package added. Goal state must survive a restart, counters
  included: a goal that spent its budget before a crash does not get a fresh one
  for free.
- **`Evaluator.Window`.** The host maps its own transcript rows onto `[]Entry`.
  `EntriesFromMessages` does it for a host whose store IS `[]agentic.Message`.
- **`Evaluator.Judge`.** One bounded, tool-less model call.  `OneShotJudge`
  builds the ordinary one over the session's own provider and model.

`State.Scope` is carried but never read here: it is the host's own marker for
where the goal started — a run id, a message id, whatever names the point the
window opens at — so a host stores one blob rather than two.

## What the evaluator reads, and what it must not

`RenderWindow` renders the goal-scoped entries newest-first within `EvalTokens`,
and heads the result with `OmittedMarker` when anything was dropped. That marker
is a line rather than nothing: an evaluator judging a truncated transcript has to
know it is truncated, or "no evidence of a test run" means two different things
it cannot tell apart. An empty window is a real state — a goal set before any
work — and says so, rather than handing the model a blank string to interpret.

A long tool result keeps 1200 runes of head and 800 of tail. Evidence of a test
run lives at both ends and nowhere in the middle.

Two things a host must NOT map onto an entry, and each exclusion is load-bearing:

- **Private reasoning.** It is not evidence that work happened.
- **Any notice goal mode itself wrote.** An evaluator that reads its own previous
  verdicts anchors on them, which turns one wrong judgment into every later one.

## The outcomes

| Outcome | What it means | What happens to the goal |
| --- | --- | --- |
| `Permitted` | The run may end: it was cancelled, or the goal is already suspended. | Kept as it is. |
| `Blocked` | The condition does not hold. `Directive` is what the model is told. | Kept; `Iterations` and the repetition run advance. |
| `Met` | The condition holds. | Cleared by the host. |
| `Failed` | The condition can never hold, or the evaluator said the same thing three times running. | Cleared by the host. |
| `Suspended` | The evaluator could not answer. | Kept, marked suspended, and it blocks nothing until the user retries. |

**Goal mode fails open.** Ending a run one iteration early is one keystroke from
being resumed; the alternative is a wedged session. So a failure to evaluate
suspends the goal loudly and permits the stop, and a cancelled run is permitted
without a model call at all — a goal a user cannot interrupt is a wedged session.

**Three identical reasons end the goal.** The evaluator has now told us the same
thing about the same non-progress three times; that is the evidence `ErrStuck`
uses one layer down and it earns the same answer. It is a verdict, not a budget,
so it CLEARS the goal rather than pausing it.

**One retry on an unusable answer, and the message says which failure it was.**
The retry sends the identical request, because the failure it is for is a model
that wrapped its object in prose — not a request that was wrong. So "the
evaluator is unreachable" and "the evaluator will not answer in JSON" never read
as the same line. An empty `reason` is a parse failure rather than a verdict with
nothing to say: the reason is the whole of what the user and the model are told,
and a block carrying no reason is a refusal nobody can act on.

## Hooking it to a run

`StopListener.Attach(ctx, events, msgs)` subscribes to `Events.OnStop` and queues
the directive as a `SystemMessage` of `DirectiveKind` when the stop is refused,
which takes the loop round again.

Re-arming IN PLACE is the point. A fresh run would replay the transcript from the
top and pay for it, and the model would read the directive as the opening of a
new conversation rather than as the answer to the stop it just tried.

`ctx` is the RUN's context, which is what makes cancellation win. `msgs` must be
the queue the run was configured with, or a blocked stop queues into nothing and
the run ends anyway.

The listener never returns an error out of the callback: an error there aborts
the run and loses the answer the model just wrote. `Report` is how the host hears
what happened — a queued directive says only what the MODEL is told, and the
session still has to record the notice, persist the counters, clear a goal that
is met, and bill the evaluator's own call.

## Why the directive says what it does

`Directive` ends by telling the model what to do about a conflicting
instruction: say which message conflicts with which part of the condition, and
stop. A mid-run message arrives as an ordinary user message and never edits the
condition, so without that paragraph a run grinds against an instruction the user
believes they changed. With it, the conflict surfaces as a visible stop with both
texts quoted.

`Briefing` is the other half, sent once when the goal is set: treat the condition
itself as the instruction, and do not pause to ask whether to continue.
