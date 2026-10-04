# One module

Everything is `github.com/wow-look-at-my/agentic-loop`. The packages under it are ordinary packages. The dependency arrows are the layering:

```
./           the agentic loop (package agentic)      <- client, extras
core         the format, its schema, the three dialects. Depends on
             xml-validator/validator.
extras       retry, rate limiting                    <- core
client       the Go API                              <- core, extras
session      conversation storage                    <- core
search       the conversation index: FTS5 + vectors over stored
             conversations. Depends on modernc.org/sqlite.
                                                     <- core, session
http         stateless + stateful HTTP               <- core, session
socket       unix socket + websocket                 <- core, session
cli          the cai commands                        <- everything
cmd/cai      main for cai
todo/driver  main for the task-list launch check
```

Optional tool families are sibling packages that return `agentic.Tools` values a host appends itself: `vfs`, `repo`, `subagent`, `webfetch`, `todo`, `resources`.

Every `main` sits under `<area>/<binary>/` or `cmd/<binary>/`. That is not taste: go-toolchain names a binary after the MODULE when the main package is one level down, so `cli`. `todo/driver` will both resolve to the module root and collide. A couple of levels down, the leaf directory is the name.

The compiler still enforces the layering: `core` cannot reach for a policy that lives in `client`, because the import will be a cycle. That property never needed separate modules — it comes from the package graph.

## Why not several modules

It was several modules, in a repository of its own, and the cost was not theoretical.

**A module cannot honestly pin its siblings.** `client` requires `core` at a pseudo-version. That line is written *before* the commit that publishes them both exists — so it necessarily names an earlier one. At the first publish it names a commit with no such module in it at all. A consumer gets `missing go.mod at revision <hash>`. A relative `replace` hides this inside the repository and nowhere else, because a replace applies to the main module only. So every build was green and every consumer was broken, which is the worst shape a failure can take.

**Adding a module took merges.** Merge, run the toolchain so the pins move onto a commit that has the module, merge again. Only then can anything outside build against it.

**And the loop was a second repository on top of that**, so a change spanning the loop and a dialect needed branches, two PRs. This is a release ordering between them.

One `go.mod` has none of that: no sibling pins to rot, no replaces, no cross-repo ordering, one CI job, and `go-toolchain` from one directory.

The argument that bought the split was dependency weight — a program that only converts a document must not acquire cobra. That is real and it is small. Cobra reaches only `cli/`, Go builds only the packages you import, and a binary that never imports `cli/` never links it. `go.sum` grows. Nothing else does. That was not worth a broken bootstrap and a two-merge dance.

**Do not split it back up.**

## Dependencies

`search` adds `modernc.org/sqlite`, which is SQLite transpiled to Go: the index is a real database. The index needs no cgo, so a consumer still cross-compiles a static binary. It reaches `search/` alone, exactly as cobra reaches `cli/` alone, and a binary that imports neither links neither.

Org modules, both resolved from the proxy, both branch-tracked:

```
require github.com/wow-look-at-my/xml-validator/validator v0.0.0-... // go-toolchain:auto-branch
require github.com/wow-look-at-my/go-containers v0.0.0-...           // go-toolchain:auto-branch
```

`auto-branch` follows each module's default branch, and `go-toolchain` re-resolves it to that branch's head on every run, rewriting the pseudo-version. Without it the line sits at whatever version it was written with, rotting into something nobody built against. The marker goes on DIRECT requires only — `go-toolchain` warns and skips an `// indirect` line. This is because a per-line branch pin cannot mean what it looks like on a dependency Go resolved transitively.

`xml-validator` is itself cut into modules now (`validator`, `reader`, `writer`, `cli`). As a result, the require names `.../xml-validator/validator` rather than the repository root, which is no longer a module.

## Building

```sh
go-toolchain
```

No `go.work`, and nothing to check out alongside.
