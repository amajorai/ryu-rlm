<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./icon-dark.png" />
    <img src="./icon-light.png" alt="Recursive Language Model" width="144" />
  </picture>
</p>

<div align="center">

# Recursive Language Model

</div>

Answer questions about a corpus larger than any context window WITHOUT putting it in a prompt: documents live as a variable, a planner works over an outline, a cheap model reads chunks in parallel, and findings fold back up with path:line citations and a replayable trace.

> **The public home of `ryu-rlm`.** Source, builds, and releases live here —
> binaries for every platform are attached to each release.
>
> This tree is generated from the Ryu monorepo, so commits pushed here
> directly are replaced on the next sync. **Pull requests are welcome** —
> open them here and they are ported into the monorepo, then flow back out.
> Ryu as a whole: https://github.com/amajorai/ryu

## Install

**App:** [Install](ryu://apps/@ryu/rlm) (opens the Ryu desktop app and asks you to confirm)

**CLI:**

```bash
ryu apps add @ryu/rlm
```

**Crate:**

```bash
cargo install ryu-rlm
```

Prebuilt binaries for every platform are attached to [each release](https://github.com/amajorai/ryu/releases).

## License

Apache-2.0 — see [LICENSE](./LICENSE).

## The idea

The usual answers to "this doesn't fit" all lose something. Truncate and you may
drop the answer. Embed-and-retrieve gives you fragments without the argument they
sat in. Buy a bigger window and you still pay the attention tax — a fact in the
middle of a very long prompt is measurably harder for a model to use than the same
fact on its own.

This app does the other thing. Your documents are held in the sidecar as a
**variable**. A *root* model plans over an **outline** of them (chunk ids, sources,
line spans, headings) and never sees the corpus itself. It reaches the text only
through operators that summarise before returning:

| Operator  | What it does                                                              | Model calls |
| --------- | ------------------------------------------------------------------------- | ----------- |
| `outline` | Re-show the map of the corpus, optionally filtered by source              | none        |
| `grep`    | Regex across everything; reports which chunk each hit fell in             | none        |
| `peek`    | Read a window of one chunk verbatim                                       | none        |
| `map`     | Ask one cheap model the same question of each selected chunk, in parallel | one/chunk   |
| `recurse` | Give a sub-question its own plan over a subset of chunks                  | nested run  |
| `note`    | Write to the planner's scratchpad                                         | none        |
| `final`   | Answer, with citations                                                    | none        |

`map` is where reading actually happens, and it is the economic core: hundreds of
small calls on a cheap model, folded down to a few lines the planner reads. Because
the planner only ever sees outlines and folds, **its own context stays roughly
constant whether the corpus is 40 KB or 40 MB.**

## What you get back

Every answer carries `path:line` citations resolved out of the chunks it rests on,
and every run keeps a full trace (each operator, its arguments, what came back,
what it cost) that the companion renders as a tree and lets you replay.

The companion also shows the ratio of *characters actually sent to models* against
*corpus size*. Below 1× the recursion read less than the corpus. Above 1× it re-read
things and stuffing the window would have been cheaper. Both are shown, because a
tool that only reports its wins cannot be measured.

## Budgets fail loudly

Depth, planning steps, model calls and wall clock are all bounded, and a run that
hits a bound is reported as `budget_exhausted` with whatever it had — never as a
quietly shortened answer. On a long-context question nobody can catch that by eye,
which is exactly why it must not be possible.

## Surfaces

- **Companion** ("Deep Read") — load files or paste text, inspect the outline, grep
  it for free, ask questions, read the trace.
- **MCP server** (`rlm__*`) — `ask` takes paths and a question and returns an answer
  with citations, so an agent can consult a large corpus without the corpus entering
  its own context. `grep` and `peek` need no model at all, so a workflow can branch
  on a deterministic search of a 40 MB corpus.
- **Composer toggle** ("Deep read the corpus") — after each answer, ask the same
  question of the selected context and append what the documents actually say. When
  the reader tool is not reachable the hook reports that it could not run; it never
  substitutes an opinion, which would be the exact failure the app exists to prevent.

## Configuration

| Setting                | What it does                                                                          |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `rlm-active-context`   | The context id the composer toggle reads from. Copy it from the companion header.     |
| `rlm-root-model`       | The planner. Sees outlines and folds only, so a strong model is affordable here.      |
| `rlm-leaf-model`       | The reader. Called once per chunk; it dominates cost. Pick a fast, cheap model.       |

| Env var          | Default          | What it does                                                     |
| ---------------- | ---------------- | ---------------------------------------------------------------- |
| `RYU_RLM_PORT`   | `8014`           | Loopback bind port                                                |
| `RYU_RLM_BIN`    | —                | Override the sidecar binary path                                  |
| `RYU_RLM_ROOTS`  | the user's home  | `:`-separated list of directories files may be loaded from        |

Files are read only from inside a configured root, and paths are canonicalised
**before** the check, so a symlink is judged on where it lands rather than where it
sits. A recursive walk steps over hidden directories (`.ssh`, `.aws`, `.git`) and
`node_modules`; a path you name explicitly is loaded regardless.

## Architecture

The sidecar has **zero dependency on `apps/core`**. Core spawns it, proxies
`/api/rlm/*` to it on loopback behind a shared-secret bearer, and the only line back
is `POST /api/host/model/complete` — the same `host.sideModel` capability the
turn-hook sandbox uses, gated on `hook:side-model`. So the app inherits the node's
provider routing, budget and egress policy, and holds no credential of its own.

```
backend/src/
  chunk.rs    structure-aware splitting; the coordinate system citations use
  store.rs    immutable contexts, the run journal, the filesystem root allowlist
  ops.rs      the closed operator vocabulary and the observation cap
  engine.rs   the bounded planning loop, the parallel map fold, recursion, the trace
  host.rs     the one authenticated line back into Core
  api.rs      HTTP, for the companion
  mcp.rs      MCP stdio, for agents and workflows
```

## Why a closed operator set rather than a code sandbox

The most general form of context-as-a-variable is a real REPL, and this app
deliberately does not ship one. A generated program that slices the wrong range
still returns *something*, and the model reports it as evidence; a closed vocabulary
can validate every argument against the corpus, so chunk 900 of a 40-chunk context is
an error rather than an empty string that reads as "nothing there". A trace of
operators also replays without re-executing code, which is what makes the companion's
tree worth looking at.

## Development

```bash
cargo test -p ryu-rlm                      # 47 tests: chunking, roots, ops, budgets
bun run --cwd apps-store/rlm/ui build      # → dist/index.html, one self-contained file
bun run --cwd apps-store/rlm/ui test       # path/parse helpers
scripts/sync-app-fixtures.sh rlm           # refresh the compiled-in UI bundle
```
