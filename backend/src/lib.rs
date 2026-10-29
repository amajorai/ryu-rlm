//! `ryu-rlm` — Recursive Language Models for Ryu.
//!
//! The problem: a question about a corpus that does not fit in a model's context
//! window. The usual answers are to truncate it, to embed and retrieve fragments of
//! it, or to buy a bigger window — and all three degrade, because a model's
//! attention over a very long prompt is not uniform. A fact placed in the middle of
//! a million tokens is measurably harder to use than the same fact placed alone.
//!
//! The answer here is to stop putting the corpus in a prompt at all. It is held in
//! this process as a **variable**. A root model plans over an *outline* of it —
//! chunk ids, sources, line spans, labels — and reads it only through operators
//! that summarise before returning: `grep` to locate, `map` to run a cheap model
//! over selected chunks in parallel and fold the answers, `recurse` to give a
//! sub-question its own plan over a subset. The root model's own context therefore
//! stays roughly constant whether the corpus is 40 KB or 40 MB, which is the whole
//! claim.
//!
//! Module map:
//!
//! - [`chunk`] — structure-aware splitting; the coordinate system everything else
//!   addresses (a chunk never straddles a document, never splits a line).
//! - [`store`] — immutable contexts, the run journal, and the root allowlist that
//!   bounds which files may ever be read.
//! - [`ops`] — the closed operator vocabulary and the observation cap that keeps a
//!   planner from reading the corpus one `peek` at a time.
//! - [`engine`] — the bounded planning loop, the parallel `map` fold, recursion, and
//!   the replayable trace.
//! - [`host`] — the single authenticated line back into Core for model calls.
//! - [`api`] / [`mcp`] — the same engine as HTTP (for the companion) and as an MCP
//!   stdio server (for agents and workflows).
//!
//! ZERO dependency on `apps/core`: this crate is a process Core spawns, reaches
//! through the generic ext-proxy, and calls back into only over
//! `/api/host/model/complete`.

pub mod api;
pub mod chunk;
pub mod engine;
pub mod host;
pub mod mcp;
pub mod ops;
pub mod paths;
pub mod store;
