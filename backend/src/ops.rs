//! The operator vocabulary — everything the root model is allowed to do to a
//! context, and the only way any of the corpus ever reaches it.
//!
//! ## Why a closed vocabulary and not a REPL
//!
//! The idea this app implements is *context-as-a-variable*: hold the corpus outside
//! the prompt and let a model manipulate it programmatically. The most general form
//! of that is a real code sandbox, and this crate deliberately does not ship one.
//! Three reasons, in order of weight:
//!
//! 1. **The failure mode of a wrong program is invisible.** A generated line of
//!    JavaScript that slices the wrong range still returns *something*, and the
//!    model reports it as evidence. A closed vocabulary can validate every argument
//!    against the corpus — chunk 900 of a 40-chunk context is an error, not an
//!    empty string that reads as "nothing there".
//! 2. **A trace of ops replays; a trace of programs re-executes.** Every step here
//!    is a small JSON value, so a run is auditable and re-runnable without running
//!    code again, which is what makes the companion's tree meaningful.
//! 3. **A sandbox is a second product.** It needs a runtime on the machine, a
//!    permission story, and its own failure surface. The ops below cover
//!    locate-and-read, which is what long-context questions actually are.
//!
//! ## Observations are capped, and that cap is the whole point
//!
//! Every operator returns an [`Observation`] whose text is truncated to
//! [`MAX_OBSERVATION_CHARS`]. Without that cap a planner could `peek` its way to the
//! entire corpus one step at a time and re-create exactly the context rot the app
//! exists to avoid. The truncation is always *marked*, never silent, so the planner
//! knows to narrow rather than concluding the text ended.

use anyhow::{bail, Result};
use regex::RegexBuilder;
use serde::{Deserialize, Serialize};

use crate::store::ContextRecord;

/// Ceiling on one operator's returned text. Sized so a dozen steps of observations
/// still fit comfortably in a planner prompt alongside the outline.
pub const MAX_OBSERVATION_CHARS: usize = 4_000;

/// Ceiling on grep hits returned in one step, regardless of what the caller asked
/// for. A pattern like `.` matches every line in the corpus.
pub const MAX_GREP_HITS: usize = 60;

/// Ceiling on chunks one `map` may fan out over. Past this the planner is not
/// selecting, it is scanning — which is a `recurse` on a narrowed context, not a map.
pub const MAX_MAP_FANOUT: usize = 64;

/// Guard on a caller-supplied regex. Rust's engine has no backtracking, so there is
/// no ReDoS here, but a pathological pattern can still compile to a very large DFA.
const REGEX_SIZE_LIMIT: usize = 1 << 20;

/// One planner move.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum Op {
    /// Re-show the corpus outline, optionally filtered to sources matching a
    /// substring. Free — no corpus text and no model call.
    Outline {
        #[serde(default)]
        source: Option<String>,
    },
    /// Read one chunk, or a window inside it.
    Peek {
        chunk: usize,
        #[serde(default)]
        start: usize,
        #[serde(default)]
        len: Option<usize>,
    },
    /// Regex search across the whole corpus. Reports which chunk each hit fell in,
    /// which is how the planner turns a keyword into chunk ids to `map` over.
    Grep {
        pattern: String,
        #[serde(default)]
        max: Option<usize>,
        #[serde(default)]
        context: Option<usize>,
    },
    /// Ask one cheap leaf model the same question about each named chunk, in
    /// parallel, and fold the non-empty answers. The workhorse: this is where the
    /// corpus is actually read, and the planner sees only the fold.
    Map { prompt: String, chunks: Vec<usize> },
    /// Run a whole nested RLM query over a sub-corpus built from the named chunks.
    /// For when a "chunk" is itself a document-sized thing.
    Recurse { query: String, chunks: Vec<usize> },
    /// Write to the planner's own scratchpad. Notes survive observation truncation,
    /// so this is how a finding from step 2 is still available at step 11.
    Note { text: String },
    /// Answer, and stop.
    Final {
        answer: String,
        #[serde(default)]
        cites: Vec<usize>,
    },
}

impl Op {
    /// Short label for the trace and the companion's tree.
    pub fn label(&self) -> &'static str {
        match self {
            Op::Outline { .. } => "outline",
            Op::Peek { .. } => "peek",
            Op::Grep { .. } => "grep",
            Op::Map { .. } => "map",
            Op::Recurse { .. } => "recurse",
            Op::Note { .. } => "note",
            Op::Final { .. } => "final",
        }
    }

    /// Reject an op whose arguments do not address this corpus, BEFORE it costs a
    /// model call. An out-of-range chunk id is the common planner slip, and letting
    /// it through as an empty result teaches the planner the chunk was irrelevant.
    pub fn validate(&self, ctx: &ContextRecord) -> Result<()> {
        let n = ctx.chunks.len();
        let check = |ids: &[usize]| -> Result<()> {
            if ids.is_empty() {
                bail!("no chunks named; pick ids from the outline");
            }
            if let Some(bad) = ids.iter().find(|i| **i >= n) {
                bail!(
                    "chunk {bad} does not exist (this context has {n}, numbered 0..{})",
                    n - 1
                );
            }
            Ok(())
        };
        match self {
            Op::Peek { chunk, .. } => check(&[*chunk]),
            Op::Map { chunks, prompt } => {
                if prompt.trim().is_empty() {
                    bail!("map needs a prompt to ask of each chunk");
                }
                if chunks.len() > MAX_MAP_FANOUT {
                    bail!(
                        "map over {} chunks exceeds the {MAX_MAP_FANOUT} fan-out ceiling; \
                         narrow with grep, or recurse on a sub-context",
                        chunks.len()
                    );
                }
                check(chunks)
            }
            Op::Recurse { chunks, query } => {
                if query.trim().is_empty() {
                    bail!("recurse needs a query");
                }
                check(chunks)
            }
            Op::Grep { pattern, .. } => {
                if pattern.trim().is_empty() {
                    bail!("grep needs a pattern");
                }
                Ok(())
            }
            Op::Final { answer, cites } => {
                if answer.trim().is_empty() {
                    bail!("final needs an answer");
                }
                if let Some(bad) = cites.iter().find(|i| **i >= n) {
                    bail!("cited chunk {bad} does not exist");
                }
                Ok(())
            }
            Op::Outline { .. } | Op::Note { .. } => Ok(()),
        }
    }
}

/// What an operator returned to the planner.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Observation {
    pub text: String,
    /// True when [`MAX_OBSERVATION_CHARS`] cut the text. Surfaced to the planner in
    /// the text itself as well — a planner that does not know it was truncated will
    /// conclude the corpus ends where the cap did.
    pub truncated: bool,
}

impl Observation {
    pub fn new(text: impl Into<String>) -> Observation {
        let text = text.into();
        let count = text.chars().count();
        if count <= MAX_OBSERVATION_CHARS {
            return Observation {
                text,
                truncated: false,
            };
        }
        let kept: String = text.chars().take(MAX_OBSERVATION_CHARS).collect();
        Observation {
            text: format!(
                "{kept}\n\n[… truncated: {} of {count} characters shown. Narrow the range, \
                 or map over the chunk instead of reading it.]",
                MAX_OBSERVATION_CHARS
            ),
            truncated: true,
        }
    }
}

/// `outline`, optionally filtered by source substring.
pub fn outline(ctx: &ContextRecord, source: Option<&str>) -> Observation {
    let Some(needle) = source else {
        return Observation::new(ctx.outline());
    };
    let rows: String = ctx
        .chunks
        .iter()
        .filter(|c| c.source.contains(needle))
        .map(|c| {
            format!(
                "[{}] {}:{}-{} ({} chars) {}\n",
                c.id,
                c.source,
                c.start_line,
                c.end_line,
                c.chars(),
                c.label
            )
        })
        .collect();
    if rows.is_empty() {
        return Observation::new(format!("no source matches {needle:?}"));
    }
    Observation::new(rows)
}

/// `peek` — a raw window into one chunk.
pub fn peek(ctx: &ContextRecord, id: usize, start: usize, len: Option<usize>) -> Observation {
    let Some(chunk) = ctx.chunk(id) else {
        return Observation::new(format!("chunk {id} does not exist"));
    };
    let chars: Vec<char> = chunk.text.chars().collect();
    if start >= chars.len() {
        return Observation::new(format!(
            "chunk {id} is {} characters; offset {start} is past its end",
            chars.len()
        ));
    }
    let take = len
        .unwrap_or(MAX_OBSERVATION_CHARS)
        .min(MAX_OBSERVATION_CHARS);
    let window: String = chars[start..].iter().take(take).collect();
    let neighbors = chunk.neighbors(ctx.chunks.len());
    Observation::new(format!(
        "[{}] {}:{}-{} (offset {start} of {} chars; adjacent chunks: {:?})\n{window}",
        chunk.id,
        chunk.source,
        chunk.start_line,
        chunk.end_line,
        chars.len(),
        neighbors
    ))
}

/// `grep` — regex across the corpus, reported with chunk ids so the planner can act
/// on the result.
pub fn grep(
    ctx: &ContextRecord,
    pattern: &str,
    max: Option<usize>,
    context_lines: Option<usize>,
) -> Observation {
    let re = match RegexBuilder::new(pattern)
        .case_insensitive(true)
        .size_limit(REGEX_SIZE_LIMIT)
        .build()
    {
        Ok(re) => re,
        Err(e) => return Observation::new(format!("bad pattern: {e}")),
    };
    let cap = max.unwrap_or(MAX_GREP_HITS).min(MAX_GREP_HITS);
    let around = context_lines.unwrap_or(0).min(3);

    let mut out = String::new();
    let mut hits = 0usize;
    let mut chunks_hit: Vec<usize> = Vec::new();
    'outer: for chunk in &ctx.chunks {
        let lines: Vec<&str> = chunk.text.lines().collect();
        for (idx, line) in lines.iter().enumerate() {
            if !re.is_match(line) {
                continue;
            }
            if !chunks_hit.contains(&chunk.id) {
                chunks_hit.push(chunk.id);
            }
            let lo = idx.saturating_sub(around);
            let hi = (idx + around + 1).min(lines.len());
            for (n, l) in lines[lo..hi].iter().enumerate() {
                out.push_str(&format!(
                    "[{}] {}:{}: {}\n",
                    chunk.id,
                    chunk.source,
                    chunk.start_line + lo + n,
                    l.trim_end()
                ));
            }
            hits += 1;
            if hits >= cap {
                out.push_str(&format!("[… stopped at the {cap}-hit ceiling]\n"));
                break 'outer;
            }
        }
    }

    if hits == 0 {
        return Observation::new(format!("no match for {pattern:?} anywhere in the corpus"));
    }
    // The chunk id list is the actionable part — it is what a follow-up `map` takes
    // as its argument — so it leads, where truncation cannot eat it.
    Observation::new(format!("{hits} hit(s) in chunks {chunks_hit:?}\n{out}"))
}

/// Parse one planner move. Accepts the `{"op": …}` tagged form only: an untagged
/// guess would make a typo'd op name silently become some other op.
pub fn parse_op(value: &serde_json::Value) -> Result<Op> {
    serde_json::from_value(value.clone()).map_err(|e| {
        anyhow::anyhow!(
            "not a valid operator ({e}). Reply with one JSON object using an \"op\" of \
             outline, peek, grep, map, recurse, note or final."
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunk::Document;
    use crate::store::{data_dir, Store};
    use std::path::PathBuf;

    fn ctx_of(docs: Vec<Document>) -> std::sync::Arc<ContextRecord> {
        let dir: PathBuf =
            std::env::temp_dir().join(format!("ryu-rlm-ops-{}", uuid::Uuid::new_v4()));
        let store = Store::open(dir).unwrap();
        store.create("t", docs, None, Some(500)).unwrap()
    }

    fn doc(source: &str, text: &str) -> Document {
        Document::new(source, text)
    }

    #[test]
    fn data_dir_is_under_the_node_directory() {
        // Guards the RYU_DIR-env-first rule: a sidecar that wrote to ~/.ryu while
        // Core ran on RYU_PROFILE=dev would put dev data in the release install.
        std::env::set_var("RYU_DIR", "/tmp/ryu-rlm-probe");
        assert!(data_dir().ends_with("rlm"));
        assert!(data_dir().starts_with("/tmp/ryu-rlm-probe"));
        std::env::remove_var("RYU_DIR");
    }

    #[test]
    fn an_out_of_range_chunk_is_an_error_not_an_empty_result() {
        // The important half: it must fail BEFORE a map spends a model call per
        // chunk, and the message must say what the valid range is.
        let ctx = ctx_of(vec![doc("a.md", "alpha")]);
        let op = Op::Map {
            prompt: "what?".into(),
            chunks: vec![0, 99],
        };
        let err = op.validate(&ctx).expect_err("must reject");
        assert!(err.to_string().contains("chunk 99 does not exist"));
    }

    #[test]
    fn map_refuses_a_fan_out_that_is_really_a_scan() {
        let body = ("p".repeat(600) + "\n\n").repeat(100);
        let ctx = ctx_of(vec![doc("big.md", &body)]);
        let all: Vec<usize> = (0..ctx.chunks.len()).collect();
        assert!(
            all.len() > MAX_MAP_FANOUT,
            "fixture must exceed the ceiling"
        );
        let err = Op::Map {
            prompt: "q".into(),
            chunks: all,
        }
        .validate(&ctx)
        .expect_err("must refuse");
        assert!(err.to_string().contains("fan-out ceiling"));
    }

    #[test]
    fn an_observation_says_so_when_it_truncates() {
        let obs = Observation::new("y".repeat(MAX_OBSERVATION_CHARS * 2));
        assert!(obs.truncated);
        assert!(obs.text.contains("truncated"));
    }

    #[test]
    fn peek_cannot_be_used_to_exceed_the_observation_cap() {
        // The loophole that would re-create context rot one step at a time: ask for
        // a megabyte and get it.
        let ctx = ctx_of(vec![doc("a.md", &"z".repeat(20_000))]);
        let obs = peek(&ctx, 0, 0, Some(1_000_000));
        assert!(obs.text.chars().count() < MAX_OBSERVATION_CHARS + 500);
    }

    #[test]
    fn grep_reports_the_chunk_ids_first_so_truncation_cannot_eat_them() {
        let ctx = ctx_of(vec![doc("a.md", "nothing\nrefund policy here\nnothing")]);
        let obs = grep(&ctx, "refund", None, Some(0));
        assert!(obs.text.starts_with("1 hit(s) in chunks [0]"));
        assert!(obs.text.contains("a.md:2"), "line numbers must be true");
    }

    #[test]
    fn grep_line_numbers_are_absolute_in_the_source_not_relative_to_the_chunk() {
        // A citation is only useful if it opens the right line in an editor. Build a
        // corpus big enough that the match lands in a later chunk.
        let body = "filler\n".repeat(300) + "the magic word\n";
        let ctx = ctx_of(vec![doc("d.txt", &body)]);
        let obs = grep(&ctx, "magic", None, Some(0));
        assert!(
            obs.text.contains("d.txt:301"),
            "expected absolute line 301, got: {}",
            obs.text
        );
    }

    #[test]
    fn a_bad_regex_is_reported_not_panicked() {
        let ctx = ctx_of(vec![doc("a.md", "x")]);
        let obs = grep(&ctx, "((((", None, None);
        assert!(obs.text.starts_with("bad pattern:"));
    }

    #[test]
    fn a_misspelled_op_is_rejected_rather_than_coerced() {
        let err = parse_op(&serde_json::json!({ "op": "peeek", "chunk": 0 }));
        assert!(err.is_err());
    }

    #[test]
    fn every_op_round_trips_through_its_tagged_form() {
        for op in [
            Op::Outline { source: None },
            Op::Peek {
                chunk: 1,
                start: 0,
                len: None,
            },
            Op::Grep {
                pattern: "x".into(),
                max: None,
                context: None,
            },
            Op::Map {
                prompt: "p".into(),
                chunks: vec![0],
            },
            Op::Recurse {
                query: "q".into(),
                chunks: vec![0],
            },
            Op::Note { text: "n".into() },
            Op::Final {
                answer: "a".into(),
                cites: vec![],
            },
        ] {
            let json = serde_json::to_value(&op).unwrap();
            assert_eq!(
                parse_op(&json).unwrap(),
                op,
                "round trip failed for {}",
                op.label()
            );
        }
    }
}
