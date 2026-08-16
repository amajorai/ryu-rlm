//! The recursion engine — a root model that plans over a corpus it never reads.
//!
//! ## The loop
//!
//! One query against one context runs as a bounded loop. Each turn the planner is
//! shown four things and nothing else:
//!
//! - the **query**,
//! - the corpus **stats and outline** (ids, sources, line spans, labels),
//! - its own **notes** (durable, never truncated away),
//! - its recent **observations** (capped, oldest collapsed to one line).
//!
//! It replies with one JSON operator. `map` and `recurse` are the two that actually
//! read: `map` spends one cheap leaf call per chunk and folds the answers, `recurse`
//! runs this whole loop again over a sub-corpus. Everything the planner learns
//! arrives already summarised, which is the property that keeps its own context
//! bounded no matter how large the corpus is.
//!
//! ## Why the prompt is rebuilt every turn instead of appended to
//!
//! A conversational transcript would grow without limit and re-create precisely the
//! context rot this app exists to prevent — the planner would end up holding the
//! corpus after all, one observation at a time. So there is no chat history here.
//! State lives in three explicit slots (notes, findings, recent observations), each
//! individually bounded, and the prompt is assembled from them fresh every turn.
//! The cost is that the planner cannot rely on remembering something it did not
//! write down; that is why `note` exists and why the system prompt insists on it.
//!
//! ## Budgets fail loudly
//!
//! Every bound — steps, model calls, depth, wall clock — ends the run with an
//! explicit status, never with a quietly shortened answer. A partial answer labelled
//! `budget_exhausted` is useful; the same text labelled `ok` is a lie, and on a
//! long-context question nobody can check it by eye. When steps run out but calls
//! remain, the planner gets one final turn to answer from what it has, and that turn
//! is marked in the trace.

use std::sync::Arc;
use std::time::Instant;

use serde::{Deserialize, Serialize};

use crate::chunk::Document;
use crate::host::{extract_json, Host, Tier};
use crate::ops::{self, Observation, Op};
use crate::store::ContextRecord;

/// Hard bounds on one run. Defaults are sized for a corpus of a few hundred chunks:
/// enough steps to grep, map twice and answer, without a runaway costing the node's
/// whole provider quota on one question.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Budget {
    pub max_steps: usize,
    pub max_model_calls: usize,
    pub max_depth: usize,
    pub wall_secs: u64,
}

impl Default for Budget {
    fn default() -> Budget {
        Budget {
            max_steps: 14,
            max_model_calls: 400,
            // Depth 2 means: a run may recurse, and that nested run may not. Deeper
            // than that the sub-queries drift far enough from the original question
            // that the fold stops being an answer to it — and the call count grows
            // multiplicatively, which is how a single question becomes a bill.
            max_depth: 2,
            wall_secs: 600,
        }
    }
}

impl Budget {
    /// Clamp a caller-supplied budget to something a node can survive. A request
    /// body is untrusted input; `max_model_calls: 1_000_000` must not be honoured
    /// just because someone typed it.
    pub fn sanitize(self) -> Budget {
        Budget {
            max_steps: self.max_steps.clamp(1, 40),
            max_model_calls: self.max_model_calls.clamp(1, 2_000),
            max_depth: self.max_depth.clamp(0, 3),
            wall_secs: self.wall_secs.clamp(10, 3_600),
        }
    }
}

/// One recorded move, and what it cost. The companion renders these as a tree via
/// `parent`, and a run replays from them without re-executing anything.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TraceStep {
    pub index: usize,
    pub depth: usize,
    /// Index of the step that spawned this one, for nested runs. `None` at the root.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    pub op: String,
    /// The operator's arguments, verbatim — this is what makes a run auditable.
    pub args: serde_json::Value,
    /// What came back, truncated for display.
    pub observation: String,
    pub truncated: bool,
    pub model_calls: usize,
    pub elapsed_ms: u64,
    /// Set when the step failed (a bad op, an out-of-range chunk, a provider error).
    /// A failed step is recorded and fed back to the planner, not thrown away — the
    /// planner correcting itself is normal, and hiding it makes traces unreadable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// How a run ended.
pub const STATUS_OK: &str = "ok";
pub const STATUS_BUDGET: &str = "budget_exhausted";
pub const STATUS_ERROR: &str = "error";

/// A finished run.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub id: String,
    pub context_id: String,
    pub query: String,
    pub status: String,
    pub answer: String,
    /// Chunk ids the answer rests on, so a reader can go check it.
    pub cites: Vec<Citation>,
    pub trace: Vec<TraceStep>,
    pub started_at: String,
    pub elapsed_ms: u64,
    pub model_calls: usize,
    /// Characters sent to models across the whole run. The honest comparison
    /// against "just put the corpus in the prompt" — which is why it is reported
    /// next to the corpus size rather than buried.
    pub prompt_chars: u64,
    pub budget: Budget,
}

/// A resolved citation: the chunk, and where it really lives.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Citation {
    pub chunk: usize,
    pub source: String,
    pub start_line: usize,
    pub end_line: usize,
    pub label: String,
}

/// Mutable state carried across turns. Each slot is separately bounded — see the
/// module note on why there is no transcript.
struct Working {
    notes: Vec<String>,
    findings: Vec<String>,
    recent: Vec<(String, String)>,
}

/// How many recent observations are shown in full before collapsing to one line.
const RECENT_IN_FULL: usize = 4;
/// Ceiling on the notes and findings blocks, so a planner that notes compulsively
/// still cannot grow its own prompt without limit.
const MAX_NOTES: usize = 24;
const MAX_FINDINGS: usize = 60;

impl Working {
    fn new() -> Working {
        Working {
            notes: Vec::new(),
            findings: Vec::new(),
            recent: Vec::new(),
        }
    }

    fn render(&self) -> String {
        let mut out = String::new();
        if !self.notes.is_empty() {
            out.push_str("## Your notes\n");
            for n in &self.notes {
                out.push_str(&format!("- {n}\n"));
            }
            out.push('\n');
        }
        if !self.findings.is_empty() {
            out.push_str("## Findings so far (from chunks you have read)\n");
            for f in &self.findings {
                out.push_str(&format!("{f}\n"));
            }
            out.push('\n');
        }
        if !self.recent.is_empty() {
            out.push_str("## Recent steps\n");
            let split = self.recent.len().saturating_sub(RECENT_IN_FULL);
            for (i, (op, obs)) in self.recent.iter().enumerate() {
                if i < split {
                    // Older steps collapse to a single line: enough for the planner
                    // to know it already tried something, not enough to re-read it.
                    let first = obs.lines().next().unwrap_or("");
                    out.push_str(&format!("{}. {op} → {first}\n", i + 1));
                } else {
                    out.push_str(&format!("{}. {op} →\n{obs}\n\n", i + 1));
                }
            }
        }
        out
    }
}

/// The planner's instructions. Written as a contract rather than a description: the
/// failure this guards against is a planner that answers from general knowledge
/// without reading anything, which on a long-context question is indistinguishable
/// from a real answer until someone checks a citation.
fn root_system(depth: usize, budget: &Budget) -> String {
    format!(
        "You are the planner of a Recursive Language Model. A large corpus is held \
outside your context as a variable. You cannot see it and you will never be shown \
it. You interact with it ONLY by emitting one operator per turn.\n\
\n\
Reply with EXACTLY ONE JSON object and nothing else:\n\
  {{\"op\":\"outline\",\"source\":\"<optional substring filter>\"}}\n\
  {{\"op\":\"grep\",\"pattern\":\"<regex>\",\"max\":40,\"context\":1}}\n\
  {{\"op\":\"peek\",\"chunk\":<id>,\"start\":0,\"len\":2000}}\n\
  {{\"op\":\"map\",\"prompt\":\"<question asked of each chunk>\",\"chunks\":[<ids>]}}\n\
  {{\"op\":\"recurse\",\"query\":\"<sub-question>\",\"chunks\":[<ids>]}}\n\
  {{\"op\":\"note\",\"text\":\"<something you must not forget>\"}}\n\
  {{\"op\":\"final\",\"answer\":\"<your answer>\",\"cites\":[<chunk ids>]}}\n\
\n\
How to work:\n\
- `grep` first to locate, then `map` to read. `map` runs a cheap model over each \
chunk in parallel and folds the answers back to you — it is how you read a hundred \
chunks without seeing them.\n\
- `peek` is for checking an exact wording, not for reading the corpus. Observations \
are truncated; peeking your way through a document will not work and is not the \
mechanism.\n\
- `recurse` when a sub-question deserves its own plan over a subset.\n\
- You are rebuilt from your notes and findings every turn. There is no conversation \
history. If something matters, `note` it.\n\
- Answer ONLY from what the operators returned. If the corpus does not settle the \
question, say so in `final` — that is a correct answer. Never fill a gap from your \
own knowledge of the subject; you were not given this corpus to be asked what you \
already think.\n\
- Cite the chunk ids your answer rests on.\n\
\n\
You are at recursion depth {depth} of {}. You have {} planning steps.",
        budget.max_depth, budget.max_steps
    )
}

/// The reader's instructions for one `map` leaf. Short on purpose: this prompt is
/// paid for once per chunk, and it is the app's dominant cost at scale.
const LEAF_SYSTEM: &str = "You are reading ONE excerpt of a much larger corpus. \
Answer the question using only this excerpt. Be brief and concrete, and quote the \
decisive words. If this excerpt contains nothing that bears on the question, reply \
with exactly: NOTHING";

/// The token a leaf uses to say an excerpt was irrelevant. Filtered out of the fold,
/// so the planner sees only chunks that actually said something.
const LEAF_NOTHING: &str = "NOTHING";

/// Run one query against one context.
///
/// Boxed because it is genuinely recursive: `recurse` calls back into it, and an
/// `async fn` cannot name its own future without indirection.
pub fn run(
    ctx: Arc<ContextRecord>,
    query: String,
    host: Arc<Host>,
    budget: Budget,
    depth: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = RunRecord> + Send>> {
    Box::pin(async move { run_inner(ctx, query, host, budget, depth).await })
}

async fn run_inner(
    ctx: Arc<ContextRecord>,
    query: String,
    host: Arc<Host>,
    budget: Budget,
    depth: usize,
) -> RunRecord {
    let started = Instant::now();
    let calls_at_start = host.calls();
    let chars_at_start = host.prompt_chars();
    let mut rec = RunRecord {
        id: uuid::Uuid::new_v4().to_string(),
        context_id: ctx.id.clone(),
        query: query.clone(),
        status: STATUS_BUDGET.to_owned(),
        answer: String::new(),
        cites: Vec::new(),
        trace: Vec::new(),
        started_at: chrono::Utc::now().to_rfc3339(),
        elapsed_ms: 0,
        model_calls: 0,
        prompt_chars: 0,
        budget,
    };

    let mut work = Working::new();
    let system = root_system(depth, &budget);
    let header = format!(
        "# Question\n{query}\n\n# Corpus\n{}\n\n## Outline\n{}\n",
        ctx.stats_line(),
        // The outline is the planner's map. On a very large corpus it is itself the
        // biggest thing in the prompt, so it is capped like any other observation —
        // and the planner is told to filter it rather than assume it saw everything.
        Observation::new(ctx.outline()).text
    );

    for step in 0..budget.max_steps {
        if started.elapsed().as_secs() >= budget.wall_secs {
            break;
        }
        if host.calls() - calls_at_start >= budget.max_model_calls as u64 {
            break;
        }

        let prompt = format!("{header}\n{}\n# Your next operator\n", work.render());
        let step_started = Instant::now();
        let calls_before = host.calls();

        let reply = match host.complete(Tier::Root, &system, &prompt).await {
            Ok(r) => r,
            Err(e) => {
                // A provider error is terminal for the run: retrying the same prompt
                // against the same dead provider just spends the budget.
                rec.status = STATUS_ERROR.to_owned();
                rec.answer = format!("The planner could not reach a model: {e}");
                push_step(
                    &mut rec,
                    depth,
                    "plan",
                    serde_json::Value::Null,
                    String::new(),
                    false,
                    (host.calls() - calls_before) as usize,
                    step_started,
                    Some(e.to_string()),
                );
                break;
            }
        };

        let parsed = extract_json(&reply)
            .ok_or_else(|| {
                anyhow::anyhow!("no JSON operator in the reply. Reply with one JSON object.")
            })
            .and_then(|v| ops::parse_op(&v))
            .and_then(|op| op.validate(&ctx).map(|()| op));

        let op = match parsed {
            Ok(op) => op,
            Err(e) => {
                // Feed the error back as an observation. A planner that mistyped an
                // op usually fixes it next turn; a run that died on the first typo
                // would be maddening and would waste the corpus load.
                let msg = e.to_string();
                work.recent.push(("invalid".to_owned(), msg.clone()));
                push_step(
                    &mut rec,
                    depth,
                    "invalid",
                    serde_json::json!({ "reply": truncate(&reply, 400) }),
                    msg.clone(),
                    false,
                    (host.calls() - calls_before) as usize,
                    step_started,
                    Some(msg),
                );
                continue;
            }
        };

        let args = serde_json::to_value(&op).unwrap_or(serde_json::Value::Null);
        let label = op.label().to_owned();

        if let Op::Final { answer, cites } = op {
            rec.status = STATUS_OK.to_owned();
            rec.answer = answer.clone();
            rec.cites = resolve_cites(&ctx, &cites);
            push_step(
                &mut rec,
                depth,
                &label,
                args,
                truncate(&answer, 800),
                false,
                (host.calls() - calls_before) as usize,
                step_started,
                None,
            );
            break;
        }

        let obs = execute(&ctx, &host, &op, budget, depth, &mut work, &mut rec, step).await;
        work.recent.push((label.clone(), obs.text.clone()));
        push_step(
            &mut rec,
            depth,
            &label,
            args,
            obs.text.clone(),
            obs.truncated,
            (host.calls() - calls_before) as usize,
            step_started,
            None,
        );
    }

    // Out of steps but not out of calls: give the planner one turn to answer from
    // what it has. Better than reporting nothing, and marked in the trace so nobody
    // mistakes a forced answer for a planned one.
    if rec.status == STATUS_BUDGET
        && host.calls() - calls_at_start < budget.max_model_calls as u64
        && started.elapsed().as_secs() < budget.wall_secs
    {
        let forced = format!(
            "{header}\n{}\n# Out of steps\nAnswer now from your notes and findings alone. \
             State plainly what you could not establish. Reply with the `final` operator.\n",
            work.render()
        );
        let step_started = Instant::now();
        let calls_before = host.calls();
        if let Ok(reply) = host.complete(Tier::Root, &system, &forced).await {
            if let Some(Op::Final { answer, cites }) =
                extract_json(&reply).and_then(|v| ops::parse_op(&v).ok())
            {
                rec.answer = answer.clone();
                rec.cites = resolve_cites(&ctx, &cites);
                push_step(
                    &mut rec,
                    depth,
                    "final(forced)",
                    serde_json::json!({ "reason": "step budget exhausted" }),
                    truncate(&answer, 800),
                    false,
                    (host.calls() - calls_before) as usize,
                    step_started,
                    None,
                );
            }
        }
    }

    if rec.answer.is_empty() {
        rec.answer = if work.findings.is_empty() {
            "No answer: the run ended before anything was read from the corpus.".to_owned()
        } else {
            format!(
                "No answer was reached within the budget. What had been found:\n{}",
                work.findings.join("\n")
            )
        };
    }

    rec.elapsed_ms = started.elapsed().as_millis() as u64;
    rec.model_calls = (host.calls() - calls_at_start) as usize;
    rec.prompt_chars = host.prompt_chars() - chars_at_start;
    rec
}

/// Execute one non-`final` operator.
#[allow(clippy::too_many_arguments)]
async fn execute(
    ctx: &Arc<ContextRecord>,
    host: &Arc<Host>,
    op: &Op,
    budget: Budget,
    depth: usize,
    work: &mut Working,
    rec: &mut RunRecord,
    step: usize,
) -> Observation {
    match op {
        Op::Outline { source } => ops::outline(ctx, source.as_deref()),
        Op::Peek { chunk, start, len } => ops::peek(ctx, *chunk, *start, *len),
        Op::Grep {
            pattern,
            max,
            context,
        } => ops::grep(ctx, pattern, *max, *context),
        Op::Note { text } => {
            if work.notes.len() < MAX_NOTES {
                work.notes.push(text.clone());
                Observation::new("noted")
            } else {
                Observation::new(format!(
                    "note dropped: already holding {MAX_NOTES} notes. Answer, or consolidate."
                ))
            }
        }
        Op::Map { prompt, chunks } => map_chunks(ctx, host, prompt, chunks, work).await,
        Op::Recurse { query, chunks } => {
            if depth + 1 > budget.max_depth {
                return Observation::new(format!(
                    "recursion refused: already at depth {depth} of {}. Use map instead.",
                    budget.max_depth
                ));
            }
            recurse(ctx, host, query, chunks, budget, depth, rec, step).await
        }
        Op::Final { .. } => unreachable!("final is handled by the caller"),
    }
}

/// `map` — one cheap leaf call per chunk, in parallel, folded.
async fn map_chunks(
    ctx: &Arc<ContextRecord>,
    host: &Arc<Host>,
    prompt: &str,
    chunks: &[usize],
    work: &mut Working,
) -> Observation {
    let mut tasks = Vec::new();
    for id in chunks {
        let Some(chunk) = ctx.chunk(*id) else {
            continue;
        };
        let host = host.clone();
        let prompt = prompt.to_owned();
        let text = chunk.text.clone();
        let id = *id;
        let source = chunk.source.clone();
        let (start, end) = (chunk.start_line, chunk.end_line);
        tasks.push(tokio::spawn(async move {
            let user = format!("# Question\n{prompt}\n\n# Excerpt\n{text}");
            let answer = host.complete(Tier::Leaf, LEAF_SYSTEM, &user).await;
            (id, source, start, end, answer)
        }));
    }

    let mut lines = Vec::new();
    let mut read = 0usize;
    let mut errors = 0usize;
    for task in tasks {
        let Ok((id, source, start, end, answer)) = task.await else {
            errors += 1;
            continue;
        };
        read += 1;
        match answer {
            Ok(text) => {
                let t = text.trim();
                // A leaf that found nothing is dropped, not reported. Keeping the
                // NOTHINGs would fill the fold with noise and push the real findings
                // past the observation cap — the fold's whole job is to be small.
                if t.is_empty() || t.eq_ignore_ascii_case(LEAF_NOTHING) {
                    continue;
                }
                lines.push(format!("[{id}] {source}:{start}-{end} — {t}"));
            }
            Err(_) => errors += 1,
        }
    }

    let relevant = lines.len();
    for line in &lines {
        if work.findings.len() < MAX_FINDINGS {
            work.findings.push(line.clone());
        }
    }

    if lines.is_empty() {
        return Observation::new(format!(
            "read {read} chunk(s); none of them bore on that question{}",
            if errors > 0 {
                format!(" ({errors} failed)")
            } else {
                String::new()
            }
        ));
    }
    Observation::new(format!(
        "read {read} chunk(s), {relevant} relevant{}:\n{}",
        if errors > 0 {
            format!(", {errors} failed")
        } else {
            String::new()
        },
        lines.join("\n")
    ))
}

/// `recurse` — a nested run over a sub-corpus built from the named chunks.
#[allow(clippy::too_many_arguments)]
async fn recurse(
    ctx: &Arc<ContextRecord>,
    host: &Arc<Host>,
    query: &str,
    chunks: &[usize],
    budget: Budget,
    depth: usize,
    rec: &mut RunRecord,
    step: usize,
) -> Observation {
    // Each selected chunk becomes a document that remembers where it really came
    // from, so citations produced inside the nested run still point at the real
    // file and the real lines. See `Document::excerpt`.
    let docs: Vec<Document> = chunks
        .iter()
        .filter_map(|id| ctx.chunk(*id))
        .map(|c| Document::excerpt(c.source.clone(), c.text.clone(), c.start_line))
        .collect();
    if docs.is_empty() {
        return Observation::new("no readable chunks in that selection");
    }

    let sub = match ContextRecord::ephemeral(&format!("{} ▸ recursion", ctx.name), docs) {
        Ok(s) => Arc::new(s),
        Err(e) => return Observation::new(format!("could not build the sub-context: {e}")),
    };

    // The nested run gets its own step allowance but shares the parent's call and
    // wall-clock ceiling by construction: those are enforced against the same `Host`
    // counter and the sub-run's own clock is shorter.
    let sub_budget = Budget {
        max_steps: budget.max_steps.min(8),
        ..budget
    };
    let nested = run(
        sub.clone(),
        query.to_owned(),
        host.clone(),
        sub_budget,
        depth + 1,
    )
    .await;

    // Splice the nested trace under this step so the companion can draw one tree
    // rather than a list of unrelated runs.
    for mut s in nested.trace {
        s.parent = Some(step);
        s.index = rec.trace.len();
        rec.trace.push(s);
    }

    Observation::new(format!(
        "recursion over {} chunk(s) ({}): {}",
        chunks.len(),
        nested.status,
        nested.answer
    ))
}

fn resolve_cites(ctx: &ContextRecord, ids: &[usize]) -> Vec<Citation> {
    ids.iter()
        .filter_map(|id| ctx.chunk(*id))
        .map(|c| Citation {
            chunk: c.id,
            source: c.source.clone(),
            start_line: c.start_line,
            end_line: c.end_line,
            label: c.label.clone(),
        })
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn push_step(
    rec: &mut RunRecord,
    depth: usize,
    op: &str,
    args: serde_json::Value,
    observation: String,
    truncated: bool,
    model_calls: usize,
    started: Instant,
    error: Option<String>,
) {
    rec.trace.push(TraceStep {
        index: rec.trace.len(),
        depth,
        parent: None,
        op: op.to_owned(),
        args,
        observation,
        truncated,
        model_calls,
        elapsed_ms: started.elapsed().as_millis() as u64,
        error,
    });
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        return s.to_owned();
    }
    s.chars().take(n).collect::<String>() + "…"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_caller_cannot_ask_for_an_unbounded_budget() {
        // The request body is untrusted input. Without the clamp one query could
        // spend the node's entire provider quota.
        let wild = Budget {
            max_steps: 100_000,
            max_model_calls: 1_000_000,
            max_depth: 99,
            wall_secs: 86_400,
        }
        .sanitize();
        assert_eq!(wild.max_steps, 40);
        assert_eq!(wild.max_model_calls, 2_000);
        assert_eq!(wild.max_depth, 3);
        assert_eq!(wall_hours(wild.wall_secs), 1);
    }

    fn wall_hours(secs: u64) -> u64 {
        secs / 3_600
    }

    #[test]
    fn a_zero_budget_still_clamps_to_something_runnable() {
        let tiny = Budget {
            max_steps: 0,
            max_model_calls: 0,
            max_depth: 0,
            wall_secs: 0,
        }
        .sanitize();
        assert_eq!(tiny.max_steps, 1);
        assert_eq!(tiny.max_model_calls, 1);
        assert_eq!(tiny.max_depth, 0, "depth 0 is legal: no recursion at all");
        assert_eq!(tiny.wall_secs, 10);
    }

    #[test]
    fn older_observations_collapse_so_the_prompt_cannot_grow_without_limit() {
        // The property the whole design rests on: a planner running many steps must
        // not accumulate every observation, or it ends up holding the corpus.
        let mut work = Working::new();
        for i in 0..20 {
            work.recent.push((
                format!("grep{i}"),
                format!("line one of {i}\n{}", "x".repeat(3_000)),
            ));
        }
        let rendered = work.render();
        let full_size = 20 * 3_000;
        assert!(
            rendered.chars().count() < full_size / 3,
            "expected collapse, got {} chars",
            rendered.chars().count()
        );
        assert!(
            rendered.contains("grep19"),
            "the most recent step stays in full"
        );
        assert!(
            rendered.contains("grep0"),
            "older steps stay listed, one line each"
        );
    }

    #[test]
    fn notes_are_never_collapsed_because_they_are_the_planners_only_memory() {
        let mut work = Working::new();
        work.notes.push("the refund window is 30 days".into());
        for i in 0..20 {
            work.recent.push((format!("op{i}"), "x".repeat(3_000)));
        }
        assert!(work.render().contains("the refund window is 30 days"));
    }

    #[test]
    fn the_planner_is_told_not_to_answer_from_its_own_knowledge() {
        // The failure this guards is the one nobody catches by eye: a fluent answer
        // to a long-context question that was never read from the corpus.
        let sys = root_system(0, &Budget::default());
        assert!(sys.contains("Answer ONLY from what the operators returned"));
        assert!(sys.contains("Never fill a gap from your own knowledge"));
    }

    #[test]
    fn the_leaf_prompt_gives_an_explicit_way_to_say_nothing() {
        // Without this the fold fills with "this excerpt does not mention…" for every
        // irrelevant chunk, and the real findings are pushed past the cap.
        assert!(LEAF_SYSTEM.contains(LEAF_NOTHING));
    }
}
