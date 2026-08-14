//! The context store — where the "variable" in *context-as-a-variable* actually
//! lives, plus the run journal the companion renders.
//!
//! A **context** is immutable once built. That is not tidiness: chunk ids are the
//! coordinate system every operator, every citation and every replayed trace is
//! written in, so a context that could grow would silently renumber answers already
//! given against it. Adding documents produces a NEW context that records its
//! parent, and the old one stays valid.
//!
//! ## Reading files is gated on a root allowlist
//!
//! [`Roots::load`] is the only path in this crate that touches the filesystem for
//! user-named files, and it refuses any path that does not canonicalize to a
//! descendant of a configured root. Canonicalisation happens BEFORE the check, so a
//! symlink pointing out of the root is rejected on where it lands, not where it
//! sits — the `..`-in-a-blocklist mistake ([[social-outpost-app]]'s `%2e%2e`
//! bypass) is not available here because nothing is pattern-matched at all.
//!
//! Roots come from `RYU_RLM_ROOTS` (a `:`-separated list on unix, `;` on Windows)
//! and default to the process's current directory when unset. There is deliberately
//! no "allow everything" value: a context is something a model then reads and
//! summarises, so an unbounded root is an unbounded exfiltration surface, and the
//! app is opt-in precisely because the operator should choose that boundary.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context as _, Result};
use serde::{Deserialize, Serialize};

use crate::chunk::{chunk_documents, Chunk, Document, DEFAULT_TARGET_CHARS};
use crate::engine::RunRecord;

/// Ceiling on one context's total characters. A context bigger than this is not a
/// long-context problem any more, it is a corpus that wants an index — and letting
/// it through would mean an `outline` with tens of thousands of rows, which is
/// itself too big for the root model's prompt. Refusing loudly beats degrading.
pub const MAX_CONTEXT_CHARS: usize = 40_000_000;

/// Ceiling on files read in one `from-paths` call, so a glob that matched a
/// `node_modules` cannot turn into a hundred thousand `open()` calls.
pub const MAX_FILES: usize = 5_000;

/// Per-file ceiling. Larger files are truncated with an explicit marker rather than
/// skipped silently — a context that quietly omitted a file would produce a
/// confident "the corpus does not mention X".
pub const MAX_FILE_CHARS: usize = 4_000_000;

/// `$RYU_DIR/rlm`, honoring Core's `RYU_DIR`-env-first paths rule so a
/// `RYU_PROFILE=dev` stack keeps its data separate from a release install.
pub fn data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("RYU_DIR") {
        if !dir.is_empty() {
            return PathBuf::from(dir).join("rlm");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".ryu")
        .join("rlm")
}

/// A built, immutable context.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ContextRecord {
    pub id: String,
    pub name: String,
    /// The context this one was derived from, when it was built by adding to or
    /// narrowing an existing one. Lets the companion show provenance instead of a
    /// flat list of near-identical corpora.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<String>,
    pub created_at: String,
    pub sources: Vec<String>,
    pub total_chars: usize,
    pub chunks: Vec<Chunk>,
}

impl ContextRecord {
    /// The outline the root model plans over: one line per chunk, id + source +
    /// line span + label. This is the ONLY view of the corpus the root model gets
    /// for free, and it is bounded by chunk COUNT, not by corpus size — which is
    /// the property that keeps a 40 MB context and a 40 KB one costing the root
    /// model the same order of prompt.
    pub fn outline(&self) -> String {
        let mut out = String::new();
        for c in &self.chunks {
            out.push_str(&format!(
                "[{}] {}:{}-{} ({} chars) {}\n",
                c.id,
                c.source,
                c.start_line,
                c.end_line,
                c.chars(),
                c.label
            ));
        }
        out
    }

    /// A compact header for prompts: what this corpus IS, without any of it.
    pub fn stats_line(&self) -> String {
        format!(
            "{} chunks across {} source(s), {} characters total (~{} tokens). \
             You have NOT been shown any of it.",
            self.chunks.len(),
            self.sources.len(),
            self.total_chars,
            self.total_chars / 4
        )
    }

    pub fn chunk(&self, id: usize) -> Option<&Chunk> {
        self.chunks.get(id)
    }

    /// Build a context that is never written to disk.
    ///
    /// This is what a `recurse` step runs against. It is deliberately NOT stored:
    /// a nested sub-corpus is an implementation detail of one step, and persisting
    /// one per recursion would bury the user's real contexts under machine-generated
    /// ones within a day of use. The nested run's trace is spliced into the parent's
    /// record instead, which is the part worth keeping.
    ///
    /// Chunks are re-cut finer than the parent's: the sub-corpus is small, and a
    /// nested planner with only two chunks to choose between cannot do anything a
    /// plain `map` would not have done.
    pub fn ephemeral(name: &str, docs: Vec<Document>) -> Result<ContextRecord> {
        if docs.is_empty() {
            bail!("a sub-context needs at least one document");
        }
        let total: usize = docs.iter().map(|d| d.text.chars().count()).sum();
        let chunks = chunk_documents(&docs, DEFAULT_TARGET_CHARS / 3);
        if chunks.is_empty() {
            bail!("the selected chunks were empty");
        }
        let mut sources: Vec<String> = docs.iter().map(|d| d.source.clone()).collect();
        sources.dedup();
        Ok(ContextRecord {
            id: format!("ephemeral:{}", uuid::Uuid::new_v4()),
            name: name.to_owned(),
            parent: None,
            created_at: chrono::Utc::now().to_rfc3339(),
            sources,
            total_chars: total,
            chunks,
        })
    }
}

/// The configured filesystem boundary for `from-paths` loads.
#[derive(Clone, Debug)]
pub struct Roots(Vec<PathBuf>);

impl Roots {
    /// From `RYU_RLM_ROOTS`, else the user's home directory.
    ///
    /// Home rather than the process's working directory, which is Core's and not the
    /// user's: a default of cwd would put every document the user actually wants to
    /// ask about outside the boundary, and the feature would read as broken. Home is
    /// paired with the hidden-entry rule in [`Roots::load`] — a recursive load never
    /// wanders into `~/.ssh` or `~/.aws` — and narrowed by setting `RYU_RLM_ROOTS`.
    pub fn from_env() -> Roots {
        let raw = std::env::var("RYU_RLM_ROOTS").unwrap_or_default();
        let parts: Vec<PathBuf> = raw
            .split(if cfg!(windows) { ';' } else { ':' })
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect();
        let chosen = if parts.is_empty() {
            dirs::home_dir().into_iter().collect()
        } else {
            parts
        };
        // Canonicalise the roots too. A root given as a symlink would otherwise
        // never match a canonicalised candidate, and the failure would look like a
        // permissions bug rather than a configuration one.
        Roots(chosen.iter().filter_map(|p| p.canonicalize().ok()).collect())
    }

    pub fn display(&self) -> Vec<String> {
        self.0.iter().map(|p| p.display().to_string()).collect()
    }

    /// Resolve one caller-named path, or refuse. Returns the canonical path.
    pub fn resolve(&self, candidate: &Path) -> Result<PathBuf> {
        let full = candidate
            .canonicalize()
            .with_context(|| format!("cannot read {}", candidate.display()))?;
        if self.0.iter().any(|r| full.starts_with(r)) {
            return Ok(full);
        }
        bail!(
            "{} is outside every configured context root ({}). Set RYU_RLM_ROOTS to \
             widen the boundary deliberately.",
            full.display(),
            self.display().join(", ")
        )
    }

    /// Read a set of paths into documents. A directory expands to its files,
    /// recursively; anything unreadable as UTF-8 is skipped with a note rather than
    /// failing the whole load, because one binary file in a directory should not
    /// cost the caller the other four hundred.
    pub fn load(&self, paths: &[String]) -> Result<(Vec<Document>, Vec<String>)> {
        let mut docs = Vec::new();
        let mut skipped = Vec::new();
        let mut queue: Vec<PathBuf> = Vec::new();
        for p in paths {
            match self.resolve(Path::new(p)) {
                Ok(full) => queue.push(full),
                Err(e) => skipped.push(e.to_string()),
            }
        }

        while let Some(path) = queue.pop() {
            if docs.len() >= MAX_FILES {
                skipped.push(format!("stopped at the {MAX_FILES}-file ceiling"));
                break;
            }
            let meta = match std::fs::metadata(&path) {
                Ok(m) => m,
                Err(e) => {
                    skipped.push(format!("{}: {e}", path.display()));
                    continue;
                }
            };
            if meta.is_dir() {
                let entries = match std::fs::read_dir(&path) {
                    Ok(e) => e,
                    Err(e) => {
                        skipped.push(format!("{}: {e}", path.display()));
                        continue;
                    }
                };
                for entry in entries.flatten() {
                    let child = entry.path();
                    if skip_when_walking(&child) {
                        continue;
                    }
                    // Re-check every descendant rather than trusting the parent:
                    // a symlink inside an allowed directory can still point out of
                    // it, and that is exactly the case the check exists for.
                    match self.resolve(&child) {
                        Ok(full) => queue.push(full),
                        Err(e) => skipped.push(e.to_string()),
                    }
                }
                continue;
            }
            match std::fs::read_to_string(&path) {
                Ok(mut text) => {
                    if text.chars().count() > MAX_FILE_CHARS {
                        text = text.chars().take(MAX_FILE_CHARS).collect::<String>()
                            + "\n\n[truncated by ryu-rlm at the per-file ceiling]";
                    }
                    docs.push(Document::new(path.display().to_string(), text));
                }
                Err(e) => skipped.push(format!("{}: {e}", path.display())),
            }
        }

        // Stable order: the queue above is a stack, so without this the chunk ids
        // for the same directory would differ run to run and no trace would replay.
        docs.sort_by(|a, b| a.source.cmp(&b.source));
        Ok((docs, skipped))
    }
}

/// Entries a *recursive walk* steps over. Two different reasons, deliberately kept
/// in one place so neither is mistaken for the other:
///
/// - **Hidden entries** (`.ssh`, `.aws`, `.gnupg`, `.git`) — this is the safety half.
///   The default root is the user's home directory, and a recursive load that swept
///   their private keys into a corpus a model then summarises would be an
///   exfiltration path opened by a convenience default. Skipping dotted entries
///   while walking is the same rule `rg` and `fd` use, so it will not surprise
///   anyone. It applies to *walking* only: a path the user names explicitly is
///   loaded, because at that point they have said what they mean.
/// - **`node_modules`** — this is only a size heuristic. One of these exhausts the
///   file ceiling on its own and yields a corpus of vendored code nobody asked
///   about. It is not a security control and must not be read as one.
fn skip_when_walking(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    name.starts_with('.') || name == "node_modules"
}

/// Contexts and runs, in memory with a JSON mirror on disk.
pub struct Store {
    dir: PathBuf,
    contexts: Mutex<BTreeMap<String, Arc<ContextRecord>>>,
    runs: Mutex<BTreeMap<String, RunRecord>>,
}

impl Store {
    pub fn open(dir: PathBuf) -> Result<Store> {
        std::fs::create_dir_all(dir.join("contexts"))?;
        std::fs::create_dir_all(dir.join("runs"))?;
        let store = Store {
            dir,
            contexts: Mutex::new(BTreeMap::new()),
            runs: Mutex::new(BTreeMap::new()),
        };
        store.rehydrate();
        Ok(store)
    }

    /// Read back what a previous process wrote. Best-effort per file: one corrupt
    /// record must not cost the user every other context they built.
    fn rehydrate(&self) {
        for (sub, is_ctx) in [("contexts", true), ("runs", false)] {
            let Ok(entries) = std::fs::read_dir(self.dir.join(sub)) else {
                continue;
            };
            for entry in entries.flatten() {
                let Ok(raw) = std::fs::read_to_string(entry.path()) else {
                    continue;
                };
                if is_ctx {
                    if let Ok(rec) = serde_json::from_str::<ContextRecord>(&raw) {
                        self.contexts
                            .lock()
                            .unwrap()
                            .insert(rec.id.clone(), Arc::new(rec));
                    }
                } else if let Ok(rec) = serde_json::from_str::<RunRecord>(&raw) {
                    self.runs.lock().unwrap().insert(rec.id.clone(), rec);
                }
            }
        }
    }

    /// Build and persist a context from documents.
    pub fn create(
        &self,
        name: &str,
        docs: Vec<Document>,
        parent: Option<String>,
        target_chars: Option<usize>,
    ) -> Result<Arc<ContextRecord>> {
        if docs.is_empty() {
            bail!("a context needs at least one document");
        }
        let total: usize = docs.iter().map(|d| d.text.chars().count()).sum();
        if total > MAX_CONTEXT_CHARS {
            bail!(
                "context is {total} characters, over the {MAX_CONTEXT_CHARS} ceiling; \
                 split it or narrow the paths"
            );
        }
        let chunks = chunk_documents(&docs, target_chars.unwrap_or(DEFAULT_TARGET_CHARS));
        if chunks.is_empty() {
            bail!("every document was empty after chunking");
        }
        let mut sources: Vec<String> = docs.iter().map(|d| d.source.clone()).collect();
        sources.dedup();
        let rec = ContextRecord {
            id: uuid::Uuid::new_v4().to_string(),
            name: name.to_owned(),
            parent,
            created_at: chrono::Utc::now().to_rfc3339(),
            sources,
            total_chars: total,
            chunks,
        };
        self.write_json(&self.dir.join("contexts").join(format!("{}.json", rec.id)), &rec)?;
        let arc = Arc::new(rec);
        self.contexts
            .lock()
            .unwrap()
            .insert(arc.id.clone(), arc.clone());
        Ok(arc)
    }

    pub fn get(&self, id: &str) -> Option<Arc<ContextRecord>> {
        self.contexts.lock().unwrap().get(id).cloned()
    }

    /// Contexts newest-first, without their chunk bodies — the list view must not
    /// serialise a 40 MB corpus to render a row.
    pub fn list(&self) -> Vec<serde_json::Value> {
        let mut out: Vec<serde_json::Value> = self
            .contexts
            .lock()
            .unwrap()
            .values()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "name": c.name,
                    "parent": c.parent,
                    "created_at": c.created_at,
                    "sources": c.sources,
                    "total_chars": c.total_chars,
                    "chunks": c.chunks.len(),
                })
            })
            .collect();
        out.sort_by(|a, b| b["created_at"].as_str().cmp(&a["created_at"].as_str()));
        out
    }

    pub fn delete(&self, id: &str) -> bool {
        let removed = self.contexts.lock().unwrap().remove(id).is_some();
        if removed {
            let _ = std::fs::remove_file(self.dir.join("contexts").join(format!("{id}.json")));
        }
        removed
    }

    /// Drop every context and run. Backs the manifest's `data_categories` entry, so
    /// the Danger Zone's promise is a real deletion and not a UI-only one.
    pub fn purge(&self) -> usize {
        let mut ctx = self.contexts.lock().unwrap();
        let n = ctx.len();
        ctx.clear();
        self.runs.lock().unwrap().clear();
        for sub in ["contexts", "runs"] {
            if let Ok(entries) = std::fs::read_dir(self.dir.join(sub)) {
                for entry in entries.flatten() {
                    let _ = std::fs::remove_file(entry.path());
                }
            }
        }
        n
    }

    pub fn save_run(&self, rec: &RunRecord) {
        let _ = self.write_json(&self.dir.join("runs").join(format!("{}.json", rec.id)), rec);
        self.runs.lock().unwrap().insert(rec.id.clone(), rec.clone());
    }

    pub fn run(&self, id: &str) -> Option<RunRecord> {
        self.runs.lock().unwrap().get(id).cloned()
    }

    /// Runs newest-first, headers only.
    pub fn list_runs(&self, context_id: Option<&str>) -> Vec<serde_json::Value> {
        let mut out: Vec<serde_json::Value> = self
            .runs
            .lock()
            .unwrap()
            .values()
            .filter(|r| context_id.is_none_or(|c| r.context_id == c))
            .map(|r| {
                serde_json::json!({
                    "id": r.id,
                    "context_id": r.context_id,
                    "query": r.query,
                    "status": r.status,
                    "started_at": r.started_at,
                    "elapsed_ms": r.elapsed_ms,
                    "model_calls": r.model_calls,
                    "steps": r.trace.len(),
                })
            })
            .collect();
        out.sort_by(|a, b| b["started_at"].as_str().cmp(&a["started_at"].as_str()));
        out
    }

    fn write_json<T: Serialize>(&self, path: &Path, value: &T) -> Result<()> {
        let body = serde_json::to_vec_pretty(value)?;
        // Write-then-rename: a process killed mid-write must not leave a half JSON
        // file that `rehydrate` then silently drops on the next boot.
        let tmp = path.with_extension("json.tmp");
        std::fs::write(&tmp, &body).map_err(|e| anyhow!("{}: {e}", tmp.display()))?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("ryu-rlm-test-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn doc(source: &str, text: &str) -> Document {
        Document::new(source, text)
    }

    #[test]
    fn a_path_outside_every_root_is_refused() {
        let root = tmpdir("root");
        let outside = tmpdir("outside");
        std::fs::write(outside.join("secret.txt"), "shh").unwrap();
        let roots = Roots(vec![root.canonicalize().unwrap()]);
        let err = roots
            .resolve(&outside.join("secret.txt"))
            .expect_err("a path outside the root must be refused");
        assert!(err.to_string().contains("outside every configured context root"));
    }

    #[test]
    fn a_symlink_is_judged_on_where_it_lands_not_where_it_sits() {
        // The whole reason resolution happens before the prefix check. A blocklist
        // of `..` would pass this and leak the file.
        let root = tmpdir("symroot");
        let outside = tmpdir("symoutside");
        let secret = outside.join("secret.txt");
        std::fs::write(&secret, "shh").unwrap();
        let link = root.join("innocent.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&secret, &link).unwrap();
        #[cfg(not(unix))]
        return;

        let roots = Roots(vec![root.canonicalize().unwrap()]);
        assert!(
            roots.resolve(&link).is_err(),
            "a symlink out of the root must be refused on its target"
        );
    }

    #[test]
    fn a_path_inside_a_root_resolves() {
        let root = tmpdir("okroot");
        let f = root.join("notes.md");
        std::fs::write(&f, "hello").unwrap();
        let roots = Roots(vec![root.canonicalize().unwrap()]);
        assert!(roots.resolve(&f).is_ok());
    }

    #[test]
    fn directory_loads_are_ordered_so_chunk_ids_are_reproducible() {
        let root = tmpdir("order");
        for name in ["c.txt", "a.txt", "b.txt"] {
            std::fs::write(root.join(name), format!("body of {name}")).unwrap();
        }
        let roots = Roots(vec![root.canonicalize().unwrap()]);
        let (docs, _) = roots.load(&[root.display().to_string()]).unwrap();
        let names: Vec<String> = docs
            .iter()
            .map(|d| Path::new(&d.source).file_name().unwrap().to_string_lossy().into())
            .collect();
        assert_eq!(names, vec!["a.txt", "b.txt", "c.txt"]);
    }

    #[test]
    fn a_recursive_walk_steps_over_hidden_directories() {
        // The default root is the user's home, so a walk that descended into ~/.ssh
        // would sweep private keys into a corpus a model then reads and summarises.
        let root = tmpdir("walk");
        std::fs::create_dir_all(root.join(".ssh")).unwrap();
        std::fs::write(root.join(".ssh").join("id_rsa"), "PRIVATE KEY").unwrap();
        std::fs::create_dir_all(root.join("node_modules")).unwrap();
        std::fs::write(root.join("node_modules").join("dep.js"), "vendored").unwrap();
        std::fs::write(root.join("notes.md"), "real content").unwrap();

        let roots = Roots(vec![root.canonicalize().unwrap()]);
        let (docs, _) = roots.load(&[root.display().to_string()]).unwrap();
        let sources: Vec<&str> = docs.iter().map(|d| d.source.as_str()).collect();
        assert_eq!(sources.len(), 1, "got {sources:?}");
        assert!(sources[0].ends_with("notes.md"));
    }

    #[test]
    fn an_explicitly_named_hidden_file_is_still_loaded() {
        // The rule is about walking, not about forbidding. Someone who types the
        // path to their own dotfile has said what they mean.
        let root = tmpdir("explicit");
        std::fs::create_dir_all(root.join(".config")).unwrap();
        let f = root.join(".config").join("app.toml");
        std::fs::write(&f, "key = 1").unwrap();
        let roots = Roots(vec![root.canonicalize().unwrap()]);
        let (docs, _) = roots.load(&[f.display().to_string()]).unwrap();
        assert_eq!(docs.len(), 1);
    }

    #[test]
    fn a_context_survives_a_restart() {
        let dir = tmpdir("persist");
        let id = {
            let store = Store::open(dir.clone()).unwrap();
            store
                .create("notes", vec![doc("a.md", "alpha beta")], None, None)
                .unwrap()
                .id
                .clone()
        };
        let reopened = Store::open(dir).unwrap();
        assert!(
            reopened.get(&id).is_some(),
            "contexts must rehydrate; a lazy sidecar is stopped after idle_stop_secs"
        );
    }

    #[test]
    fn an_oversized_context_is_refused_rather_than_degraded() {
        let dir = tmpdir("toobig");
        let store = Store::open(dir).unwrap();
        let huge = "x".repeat(MAX_CONTEXT_CHARS + 1);
        let err = store.create("huge", vec![doc("big.txt", &huge)], None, None);
        assert!(err.is_err());
    }

    #[test]
    fn the_outline_grows_with_chunk_count_not_corpus_size() {
        let dir = tmpdir("outline");
        let store = Store::open(dir).unwrap();
        let body = "# Section\n".to_owned() + &"filler line\n".repeat(2_000);
        let ctx = store.create("doc", vec![doc("d.md", &body)], None, None).unwrap();
        let outline = ctx.outline();
        assert_eq!(outline.lines().count(), ctx.chunks.len());
        assert!(
            outline.chars().count() < ctx.total_chars / 4,
            "the outline must be far smaller than the corpus, or the root model is \
             reading the corpus after all"
        );
    }
}
