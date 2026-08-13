//! The HTTP surface Core proxies at `/api/rlm/*`.
//!
//! Every route here is behind the shared-secret bearer gate in `main.rs`, and every
//! path is separately declared in the manifest's `http.routes` allowlist. That
//! allowlist matches on **exact segment count**, so a route added here without its
//! manifest row is a hard 404 that reads like a router bug — the two must move
//! together.
//!
//! The surface splits in two, and the split is the app's whole shape:
//!
//! - **Corpus routes** (`outline`, `peek`, `grep`) cost nothing and need no model.
//!   They work on a node with no provider configured at all, which is what makes
//!   the companion useful for inspecting a context before spending anything on it.
//! - **`/query`** is the one route that runs the recursion engine, and the only one
//!   that needs `hook:side-model`. With no host callback it reports 503 rather than
//!   pretending — an app that answered a long-context question without reading the
//!   context would be worse than one that refused.

use std::sync::Arc;

use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::json;

use crate::chunk::Document;
use crate::engine::{self, Budget};
use crate::host::Host;
use crate::ops;
use crate::store::{Roots, Store};

pub struct Ctx {
    pub store: Store,
    pub host: Option<Arc<Host>>,
    pub roots: Roots,
    pub budget: Budget,
}

pub fn routes(ctx: Arc<Ctx>) -> Router {
    Router::new()
        .route("/contexts", get(list_contexts).post(create_context))
        .route("/contexts/from-paths", post(create_from_paths))
        .route("/contexts/:id", get(get_context).delete(delete_context))
        .route("/contexts/:id/outline", get(get_outline))
        .route("/contexts/:id/peek", post(post_peek))
        .route("/contexts/:id/grep", post(post_grep))
        .route("/query", post(post_query))
        .route("/runs", get(list_runs))
        .route("/runs/:id", get(get_run))
        .route("/purge", delete(post_purge))
        .with_state(ctx)
}

/// Advertised at `/capability` so a node can see what this sidecar offers — and,
/// more usefully, what it currently CANNOT do, which is how a missing
/// `hook:side-model` grant becomes visible before someone waits on a query.
pub fn capability_descriptor(has_host: bool, roots: &Roots) -> serde_json::Value {
    json!({
        "service": "ryu-rlm",
        "capability": "rlm.query",
        "model_calls_available": has_host,
        "context_roots": roots.display(),
        "operators": ["outline", "peek", "grep", "map", "recurse", "note", "final"],
        "limits": {
            "max_context_chars": crate::store::MAX_CONTEXT_CHARS,
            "max_files_per_load": crate::store::MAX_FILES,
            "max_observation_chars": ops::MAX_OBSERVATION_CHARS,
            "max_map_fanout": ops::MAX_MAP_FANOUT,
        },
    })
}

fn err(status: StatusCode, msg: impl std::fmt::Display) -> Response {
    (status, Json(json!({ "error": msg.to_string() }))).into_response()
}

#[derive(Deserialize)]
struct CreateBody {
    #[serde(default)]
    name: Option<String>,
    documents: Vec<DocumentBody>,
    #[serde(default)]
    parent: Option<String>,
    #[serde(default)]
    target_chars: Option<usize>,
}

#[derive(Deserialize)]
struct DocumentBody {
    source: String,
    text: String,
}

async fn create_context(State(ctx): State<Arc<Ctx>>, Json(body): Json<CreateBody>) -> Response {
    let docs: Vec<Document> = body
        .documents
        .into_iter()
        .map(|d| Document::new(d.source, d.text))
        .collect();
    match ctx.store.create(
        body.name.as_deref().unwrap_or("context"),
        docs,
        body.parent,
        body.target_chars,
    ) {
        Ok(rec) => Json(json!({
            "id": rec.id,
            "name": rec.name,
            "chunks": rec.chunks.len(),
            "total_chars": rec.total_chars,
            "sources": rec.sources,
        }))
        .into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, e),
    }
}

#[derive(Deserialize)]
struct PathsBody {
    #[serde(default)]
    name: Option<String>,
    paths: Vec<String>,
    #[serde(default)]
    target_chars: Option<usize>,
}

async fn create_from_paths(State(ctx): State<Arc<Ctx>>, Json(body): Json<PathsBody>) -> Response {
    let (docs, skipped) = match ctx.roots.load(&body.paths) {
        Ok(v) => v,
        Err(e) => return err(StatusCode::BAD_REQUEST, e),
    };
    if docs.is_empty() {
        // Report what was refused. A silent empty context here is the worst outcome:
        // every later answer would be a confident "the corpus does not mention that".
        return err(
            StatusCode::BAD_REQUEST,
            format!(
                "nothing readable was loaded. Refused or unreadable: {}",
                if skipped.is_empty() {
                    "(none reported)".to_owned()
                } else {
                    skipped.join("; ")
                }
            ),
        );
    }
    match ctx.store.create(
        body.name.as_deref().unwrap_or("files"),
        docs,
        None,
        body.target_chars,
    ) {
        Ok(rec) => Json(json!({
            "id": rec.id,
            "name": rec.name,
            "chunks": rec.chunks.len(),
            "total_chars": rec.total_chars,
            "sources": rec.sources,
            "skipped": skipped,
        }))
        .into_response(),
        Err(e) => err(StatusCode::BAD_REQUEST, e),
    }
}

async fn list_contexts(State(ctx): State<Arc<Ctx>>) -> Response {
    Json(json!({ "contexts": ctx.store.list() })).into_response()
}

async fn get_context(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    match ctx.store.get(&id) {
        // Chunk bodies are deliberately omitted: this is the metadata view, and a
        // 40 MB JSON response would be unusable in the companion anyway. Bodies come
        // from `peek`, one window at a time.
        Some(rec) => Json(json!({
            "id": rec.id,
            "name": rec.name,
            "parent": rec.parent,
            "created_at": rec.created_at,
            "sources": rec.sources,
            "total_chars": rec.total_chars,
            "chunks": rec.chunks.iter().map(|c| json!({
                "id": c.id,
                "source": c.source,
                "start_line": c.start_line,
                "end_line": c.end_line,
                "chars": c.chars(),
                "label": c.label,
            })).collect::<Vec<_>>(),
        }))
        .into_response(),
        None => err(StatusCode::NOT_FOUND, "no such context"),
    }
}

async fn delete_context(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    Json(json!({ "deleted": ctx.store.delete(&id) })).into_response()
}

async fn get_outline(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    match ctx.store.get(&id) {
        Some(rec) => Json(json!({ "outline": rec.outline(), "stats": rec.stats_line() }))
            .into_response(),
        None => err(StatusCode::NOT_FOUND, "no such context"),
    }
}

#[derive(Deserialize)]
struct PeekBody {
    chunk: usize,
    #[serde(default)]
    start: usize,
    #[serde(default)]
    len: Option<usize>,
}

async fn post_peek(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
    Json(body): Json<PeekBody>,
) -> Response {
    let Some(rec) = ctx.store.get(&id) else {
        return err(StatusCode::NOT_FOUND, "no such context");
    };
    let obs = ops::peek(&rec, body.chunk, body.start, body.len);
    Json(json!({ "text": obs.text, "truncated": obs.truncated })).into_response()
}

#[derive(Deserialize)]
struct GrepBody {
    pattern: String,
    #[serde(default)]
    max: Option<usize>,
    #[serde(default)]
    context: Option<usize>,
}

async fn post_grep(
    State(ctx): State<Arc<Ctx>>,
    Path(id): Path<String>,
    Json(body): Json<GrepBody>,
) -> Response {
    let Some(rec) = ctx.store.get(&id) else {
        return err(StatusCode::NOT_FOUND, "no such context");
    };
    let obs = ops::grep(&rec, &body.pattern, body.max, body.context);
    Json(json!({ "text": obs.text, "truncated": obs.truncated })).into_response()
}

#[derive(Deserialize)]
pub struct QueryBody {
    pub context_id: String,
    pub query: String,
    #[serde(default)]
    pub max_steps: Option<usize>,
    #[serde(default)]
    pub max_model_calls: Option<usize>,
    #[serde(default)]
    pub max_depth: Option<usize>,
    #[serde(default)]
    pub wall_secs: Option<u64>,
}

impl QueryBody {
    /// Overlay the caller's bounds on the node default, then clamp. Every field is
    /// optional so the common call is `{context_id, query}`.
    pub fn budget(&self, base: Budget) -> Budget {
        Budget {
            max_steps: self.max_steps.unwrap_or(base.max_steps),
            max_model_calls: self.max_model_calls.unwrap_or(base.max_model_calls),
            max_depth: self.max_depth.unwrap_or(base.max_depth),
            wall_secs: self.wall_secs.unwrap_or(base.wall_secs),
        }
        .sanitize()
    }
}

async fn post_query(State(ctx): State<Arc<Ctx>>, Json(body): Json<QueryBody>) -> Response {
    let Some(rec) = ctx.store.get(&body.context_id) else {
        return err(StatusCode::NOT_FOUND, "no such context");
    };
    let Some(host) = ctx.host.clone() else {
        return err(
            StatusCode::SERVICE_UNAVAILABLE,
            "no model callback is available to this sidecar, so nothing can be read from \
             the corpus. Check that @ryu/rlm is granted hook:side-model.",
        );
    };
    if body.query.trim().is_empty() {
        return err(StatusCode::BAD_REQUEST, "query is empty");
    }
    let budget = body.budget(ctx.budget);
    let run = engine::run(rec, body.query.clone(), host, budget, 0).await;
    ctx.store.save_run(&run);
    Json(run).into_response()
}

async fn list_runs(State(ctx): State<Arc<Ctx>>) -> Response {
    Json(json!({ "runs": ctx.store.list_runs(None) })).into_response()
}

async fn get_run(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    match ctx.store.run(&id) {
        Some(r) => Json(r).into_response(),
        None => err(StatusCode::NOT_FOUND, "no such run"),
    }
}

async fn post_purge(State(ctx): State<Arc<Ctx>>) -> Response {
    Json(json!({ "deleted_contexts": ctx.store.purge() })).into_response()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_omitted_budget_field_inherits_the_node_default() {
        let base = Budget::default();
        let body = QueryBody {
            context_id: "c".into(),
            query: "q".into(),
            max_steps: Some(3),
            max_model_calls: None,
            max_depth: None,
            wall_secs: None,
        };
        let b = body.budget(base);
        assert_eq!(b.max_steps, 3);
        assert_eq!(b.max_model_calls, base.max_model_calls);
        assert_eq!(b.max_depth, base.max_depth);
    }

    #[test]
    fn a_caller_supplied_budget_is_still_clamped() {
        let body = QueryBody {
            context_id: "c".into(),
            query: "q".into(),
            max_steps: Some(999),
            max_model_calls: Some(999_999),
            max_depth: Some(9),
            wall_secs: Some(999_999),
        };
        let b = body.budget(Budget::default());
        assert_eq!(b.max_steps, 40);
        assert_eq!(b.max_model_calls, 2_000);
        assert_eq!(b.max_depth, 3);
    }

    #[test]
    fn the_capability_descriptor_says_when_model_calls_are_unavailable() {
        // This is how a missing grant becomes visible before a user waits two
        // minutes on a query that was never going to run.
        let d = capability_descriptor(false, &Roots::from_env());
        assert_eq!(d["model_calls_available"], false);
        assert_eq!(d["capability"], "rlm.query");
    }
}
