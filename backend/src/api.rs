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

/// The OpenAPI document Core fetches from `/openapi.json` and lowers into LLM tools.
///
/// The `#[utoipa::path]` annotations below carry the ABSOLUTE external path
/// (`/api/rlm/...`, `{id}` in brace form) while the router above registers paths
/// relative to the mount in axum's `:id` form. The two forms differ ON PURPOSE — Core
/// nests this router at `/api/rlm`, and the document has to describe the URL a caller
/// actually hits. Do not "align" either side; `every_declared_route_appears_in_the_openapi_doc`
/// normalises between them.
///
/// This app also ships a stdio `mcp_servers` server ([`crate::mcp`]), and five of its
/// six tools are the same operations as routes here: `contexts` ↔ `GET /contexts`,
/// `load` ↔ `POST /contexts/from-paths` (or `POST /contexts` when it is handed inline
/// text), `grep` and `peek` ↔ the matching `POST /contexts/{id}/…`, and `query` ↔
/// `POST /query`. `ask` has no single twin — it fuses a load and a query. That overlap
/// is deliberate and harmless: both surfaces call the SAME functions in
/// [`crate::ops`] / [`crate::engine`], so they cannot drift, and a node with the MCP
/// server disabled still gets the whole surface as derived tools. Seven routes exist
/// ONLY here — `POST /contexts`, `GET`/`DELETE /contexts/{id}`, the outline, both run
/// routes and the purge.
#[must_use]
pub fn openapi() -> utoipa::openapi::OpenApi {
    <RlmApiDoc as utoipa::OpenApi>::openapi()
}

/// `components(schemas(...))` is what turns each `request_body = T` into a resolvable
/// `#/components/schemas/T`: without the entry the operation still carries a `$ref`
/// whose target is missing, and Core derives a write tool with zero visible arguments —
/// discoverable and uncallable. utoipa 5 also auto-collects schemas reachable from the
/// annotated paths, so these rows are belt-and-braces; they are listed anyway so the
/// registration is greppable and cannot be lost to an attribute edit.
///
/// `DocumentBody` is here because it is reachable only TRANSITIVELY, through
/// `CreateBody::documents`.
#[derive(utoipa::OpenApi)]
#[openapi(
    paths(
        list_contexts,
        create_context,
        create_from_paths,
        get_context,
        delete_context,
        get_outline,
        post_peek,
        post_grep,
        post_query,
        list_runs,
        get_run,
        post_purge,
    ),
    components(schemas(CreateBody, DocumentBody, PathsBody, PeekBody, GrepBody, QueryBody))
)]
struct RlmApiDoc;

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

#[derive(Deserialize, utoipa::ToSchema)]
struct CreateBody {
    /// A label for the corpus, shown in the context list. Defaults to "context".
    #[serde(default)]
    name: Option<String>,
    /// The documents to hold. Each carries its own text — nothing is read from disk
    /// on this route; use `/api/rlm/contexts/from-paths` for that.
    // Inlined so the model sees `source` and `text` rather than a `$ref` it cannot
    // follow: Core resolves refs only one level into a schema, so a property-level
    // reference reaches the model as an empty object.
    #[schema(inline)]
    documents: Vec<DocumentBody>,
    /// Id of an existing context this one is carved out of, recorded for provenance.
    #[serde(default)]
    parent: Option<String>,
    /// Target size in characters for each chunk the corpus is split into. Omit for
    /// the node default.
    #[serde(default)]
    target_chars: Option<usize>,
}

#[derive(Deserialize, utoipa::ToSchema)]
struct DocumentBody {
    /// Where this text came from — a file path, a URL, anything citable. Answers
    /// quote it back as `source:line`, so a meaningless value makes them unusable.
    source: String,
    /// The full document text.
    text: String,
}

/// `POST /api/rlm/contexts` — hold documents as a context and return its id.
#[utoipa::path(
    post,
    path = "/api/rlm/contexts",
    tag = "RLM",
    summary = "Hold documents as a queryable context, far larger than a model window, and return its id.",
    request_body = CreateBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[derive(Deserialize, utoipa::ToSchema)]
struct PathsBody {
    /// A label for the corpus, shown in the context list. Defaults to "files".
    #[serde(default)]
    name: Option<String>,
    /// Files and directories to read. Paths outside this node's configured readable
    /// roots are refused and reported back in `skipped` rather than silently dropped.
    paths: Vec<String>,
    /// Target size in characters for each chunk. Omit for the node default.
    #[serde(default)]
    target_chars: Option<usize>,
}

/// `POST /api/rlm/contexts/from-paths` — build a context by reading files off disk.
#[utoipa::path(
    post,
    path = "/api/rlm/contexts/from-paths",
    tag = "RLM",
    summary = "Read files and directories into a queryable context without putting them in the conversation.",
    request_body = PathsBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

/// `GET /api/rlm/contexts` — every context held on this node.
#[utoipa::path(
    get,
    path = "/api/rlm/contexts",
    tag = "RLM",
    summary = "List every context held on this node, with its size and sources.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_contexts(State(ctx): State<Arc<Ctx>>) -> Response {
    Json(json!({ "contexts": ctx.store.list() })).into_response()
}

/// `GET /api/rlm/contexts/{id}` — one context's metadata and chunk index.
#[utoipa::path(
    get,
    path = "/api/rlm/contexts/{id}",
    tag = "RLM",
    summary = "Read one context's sources and chunk index (chunk bodies come from peek).",
    params(("id" = String, Path, description = "Context id returned when it was created")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

/// `DELETE /api/rlm/contexts/{id}` — drop one context and its stored text.
#[utoipa::path(
    delete,
    path = "/api/rlm/contexts/{id}",
    tag = "RLM",
    summary = "Delete one context and the text it holds.",
    params(("id" = String, Path, description = "Context id to delete")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn delete_context(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    Json(json!({ "deleted": ctx.store.delete(&id) })).into_response()
}

/// `GET /api/rlm/contexts/{id}/outline` — the map to navigate by. No model call.
#[utoipa::path(
    get,
    path = "/api/rlm/contexts/{id}/outline",
    tag = "RLM",
    summary = "Outline a context — chunk numbers, sources and line ranges — with no model call at all.",
    params(("id" = String, Path, description = "Context id to outline")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_outline(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    match ctx.store.get(&id) {
        Some(rec) => {
            Json(json!({ "outline": rec.outline(), "stats": rec.stats_line() })).into_response()
        }
        None => err(StatusCode::NOT_FOUND, "no such context"),
    }
}

#[derive(Deserialize, utoipa::ToSchema)]
struct PeekBody {
    /// Which chunk to read, by the number the outline gives it.
    chunk: usize,
    /// Character offset within the chunk to start at. Defaults to the beginning.
    #[serde(default)]
    start: usize,
    /// How many characters to return. Omit for the largest window the observation
    /// budget allows; the response reports whether it was truncated.
    #[serde(default)]
    len: Option<usize>,
}

/// `POST /api/rlm/contexts/{id}/peek` — read one window of one chunk. No model call.
#[utoipa::path(
    post,
    path = "/api/rlm/contexts/{id}/peek",
    tag = "RLM",
    summary = "Read one window of one chunk verbatim, with no model call — safe to branch a workflow on.",
    params(("id" = String, Path, description = "Context id to read from")),
    request_body = PeekBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[derive(Deserialize, utoipa::ToSchema)]
struct GrepBody {
    /// Regular expression to search the whole corpus for.
    pattern: String,
    /// Cap on how many matches to return. Omit for the node default.
    #[serde(default)]
    max: Option<usize>,
    /// How many lines of surrounding text to include with each match.
    #[serde(default)]
    context: Option<usize>,
}

/// `POST /api/rlm/contexts/{id}/grep` — search the corpus. No model call.
#[utoipa::path(
    post,
    path = "/api/rlm/contexts/{id}/grep",
    tag = "RLM",
    summary = "Search a whole context by regex and get the matching lines with citations, with no model call.",
    params(("id" = String, Path, description = "Context id to search")),
    request_body = GrepBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

#[derive(Deserialize, utoipa::ToSchema)]
pub struct QueryBody {
    /// The context to answer from. Load one first — nothing is read from disk here.
    pub context_id: String,
    /// The question to answer. The corpus never enters the caller's own context: a
    /// root model plans over it and sub-model calls read one slice each.
    pub query: String,
    /// Ceiling on planning steps. Omit for the node default; values above the node
    /// maximum are clamped down, never up.
    #[serde(default)]
    pub max_steps: Option<usize>,
    /// Ceiling on sub-model calls, which is what the run actually costs.
    #[serde(default)]
    pub max_model_calls: Option<usize>,
    /// How deep recursion may go before a slice must answer for itself.
    #[serde(default)]
    pub max_depth: Option<usize>,
    /// Wall-clock deadline in seconds. The run returns what it has when it expires.
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

/// `POST /api/rlm/query` — the one route that runs the recursion engine.
#[utoipa::path(
    post,
    path = "/api/rlm/query",
    tag = "RLM",
    summary = "Answer a question over a whole context with citations, without reading the corpus into this conversation.",
    request_body = QueryBody,
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

/// `GET /api/rlm/runs` — past queries, newest first.
#[utoipa::path(
    get,
    path = "/api/rlm/runs",
    tag = "RLM",
    summary = "List past queries with their answers and what each one cost.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn list_runs(State(ctx): State<Arc<Ctx>>) -> Response {
    Json(json!({ "runs": ctx.store.list_runs(None) })).into_response()
}

/// `GET /api/rlm/runs/{id}` — one run's full replayable trace.
#[utoipa::path(
    get,
    path = "/api/rlm/runs/{id}",
    tag = "RLM",
    summary = "Read one past query's full trace — every operator, slice read and sub-answer.",
    params(("id" = String, Path, description = "Run id, as returned by a query or the run list")),
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
async fn get_run(State(ctx): State<Arc<Ctx>>, Path(id): Path<String>) -> Response {
    match ctx.store.run(&id) {
        Some(r) => Json(r).into_response(),
        None => err(StatusCode::NOT_FOUND, "no such run"),
    }
}

/// `DELETE /api/rlm/purge` — drop every stored context.
#[utoipa::path(
    delete,
    path = "/api/rlm/purge",
    tag = "RLM",
    summary = "Delete every context held on this node. Past run traces are kept.",
    responses((status = 200, description = "OK", body = serde_json::Value))
)]
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

    // ── OpenAPI document ───────────────────────────────────────────────────────

    /// This app's own manifest, read at compile time. The route contract lives there,
    /// so the invariants below compare the document against the real declaration
    /// rather than against a second list that could drift from it.
    fn manifest() -> serde_json::Value {
        serde_json::from_str(include_str!("../../manifest.json")).expect("valid JSON")
    }

    /// The sidecar that declares an `http.mount` — selected by mount rather than by
    /// index so a future mountless sidecar cannot silently redirect the assertion.
    fn mounted_sidecar() -> serde_json::Value {
        manifest()["sidecars"]
            .as_array()
            .expect("sidecars must be an array")
            .iter()
            .find(|s| s["http"]["mount"].is_string())
            .expect("one sidecar must declare an http.mount")
            .clone()
    }

    /// A manifest route (relative to the mount, in axum's `:param` form) rewritten into
    /// the form the OpenAPI document uses (absolute, `{param}`). The two forms differ
    /// deliberately; normalise here rather than "aligning" either side.
    fn doc_path_for(mount: &str, route: &str) -> String {
        let joined = if route == "/" {
            mount.to_owned()
        } else {
            format!("{mount}{route}")
        };
        joined
            .split('/')
            .map(|seg| match seg.strip_prefix(':') {
                Some(name) => format!("{{{name}}}"),
                None => seg.to_owned(),
            })
            .collect::<Vec<_>>()
            .join("/")
    }

    #[test]
    fn openapi_doc_covers_the_served_routes() {
        let doc = openapi();
        assert!(!doc.paths.paths.is_empty());
    }

    #[test]
    fn every_declared_route_appears_in_the_openapi_doc() {
        // The direction that decides tool yield. Core keeps only the document
        // operations the manifest ALSO declares, so a declared route with no
        // `#[utoipa::path]` annotation is a tool that silently never exists — nothing
        // errors, an agent simply cannot call it.
        let sidecar = mounted_sidecar();
        let mount = sidecar["http"]["mount"].as_str().expect("an http.mount");
        let doc = openapi();
        for route in sidecar["http"]["routes"]
            .as_array()
            .expect("routes must be an array")
        {
            let path = route["path"].as_str().expect("a route path");
            let expected = doc_path_for(mount, path);
            assert!(
                doc.paths.paths.contains_key(&expected),
                "'{path}' is declared in manifest.json but the OpenAPI document has no \
                 '{expected}' operation — Core derives no tool for it"
            );
        }
    }

    #[test]
    fn a_documents_argument_spells_out_its_fields_instead_of_hiding_behind_a_ref() {
        // Core resolves a `$ref` only ONE level into a schema, so `documents` holding a
        // property-level reference to `DocumentBody` would reach the model as an array
        // of empty objects — visible and unfillable. `#[schema(inline)]` prevents that.
        let doc = serde_json::to_value(openapi()).expect("the document serializes");
        let create = doc["components"]["schemas"]["CreateBody"].to_string();
        assert!(
            create.contains("source") && create.contains("text"),
            "the create body hides the document fields: {create}"
        );
    }

    #[test]
    fn every_write_route_documents_a_typed_request_body() {
        // An untyped body (`request_body = serde_json::Value`) lowers to a tool with
        // zero visible arguments: the model can see it and can never fill it in. This
        // pins that each body names a real schema Core can resolve.
        let doc = serde_json::to_value(openapi()).expect("the document serializes");
        for (path, verb) in [
            ("/api/rlm/contexts", "post"),
            ("/api/rlm/contexts/from-paths", "post"),
            ("/api/rlm/contexts/{id}/peek", "post"),
            ("/api/rlm/contexts/{id}/grep", "post"),
            ("/api/rlm/query", "post"),
        ] {
            let schema =
                &doc["paths"][path][verb]["requestBody"]["content"]["application/json"]["schema"];
            let named = schema["$ref"].is_string() || schema["properties"].is_object();
            assert!(named, "{verb} {path} has no typed request body: {schema}");
        }
    }
}
