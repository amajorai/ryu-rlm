//! Handler-level tests for purge, the run list/trace reads and the outline.
//! They call the real handlers against a real on-disk [`Store`].

use std::path::PathBuf;

use axum::body::to_bytes;
use serde_json::Value;

use super::*;
use crate::engine::RunRecord;

fn scratch() -> PathBuf {
    std::env::temp_dir().join(format!("ryu-rlm-api-{}", uuid::Uuid::new_v4()))
}

fn ctx_at(dir: PathBuf) -> Arc<Ctx> {
    Arc::new(Ctx {
        store: Store::open(dir).unwrap(),
        host: None,
        roots: Roots::from_env(),
        budget: Budget::default(),
    })
}

async fn body(response: Response) -> (StatusCode, Value) {
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, serde_json::from_slice(&bytes).unwrap())
}

fn context(ctx: &Ctx, name: &str, text: &str) -> String {
    ctx.store
        .create(
            name,
            vec![Document::new(format!("{name}.md"), text)],
            None,
            None,
        )
        .unwrap()
        .id
        .clone()
}

fn run(context_id: &str, query: &str, started_at: &str) -> RunRecord {
    serde_json::from_value(json!({
        "id": format!("run-{query}"),
        "context_id": context_id,
        "query": query,
        "status": "ok",
        "answer": format!("answer to {query}"),
        "cites": [],
        "trace": [],
        "started_at": started_at,
        "elapsed_ms": 12,
        "model_calls": 3,
        "prompt_chars": 900,
        "budget": Budget::default(),
    }))
    .unwrap()
}

fn files_in(dir: &std::path::Path) -> usize {
    std::fs::read_dir(dir).map_or(0, |entries| entries.count())
}

#[tokio::test]
async fn purge_deletes_every_context_and_run_and_reports_the_context_count() {
    let dir = scratch();
    let ctx = ctx_at(dir.clone());
    let first = context(&ctx, "alpha", "# Alpha\nfirst body\n");
    context(&ctx, "beta", "# Beta\nsecond body\n");
    ctx.store
        .save_run(&run(&first, "q1", "2026-01-01T00:00:00Z"));
    assert_eq!(files_in(&dir.join("contexts")), 2);
    assert_eq!(files_in(&dir.join("runs")), 1);

    let (status, purged) = body(post_purge(State(Arc::clone(&ctx))).await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(purged, json!({ "deleted_contexts": 2 }));

    let (_, contexts) = body(list_contexts(State(Arc::clone(&ctx))).await).await;
    assert_eq!(contexts, json!({ "contexts": [] }));
    let (_, runs) = body(list_runs(State(Arc::clone(&ctx))).await).await;
    assert_eq!(runs, json!({ "runs": [] }));
    let (status, _) = body(get_outline(State(Arc::clone(&ctx)), Path(first.clone())).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);

    // The on-disk mirror is gone too, so a restart does not bring anything back.
    assert_eq!(files_in(&dir.join("contexts")), 0);
    assert_eq!(files_in(&dir.join("runs")), 0);
    let reopened = ctx_at(dir.clone());
    assert!(reopened.store.list().is_empty());
    assert!(reopened.store.list_runs(None).is_empty());

    // A second purge has nothing left to delete.
    let (_, again) = body(post_purge(State(reopened)).await).await;
    assert_eq!(again, json!({ "deleted_contexts": 0 }));
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn runs_list_newest_first_and_a_run_reads_back_in_full() {
    let dir = scratch();
    let ctx = ctx_at(dir.clone());
    let id = context(&ctx, "alpha", "# Alpha\nbody\n");
    ctx.store
        .save_run(&run(&id, "older", "2026-01-01T00:00:00Z"));
    ctx.store
        .save_run(&run(&id, "newer", "2026-02-01T00:00:00Z"));

    let (status, listed) = body(list_runs(State(Arc::clone(&ctx))).await).await;
    assert_eq!(status, StatusCode::OK);
    let rows = listed["runs"].as_array().unwrap();
    let queries: Vec<&str> = rows.iter().map(|r| r["query"].as_str().unwrap()).collect();
    assert_eq!(queries, vec!["newer", "older"]);
    assert_eq!(rows[0]["context_id"], id.as_str());
    assert_eq!(rows[0]["model_calls"], 3);
    assert_eq!(rows[0]["steps"], 0);
    assert!(rows[0].get("answer").is_none(), "the list is headers only");

    let (status, full) =
        body(get_run(State(Arc::clone(&ctx)), Path("run-newer".to_owned())).await).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(full["answer"], "answer to newer");
    assert_eq!(full["prompt_chars"], 900);

    let (status, missing) =
        body(get_run(State(Arc::clone(&ctx)), Path("run-nope".to_owned())).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing, json!({ "error": "no such run" }));

    // Runs survive a restart.
    let reopened = ctx_at(dir.clone());
    let (status, _) = body(get_run(State(reopened), Path("run-older".to_owned())).await).await;
    assert_eq!(status, StatusCode::OK);
    let _ = std::fs::remove_dir_all(dir);
}

#[tokio::test]
async fn the_outline_lists_each_chunk_and_an_unknown_context_is_404() {
    let dir = scratch();
    let ctx = ctx_at(dir.clone());
    let id = context(&ctx, "alpha", "# Alpha\nline two\nline three\n");

    let (status, outline) =
        body(get_outline(State(Arc::clone(&ctx)), Path(id.clone())).await).await;
    assert_eq!(status, StatusCode::OK);
    let record = ctx.store.get(&id).unwrap();
    assert_eq!(outline["outline"], record.outline());
    let text = outline["outline"].as_str().unwrap();
    assert!(text.starts_with("[0] alpha.md:"), "{text}");
    assert_eq!(text.lines().count(), record.chunks.len());
    assert!(outline["stats"].as_str().unwrap().starts_with(&format!(
        "{} chunks across 1 source(s)",
        record.chunks.len()
    )));

    let (status, missing) =
        body(get_outline(State(Arc::clone(&ctx)), Path("no-such-context".to_owned())).await).await;
    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(missing, json!({ "error": "no such context" }));
    let _ = std::fs::remove_dir_all(dir);
}
