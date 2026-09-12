//! `ryu-rlm mcp` — the same engine, spoken as an MCP stdio server.
//!
//! This is the seam that makes the app usable from an **agent** and from a
//! **workflow** without Core knowing anything about it. The manifest declares the
//! server under `mcp_servers`, Core spawns it like any other MCP process, and the
//! tools appear as `rlm.ask`, `rlm.query`, … — which is exactly the
//! `<server>.<tool>` id a workflow's `mcp` node takes. No route, no node kind and
//! no Core edit is added for this app.
//!
//! Tool names here are **bare**. Core forms the public id by prefixing the server
//! name, so a self-prefixed `rlm_ask` would register as `rlm.rlm_ask`.
//!
//! ## Why an agent should reach for this
//!
//! An agent that wants to answer a question about a large corpus has two options:
//! read the corpus into its own context (which is what it will do by default, and
//! what degrades its answers on everything else in the conversation), or hand the
//! job here and get back an answer plus citations. `ask` is deliberately the
//! shortest path — paths in, answer out, corpus never touching the caller's
//! context — because a tool that takes three calls to be useful does not get used.
//!
//! `grep` and `peek` need no model at all, so a workflow can locate something in a
//! 40 MB corpus deterministically and branch on it.
//!
//! Framing is newline-delimited JSON-RPC 2.0 over stdin/stdout, protocol
//! `2024-11-05` — what Core's MCP client speaks.

use std::sync::Arc;

use anyhow::{anyhow, Result};
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::api::Ctx;
use crate::engine::{self, Budget, EvidencePolicy};
use crate::ops;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Tool descriptors, in the shape `tools/list` returns.
fn tool_list() -> Value {
    json!([
        {
            "name": "ask",
            "description":
                "Answer a question about files far larger than your context window, WITHOUT \
                 reading them into your context. Give it paths (files or directories) and a \
                 question; it chunks the corpus, plans over an outline, runs a cheap model over \
                 the chunks that matter, and returns an answer with `path:line` citations plus \
                 what it cost. Reach for this instead of reading a large directory yourself — \
                 the corpus never enters this conversation. Returns a `context_id` you can ask \
                 follow-up questions against with `query`, which skips reloading.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": {
                        "type": "array", "items": { "type": "string" },
                        "description": "Files or directories to read. Must be inside a configured context root."
                    },
                    "query": { "type": "string", "description": "The question to answer from those files." },
                    "max_steps": { "type": "integer", "description": "Planning steps before it must answer. Default 14." },
                    "max_model_calls": { "type": "integer", "description": "One root-owned ceiling shared by planner, recursion and map leaves." },
                    "max_depth": { "type": "integer", "description": "How deep it may recurse. Default 2." },
                    "wall_secs": { "type": "integer", "description": "One root deadline shared by every nested call." },
                    "evidence_policy": evidence_policy_schema()
                },
                "required": ["paths", "query"]
            }
        },
        {
            "name": "load",
            "description":
                "Load files or directories into a reusable context and return its id, chunk count \
                 and outline — without answering anything and without any model call. Use when you \
                 want to inspect what was loaded, or to ask several questions against one corpus.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "paths": { "type": "array", "items": { "type": "string" } },
                    "name": { "type": "string", "description": "A label for this corpus." },
                    "text": { "type": "string", "description": "Instead of paths: literal text to load as the corpus." }
                }
            }
        },
        {
            "name": "query",
            "description":
                "Ask a question against a context that is already loaded. Same engine as `ask`, \
                 without re-reading the files. Returns the answer, the chunks it rests on, and the \
                 number of model calls it took.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context_id": { "type": "string" },
                    "query": { "type": "string" },
                    "max_steps": { "type": "integer" },
                    "max_model_calls": { "type": "integer" },
                    "max_depth": { "type": "integer" },
                    "wall_secs": { "type": "integer" },
                    "evidence_policy": evidence_policy_schema()
                },
                "required": ["context_id", "query"]
            }
        },
        {
            "name": "grep",
            "description":
                "Regex search across a loaded context. NO model call — deterministic, so a \
                 workflow branch gated on it takes the same path for the same corpus. Reports \
                 which chunk each hit fell in, with true `path:line` numbers.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context_id": { "type": "string" },
                    "pattern": { "type": "string", "description": "Case-insensitive regular expression." },
                    "max": { "type": "integer" },
                    "context": { "type": "integer", "description": "Lines of surrounding context per hit, 0–3." }
                },
                "required": ["context_id", "pattern"]
            }
        },
        {
            "name": "peek",
            "description":
                "Read a window of one chunk verbatim. NO model call. Use to check an exact wording \
                 a citation points at — not to read the corpus, which is what `ask` is for.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "context_id": { "type": "string" },
                    "chunk": { "type": "integer" },
                    "start": { "type": "integer" },
                    "len": { "type": "integer" }
                },
                "required": ["context_id", "chunk"]
            }
        },
        {
            "name": "contexts",
            "description": "List the loaded contexts on this node with their sizes and sources.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

fn evidence_policy_schema() -> Value {
    json!({
        "type": "object",
        "description": "Optional structural grounding requirements for the final answer.",
        "properties": {
            "minimum_citations": { "type": "integer", "minimum": 0 },
            "minimum_successful_recursions": { "type": "integer", "minimum": 0 },
            "allowed_source_prefix": { "type": "string" }
        }
    })
}

/// Serve MCP on stdin/stdout until the stream closes.
pub async fn serve(ctx: Arc<Ctx>) -> Result<()> {
    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();
    let mut stdout = tokio::io::stdout();

    while let Some(line) = lines.next_line().await? {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Ok(frame) = serde_json::from_str::<Value>(line) else {
            continue; // A frame we cannot parse has no id to answer on.
        };
        let method = frame.get("method").and_then(Value::as_str).unwrap_or("");
        let id = frame.get("id").cloned();
        // Notifications carry no id and take no response.
        let Some(id) = id else { continue };

        let response = match method {
            "initialize" => json!({
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": { "tools": {} },
                "serverInfo": { "name": "ryu-rlm", "version": env!("CARGO_PKG_VERSION") }
            }),
            "ping" => json!({}),
            "tools/list" => json!({ "tools": tool_list() }),
            "tools/call" => {
                let params = frame.get("params").cloned().unwrap_or_else(|| json!({}));
                let name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params
                    .get("arguments")
                    .cloned()
                    .unwrap_or_else(|| json!({}));
                match call_tool(&ctx, name, args).await {
                    Ok(value) => tool_result(&value, false),
                    Err(e) => tool_result(&json!({ "error": e.to_string() }), true),
                }
            }
            other => {
                write_frame(
                    &mut stdout,
                    &json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": { "code": -32601, "message": format!("unknown method '{other}'") }
                    }),
                )
                .await?;
                continue;
            }
        };
        write_frame(
            &mut stdout,
            &json!({ "jsonrpc": "2.0", "id": id, "result": response }),
        )
        .await?;
    }
    Ok(())
}

/// MCP returns tool output as content blocks; JSON goes in a text block so a client
/// that only renders text still shows something readable.
fn tool_result(value: &Value, is_error: bool) -> Value {
    json!({
        "content": [{
            "type": "text",
            "text": serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
        }],
        "isError": is_error
    })
}

async fn call_tool(ctx: &Arc<Ctx>, name: &str, args: Value) -> Result<Value> {
    let string =
        |key: &str| -> Option<String> { args.get(key).and_then(Value::as_str).map(str::to_owned) };
    let number =
        |key: &str| -> Option<usize> { args.get(key).and_then(Value::as_u64).map(|n| n as usize) };
    let context = |key: &str| {
        let id = string(key).ok_or_else(|| anyhow!("{key} is required"))?;
        ctx.store
            .get(&id)
            .ok_or_else(|| anyhow!("no context with the id '{id}' — call `contexts` to list them"))
    };
    let paths = || -> Vec<String> {
        args.get("paths")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default()
    };

    match name {
        "contexts" => Ok(json!({ "contexts": ctx.store.list() })),

        "load" => {
            let name_of = string("name").unwrap_or_else(|| "corpus".to_owned());
            let record = if let Some(text) = string("text") {
                ctx.store.create(
                    &name_of,
                    vec![crate::chunk::Document::new("text", text)],
                    None,
                    None,
                )?
            } else {
                let want = paths();
                if want.is_empty() {
                    return Err(anyhow!("give either `paths` or `text`"));
                }
                let (docs, skipped) = ctx.roots.load(&want)?;
                if docs.is_empty() {
                    return Err(anyhow!(
                        "nothing readable was loaded. Refused or unreadable: {}",
                        skipped.join("; ")
                    ));
                }
                ctx.store.create(&name_of, docs, None, None)?
            };
            Ok(json!({
                "context_id": record.id,
                "chunks": record.chunks.len(),
                "total_chars": record.total_chars,
                "sources": record.sources,
                "outline": ops::Observation::new(record.outline()).text,
            }))
        }

        "grep" => {
            let record = context("context_id")?;
            let pattern = string("pattern").ok_or_else(|| anyhow!("pattern is required"))?;
            let obs = ops::grep(&record, &pattern, number("max"), number("context"));
            Ok(json!({ "text": obs.text, "truncated": obs.truncated }))
        }

        "peek" => {
            let record = context("context_id")?;
            let chunk = number("chunk").ok_or_else(|| anyhow!("chunk is required"))?;
            let obs = ops::peek(&record, chunk, number("start").unwrap_or(0), number("len"));
            Ok(json!({ "text": obs.text, "truncated": obs.truncated }))
        }

        "query" | "ask" => {
            let query = string("query").ok_or_else(|| anyhow!("query is required"))?;
            let host = ctx.host.clone().ok_or_else(|| {
                anyhow!(
                    "no model callback is available to this sidecar, so nothing can be read \
                     from the corpus. Check that @ryu/rlm is granted hook:side-model."
                )
            })?;

            let record = if name == "ask" {
                let want = paths();
                if want.is_empty() {
                    return Err(anyhow!(
                        "`ask` needs `paths`; use `query` for a loaded context"
                    ));
                }
                let (docs, skipped) = ctx.roots.load(&want)?;
                if docs.is_empty() {
                    return Err(anyhow!(
                        "nothing readable was loaded. Refused or unreadable: {}",
                        skipped.join("; ")
                    ));
                }
                ctx.store.create("ask", docs, None, None)?
            } else {
                context("context_id")?
            };

            let budget = Budget {
                max_steps: number("max_steps").unwrap_or(ctx.budget.max_steps),
                max_model_calls: number("max_model_calls").unwrap_or(ctx.budget.max_model_calls),
                max_depth: number("max_depth").unwrap_or(ctx.budget.max_depth),
                wall_secs: args
                    .get("wall_secs")
                    .and_then(Value::as_u64)
                    .unwrap_or(ctx.budget.wall_secs),
            }
            .sanitize();
            let evidence_policy = args
                .get("evidence_policy")
                .cloned()
                .map(serde_json::from_value::<EvidencePolicy>)
                .transpose()?
                .unwrap_or_default()
                .validate()
                .map_err(|message| anyhow!(message))?;

            let run = engine::run(record.clone(), query, host, budget, evidence_policy).await;
            ctx.store.save_run(&run);
            // The trace is deliberately NOT returned to the caller: it is large, and
            // an agent that pasted it into its context would undo the entire point of
            // calling this tool. It is kept on the node and rendered in the companion,
            // and `run_id` is how a person finds it.
            Ok(json!({
                "answer": run.answer,
                "status": run.status,
                "cites": run.cites,
                "context_id": record.id,
                "run_id": run.id,
                "input_digest": run.input_digest,
                "evidence": run.evidence,
                "cost": {
                    "model_calls": run.model_calls,
                    "prompt_chars": run.prompt_chars,
                    "corpus_chars": record.total_chars,
                    "elapsed_ms": run.elapsed_ms,
                },
            }))
        }

        other => Err(anyhow!("unknown tool '{other}'")),
    }
}

async fn write_frame(out: &mut tokio::io::Stdout, value: &Value) -> Result<()> {
    let mut line = serde_json::to_vec(value)?;
    line.push(b'\n');
    out.write_all(&line).await?;
    out.flush().await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tool_names_are_bare_so_core_can_prefix_them() {
        // Core forms `{server}.{tool}`. A self-prefixed name becomes `rlm.rlm_ask`.
        for tool in tool_list().as_array().unwrap() {
            let name = tool["name"].as_str().unwrap();
            assert!(
                !name.starts_with("rlm"),
                "{name} is self-prefixed and would register as rlm.{name}"
            );
        }
    }

    #[test]
    fn every_tool_declares_an_object_input_schema() {
        for tool in tool_list().as_array().unwrap() {
            assert_eq!(
                tool["inputSchema"]["type"], "object",
                "{} has no object input schema",
                tool["name"]
            );
            assert!(
                tool["description"].as_str().is_some_and(|d| d.len() > 40),
                "{} needs a description an agent can select on",
                tool["name"]
            );
        }
    }

    #[test]
    fn the_deterministic_tools_are_advertised_as_needing_no_model() {
        // A workflow author choosing a node to branch on needs to know which of these
        // is reproducible. If the wording drifts, the affordance is lost.
        let tools = tool_list();
        let grep = tools
            .as_array()
            .unwrap()
            .iter()
            .find(|t| t["name"] == "grep")
            .unwrap();
        assert!(grep["description"]
            .as_str()
            .unwrap()
            .contains("NO model call"));
    }
}
