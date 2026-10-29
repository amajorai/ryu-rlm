//! The sidecar's one line back into Core.
//!
//! Every model call this app makes — the root planner and every leaf reader —
//! goes through Core's generic sidecar callback:
//!
//! ```text
//! POST http://127.0.0.1:$RYU_CORE_PORT/api/host/model/complete
//!   authorization: Bearer $RYU_EXT_TOKEN
//!   x-ryu-plugin-id: $RYU_EXT_PLUGIN_ID
//!   { "prompt": …, "system": …, "model_pref_key": … }
//! ```
//!
//! Core authenticates the minted per-plugin token, intersects the manifest's
//! declared `host_api.grants` with the Gateway-*approved* grants, and only then runs
//! the completion through the same `host.sideModel` capability the turn-hook sandbox
//! uses. So the app inherits the node's provider routing, its budget and its egress
//! policy, and holds no credential of its own. Both halves of the grant matter:
//! `hook:side-model` must appear in the sidecar's `host_api.grants` **and** be
//! approved for the plugin; missing either side is a 403.
//!
//! ## Two model tiers, on purpose
//!
//! Recursion only pays if the leaves are cheap. [`Tier::Leaf`] resolves through the
//! `rlm-leaf-model` preference and does the bulk reading — one chunk each, hundreds
//! of calls on a large corpus. [`Tier::Root`] resolves through `rlm-root-model` and
//! runs the planner, which sees only outlines and folded findings and therefore
//! stays small enough to afford a strong model. Pointing both at the same model is
//! valid and is what happens when the user configures neither; the split is an
//! economic lever, not a correctness requirement.
//!
//! ## Concurrency is bounded here, not at the call site
//!
//! A `map` over four hundred chunks must not open four hundred simultaneous
//! completions: that stampedes the node's provider rate limit and turns one query
//! into a global outage for every other agent on the node. The semaphore lives in
//! [`Host`] so *every* caller inherits it, rather than in the engine where a second
//! call site could forget it.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use anyhow::{anyhow, Context};
use serde_json::{json, Value};
use tokio::sync::Semaphore;

/// Env keys Core injects into every manifest sidecar at spawn.
const ENV_TOKEN: &str = "RYU_EXT_TOKEN";
const ENV_PLUGIN_ID: &str = "RYU_EXT_PLUGIN_ID";
const ENV_CORE_PORT: &str = "RYU_CORE_PORT";

/// How many completions may be in flight at once. Deliberately modest: the point of
/// the leaf tier is many small cheap calls, and the node's provider quota is shared
/// with every other agent running on it.
const MAX_IN_FLIGHT: usize = 8;

/// Which of the two configured models a call should use.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// The planner. Sees outlines, findings and its own notes — never the corpus.
    Root,
    /// The reader. Sees exactly one chunk and one instruction.
    Leaf,
}

/// Why a completion did not run. Budget failures remain distinct from provider
/// failures so the engine can report `budget_exhausted` instead of claiming Core or
/// the provider broke.
#[derive(Debug)]
pub enum CompleteError {
    CallBudget,
    Deadline,
    Host(anyhow::Error),
}

impl std::fmt::Display for CompleteError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompleteError::CallBudget => formatter.write_str("root model-call budget exhausted"),
            CompleteError::Deadline => formatter.write_str("root wall-clock deadline exhausted"),
            CompleteError::Host(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CompleteError {}

/// The one quota owned by a root query. Every planner, recursion and map leaf gets
/// this same value by `Arc`; no nested call can rebase either counter.
#[derive(Debug)]
pub struct RunQuota {
    remaining_calls: AtomicUsize,
    reserved_calls: AtomicUsize,
    prompt_chars: AtomicU64,
    deadline: tokio::time::Instant,
}

impl RunQuota {
    pub fn new(max_model_calls: usize, wall: Duration) -> RunQuota {
        RunQuota {
            remaining_calls: AtomicUsize::new(max_model_calls),
            reserved_calls: AtomicUsize::new(0),
            prompt_chars: AtomicU64::new(0),
            deadline: tokio::time::Instant::now() + wall,
        }
    }

    /// Claim one model call atomically. This happens before waiting on the global
    /// Host semaphore, so a large map cannot queue more work than the root budget.
    pub(crate) fn reserve_call(&self) -> Result<(), CompleteError> {
        if self.remaining_time().is_none() {
            return Err(CompleteError::Deadline);
        }
        self.remaining_calls
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |remaining| {
                remaining.checked_sub(1)
            })
            .map_err(|_| CompleteError::CallBudget)?;
        self.reserved_calls.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    fn remaining_time(&self) -> Option<Duration> {
        let remaining = self
            .deadline
            .saturating_duration_since(tokio::time::Instant::now());
        (!remaining.is_zero()).then_some(remaining)
    }

    pub fn calls(&self) -> usize {
        self.reserved_calls.load(Ordering::Relaxed)
    }

    pub fn calls_remaining(&self) -> usize {
        self.remaining_calls.load(Ordering::Acquire)
    }

    pub fn prompt_chars(&self) -> u64 {
        self.prompt_chars.load(Ordering::Relaxed)
    }

    pub fn deadline_elapsed(&self) -> bool {
        self.remaining_time().is_none()
    }
}

impl Tier {
    /// The settings key Core resolves to an actual model. Unset means "the node's
    /// default", which is why an un-configured install still works.
    pub fn pref_key(self) -> &'static str {
        match self {
            Tier::Root => "rlm-root-model",
            Tier::Leaf => "rlm-leaf-model",
        }
    }
}

#[derive(Clone)]
pub struct Host {
    base: String,
    plugin_id: String,
    token: String,
    http: reqwest::Client,
    gate: Arc<Semaphore>,
    /// Completions issued since boot, and the characters sent to them. Reported on
    /// every run so a person can see what a query actually cost — the number that
    /// decides whether recursion beat stuffing the window.
    calls: Arc<AtomicU64>,
    prompt_chars: Arc<AtomicU64>,
    #[cfg(test)]
    mock: Option<Arc<MockHost>>,
}

impl Host {
    /// Build from the injected environment. `None` when the process was not spawned
    /// by Core — the corpus routes (`outline`, `peek`, `grep`) still work, and the
    /// model-backed ones report that they are unavailable rather than pretending.
    pub fn from_env() -> Option<Host> {
        let token = std::env::var(ENV_TOKEN).ok().filter(|s| !s.is_empty())?;
        let plugin_id = std::env::var(ENV_PLUGIN_ID)
            .ok()
            .filter(|s| !s.is_empty())?;
        let port = std::env::var(ENV_CORE_PORT)
            .ok()
            .and_then(|p| p.parse::<u16>().ok())?;
        Some(Host {
            base: format!("http://127.0.0.1:{port}"),
            plugin_id,
            token,
            // The root quota supplies the request timeout on every call. A client
            // default here would create a second deadline that nested runs could
            // observe differently from their parent.
            http: reqwest::Client::builder().build().ok()?,
            gate: Arc::new(Semaphore::new(MAX_IN_FLIGHT)),
            calls: Arc::new(AtomicU64::new(0)),
            prompt_chars: Arc::new(AtomicU64::new(0)),
            #[cfg(test)]
            mock: None,
        })
    }

    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }

    pub fn prompt_chars(&self) -> u64 {
        self.prompt_chars.load(Ordering::Relaxed)
    }

    /// One completion at the given tier under a root query's quota.
    pub async fn complete(
        &self,
        quota: &RunQuota,
        tier: Tier,
        system: &str,
        prompt: &str,
    ) -> Result<String, CompleteError> {
        quota.reserve_call()?;
        let remaining = quota.remaining_time().ok_or(CompleteError::Deadline)?;
        let _permit = tokio::time::timeout(remaining, self.gate.acquire())
            .await
            .map_err(|_| CompleteError::Deadline)?
            .map_err(|_| CompleteError::Host(anyhow!("model gate closed")))?;

        let prompt_chars = (system.len() + prompt.len()) as u64;
        quota
            .prompt_chars
            .fetch_add(prompt_chars, Ordering::Relaxed);
        self.calls.fetch_add(1, Ordering::Relaxed);
        self.prompt_chars.fetch_add(prompt_chars, Ordering::Relaxed);

        let remaining = quota.remaining_time().ok_or(CompleteError::Deadline)?;
        #[cfg(test)]
        if let Some(mock) = &self.mock {
            return tokio::time::timeout(remaining, mock.complete())
                .await
                .map_err(|_| CompleteError::Deadline)?;
        }

        let args = json!({
            "system": system,
            "prompt": prompt,
            "model_pref_key": tier.pref_key(),
        });
        let request = self
            .http
            .post(format!("{}/api/host/model/complete", self.base))
            .bearer_auth(&self.token)
            .header("x-ryu-plugin-id", &self.plugin_id)
            .json(&args)
            .send();
        let resp = tokio::time::timeout(remaining, request)
            .await
            .map_err(|_| CompleteError::Deadline)?
            .map_err(|error| {
                CompleteError::Host(anyhow!(error).context("calling the host model callback"))
            })?;

        let status = resp.status();
        let body: Value = resp
            .json()
            .await
            .context("host model callback returned a non-JSON body")
            .map_err(CompleteError::Host)?;
        if !status.is_success() {
            let msg = body
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("host model callback failed");
            // 403 here almost always means the grant is declared but not approved.
            return Err(CompleteError::Host(anyhow!("{msg} (HTTP {status})")));
        }
        let text = body.get("result").map(render_result).unwrap_or_default();
        if text.trim().is_empty() {
            return Err(CompleteError::Host(anyhow!(
                "the model returned an empty completion"
            )));
        }
        Ok(text)
    }
}

#[cfg(test)]
#[derive(Clone)]
pub(crate) struct MockReply {
    pub text: Result<String, String>,
    pub delay: Duration,
}

#[cfg(test)]
struct MockHost {
    replies: tokio::sync::Mutex<std::collections::VecDeque<MockReply>>,
}

#[cfg(test)]
impl MockHost {
    async fn complete(&self) -> Result<String, CompleteError> {
        let reply = self
            .replies
            .lock()
            .await
            .pop_front()
            .ok_or_else(|| CompleteError::Host(anyhow!("mock host ran out of replies")))?;
        tokio::time::sleep(reply.delay).await;
        reply
            .text
            .map_err(|message| CompleteError::Host(anyhow!(message)))
    }
}

#[cfg(test)]
impl Host {
    pub(crate) fn for_test(replies: Vec<MockReply>, max_in_flight: usize) -> Host {
        Host {
            base: String::new(),
            plugin_id: "test".to_owned(),
            token: "test".to_owned(),
            http: reqwest::Client::new(),
            gate: Arc::new(Semaphore::new(max_in_flight)),
            calls: Arc::new(AtomicU64::new(0)),
            prompt_chars: Arc::new(AtomicU64::new(0)),
            mock: Some(Arc::new(MockHost {
                replies: tokio::sync::Mutex::new(replies.into()),
            })),
        }
    }
}

/// The bridge returns either a bare string or a `{ text }`-shaped object depending
/// on the provider; accept both rather than depending on one provider's shape.
fn render_result(value: &Value) -> String {
    match value {
        Value::String(s) => s.clone(),
        Value::Object(map) => map
            .get("text")
            .or_else(|| map.get("content"))
            .or_else(|| map.get("output"))
            .map(render_result)
            .unwrap_or_default(),
        Value::Array(items) => items.iter().map(render_result).collect::<Vec<_>>().join(""),
        _ => String::new(),
    }
}

/// Pull the first balanced JSON object or array out of a completion.
///
/// The planner is asked for JSON, and models wrap JSON in prose or a ```json fence
/// often enough that a strict parse would waste a planning call per step. The scan
/// is string- and escape-aware, so a `{` inside a quoted excerpt does not throw off
/// the depth count.
pub fn extract_json(text: &str) -> Option<Value> {
    let bytes = text.as_bytes();
    for (start, open) in bytes.iter().enumerate() {
        let close = match open {
            b'{' => b'}',
            b'[' => b']',
            _ => continue,
        };
        let mut depth = 0i32;
        let mut in_string = false;
        let mut escaped = false;
        for (idx, byte) in bytes.iter().enumerate().skip(start) {
            if in_string {
                if escaped {
                    escaped = false;
                } else if *byte == b'\\' {
                    escaped = true;
                } else if *byte == b'"' {
                    in_string = false;
                }
                continue;
            }
            match *byte {
                b'"' => in_string = true,
                b if b == *open => depth += 1,
                b if b == close => {
                    depth -= 1;
                    if depth == 0 {
                        if let Ok(v) = serde_json::from_str::<Value>(&text[start..=idx]) {
                            return Some(v);
                        }
                        break;
                    }
                }
                _ => {}
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_two_tiers_resolve_to_different_preference_keys() {
        // If these collided, the cheap-leaf economics the whole app rests on would
        // silently become "run everything on the strong model".
        assert_ne!(Tier::Root.pref_key(), Tier::Leaf.pref_key());
    }

    #[test]
    fn a_bare_string_result_and_a_wrapped_one_render_the_same() {
        assert_eq!(render_result(&json!("hi")), "hi");
        assert_eq!(render_result(&json!({ "text": "hi" })), "hi");
        assert_eq!(render_result(&json!({ "content": "hi" })), "hi");
        assert_eq!(
            render_result(&json!([{ "text": "a" }, { "text": "b" }])),
            "ab"
        );
    }

    #[test]
    fn json_is_recovered_from_a_fenced_or_chatty_completion() {
        let v = extract_json("Sure! ```json\n{\"op\":\"peek\",\"chunk\":3}\n``` hope that helps")
            .expect("payload");
        assert_eq!(v["op"], "peek");
        assert_eq!(v["chunk"], 3);
    }

    #[test]
    fn a_brace_inside_a_quoted_excerpt_does_not_unbalance_the_scan() {
        let v =
            extract_json(r#"{"op":"note","text":"the config was { broken }"}"#).expect("payload");
        assert_eq!(v["text"], "the config was { broken }");
    }

    #[test]
    fn no_json_at_all_is_none_rather_than_a_panic() {
        assert!(extract_json("I could not answer that.").is_none());
    }
}
