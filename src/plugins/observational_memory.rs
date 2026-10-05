//! Observational Memory: context compaction in the style of Mastra's OM.
//!
//! Two background roles keep the context window small:
//! - the **Observer** runs when the unobserved messages pass `message_tokens`
//!   (default 30k). It turns the oldest messages into dated, prioritised notes
//!   (🔴 high, 🟡 medium, 🟢 low) appended to the session's observation log.
//!   Only about `keep_ratio` (default 20%) of the threshold stays as raw
//!   messages.
//! - the **Reflector** runs when the log passes `observation_tokens` (default
//!   40k). It rewrites the whole log into a shorter one; new observations then
//!   append after it.
//!
//! Every request then carries: system prompt + `<observations>` log + the
//! unobserved tail of the conversation.
//!
//! Hook: a listener on the `llm/stream` waterfall rewrites the request before
//! it reaches the adapter. Observer and Reflector calls go straight to the LLM
//! runtime, so they never pass through this listener again.
//!
//! Models are tried in order and the first one that answers wins: by default
//! Claude through the subscription (`claude` route), then a local model if a
//! route for it is configured. A failed compaction never fails the turn; the
//! request just goes out uncompacted.
//!
//! Config (`plugins.observational-memory` in dsh.json):
//! `{ "models": [{ "provider": "claude", "model": "claude-haiku-4-5-20251001" },
//!               { "provider": "lmstudio", "model": "google/gemma-4-26b-a4b" }],
//!    "message_tokens": 30000, "observation_tokens": 40000, "keep_ratio": 0.2,
//!    "dir": "/optional/persistence/dir" }`

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use cordis::{plugin_with, Context, Injection, Plugin};
use futures::StreamExt;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::api::services::LlmService;
use crate::api::LLM_SERVICE;
use crate::types::{ContentBlock, FinishReason, GenerateOptions, Message, StreamChunk};

use super::claude_code::render_transcript;

#[derive(Debug, Clone, Deserialize, Serialize, PartialEq)]
pub struct ModelRef {
    pub provider: String,
    pub model: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct OmConfig {
    pub models: Vec<ModelRef>,
    pub message_tokens: usize,
    pub observation_tokens: usize,
    pub keep_ratio: f64,
    pub dir: Option<PathBuf>,
}

impl Default for OmConfig {
    fn default() -> Self {
        OmConfig {
            models: vec![
                ModelRef { provider: "claude".into(), model: "claude-haiku-4-5-20251001".into() },
                ModelRef { provider: "lmstudio".into(), model: "google/gemma-4-26b-a4b".into() },
            ],
            message_tokens: 30_000,
            observation_tokens: 40_000,
            keep_ratio: 0.2,
            dir: None,
        }
    }
}

/// Per-session memory: the observation log and the last message it covers.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct OmState {
    pub observations: String,
    pub observed_upto: Option<String>,
}

// ---------------------------------------------------------------------------
// Pure core
// ---------------------------------------------------------------------------

/// Rough token estimate (4 chars per token), the same budget scale as Mastra's
/// fast local estimator.
pub fn estimate_tokens(text: &str) -> usize {
    text.chars().count().div_ceil(4)
}

pub fn message_tokens(message: &Message) -> usize {
    let mut chars = 0;
    for block in &message.content {
        chars += match block {
            ContentBlock::Text { text } | ContentBlock::Reasoning { text } => text.len(),
            ContentBlock::ToolCall { name, arguments, .. } => name.len() + arguments.len(),
            ContentBlock::ToolResult { content, .. } => {
                content.iter().filter_map(|b| b.as_text()).map(str::len).sum()
            }
        };
    }
    chars.div_ceil(4)
}

/// Index of the first message not yet covered by the log. `None` means the
/// recorded boundary is not in this history: the state belongs to another
/// conversation and must be dropped.
pub fn first_unobserved(messages: &[Message], observed_upto: Option<&str>) -> Option<usize> {
    match observed_upto {
        None => Some(0),
        Some(id) => messages.iter().position(|m| m.id == id).map(|i| i + 1),
    }
}

/// A message may start the kept tail only if it carries no tool result:
/// otherwise its tool call would be compacted away and the result orphaned.
fn can_start_tail(message: &Message) -> bool {
    !message.content.iter().any(|b| matches!(b, ContentBlock::ToolResult { .. }))
}

/// Where to cut `messages[start..]` so roughly `keep_tokens` stay raw.
/// Returns the index of the first kept message, or `None` if nothing can be
/// observed.
pub fn choose_cut(messages: &[Message], start: usize, keep_tokens: usize) -> Option<usize> {
    if messages.len() <= start + 1 {
        return None;
    }
    // Walk back from the end until the tail holds `keep_tokens`.
    let mut kept = 0;
    let mut cut = messages.len() - 1;
    while cut > start {
        kept += message_tokens(&messages[cut]);
        if kept >= keep_tokens {
            break;
        }
        cut -= 1;
    }
    // Move to a safe boundary: back first (keeps more), else forward.
    let back = (start + 1..=cut).rev().find(|&i| can_start_tail(&messages[i]));
    let forward = (cut..messages.len()).find(|&i| can_start_tail(&messages[i]));
    back.or(forward).filter(|&i| i > start)
}

/// Build the rewritten request: observations in the system prompt, only the
/// unobserved tail as messages.
pub fn apply_memory(options: &mut GenerateOptions, state: &OmState, start: usize) {
    if state.observations.trim().is_empty() {
        return;
    }
    options.messages.drain(..start);
    let block = format!(
        "<observations>\nThis conversation started before the messages below. These are \
         your observations of the earlier part, oldest first. Treat them as your memory.\n\n{}\n</observations>",
        state.observations.trim()
    );
    options.system = Some(match options.system.take() {
        Some(system) if !system.is_empty() => format!("{system}\n\n{block}"),
        _ => block,
    });
}

/// `YYYY-MM-DD` and `HH:MM` (UTC) for observation headers.
pub fn utc_now() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = yoe + era * 400 + i64::from(month <= 2);
    (
        format!("{year:04}-{month:02}-{day:02}"),
        format!("{:02}:{:02}", rem / 3600, (rem % 3600) / 60),
    )
}

pub const OBSERVER_SYSTEM: &str = "You are the Observer, the memory of an AI coding agent. \
You turn raw conversation history into compact observations the agent will read instead of \
the original messages, which are deleted afterwards. Anything you leave out is forgotten.

Write only NEW observations for the messages given, in this exact format:

Date: YYYY-MM-DD
- 🔴 HH:MM observation
  - 🟡 HH:MM detail
- 🟢 HH:MM observation

Priorities:
- 🔴 high: the user's goals, requirements, constraints, preferences and decisions; names, \
paths, versions, commands and error messages that matter later; what was finished and what failed.
- 🟡 medium: useful context, intermediate findings, approaches tried.
- 🟢 low: minor detail worth keeping only if space allows.

Rules:
- One fact per line, terse, concrete, past tense for what happened. Quote exact identifiers.
- Record outcomes of tool calls (what was read, changed, run and the result), not the raw output.
- Do not repeat facts already in the existing observations.
- End with two lines: `Current task: ...` and `Next step: ...`.
- Output the observations only, no preamble.";

pub const REFLECTOR_SYSTEM: &str = "You are the Reflector, the memory of an AI coding agent. \
The observation log below has grown too long. Rewrite it as one shorter log that replaces it.

Rules:
- Keep every 🔴 fact unless a later observation supersedes it.
- Merge duplicates and related lines; drop superseded facts and most 🟢 lines.
- Keep the `Date:` headers and HH:MM times, oldest first, same emoji format.
- Keep the latest `Current task:` and `Next step:` lines at the end.
- Aim for half the length or less. Output the log only, no preamble.";

fn observer_input(existing: &str, transcript: &str, date: &str, time: &str) -> String {
    format!(
        "Current time: {date} {time} UTC\n\n<existing_observations>\n{}\n</existing_observations>\n\n\
         <messages>\n{transcript}</messages>\n\nWrite the new observations.",
        if existing.trim().is_empty() { "(none yet)" } else { existing.trim() }
    )
}

// ---------------------------------------------------------------------------
// Model calls with ordered fallback
// ---------------------------------------------------------------------------

/// Ask each configured model in turn; the first non-empty answer wins.
/// Models whose provider route is not registered are skipped.
pub async fn complete(llm: &LlmService, models: &[ModelRef], system: &str, user: String) -> Result<String, String> {
    let mut errors = Vec::new();
    for model in models {
        if !llm.has_provider(&model.provider) {
            errors.push(format!("{}: no such provider route", model.provider));
            continue;
        }
        let options = GenerateOptions {
            provider: model.provider.clone(),
            model: model.model.clone(),
            messages: vec![Message::user("om-input", vec![ContentBlock::text(user.clone())])],
            system: Some(system.to_string()),
            tools: None,
            temperature: None,
            max_tokens: None,
            stop: None,
            session_id: None,
        };
        match collect_text(llm, options).await {
            Ok(text) if !text.trim().is_empty() => return Ok(text.trim().to_string()),
            Ok(_) => errors.push(format!("{}/{}: empty answer", model.provider, model.model)),
            Err(err) => errors.push(format!("{}/{}: {err}", model.provider, model.model)),
        }
    }
    Err(errors.join("; "))
}

async fn collect_text(llm: &LlmService, options: GenerateOptions) -> Result<String, String> {
    let mut stream = llm.stream(options).await.map_err(|err| format!("{err:?}"))?;
    let mut text = String::new();
    while let Some(chunk) = stream.next().await {
        match chunk {
            StreamChunk::TextDelta { text: delta, .. } => text.push_str(&delta),
            StreamChunk::Finish { reason: FinishReason::Error { failure } | FinishReason::Aborted { failure } } => {
                return Err(failure.to_string());
            }
            _ => {}
        }
    }
    Ok(text)
}

// ---------------------------------------------------------------------------
// Runtime: per-session state, observe/reflect, persistence
// ---------------------------------------------------------------------------

#[derive(Default)]
struct Memory {
    sessions: Mutex<HashMap<String, Arc<tokio::sync::Mutex<OmState>>>>,
}

impl Memory {
    fn session(&self, id: &str, dir: Option<&PathBuf>) -> Arc<tokio::sync::Mutex<OmState>> {
        self.sessions
            .lock()
            .unwrap()
            .entry(id.to_string())
            .or_insert_with(|| Arc::new(tokio::sync::Mutex::new(load(dir, id).unwrap_or_default())))
            .clone()
    }
}

fn state_path(dir: &std::path::Path, session: &str) -> PathBuf {
    let safe: String = session.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    dir.join(format!("{safe}.om.json"))
}

fn load(dir: Option<&PathBuf>, session: &str) -> Option<OmState> {
    let text = std::fs::read_to_string(state_path(dir?, session)).ok()?;
    serde_json::from_str(&text).ok()
}

fn save(dir: Option<&PathBuf>, session: &str, state: &OmState) {
    let Some(dir) = dir else { return };
    if std::fs::create_dir_all(dir).is_ok() {
        if let Ok(text) = serde_json::to_string_pretty(state) {
            let _ = std::fs::write(state_path(dir, session), text);
        }
    }
}

/// One compaction pass for a request: observe and reflect if thresholds are
/// crossed, then rewrite `options`. Returns what happened, for the event log.
async fn compact(
    llm: &LlmService,
    config: &OmConfig,
    state: &mut OmState,
    options: &mut GenerateOptions,
) -> Value {
    let mut report = json!({});
    let start = match first_unobserved(&options.messages, state.observed_upto.as_deref()) {
        Some(start) => start,
        None => {
            *state = OmState::default();
            report["reset"] = json!(true);
            0
        }
    };
    let mut start = start;

    let unobserved: usize = options.messages[start..].iter().map(message_tokens).sum();
    if unobserved >= config.message_tokens {
        let keep = (config.message_tokens as f64 * config.keep_ratio) as usize;
        if let Some(cut) = choose_cut(&options.messages, start, keep) {
            let (date, time) = utc_now();
            let transcript = render_transcript(&options.messages[start..cut]);
            let input = observer_input(&state.observations, &transcript, &date, &time);
            match complete(llm, &config.models, OBSERVER_SYSTEM, input).await {
                Ok(notes) => {
                    if !state.observations.is_empty() {
                        state.observations.push_str("\n\n");
                    }
                    state.observations.push_str(&notes);
                    state.observed_upto = Some(options.messages[cut - 1].id.clone());
                    report["observed"] = json!({ "messages": cut - start, "tokens": unobserved });
                    start = cut;
                }
                Err(err) => report["observer_error"] = json!(err),
            }
        }
    }

    let log_tokens = estimate_tokens(&state.observations);
    if log_tokens >= config.observation_tokens {
        match complete(llm, &config.models, REFLECTOR_SYSTEM, state.observations.clone()).await {
            Ok(log) => {
                report["reflected"] = json!({ "from": log_tokens, "to": estimate_tokens(&log) });
                state.observations = log;
            }
            Err(err) => report["reflector_error"] = json!(err),
        }
    }

    apply_memory(options, state, start);
    report
}

pub fn plugin() -> Arc<dyn Plugin> {
    plugin_with(
        "observational-memory",
        vec![Injection::from(LLM_SERVICE)],
        |ctx: Context, config: Value| async move {
            let config: OmConfig = if config.is_null() {
                OmConfig::default()
            } else {
                serde_json::from_value(config)
                    .map_err(|err| cordis::Error::msg(format!("observational-memory config: {err}")))?
            };
            let llm = ctx.require::<LlmService>(LLM_SERVICE)?;
            let memory = Arc::new(Memory::default());
            let config = Arc::new(config);
            let events = ctx.clone();

            ctx.on("llm/stream", move |_ctx, payload, next| {
                let llm = llm.clone();
                let memory = memory.clone();
                let config = config.clone();
                let events = events.clone();
                Box::pin(async move {
                    let Ok(mut options) = serde_json::from_value::<GenerateOptions>(payload.clone()) else {
                        return next.run(payload).await;
                    };
                    let Some(session) = options.session_id.clone() else {
                        return next.run(payload).await;
                    };
                    let cell = memory.session(&session, config.dir.as_ref());
                    let mut state = cell.lock().await;
                    let before = state.clone();
                    let report = compact(&llm, &config, &mut state, &mut options).await;
                    if *state != before {
                        save(config.dir.as_ref(), &session, &state);
                    }
                    drop(state);
                    if report.as_object().is_some_and(|r| !r.is_empty()) {
                        events.emit("om/compacted", json!({ "session": session, "report": report }));
                    }
                    let payload = serde_json::to_value(&options).map_err(|err| cordis::Error::msg(err.to_string()))?;
                    next.run(payload).await
                })
            })
            .await?;
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{MessageSource, Role};

    fn text(id: &str, role: Role, body: &str) -> Message {
        Message { id: id.into(), role, content: vec![ContentBlock::text(body)], source: MessageSource::User }
    }

    fn tool_result(id: &str, call: &str) -> Message {
        Message {
            id: id.into(),
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_call_id: call.into(),
                content: vec![ContentBlock::text("x".repeat(400))],
                is_error: None,
            }],
            source: MessageSource::User,
        }
    }

    #[test]
    fn utc_date_is_well_formed() {
        let (date, time) = utc_now();
        assert_eq!(date.len(), 10);
        assert_eq!(time.len(), 5);
        assert!(date.starts_with("20"));
    }

    #[test]
    fn boundary_lookup_resets_on_foreign_history() {
        let messages = vec![text("a", Role::User, "hi"), text("b", Role::Assistant, "yo")];
        assert_eq!(first_unobserved(&messages, None), Some(0));
        assert_eq!(first_unobserved(&messages, Some("a")), Some(1));
        assert_eq!(first_unobserved(&messages, Some("zzz")), None);
    }

    #[test]
    fn cut_never_orphans_a_tool_result() {
        let messages = vec![
            text("u1", Role::User, &"a".repeat(400)),
            text("a1", Role::Assistant, "calling"),
            tool_result("t1", "c1"),
            tool_result("t2", "c2"),
        ];
        // A tiny keep budget would cut at t2; the cut must back up to a1.
        let cut = choose_cut(&messages, 0, 10).unwrap();
        assert_eq!(messages[cut].id, "a1");
    }

    #[test]
    fn nothing_to_observe_in_a_single_message() {
        let messages = vec![text("u1", Role::User, "hi")];
        assert_eq!(choose_cut(&messages, 0, 10), None);
    }

    #[test]
    fn apply_memory_moves_observations_into_system() {
        let mut options = GenerateOptions {
            provider: "mock".into(),
            model: "m".into(),
            messages: vec![text("u1", Role::User, "old"), text("u2", Role::User, "new")],
            system: Some("base".into()),
            tools: None,
            temperature: None,
            max_tokens: None,
            stop: None,
            session_id: Some("s".into()),
        };
        let state = OmState { observations: "- 🔴 10:00 user likes Rust".into(), observed_upto: Some("u1".into()) };
        apply_memory(&mut options, &state, 1);
        assert_eq!(options.messages.len(), 1);
        assert_eq!(options.messages[0].id, "u2");
        let system = options.system.unwrap();
        assert!(system.starts_with("base\n\n<observations>"));
        assert!(system.contains("user likes Rust"));
    }
}
