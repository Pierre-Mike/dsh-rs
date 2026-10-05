//! `claude-code` provider: Claude on your subscription, through the real
//! Claude Code CLI (`claude -p`).
//!
//! Why the CLI and not the Messages API: `claude -p` draws from the logged-in
//! plan's limits, while a direct API call is billed per token. Two guards keep
//! it that way:
//! - `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` are removed from the child
//!   environment, so Claude Code can only use its OAuth login;
//! - the `system/init` event must report `apiKeySource: "none"`, otherwise the
//!   call is refused before any answer is used.
//!
//! Claude Code runs with every built-in tool disabled, so it acts as a plain
//! model. dsh's own tools are offered through a small text protocol: the model
//! writes `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` and this
//! adapter turns each block into a native `ToolCall`.
//!
//! Config (`plugins.claude-code` in dsh.json, all optional):
//! `{ "command": "claude", "model": "claude-sonnet-5-5", "providers": ["claude"],
//!    "effort": "low", "timeout_secs": 600 }`

use std::process::Stdio;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use cordis::plugin::BoxFuture;
use cordis::{plugin_with, Context, Injection, Plugin};
use serde::Deserialize;
use serde_json::{json, Value};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::api::services::{LlmAdapterApi, LlmService};
use crate::api::LLM_SERVICE;
use crate::types::{
    BoxStream, ContentBlock, FinishReason, GenerateOptions, LlmError, Message, Role,
    StreamChunk, TokenUsage, ToolSchema, stream_from_chunks,
};

const DEFAULT_MODEL: &str = "claude-sonnet-5-5";
const BILLING_ENV: [&str; 2] = ["ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"];
const TURN_INSTRUCTION: &str = "The conversation so far is on stdin. Write only the next assistant turn.";

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClaudeCodeConfig {
    pub command: String,
    pub model: String,
    pub providers: Vec<String>,
    pub effort: Option<String>,
    pub timeout_secs: u64,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        ClaudeCodeConfig {
            command: "claude".to_string(),
            model: DEFAULT_MODEL.to_string(),
            providers: vec!["claude".to_string()],
            effort: None,
            timeout_secs: 600,
        }
    }
}

pub struct ClaudeCodeAdapter {
    config: ClaudeCodeConfig,
}

impl ClaudeCodeAdapter {
    pub fn new(config: ClaudeCodeConfig) -> Self {
        ClaudeCodeAdapter { config }
    }

    fn args(&self, system: &str, model: &str) -> Vec<String> {
        let mut args: Vec<String> = [
            "-p", TURN_INSTRUCTION,
            "--system-prompt", system,
            "--model", model,
            "--tools", "",
            "--setting-sources", "",
            "--strict-mcp-config",
            "--disable-slash-commands",
            "--no-session-persistence",
            "--output-format", "stream-json",
            "--verbose",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        if let Some(effort) = &self.config.effort {
            args.push("--effort".to_string());
            args.push(effort.clone());
        }
        args
    }

    async fn run(&self, options: GenerateOptions) -> Result<Vec<StreamChunk>, LlmError> {
        let model = if options.model.is_empty() { self.config.model.clone() } else { options.model.clone() };
        let tools = options.tools.clone().unwrap_or_default();
        let system = system_prompt(options.system.as_deref(), &tools);
        let transcript = render_transcript(&options.messages);

        let mut command = tokio::process::Command::new(&self.config.command);
        command
            .args(self.args(&system, &model))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for var in BILLING_ENV {
            command.env_remove(var);
        }
        let mut child = command
            .spawn()
            .map_err(|err| LlmError::code("SPAWN", format!("cannot run {}: {err}", self.config.command)))?;

        let mut stdin = child.stdin.take().expect("piped stdin");
        stdin
            .write_all(transcript.as_bytes())
            .await
            .map_err(|err| LlmError::code("STDIN", err.to_string()))?;
        drop(stdin);

        let stdout = child.stdout.take().expect("piped stdout");
        let read = async {
            let mut lines = BufReader::new(stdout).lines();
            let mut result: Option<Value> = None;
            while let Some(line) = lines.next_line().await.map_err(|err| LlmError::code("STDOUT", err.to_string()))? {
                let Ok(event) = serde_json::from_str::<Value>(&line) else { continue };
                match event.get("type").and_then(Value::as_str) {
                    Some("system") if event.get("subtype").and_then(Value::as_str) == Some("init") => {
                        check_auth(&event)?;
                    }
                    Some("result") => result = Some(event),
                    _ => {}
                }
            }
            result.ok_or_else(|| LlmError::code("NO_RESULT", "claude -p exited without a result event"))
        };
        let timeout = Duration::from_secs(self.config.timeout_secs);
        let outcome = tokio::time::timeout(timeout, read)
            .await
            .map_err(|_| LlmError::code("TIMEOUT", format!("claude -p took longer than {}s", timeout.as_secs())));
        let result = match outcome {
            Ok(Ok(result)) => result,
            Ok(Err(err)) | Err(err) => {
                let _ = child.kill().await;
                return Err(err);
            }
        };
        let _ = child.wait().await;

        if result.get("is_error").and_then(Value::as_bool).unwrap_or(false) {
            let text = result.get("result").and_then(Value::as_str).unwrap_or("claude -p reported an error");
            return Err(LlmError::code("CLAUDE_CODE", text.to_string()));
        }
        let text = result.get("result").and_then(Value::as_str).unwrap_or_default();
        Ok(response_chunks(text, usage(&result)))
    }
}

impl LlmAdapterApi for ClaudeCodeAdapter {
    fn name(&self) -> &'static str {
        "claude-code"
    }

    fn stream(&self, options: GenerateOptions) -> BoxFuture<Result<BoxStream<StreamChunk>, LlmError>> {
        let adapter = ClaudeCodeAdapter::new(self.config.clone());
        Box::pin(async move { adapter.run(options).await.map(stream_from_chunks) })
    }
}

/// Refuse anything but the subscription login.
fn check_auth(init: &Value) -> Result<(), LlmError> {
    match init.get("apiKeySource").and_then(Value::as_str) {
        Some("none") => Ok(()),
        other => Err(LlmError::code(
            "BILLING_GUARD",
            format!("claude -p would authenticate with {other:?}, not the subscription login; refusing"),
        )),
    }
}

fn usage(result: &Value) -> TokenUsage {
    let u = &result["usage"];
    let n = |key: &str| u.get(key).and_then(Value::as_u64);
    TokenUsage {
        input_tokens: n("input_tokens").unwrap_or(0),
        output_tokens: n("output_tokens").unwrap_or(0),
        cache_read_tokens: n("cache_read_input_tokens"),
        cache_write_tokens: n("cache_creation_input_tokens"),
        reasoning_tokens: None,
    }
}

// ---------------------------------------------------------------------------
// Pure core: prompt rendering and response parsing.
// ---------------------------------------------------------------------------

/// The dsh system prompt plus, when tools exist, the tool-call protocol.
pub fn system_prompt(system: Option<&str>, tools: &[ToolSchema]) -> String {
    let mut out = system.unwrap_or("You are a helpful assistant.").to_string();
    if tools.is_empty() {
        return out;
    }
    out.push_str(
        "\n\n# Tools\n\
         You can call these tools. To call one, write a block exactly like:\n\
         <tool_call>{\"name\": \"TOOL_NAME\", \"arguments\": {...}}</tool_call>\n\
         Use exactly the tag <tool_call>. You may write several blocks. End your reply \
         immediately after the last </tool_call>: you do not know the result yet. The \
         harness runs the tool and sends the real output in the next turn as a \
         <tool_result> block. Never write <tool_result> yourself.\n\n\
         Available tools (JSON Schema for arguments):\n",
    );
    for tool in tools {
        out.push_str(&format!(
            "- {}: {}\n  arguments: {}\n",
            tool.name, tool.description, tool.parameters
        ));
    }
    out
}

/// Render the message history as a plain transcript for stdin.
pub fn render_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<{role}>\n"));
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => {
                    out.push_str(text);
                    out.push('\n');
                }
                ContentBlock::Reasoning { .. } => {}
                ContentBlock::ToolCall { name, arguments, .. } => {
                    let arguments: Value = serde_json::from_str(arguments).unwrap_or(Value::Null);
                    out.push_str(&format!(
                        "<tool_call>{}</tool_call>\n",
                        json!({ "name": name, "arguments": arguments })
                    ));
                }
                ContentBlock::ToolResult { tool_call_id, content, is_error } => {
                    let text: Vec<&str> = content.iter().filter_map(|b| b.as_text()).collect();
                    let error = if is_error.unwrap_or(false) { " error=\"true\"" } else { "" };
                    out.push_str(&format!(
                        "<tool_result id=\"{tool_call_id}\"{error}>\n{}\n</tool_result>\n",
                        text.join("\n")
                    ));
                }
            }
        }
        out.push_str(&format!("</{role}>\n\n"));
    }
    out
}

/// Tag names models actually use for the protocol; `tool_call` is the one we ask for.
const CALL_TAGS: [&str; 3] = ["tool_call", "function_call", "call"];

/// The earliest `<tag>…</tag>` block in `text`: (start, end, body). Open and
/// close tags may be different variants (small models mix them up).
fn next_block(text: &str) -> Option<(usize, usize, &str)> {
    let (start, open_len) = CALL_TAGS
        .iter()
        .filter_map(|tag| text.find(&format!("<{tag}>")).map(|i| (i, tag.len() + 2)))
        .min_by_key(|(i, _)| *i)?;
    let body_start = start + open_len;
    let (body_len, close_len) = CALL_TAGS
        .iter()
        .filter_map(|tag| text[body_start..].find(&format!("</{tag}>")).map(|i| (i, tag.len() + 3)))
        .min_by_key(|(i, _)| *i)?;
    Some((start, body_start + body_len + close_len, &text[body_start..body_start + body_len]))
}

/// Split a reply into its prose and its tool-call blocks.
///
/// Anything after the last tool call is dropped: the model cannot know a
/// tool's result yet, so text there is invented (typically a fake
/// `<tool_result>`).
pub fn parse_reply(text: &str) -> (String, Vec<(String, Value)>) {
    let mut prose = String::new();
    let mut calls = Vec::new();
    let mut rest = text;
    while let Some((start, end, body)) = next_block(rest) {
        match serde_json::from_str::<Value>(body.trim()) {
            Ok(call) if call.get("name").and_then(Value::as_str).is_some() => {
                prose.push_str(&rest[..start]);
                let name = call["name"].as_str().unwrap_or_default().to_string();
                let arguments = call.get("arguments").cloned().unwrap_or_else(|| json!({}));
                calls.push((name, arguments));
            }
            // Not valid JSON: keep it as text so nothing silently disappears.
            _ => prose.push_str(&rest[..end]),
        }
        rest = &rest[end..];
    }
    if calls.is_empty() {
        prose.push_str(rest);
    }
    (prose.trim().to_string(), calls)
}

static CALL_SEQ: AtomicU64 = AtomicU64::new(0);

fn call_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("cc_{nanos:x}_{}", CALL_SEQ.fetch_add(1, Ordering::Relaxed))
}

/// Turn a finished reply into the chunk sequence the agent loop expects.
pub fn response_chunks(text: &str, usage: TokenUsage) -> Vec<StreamChunk> {
    let (prose, calls) = parse_reply(text);
    let mut chunks = Vec::new();
    let mut index = 0;
    if !prose.is_empty() {
        chunks.push(StreamChunk::BlockStart { index, block_type: "text".to_string() });
        chunks.push(StreamChunk::TextDelta { index, text: prose.clone() });
        chunks.push(StreamChunk::BlockEnd { index, block: ContentBlock::text(prose) });
        index += 1;
    }
    let has_calls = !calls.is_empty();
    for (name, arguments) in calls {
        let id = call_id();
        let arguments = arguments.to_string();
        chunks.push(StreamChunk::BlockStart { index, block_type: "tool-call".to_string() });
        chunks.push(StreamChunk::ToolCallDelta {
            index,
            id: id.clone(),
            name: Some(name.clone()),
            arguments_delta: arguments.clone(),
        });
        chunks.push(StreamChunk::BlockEnd { index, block: ContentBlock::ToolCall { id, name, arguments } });
        index += 1;
    }
    chunks.push(StreamChunk::Usage { usage });
    chunks.push(StreamChunk::Finish {
        reason: if has_calls { FinishReason::ToolCalls } else { FinishReason::Stop },
    });
    chunks
}

// ---------------------------------------------------------------------------
// The plugin: registers the adapter on `ctx.llm` for as long as it is loaded.
// ---------------------------------------------------------------------------

pub fn plugin() -> Arc<dyn Plugin> {
    plugin_with(
        "claude-code",
        vec![Injection::from(LLM_SERVICE)],
        |ctx: Context, config: Value| async move {
            let config: ClaudeCodeConfig = if config.is_null() {
                ClaudeCodeConfig::default()
            } else {
                serde_json::from_value(config).map_err(|err| cordis::Error::msg(format!("claude-code config: {err}")))?
            };
            let llm = ctx.require::<LlmService>(LLM_SERVICE)?;
            ctx.effect("claude-code adapter", async move {
                let routes = config.providers.clone();
                let refs: Vec<&str> = routes.iter().map(String::as_str).collect();
                llm.register_adapter(&refs, Arc::new(ClaudeCodeAdapter::new(config)))
                    .map_err(|err| cordis::Error::msg(err.to_string()))?;
                let disposer: cordis::fiber::Disposer = Box::new(move || {
                    Box::pin(async move {
                        let refs: Vec<&str> = routes.iter().map(String::as_str).collect();
                        llm.unregister_adapter(&refs);
                    })
                });
                Ok(Some(disposer))
            })
            .await?;
            Ok(())
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_prose_and_tool_calls() {
        let (prose, calls) = parse_reply(
            "Let me check.\n<tool_call>{\"name\": \"bash\", \"arguments\": {\"command\": \"ls\"}}</tool_call>",
        );
        assert_eq!(prose, "Let me check.");
        assert_eq!(calls, vec![("bash".to_string(), json!({ "command": "ls" }))]);
    }

    #[test]
    fn drops_invented_results_and_accepts_tag_variants() {
        let (prose, calls) = parse_reply(
            "<call>\n{\"name\": \"count_lines\", \"arguments\": {\"path\": \"README.md\"}}\n</call>\n\
             <tool_result>\nlines: 45\n</tool_result>\n\nREADME.md contains 45 lines.",
        );
        assert_eq!(calls, vec![("count_lines".to_string(), json!({ "path": "README.md" }))]);
        assert_eq!(prose, "", "the invented result and answer are gone");
    }

    #[test]
    fn accepts_mismatched_open_and_close_tags() {
        let (_, calls) = parse_reply("<call>{\"name\": \"x\", \"arguments\": {}}</tool_call>");
        assert_eq!(calls, vec![("x".to_string(), json!({}))]);
    }

    #[test]
    fn keeps_malformed_tool_calls_as_text() {
        let (prose, calls) = parse_reply("<tool_call>not json</tool_call> done");
        assert!(calls.is_empty());
        assert_eq!(prose, "<tool_call>not json</tool_call> done");
    }

    #[test]
    fn tool_calls_finish_with_tool_calls_reason() {
        let chunks = response_chunks("<tool_call>{\"name\": \"x\"}</tool_call>", TokenUsage::default());
        assert!(matches!(chunks.last(), Some(StreamChunk::Finish { reason: FinishReason::ToolCalls })));
    }

    #[test]
    fn refuses_api_key_auth() {
        assert!(check_auth(&json!({ "apiKeySource": "none" })).is_ok());
        let err = check_auth(&json!({ "apiKeySource": "ANTHROPIC_API_KEY" })).unwrap_err();
        assert_eq!(err.failure.code, "BILLING_GUARD");
    }

    #[test]
    fn transcript_round_trips_tool_protocol() {
        let messages = vec![
            Message::user("u1", vec![ContentBlock::text("hi")]),
            Message {
                id: "a1".into(),
                role: Role::Assistant,
                content: vec![ContentBlock::ToolCall { id: "c1".into(), name: "bash".into(), arguments: "{\"command\":\"ls\"}".into() }],
                source: crate::types::MessageSource::User,
            },
        ];
        let text = render_transcript(&messages);
        assert!(text.contains("<user>\nhi\n</user>"));
        assert!(text.contains("<tool_call>{\"arguments\":{\"command\":\"ls\"},\"name\":\"bash\"}</tool_call>"));
    }
}
