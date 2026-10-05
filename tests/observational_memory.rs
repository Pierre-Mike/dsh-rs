//! End-to-end test for the observational-memory plugin through the real agent
//! loop: a recording model sees what each request contains, scripted models
//! play the Observer (one fails first, to exercise the fallback order).

mod common;

use std::sync::{Arc, Mutex};

use cordis::plugin::BoxFuture;
use dsh_rs::api::services::LlmService;
use dsh_rs::core::agent::user_message_with_text;
use dsh_rs::llm::{BoxStream, GenerateOptions, LlmAdapter, LlmError, StreamChunk, runtime::stream_from_chunks};
use serde_json::json;

/// Records every request and answers "ok".
struct Recorder(Arc<Mutex<Vec<GenerateOptions>>>);

impl LlmAdapter for Recorder {
    fn name(&self) -> &'static str {
        "recorder"
    }
    fn stream(&self, options: GenerateOptions) -> BoxFuture<Result<BoxStream<StreamChunk>, LlmError>> {
        self.0.lock().unwrap().push(options);
        Box::pin(async { Ok(stream_from_chunks(common::text_response("ok"))) })
    }
}

/// An Observer that always answers with a fixed note.
struct Observer;

impl LlmAdapter for Observer {
    fn name(&self) -> &'static str {
        "observer"
    }
    fn stream(&self, _options: GenerateOptions) -> BoxFuture<Result<BoxStream<StreamChunk>, LlmError>> {
        Box::pin(async { Ok(stream_from_chunks(common::text_response("- 🔴 00:00 OBS-NOTE user said things"))) })
    }
}

/// An Observer route that is down.
struct Broken;

impl LlmAdapter for Broken {
    fn name(&self) -> &'static str {
        "broken"
    }
    fn stream(&self, _options: GenerateOptions) -> BoxFuture<Result<BoxStream<StreamChunk>, LlmError>> {
        Box::pin(async { Err(LlmError::code("DOWN", "unavailable")) })
    }
}

#[tokio::test]
async fn old_messages_become_observations() {
    let ctx = cordis::Context::new();
    dsh_rs::bundle::install_base_default(&ctx).await.unwrap();

    let requests = Arc::new(Mutex::new(Vec::new()));
    let llm = ctx.require::<LlmService>(dsh_rs::api::LLM_SERVICE).unwrap();
    llm.unregister_adapter(&["mock"]);
    llm.register_adapter(&["mock"], Arc::new(Recorder(requests.clone()))).unwrap();
    llm.register_adapter(&["broken"], Arc::new(Broken)).unwrap();
    llm.register_adapter(&["obs"], Arc::new(Observer)).unwrap();

    // Tiny thresholds so a few turns trigger the Observer. Model order:
    // a missing route, a broken one, then the working one.
    let plugins = json!({
        "observational-memory": {
            "models": [
                { "provider": "not-registered", "model": "x" },
                { "provider": "broken", "model": "x" },
                { "provider": "obs", "model": "x" }
            ],
            "message_tokens": 60,
            "keep_ratio": 0.2
        }
    });
    for (plugin, config) in dsh_rs::plugins::extra_plugins(&plugins) {
        ctx.plugin(plugin, config).join().await.unwrap();
    }

    let agent = common::create_agent(&ctx).await;
    for i in 0..4 {
        agent.followup(user_message_with_text(format!("u-{i}"), format!("turn {i}: {}", "words ".repeat(20))));
        agent.when_idle().await;
    }

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 4, "one model request per turn");

    // Early requests go out untouched.
    assert!(requests[0].system.as_deref().is_none_or(|s| !s.contains("<observations>")));

    // Once past the threshold, old messages are replaced by the observation log.
    let last = requests.last().unwrap();
    let system = last.system.clone().unwrap_or_default();
    assert!(system.contains("<observations>"), "system: {system}");
    assert!(system.contains("OBS-NOTE"), "fell back to the working observer");
    let full_history = 4 * 2 - 1; // 4 user messages + 3 earlier answers
    assert!(last.messages.len() < full_history, "kept {} of {full_history}", last.messages.len());
    assert_eq!(last.messages.last().unwrap().id, "u-3", "the newest message is always kept");
}

#[tokio::test]
async fn compaction_failure_never_breaks_the_turn() {
    let ctx = cordis::Context::new();
    dsh_rs::bundle::install_base_default(&ctx).await.unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let llm = ctx.require::<LlmService>(dsh_rs::api::LLM_SERVICE).unwrap();
    llm.unregister_adapter(&["mock"]);
    llm.register_adapter(&["mock"], Arc::new(Recorder(requests.clone()))).unwrap();
    llm.register_adapter(&["broken"], Arc::new(Broken)).unwrap();

    let plugins = json!({
        "observational-memory": { "models": [{ "provider": "broken", "model": "x" }], "message_tokens": 10 }
    });
    for (plugin, config) in dsh_rs::plugins::extra_plugins(&plugins) {
        ctx.plugin(plugin, config).join().await.unwrap();
    }

    let agent = common::create_agent(&ctx).await;
    for i in 0..3 {
        agent.followup(user_message_with_text(format!("u-{i}"), "words ".repeat(20)));
        agent.when_idle().await;
    }
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 3);
    assert_eq!(requests[2].messages.len(), 5, "uncompacted history still goes out");
}
