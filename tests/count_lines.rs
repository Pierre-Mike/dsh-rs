//! End-to-end test for the `count_lines` example plugin: a scripted model
//! calls the tool through the real agent loop, no network, no LM Studio.
//! Copy this file as the template for testing your own plugins.

mod common;

use dsh_rs::core::agent::user_message_with_text;
use dsh_rs::session::SessionEventData;
use serde_json::json;

#[tokio::test]
async fn model_calls_count_lines_through_the_agent_loop() {
    // A file the tool will count: 3 lines, 6 words.
    let dir = std::env::temp_dir().join("dsh-count-lines-test");
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("sample.txt");
    std::fs::write(&file, "one two\nthree four\nfive six\n").unwrap();

    // 1. Boot the base bundle, then mount your plugin beside it.
    let ctx = cordis::Context::new();
    dsh_rs::bundle::install_base_default(&ctx).await.unwrap();
    for (plugin, config) in dsh_rs::plugins::extra_plugins(&serde_json::Value::Null) {
        ctx.plugin(plugin, config).join().await.unwrap();
    }

    // 2. Script the model: first request -> call the tool, second -> answer.
    common::boot_scripted(
        &ctx,
        false,
        vec![
            common::tool_call_response(
                "call-1",
                "count_lines",
                json!({ "path": file.display().to_string() }),
            ),
            common::text_response("it has 3 lines"),
        ],
    )
    .await;

    // 3. Drive one user turn and wait for the loop to settle.
    let agent = common::create_agent(&ctx).await;
    agent.followup(user_message_with_text("u-1", "count the lines"));
    agent.when_idle().await;

    // 4. Assert on the session log: the tool ran and returned real counts.
    let events = agent.session().events();
    let result = events
        .iter()
        .find_map(|e| match &e.data {
            SessionEventData::ToolResult { message, .. } => Some(serde_json::to_string(message).unwrap()),
            _ => None,
        })
        .expect("a tool/result event");
    assert!(result.contains("\\\"lines\\\":3") || result.contains("\"lines\":3"), "got: {result}");
    assert!(!result.contains("READ_FAILED"), "got: {result}");
}
