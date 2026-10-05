//! Example plugin: a `count_lines` tool.
//!
//! Shows the three moves every dsh plugin makes:
//! 1. declare what it needs (`inject: tools`), so it only starts once the
//!    tool registry is live;
//! 2. register its work inside a fiber effect;
//! 3. return a disposer, so unloading the plugin removes the tool.

use std::sync::Arc;

use cordis::{plugin_with, Context, Injection, Plugin};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::services::{DynamicToolSpec, ToolsService};
use crate::api::TOOLS_SERVICE;
use crate::types::ToolExecutionResult;

const TOOL_NAME: &str = "count_lines";

#[derive(Deserialize)]
struct Args {
    path: String,
}

/// Pure core: count lines, words and bytes of a text.
pub fn count(text: &str) -> Value {
    json!({
        "lines": text.lines().count(),
        "words": text.split_whitespace().count(),
        "bytes": text.len(),
    })
}

async fn execute(arguments: Value) -> ToolExecutionResult {
    let args: Args = match serde_json::from_value(arguments) {
        Ok(args) => args,
        Err(err) => return ToolExecutionResult::error("INVALID_ARGS", err.to_string()),
    };
    match tokio::fs::read_to_string(&args.path).await {
        Ok(text) => {
            let mut value = count(&text);
            value["path"] = Value::String(args.path);
            ToolExecutionResult::success_text(value.to_string(), value)
        }
        Err(err) => ToolExecutionResult::error(
            "READ_FAILED",
            format!("cannot read {}: {err}", args.path),
        ),
    }
}

pub fn plugin() -> Arc<dyn Plugin> {
    plugin_with(
        "count-lines",
        vec![Injection::from(TOOLS_SERVICE)],
        |ctx: Context, _config: ()| async move {
            let registry = ctx.require::<ToolsService>(TOOLS_SERVICE)?;
            ctx.effect("count_lines tool", async move {
                registry.register_dynamic_tool(DynamicToolSpec {
                    name: TOOL_NAME.to_string(),
                    description: "Count the lines, words and bytes of a text file.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "path": { "type": "string", "description": "Path of the file to count." }
                        },
                        "required": ["path"]
                    }),
                    exec: Arc::new(|arguments| Box::pin(execute(arguments))),
                });
                let disposer: cordis::fiber::Disposer = Box::new(move || {
                    Box::pin(async move { registry.unregister_dynamic_tool(TOOL_NAME) })
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
    fn counts_lines_words_bytes() {
        assert_eq!(count("a b\nc\n"), json!({ "lines": 2, "words": 3, "bytes": 6 }));
    }

    #[tokio::test]
    async fn registers_and_unregisters_with_the_plugin() {
        let ctx = Context::new();
        crate::bundle::install_base_default(&ctx).await.unwrap();
        let fiber = ctx.plugin(plugin(), None);
        fiber.join().await.unwrap();

        let tools = ctx.require::<ToolsService>(TOOLS_SERVICE).unwrap();
        assert!(tools.list().iter().any(|name| name == TOOL_NAME));

        fiber.dispose().await;
        assert!(!tools.list().iter().any(|name| name == TOOL_NAME));
    }
}
