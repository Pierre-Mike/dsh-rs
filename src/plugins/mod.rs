//! # plugins
//!
//! Your own static plugins. Each one is a normal cordis plugin compiled into
//! the `dsh` binary, so it gets the full kernel API (inject, provide, effects,
//! events). `boot()` in `main.rs` mounts everything [`extra_plugins`] returns
//! beside the base bundle.
//!
//! Each plugin reads its config from `plugins.<name>` in dsh.json. Set a
//! section to `false` to turn that plugin off.
//!
//! To add a plugin: write a module here, add it to `extra_plugins`.

pub mod claude_code;
pub mod count_lines;
pub mod observational_memory;

use std::sync::Arc;

use cordis::Plugin;
use serde_json::Value;

/// How a plugin is switched on when dsh.json says nothing about it.
enum Default {
    On,
    Off,
}

/// Every local plugin to mount, with its config section. `plugins` is the
/// `plugins` object from dsh.json (or `Value::Null`).
pub fn extra_plugins(plugins: &Value) -> Vec<(Arc<dyn Plugin>, Option<Value>)> {
    let all: Vec<(&str, Arc<dyn Plugin>, Default)> = vec![
        ("count-lines", count_lines::plugin(), Default::On),
        ("claude-code", claude_code::plugin(), Default::On),
        // Changes what the model sees, so only on when configured.
        ("observational-memory", observational_memory::plugin(), Default::Off),
    ];
    all.into_iter()
        .filter_map(|(name, plugin, default)| match (plugins.get(name), default) {
            (Some(Value::Bool(false)), _) => None,
            (None, Default::Off) => None,
            (Some(Value::Bool(true)) | None, Default::On) | (Some(Value::Bool(true)), Default::Off) => {
                Some((plugin, None))
            }
            (Some(section), _) => Some((plugin, Some(section.clone()))),
        })
        .collect()
}
