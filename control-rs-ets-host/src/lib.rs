//! Host-side library for Embedded Test Server (ETS) interaction, transport, and testing.
//!
//! Provides `ServerBridge` transport abstraction, framed packet communication,
//! interactive session state management, and headless target execution (`run_headless_ets`).

#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::cast_possible_truncation,
    clippy::doc_markdown,
    clippy::indexing_slicing,
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::multiple_crate_versions,
    clippy::needless_pass_by_value,
    clippy::or_fun_call,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]

pub mod bridge;
pub mod error;
pub mod runner;
pub mod session;
pub mod target;

pub use bridge::{BridgeMessage, ServerBridge};
pub use error::HostError;
pub use runner::{Completion, EtsRunResult, TestOutcome, run_headless_ets};
pub use session::{
    SessionAction, SessionState, SettingItem, SuiteItem, TestItem,
};
pub use target::{
    QemuArch, QemuTargetDetails, SubprocessTarget, Target, build_target_elf,
    target_elf_path,
};
