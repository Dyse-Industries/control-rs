//! Host-side library for Embedded Test Server (ETS) interaction, transport, and testing.
//!
//! Provides `ETSBridge` transport abstraction, framed packet communication,
//! interactive session state management, and headless target execution (`run_headless_ets`).

#![allow(
    clippy::indexing_slicing,
    clippy::multiple_crate_versions,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::type_complexity
)]

pub use bridge::{BridgeMessage, ETSBridge, OwnedTelemetry};
pub use error::HostError;
pub use runner::{
    Completion, EtsRunResult, RunRecord, TestOutcome, run_headless_ets,
};
pub use session::{
    SessionAction, SessionState, SettingItem, SuiteItem, TestItem,
};
pub use target::{
    QemuArch, QemuTargetDetails, SubprocessTarget, Target, build_target_elf,
    target_elf_path,
};

pub mod bridge;
pub mod error;
pub mod runner;
pub mod session;
pub mod target;
