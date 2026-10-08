//! Host-side library for Embedded Test Server (ETS) interaction, transport, and testing.
//!
//! Provides `ETSBridge` transport abstraction, framed packet communication,
//! interactive session state management, and headless target execution (`run_headless_ets`).
//!
//! # Features
//!
//! - `fake-link`: exposes an in-memory `ETSBridge` link (`ETSBridge::fake`) on
//!   Unix hosts for testing code that drives a bridge. It is for tests only.

#[cfg(all(any(test, feature = "fake-link"), unix))]
pub use bridge::FakeBridge;
pub use bridge::{BridgeMessage, ETSBridge, OwnedTelemetry};
pub use error::HostError;
pub use runner::{
    Completion, RunOptions, RunRecord, TestOutcome, run_headless_ets,
    run_headless_ets_with_options, run_headless_ets_with_tasks,
};
pub use session::{
    DEFAULT_STOP_TIMEOUT, HEARTBEAT_PERIOD, SETTINGS_READY, SUITE_INFO_READY,
    SUITE_READY_MASK, SessionAction, SessionPhase, SessionState, SettingItem,
    SuiteItem, TESTS_READY, TargetInfo, TaskItem, TaskRunRecord, TaskStart,
    TaskStartError, TestIndex, TestItem,
};
pub use target::{
    QemuArch, QemuTargetDetails, SubprocessTarget, Target, build_target_elf,
    target_elf_path,
};

pub mod bridge;
pub mod error;
pub mod runner;
pub mod session;
pub mod sim;
pub mod target;
