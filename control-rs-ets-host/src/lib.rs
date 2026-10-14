//! Host-side library for Embedded Test Server (ETS) interaction, transport, and testing.
//!
//! Provides `ETSBridge` transport abstraction, framed packet communication,
//! interactive session state management, and headless target execution (`run_headless`).
//!
//! # Features
//!
//! - `fake-link`: exposes an in-memory `ETSBridge` link (`ETSBridge::fake`) on
//!   Unix hosts for testing code that drives a bridge. It is for tests only.

#[cfg(all(any(test, feature = "fake-link"), unix))]
pub use bridge::FakeBridge;
pub use bridge::{BridgeMessage, ETSBridge, OwnedTelemetry, TestState};
pub use catalog::{Catalog, ItemId};
pub use error::HostError;
pub use headless::{
    Completion, RunOptions, RunRecord, TestOutcome, run_headless,
};
pub use link::{HostLink, LinkEvent, Polled};
pub use session::{CaseOutcome, CaseStatus, Session, SessionOptions};
pub use state::{
    SETTINGS_READY, SUITE_INFO_READY, SUITE_READY_MASK, SessionAction,
    SessionPhase, SessionState, SettingItem, SuiteItem, TESTS_READY,
    TargetInfo, TestIndex, TestItem,
};
pub use target::{
    QemuArch, QemuTargetDetails, SubprocessTarget, Target, build_target_elf,
    target_elf_path,
};
pub use task_run::{
    RunControl, RunObserver, RunStatus, TaskOptions, TaskRecord, TaskRun,
    TaskSim,
};

pub mod bridge;
pub mod catalog;
pub mod error;
pub mod headless;
pub mod link;
pub mod session;
pub mod state;
pub mod target;
pub mod task_run;

#[cfg(test)]
mod testing;
