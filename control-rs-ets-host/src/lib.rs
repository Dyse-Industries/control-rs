//! Host-side library for Embedded Test Server (ETS) interaction, transport, and testing.
//!
//! Provides `ETSBridge` transport abstraction, framed packet communication,
//! interactive session state management, and headless target execution (`run_headless_ets`).

pub use bridge::{BridgeMessage, ETSBridge, OwnedTelemetry};
pub use error::HostError;
pub use runner::{
    Completion, RunOptions, RunRecord, TestOutcome, run_headless_ets,
    run_headless_ets_with_options,
};
pub use session::{
    SETTINGS_READY, SUITE_INFO_READY, SUITE_READY_MASK, SessionAction,
    SessionPhase, SessionState, SettingItem, SuiteItem, TESTS_READY, TestIndex,
    TestItem,
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
