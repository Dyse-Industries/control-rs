//! Host testing and interactive TUI tooling for HIL (Hardware-in-the-Loop),
//! built as an `xtask`-style task runner.
#![allow(
    clippy::panic,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::indexing_slicing,
    clippy::must_use_candidate,
    clippy::multiple_crate_versions
)]

pub mod bridge;
pub mod tui;
