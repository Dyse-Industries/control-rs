//! Hardware-in-the-Loop (HIL) testing framework for control-rs.
//! Provides test runner execution, telemetry serialization, and settings management.

#![no_std]
#![allow(clippy::multiple_crate_versions)]

pub use executor::{DummyExecutor, TestExecutor};
pub use server::Server;
pub use settings::Setting;

pub mod comms;
pub mod executor;
pub mod server;
pub mod settings;
pub mod time;
pub mod util;

/// Describes a single test executable.
#[derive(Debug, Clone, Copy)]
pub struct ExecDescriptor {
    /// The doc comment description of the test.
    pub description: &'static str,
    /// The name of the test executable.
    pub name: &'static str,
    /// A function pointer to the test executable.
    pub test_fn: fn(),
}

/// A slice of settings for a test suite.
pub type SettingsSlice = &'static [&'static dyn Setting];

/// Describes a test suite.
pub struct SuiteDescriptor {
    /// The doc comment description of the test suite.
    pub description: &'static str,
    /// A slice of test executables in this suite.
    pub executables: &'static [ExecDescriptor],
    /// The name of the test suite.
    pub name: &'static str,
    /// A slice of configurable settings for this suite.
    pub settings: SettingsSlice,
}
