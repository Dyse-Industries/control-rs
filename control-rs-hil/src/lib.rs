#![no_std]

pub mod comms;
pub mod settings;
pub use settings::Setting;
pub mod server;
pub use server::Server;
pub mod time;

/// Describes a single test executable.
#[derive(Debug, Clone, Copy)]
pub struct ExecDescriptor {
    /// The name of the test executable.
    pub name: &'static str,
    /// The doc comment description of the test.
    pub description: &'static str,
    /// A function pointer to the test executable.
    pub test_fn: fn(),
}

/// A slice of settings for a test suite.
pub type SettingsSlice = &'static [&'static dyn Setting];

/// Describes a test suite.
pub struct SuiteDescriptor {
    /// A slice of test executables in this suite.
    pub executables: &'static [ExecDescriptor],
    /// The name of the test suite.
    pub name: &'static str,
    /// The doc comment description of the test suite.
    pub description: &'static str,
    /// A slice of configurable settings for this suite.
    pub settings: SettingsSlice,
}
