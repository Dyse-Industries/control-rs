//! Hardware-in-the-Loop (HIL) testing framework for `control-rs`.
//!
//! # Description
//!
//! The `control-rs-hil` crate implements a Hardware-in-the-Loop (HIL) testing framework for
//! embedded control systems. It acts as the on-target server that communicates with a host machine
//! over a structured packet-framed protocol, enabling remote test execution, settings configuration,
//! and execution profiling (CPU cycles, timing, and peak stack memory usage).
//!
//! # Core Concepts
//!
//! 1. **Test Runner Server**: A loop that listens for framed commands from a host TUI, executes
//!    targeted test functions in critical sections (with interrupts disabled), and reports status.
//! 2. **Settings Registry**: Test suites declare dynamic settings (represented by the `Setting` trait)
//!    which the host can query and update in real-time.
//! 3. **Hardware Profiling**: Tracks CPU clock cycles, wall-clock execution time, and paint-based
//!    peak stack depth using target-specific mechanisms (like DWT on ARM Cortex-M or CSRs on RISC-V).
//! 4. **Structured Telemetry**: Frames and serializes protocol events (like panic alerts, test outcome
//!    transitions, and metrics reports) using the Postcard wire format.
//!
//! # Usage
//!
//! ```
//! use control_rs_hil::{ExecDescriptor, SuiteDescriptor, SettingsSlice};
//!
//! // Define a mock test function
//! fn dummy_test() {}
//!
//! // Define test executable descriptor
//! static EXEC: ExecDescriptor = ExecDescriptor {
//!     description: "A dummy test executable",
//!     name: "dummy_test",
//!     test_fn: dummy_test,
//! };
//!
//! // Define a suite descriptor with the test
//! static SUITE: SuiteDescriptor = SuiteDescriptor {
//!     description: "A suite containing dummy tests",
//!     executables: &[EXEC],
//!     name: "dummy_suite",
//!     settings: &[],
//! };
//!
//! assert_eq!(SUITE.name, "dummy_suite");
//! assert_eq!(SUITE.executables.len(), 1);
//! ```
//!
//! # Features
//!
//! By default, the crate compiles with the following cargo features:
//! - `stack-paint`: Enables painting the stack memory space with sentinel bytes (`0xCDCD_CDCD`)
//!   to perform high-water-mark peak stack depth tracking. If disabled, stack profiling reads will report 0.
//!
//! # Limitations
//!
//! - **Single Threaded Execution**: The HIL runner executes tests sequentially in a single-threaded
//!   environment with global interrupts disabled.
//! - **Hardware Dependency**: Access to low-level registers (`SysTick`, `DWT`, `CSRs`) assumes exclusive control
//!   over target-specific profiling hardware.
//! - **No dynamic allocation**: Designed for `no-std` contexts without a heap allocator.

#![no_std]
#![allow(clippy::multiple_crate_versions)]

pub use profiler::CPUProfiler;
#[cfg(target_arch = "arm")]
pub use profiler::CortexMProfiler;
#[cfg(target_arch = "riscv32")]
pub use profiler::RiscvProfiler;
pub use server::{Context, Server};
pub use settings::Setting;

pub mod comms;
pub mod profiler;
pub mod server;
pub mod settings;
pub mod util;

/// Describes a single test executable.
///
/// This structure holds metadata about an individual test function, including
/// its name, doc description, and the function pointer itself.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_hil::ExecDescriptor;
/// fn my_test() {}
/// let exec = ExecDescriptor {
///     description: "A simple unit test",
///     name: "my_test",
///     test_fn: my_test,
/// };
/// assert_eq!(exec.name, "my_test");
/// ```
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
///
/// Refers to a static slice of trait objects implementing [Setting].
pub type SettingsSlice = &'static [&'static dyn Setting];

/// Describes a test suite.
///
/// A test suite aggregates a group of test executables and a set of configurable
/// parameters/settings that alter the suite's behavior.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_hil::{ExecDescriptor, SuiteDescriptor};
/// fn test_a() {}
/// static EXECS: &[ExecDescriptor] = &[
///     ExecDescriptor {
///         description: "Test A",
///         name: "test_a",
///         test_fn: test_a,
///     }
/// ];
/// static SUITE: SuiteDescriptor = SuiteDescriptor {
///     description: "My Test Suite",
///     executables: EXECS,
///     name: "my_suite",
///     settings: &[],
/// };
/// assert_eq!(SUITE.executables.len(), 1);
/// ```
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
