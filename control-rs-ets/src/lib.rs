//! Embedded Test Server (ETS) testing framework for `control-rs`.
//!
//! # Description
//!
//! The `control-rs-ets` crate implements an Embedded Test Server (ETS) testing framework for
//! embedded control systems. It acts as the on-target server that communicates with a host machine
//! over a structured packet-framed protocol, enabling remote test execution, parameter configuration,
//! and execution profiling (CPU cycles, timing and peak stack memory usage).
//!
//! # Core Concepts
//!
//! 1. **Server**: A loop that listens for framed commands from a host TUI, executes
//!    targeted test functions in critical sections (with interrupts disabled) and reports status.
//! 2. **Parameter Table**: Test suites declare named parameters (`ParamEntry` entries in a flat table)
//!    which the host can query and update in real-time.
//! 3. **Hardware Profiling**: Tracks CPU clock cycles, wall-clock execution time and paint-based
//!    peak stack depth using target-specific mechanisms (like DWT on ARM Cortex-M or CSRs on RISC-V).
//! 4. **Structured Telemetry**: Frames and serializes protocol events (like panic alerts, test outcome
//!    transitions and metrics reports) using the Postcard wire format.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::{ExecDescriptor, SuiteDescriptor};
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
//!     params: &[],
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
//!   to perform stack depth tracking. If disabled, stack profiling reads report 0.
//!
//! # Limitations
//!
//! - **Single Threaded Execution**: ETS executes tests sequentially in a single-threaded
//!   environment with global interrupts disabled.
//! - **Hardware Dependency**: Access to low-level registers (`SysTick`, `DWT`, `CSRs`) assumes exclusive control
//!   over target-specific profiling hardware.
//! - **No dynamic allocation**: Designed for `no-std` contexts without a heap allocator.

#![no_std]
#![allow(clippy::multiple_crate_versions)]

pub use params::{ParamEntry, ParamTable};
pub use profiler::CPUProfiler;
#[cfg(target_arch = "arm")]
pub use profiler::CortexMProfiler;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub use profiler::RiscvProfiler;
pub use server::{Context, Server};

pub mod link;
pub mod params;
pub mod profiler;
pub mod server;
pub mod util;
pub mod wire;

/// Maximum bytes of one task input or output packet.
///
/// `MAX_MESSAGE_SIZE` (512) minus the worst-case `TaskInput` or `TaskSample` header: variant tag 1,
/// `suite` 3, `item` 3, `seq` 10, length prefix 2.
pub const MAX_PACKET_SIZE: usize = 493;

/// Describes a single test executable.
///
/// This structure holds metadata about an individual test function, including
/// its name, doc description and the function pointer itself.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::ExecDescriptor;
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

/// Describes a test suite.
///
/// A test suite aggregates a group of test executables and a set of configurable
/// parameters that alter the suite's behavior.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::{ExecDescriptor, SuiteDescriptor};
/// fn test_a() {}
/// static EXECS: &[ExecDescriptor] = &[
///     ExecDescriptor {
///         description: "Test A",
///         name: "test_a",
///         test_fn: test_a,
///     }
/// ];
/// static SUITE: SuiteDescriptor = SuiteDescriptor {
///     description: "Example Test Suite",
///     executables: EXECS,
///     name: "my_suite",
///     params: &[],
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
    /// The parameter table of this suite.
    pub params: ParamTable,
}
