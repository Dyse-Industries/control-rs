//! Embedded Test Server (ETS) testing framework for `control-rs`.
//!
//! # Description
//!
//! The `control-rs-ets` crate implements an Embedded Test Server (ETS) testing framework for
//! embedded control systems. It acts as the on-target server that communicates with a host machine
//! over a structured packet-framed protocol, enabling remote test execution, settings configuration,
//! and execution profiling (CPU cycles, timing and peak stack memory usage).
//!
//! # Core Concepts
//!
//! 1. **Server**: A loop that listens for framed commands from a host TUI, executes
//!    targeted test functions in critical sections (with interrupts disabled) and reports status.
//! 2. **Settings Registry**: Test suites declare dynamic settings (represented by the `Setting` trait)
//!    which the host can query and update in real-time.
//! 3. **Hardware Profiling**: Tracks CPU clock cycles, wall-clock execution time and paint-based
//!    peak stack depth using target-specific mechanisms (like DWT on ARM Cortex-M or CSRs on RISC-V).
//! 4. **Structured Telemetry**: Frames and serializes protocol events (like panic alerts, test outcome
//!    transitions and metrics reports) using the Postcard wire format.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::{ExecDescriptor, SuiteDescriptor, SettingsSlice};
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

pub use comms::TaskRunState;
pub use profiler::CPUProfiler;
#[cfg(target_arch = "arm")]
pub use profiler::CortexMProfiler;
#[cfg(any(target_arch = "riscv32", target_arch = "riscv64"))]
pub use profiler::RiscvProfiler;
pub use server::{Context, Server};
pub use settings::Setting;

pub mod comms;
pub mod profiler;
pub mod server;
pub mod settings;
pub mod util;

/// Message of the `Error` outcome returned when an input packet fails to decode.
pub const INPUT_DECODE_ERROR: &str = "input decode";

/// Maximum bytes of one task input or output packet.
///
/// `MAX_PAYLOAD_SIZE` (512) minus the worst-case `TaskInput` or `TaskSample`
/// header: variant tag 1, `suite_id` 3, `test_id` 3, `seq` 10, length prefix 2.
pub const MAX_PACKET_SIZE: usize = 493;

/// Message of the `Error` outcome returned when an output packet exceeds
/// [`MAX_PACKET_SIZE`].
pub const OUTPUT_OVERFLOW_ERROR: &str = "output overflow";

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

/// A task's setup, reset or teardown function.
pub type TaskHook = fn() -> Result<(), &'static str>;

/// The untyped step function of a task.
pub type TaskStepFn = fn(&mut TaskIo<'_>) -> TaskOutcome;

/// The newest input packet bytes a step receives, if any.
pub type InputBytes<'a> = Option<&'a [u8]>;

/// A typed step function over input `I` and output `O`.
pub type TypedStepFn<I, O> = fn(&TaskContext<'_, I>) -> TaskStatus<O>;

/// Describes the task of a suite: a setup, a step, a reset and a teardown
/// function over typed input and output packets.
///
/// The descriptor is untyped so that every task fits one linker section; the
/// `#[ets_suite]` macro generates the `step` wrapper that decodes the input
/// packet and encodes the output packet.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::{TaskDescriptor, TaskIo, TaskOutcome, TaskRunState, SuiteDescriptor};
///
/// fn ok() -> Result<(), &'static str> { Ok(()) }
/// fn step(_: &mut TaskIo<'_>) -> TaskOutcome {
///     TaskOutcome { status: TaskRunState::Pass, message: None }
/// }
/// static SUITE: SuiteDescriptor = SuiteDescriptor {
///     description: "", executables: &[], name: "s", settings: &[],
/// };
/// static TASK: TaskDescriptor = TaskDescriptor {
///     suite: &SUITE,
///     name: "l",
///     description: "",
///     input_type: "()",
///     output_type: "()",
///     setup: ok,
///     step,
///     reset: ok,
///     teardown: ok,
///     link_timeout_ms: 0,
/// };
/// assert_eq!(TASK.link_timeout_ms, 0);
/// ```
pub struct TaskDescriptor {
    /// The doc comment description of the task.
    pub description: &'static str,
    /// The type name of the task's input packet.
    pub input_type: &'static str,
    /// The host link timeout in milliseconds (`0` disables supervision).
    pub link_timeout_ms: u32,
    /// The name of the task.
    pub name: &'static str,
    /// The type name of the task's output packet.
    pub output_type: &'static str,
    /// Returns the hardware to a safe state under the current settings.
    pub reset: TaskHook,
    /// Prepares the hardware once at the start of a run.
    pub setup: TaskHook,
    /// One iteration of the task.
    pub step: TaskStepFn,
    /// The suite that owns the task and its settings.
    pub suite: &'static SuiteDescriptor,
    /// Returns the hardware to a safe state at the end of a run.
    pub teardown: TaskHook,
}

/// The untyped arguments and results of one task step.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Instantiating or accessing this struct does not panic.
pub struct TaskIo<'a> {
    /// The newest received input packet bytes, if any.
    pub input: InputBytes<'a>,
    /// The step index the newest input is for.
    pub input_seq: Option<u64>,
    /// Buffer the step writes its encoded output packet into.
    pub output: &'a mut [u8],
    /// Number of bytes of `output` the step wrote.
    pub output_len: usize,
    /// The index `k` of this step, counted from `0` per run.
    pub step: u64,
}

/// The untyped status of one task step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TaskOutcome {
    /// An optional message accompanying the status.
    pub message: Option<&'static str>,
    /// One of `Running`, `Warn`, `Pass`, `Fail` or `Error`.
    pub status: comms::TaskRunState,
}

/// The status a typed step returns. `Running` and `Warn` continue the run and
/// carry the step's output packet.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskStatus<O> {
    /// The step failed the run.
    Fail(Option<&'static str>),
    /// The step passed the run.
    Pass(Option<&'static str>),
    /// The step ended the run with an error.
    Error(Option<&'static str>),
    /// Continue and send the output packet.
    Running(O),
    /// Continue, send the output packet and report a message.
    Warn(O, Option<&'static str>),
}

/// The view of a step's input a typed step receives.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Accessing the context does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::TaskContext;
///
/// let ctx = TaskContext::new(3, Some(&1.5_f32), Some(3));
/// assert_eq!(ctx.step(), 3);
/// assert_eq!(ctx.input(), Some(&1.5));
/// assert_eq!(ctx.input_seq(), Some(3));
/// ```
pub struct TaskContext<'a, I> {
    input: Option<&'a I>,
    input_seq: Option<u64>,
    step: u64,
}

impl<'a, I> TaskContext<'a, I> {
    /// The newest received input, if any.
    #[must_use]
    pub const fn input(&self) -> Option<&'a I> {
        self.input
    }

    /// The step index the newest input is for.
    #[must_use]
    pub const fn input_seq(&self) -> Option<u64> {
        self.input_seq
    }

    /// Creates a context.
    #[must_use]
    pub const fn new(
        step: u64,
        input: Option<&'a I>,
        input_seq: Option<u64>,
    ) -> Self {
        Self {
            input,
            input_seq,
            step,
        }
    }

    /// The index `k` of this step, counted from `0` per run.
    #[must_use]
    pub const fn step(&self) -> u64 {
        self.step
    }
}

/// Runs a typed step over the untyped [`TaskIo`]: decodes the input packet,
/// calls `step` and encodes the output packet.
///
/// An input that fails to decode skips the step and returns `Error` with
/// [`INPUT_DECODE_ERROR`]; an output larger than the output buffer returns
/// `Error` with [`OUTPUT_OVERFLOW_ERROR`]. The `#[ets_suite]` macro generates
/// the task's `step` wrapper as a call to this function.
///
/// # Arguments
/// * `io` - The untyped step arguments and output buffer.
/// * `step` - The user's typed step function.
///
/// # Returns
/// * `TaskOutcome` - The status and message of the step.
///
/// # Example
/// ```
/// use control_rs_ets::{TaskContext, TaskIo, TaskRunState, TaskStatus, task_step};
///
/// fn double(ctx: &TaskContext<'_, f32>) -> TaskStatus<f32> {
///     TaskStatus::Running(ctx.input().copied().unwrap_or(0.0) * 2.0)
/// }
/// let mut input = [0u8; 8];
/// let input = postcard::to_slice(&1.5_f32, &mut input).unwrap();
/// let mut out = [0u8; 16];
/// let mut io = TaskIo {
///     input: Some(input),
///     input_seq: Some(0),
///     output: &mut out,
///     output_len: 0,
///     step: 0,
/// };
/// let outcome = task_step::<f32, f32>(&mut io, double);
/// assert_eq!(outcome.status, TaskRunState::Running);
/// let len = io.output_len;
/// assert_eq!(postcard::from_bytes::<f32>(&out[..len]), Ok(3.0));
/// ```
pub fn task_step<'b, I, O>(
    io: &mut TaskIo<'b>,
    step: TypedStepFn<I, O>,
) -> TaskOutcome
where
    I: serde::Deserialize<'b>,
    O: serde::Serialize,
{
    let decoded = match io.input {
        Some(bytes) => match postcard::from_bytes::<I>(bytes) {
            Ok(value) => Some(value),
            Err(_) => {
                return TaskOutcome {
                    message: Some(INPUT_DECODE_ERROR),
                    status: comms::TaskRunState::Error,
                };
            }
        },
        None => None,
    };
    let ctx = TaskContext::new(io.step, decoded.as_ref(), io.input_seq);

    let (status, message, packet) = match step(&ctx) {
        TaskStatus::Running(out) => {
            (comms::TaskRunState::Running, None, Some(out))
        }
        TaskStatus::Warn(out, msg) => {
            (comms::TaskRunState::Warn, msg, Some(out))
        }
        TaskStatus::Pass(msg) => (comms::TaskRunState::Pass, msg, None),
        TaskStatus::Fail(msg) => (comms::TaskRunState::Fail, msg, None),
        TaskStatus::Error(msg) => (comms::TaskRunState::Error, msg, None),
    };

    io.output_len = 0;
    if let Some(out) = packet {
        match postcard::to_slice(&out, io.output) {
            Ok(written) => io.output_len = written.len(),
            Err(_) => {
                return TaskOutcome {
                    message: Some(OUTPUT_OVERFLOW_ERROR),
                    status: comms::TaskRunState::Error,
                };
            }
        }
    }
    TaskOutcome { message, status }
}
