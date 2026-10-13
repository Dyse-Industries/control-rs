//! Runner for one executable.
//!
//! # Description
//!
//! A `Runner` executes one executable on the target and reports its state transitions and
//! metrics through the borrowed [`Context`]. The server resolves identifiers and constructs a
//! runner; it holds no execution logic. The per-run variant is the type parameter `M: Mode`, so a
//! new execution model is a new `Mode` and one dispatcher arm, not a flag in a loop.
//!
//! # Core Concepts
//!
//! - **Mode**: [`Mode`] supplies how the executable is executed and measured. [`Atomic`] runs the
//!   executable once with interrupts masked and profiles it.
//! - **Active item**: The runner publishes the running suite and test indices in
//!   [`CURRENT_SUITE`] and [`CURRENT_TEST`] so the fault path can name the item that failed.

use core::marker::PhantomData;

use crate::ExecDescriptor;
use crate::comms::{HostComms, Telemetry, TestState};
use crate::context::{Context, TestMetrics};
use crate::profiler::CPUProfiler;
use crate::server::{CURRENT_SUITE, CURRENT_TEST};

/// Outcome of a run: success or the transport error of `C`.
pub type RunResult<C> = Result<(), <C as HostComms>::Error>;
/// Shared borrow of the context.
type Ctx<'a, C, P> = &'a Context<C, P>;
/// Exclusive borrow of the context.
type CtxMut<'a, C, P> = &'a mut Context<C, P>;

/// How a runner executes and measures an executable.
///
/// # Safety
/// Implementations must not use `unsafe` code that relies on state the runner does not provide.
///
/// # Panics
/// Implementations do not panic.
pub trait Mode {
    /// Executes `exec` once under this mode and returns what was measured.
    fn execute<C: HostComms, P: CPUProfiler>(
        context: Ctx<'_, C, P>,
        exec: &ExecDescriptor,
    ) -> TestMetrics;
}

/// Runs the executable once with interrupts masked and profiles cycles, time and stack.
#[derive(Debug, Clone, Copy, Default)]
pub struct Atomic;

/// Executes one executable and reports its progress.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Running does not panic.
///
/// # Example
/// ```
/// use control_rs_ets::ExecDescriptor;
/// use control_rs_ets::run::{Atomic, Runner};
/// # use control_rs_ets::server::Context;
/// # use control_rs_ets::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
/// # use control_rs_ets::profiler::CPUProfiler;
/// # struct MockComms;
/// # impl HostComms for MockComms {
/// #     type Error = &'static str;
/// #     fn flush(&mut self) -> SendResult<Self::Error> { Ok(()) }
/// #     fn poll_command(&mut self) -> PollResult<Self::Error> { Ok(None) }
/// #     fn send_telemetry(&mut self, _: &Telemetry<'_>) -> SendResult<Self::Error> { Ok(()) }
/// # }
/// # struct MockProfiler;
/// # impl CPUProfiler for MockProfiler {
/// #     fn get_cycles(&self) -> u64 { 0 }
/// #     fn get_nanos(&self) -> u64 { 0 }
/// #     fn get_sp(&self) -> usize { 0 }
/// #     fn get_stack_end(&self) -> usize { 0 }
/// # }
/// fn case() {}
/// let exec = ExecDescriptor { description: "case", name: "case", test_fn: case };
/// let mut context = Context::new(MockComms, MockProfiler);
/// assert!(Runner::<_, _, Atomic>::new(&mut context, 0, 0, &exec).run().is_ok());
/// ```
pub struct Runner<'a, C, P, M> {
    context: CtxMut<'a, C, P>,
    exec: &'a ExecDescriptor,
    mode: PhantomData<M>,
    suite_id: u16,
    test_id: u16,
}

impl Mode for Atomic {
    fn execute<C: HostComms, P: CPUProfiler>(
        context: Ctx<'_, C, P>,
        exec: &ExecDescriptor,
    ) -> TestMetrics {
        context.profile_test(exec.test_fn)
    }
}

impl<'a, C, P, M> Runner<'a, C, P, M> {
    /// Creates a runner for the executable at `suite_id` and `test_id`.
    pub const fn new(
        context: CtxMut<'a, C, P>,
        suite_id: u16,
        test_id: u16,
        exec: &'a ExecDescriptor,
    ) -> Self {
        Self {
            context,
            exec,
            mode: PhantomData,
            suite_id,
            test_id,
        }
    }
}

impl<C: HostComms, P: CPUProfiler, M: Mode> Runner<'_, C, P, M> {
    /// Executes the executable, publishing the active item and reporting `Running`, `Passed`
    /// and the metric report.
    ///
    /// # Errors
    /// Returns a transport error if reporting fails.
    pub fn run(self) -> RunResult<C> {
        let Self {
            context,
            exec,
            suite_id,
            test_id,
            ..
        } = self;

        let _ = context.send_telemetry_and_flush_locked(
            &Telemetry::TestStateChange {
                suite_id,
                test_id,
                state: TestState::Running,
            },
        )?;

        // Track globally in case of panic during execution.
        CURRENT_SUITE.set_active(suite_id as usize);
        CURRENT_TEST.set_active(test_id as usize);

        let metrics = M::execute(context, exec);

        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();

        let _ = context.send_telemetry_locked(&Telemetry::TestStateChange {
            suite_id,
            test_id,
            state: TestState::Passed,
        })?;

        let _ = context.send_telemetry_and_flush_locked(
            &Telemetry::MetricReport {
                suite_id,
                test_id,
                cycles: metrics.cycles,
                time_us: metrics.time_us,
                stack_peak: metrics.stack_peak,
            },
        )?;

        Ok(())
    }
}
