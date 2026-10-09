//! On-target Server event loop.
//!
//! # Description
//!
//! This module implements the interactive Server, coordination contexts,
//! and global run state indicators. The server processes incoming commands from the host,
//! dynamically edits settings, executes target test routines and profiles their cycles/memory.
//!
//! # Core Concepts
//!
//! - **ETS**: The central coordinator (`Server`) executing suites, managing settings updates,
//!   and reporting telemetry outcomes back to the host machine.
//! - **Execution Context**: `Context` wraps host communication mechanisms and CPU profiling metrics
//!   using thread-safe mutex-free locking (`CommsLock`).
//! - **Test Index Indicator**: Thread-safe atomic indicator (`TestIndexIndicator`) tracking the active suite
//!   and test indexes to let exception/panic handlers report where a crash happened.

use core::sync::atomic::{AtomicBool, AtomicIsize, AtomicPtr, Ordering};

use crate::comms::{
    Command, CommsLock, HostComms, MAX_MESSAGE_SIZE, PROTOCOL_VERSION,
    TaskRunState, Telemetry, TestState,
};
use crate::settings::SettingValue;
use crate::{MAX_PACKET_SIZE, SuiteDescriptor, TaskDescriptor, TaskIo};

/// Serializes the tests that run a task or read the run indicators.
#[cfg(test)]
pub(crate) mod test_lock {
    extern crate std;

    static RUN_STATE: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// Holds the run-state lock until the guard drops.
    pub fn hold() -> std::sync::MutexGuard<'static, ()> {
        RUN_STATE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

// --- Static variables ---

/// The task run in progress, or null when no task runs.
///
/// Used by the panic handler to run the task's teardown.
///
/// # Safety
/// The pointer is stored from a `&'static TaskDescriptor` and is only
/// dereferenced as one.
pub static ACTIVE_TASK: AtomicPtr<TaskDescriptor> =
    AtomicPtr::new(core::ptr::null_mut());

/// Global tracker for the currently executing suite ID.
/// Used by the panic handler to report test failures.
///
/// # Safety
/// Accessing or updating this indicator is thread-safe and atomic.
pub static CURRENT_SUITE: TestIndexIndicator = TestIndexIndicator::new();

/// Global tracker for the currently executing test ID.
/// Used by the panic handler to report test failures.
///
/// # Safety
/// Accessing or updating this indicator is thread-safe and atomic.
pub static CURRENT_TEST: TestIndexIndicator = TestIndexIndicator::new();

/// Set once the active task's teardown has started, so that it is never
/// entered twice.
pub static TEARDOWN_STARTED: AtomicBool = AtomicBool::new(false);

// --- Type definitions ---

/// Context object that encapsulates communication, CPU profiling utilities and the communication lock.
///
/// This type coordinates accesses to low-level target hardware. To ensure thread safety and avoid race
/// conditions (for example, between the main event loop and interrupt/exception/panic handlers that
/// also need to send telemetry), the `Context` holds a `CommsLock`. All public telemetry and poll methods
/// are run via `_locked` helper methods, ensuring mutual exclusion without requiring blocking mutexes
/// that could cause deadlocks in interrupt-disabled contexts.
///
/// # Safety
/// Access to the underlying communication interface is protected by the internal `CommsLock`.
/// This struct does not use `unsafe` code directly.
///
/// # Panics
/// Context operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::server::Context;
/// # use control_rs_ets::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
/// # use control_rs_ets::profiler::CPUProfiler;
///
/// # struct MockComms;
/// # impl HostComms for MockComms {
/// #     type Error = &'static str;
/// #     fn flush(&mut self) -> SendResult<Self::Error> { Ok(()) }
/// #     fn poll_command(&mut self) -> PollResult<'_, Self::Error> { Ok(None) }
/// #     fn send_telemetry(&mut self, _: &Telemetry<'_>) -> SendResult<Self::Error> { Ok(()) }
/// # }
/// # struct MockProfiler;
/// # impl CPUProfiler for MockProfiler {
/// #     fn get_cycles(&self) -> u64 { 0 }
/// #     fn get_nanos(&self) -> u64 { 0 }
/// #     fn get_sp(&self) -> usize { 0 }
/// #     fn get_stack_end(&self) -> usize { 0 }
/// # }
///
/// let mut ctx = Context::new(MockComms, MockProfiler);
/// assert!(ctx.flush_locked().is_ok());
/// ```
pub struct Context<C, P> {
    /// Host communication channel.
    pub comms: C,
    /// Thread-safe communications lock to protect comms.
    pub comms_lock: CommsLock,
    /// CPU profiling and execution utilities.
    pub cpu_utils: P,
}

/// Interactive Embedded Test Server.
///
/// The `Server` coordinates the execution of ETS tests on the target. It manages the boot discovery phase,
/// processes settings updates, runs tests inside critical sections (interrupts disabled) and returns
/// telemetry and cycle/stack usage metrics.
///
/// # Safety
/// This structure does not use `unsafe` code.
///
/// # Panics
/// Server loop functions do not panic under normal operations.
///
/// # Example
/// ```
/// use control_rs_ets::server::{Server, Context};
/// # use control_rs_ets::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
/// # use control_rs_ets::profiler::CPUProfiler;
///
/// # struct MockComms;
/// # impl HostComms for MockComms {
/// #     type Error = &'static str;
/// #     fn flush(&mut self) -> SendResult<Self::Error> { Ok(()) }
/// #     fn poll_command(&mut self) -> PollResult<'_, Self::Error> { Ok(None) }
/// #     fn send_telemetry(&mut self, _: &Telemetry<'_>) -> SendResult<Self::Error> { Ok(()) }
/// # }
/// # struct MockProfiler;
/// # impl CPUProfiler for MockProfiler {
/// #     fn get_cycles(&self) -> u64 { 0 }
/// #     fn get_nanos(&self) -> u64 { 0 }
/// #     fn get_sp(&self) -> usize { 0 }
/// #     fn get_stack_end(&self) -> usize { 0 }
/// # }
///
/// let ctx = Context::new(MockComms, MockProfiler);
/// let server = Server::new(ctx, &[]);
/// assert_eq!(server.suites.len(), 0);
/// ```
pub struct Server<'a, C, P> {
    /// The global context containing comms, `cpu_utils` and the comms lock.
    pub context: Context<C, P>,
    /// The registered tasks, one per lifecycle suite.
    pub tasks: &'a [&'static TaskDescriptor],
    /// The registered test suites.
    pub suites: &'a [&'static SuiteDescriptor],
}

/// The end state of a task run and its optional message.
type Verdict = (TaskRunState, Option<&'static str>);

/// Mutable state of one task run, held on the run task's stack frame.
struct TaskRun {
    deadline_ns: u64,
    input: [u8; MAX_PACKET_SIZE],
    input_len: usize,
    input_seq: Option<u64>,
    k: u64,
    last_sent: Verdict,
    lockstep: bool,
    started_ns: u64,
    steps: u64,
    suite_id: u16,
    test_id: u16,
    timeout_ns: u64,
    verdict: Option<Verdict>,
}

/// The parameters of a `StartTask` command.
#[derive(Clone, Copy)]
struct StartRequest {
    lockstep: bool,
    max_steps: u64,
    suite_id: u16,
    test_id: u16,
}

/// What the server does after a polled command was accepted by a run.
enum Boundary {
    Log(&'static str),
    Nothing,
    Setting {
        setting_id: u16,
        value: SettingValue,
    },
}

/// Result of server operations.
///
/// Indicates success or contains transport error `E`.
pub type ServerResult<E> = Result<(), E>;

/// Performance metrics measured during a test execution.
pub struct TestMetrics {
    /// Number of elapsed CPU cycles.
    pub cycles: u64,
    /// Peak stack usage in bytes.
    pub stack_peak: u32,
    /// Elapsed wall-clock time in microseconds.
    pub time_us: u64,
}

/// Represents the active state of a Server execution indicator.
///
/// Tracks execution indexes or flags idle state via atomic operations.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::server::TestIndexIndicator;
///
/// let indicator = TestIndexIndicator::new();
/// assert!(indicator.get().is_none());
/// indicator.set_active(5);
/// assert_eq!(indicator.get(), Some(5));
/// indicator.set_idle();
/// assert!(indicator.get().is_none());
/// ```
pub struct TestIndexIndicator {
    /// Holds the active test index (>= 0) or `IDLE_STATE` (< 0).
    state: AtomicIsize,
}

#[allow(clippy::type_complexity)]
impl<C: HostComms, P> Context<C, P> {
    /// Flushes comms safely by acquiring the comms lock first.
    ///
    /// If the lock is already held (for example, by a panic handler or an interrupt), this method returns `Ok(false)`
    /// immediately to avoid reentrancy deadlock or state corruption.
    ///
    /// # Returns
    /// * `Result<bool, C::Error>` - Success flag (true if locked and flushed, false if lock was busy) or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if flushing fails.
    pub fn flush_locked(&mut self) -> Result<bool, C::Error> {
        if self.comms_lock.try_lock() {
            let res = self.comms.flush();
            self.comms_lock.unlock();
            res.map(|()| true)
        } else {
            Ok(false)
        }
    }

    /// Creates a new `Context`.
    ///
    /// # Arguments
    /// * `comms` - Communication peripheral channel.
    /// * `cpu_utils` - CPU execution/profiler.
    ///
    /// # Returns
    /// * `Self` - Context instance.
    pub const fn new(comms: C, cpu_utils: P) -> Self {
        Self {
            comms,
            comms_lock: CommsLock::new(),
            cpu_utils,
        }
    }

    /// Polls a command safely by acquiring the comms lock first.
    ///
    /// If the lock is already held, this returns `Ok(None)` immediately to prevent nested/concurrent polls.
    ///
    /// # Returns
    /// * `PollResult<C::Error>` - Success (optionally wrapped command) or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if polling fails.
    pub fn poll_command_locked(
        &mut self,
    ) -> Result<Option<Command<'_>>, C::Error> {
        if self.comms_lock.try_lock() {
            let res = self.comms.poll_command();
            self.comms_lock.unlock();
            res
        } else {
            Ok(None)
        }
    }

    /// Sends telemetry and flushes safely by acquiring the comms lock first.
    ///
    /// If the lock is already held, this returns `Ok(false)` immediately. This is the primary safe interface
    /// for regular telemetry logs.
    ///
    /// # Arguments
    /// * `telemetry` - Reference to the telemetry data structure.
    ///
    /// # Returns
    /// * `Result<bool, C::Error>` - Success flag (true if locked and sent, false if lock was busy) or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if writing or flushing fails.
    pub fn send_telemetry_and_flush_locked(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> Result<bool, C::Error> {
        if self.comms_lock.try_lock() {
            let res = self
                .comms
                .send_telemetry(telemetry)
                .and_then(|()| self.comms.flush());
            self.comms_lock.unlock();
            res.map(|()| true)
        } else {
            Ok(false)
        }
    }

    /// Sends telemetry safely by acquiring the comms lock first.
    ///
    /// If the lock is already held, this returns `Ok(false)` immediately.
    ///
    /// # Arguments
    /// * `telemetry` - Reference to the telemetry data structure.
    ///
    /// # Returns
    /// * `Result<bool, C::Error>` - Success flag (true if locked and sent, false if lock was busy) or transport error status.
    ///
    /// # Errors
    /// Returns a transport error if writing fails.
    pub fn send_telemetry_locked(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> Result<bool, C::Error> {
        if self.comms_lock.try_lock() {
            let res = self.comms.send_telemetry(telemetry);
            self.comms_lock.unlock();
            res.map(|()| true)
        } else {
            Ok(false)
        }
    }
}

#[allow(clippy::type_complexity)]
impl<C: HostComms, P: crate::profiler::CPUProfiler> Context<C, P> {
    /// Profile the execution of a test function, measuring cycles, stack and time.
    pub fn profile_test<F>(&self, test_fn: F) -> TestMetrics
    where
        F: FnOnce(),
    {
        let sp = self.cpu_utils.get_sp();
        // SAFETY: The active stack pointer `sp` is retrieved immediately before painting.
        // The implementation of `paint_stack` handles bounds calculations and enforces a safety
        // margin to protect active call frames.
        unsafe {
            self.cpu_utils.paint_stack(sp);
        }

        let mut start_cycles = 0;
        let mut start_time_ns = 0;
        let mut end_cycles = 0;
        let mut end_time_ns = 0;

        self.cpu_utils.disable_interrupts(|| {
            start_cycles = self.cpu_utils.get_cycles();
            start_time_ns = self.cpu_utils.get_nanos();
            test_fn();
            end_time_ns = self.cpu_utils.get_nanos();
            end_cycles = self.cpu_utils.get_cycles();
        });

        // SAFETY: The peak stack usage is queried relative to the same stack pointer `sp` used
        // to paint the stack. The stack was painted with sentinel bytes and reading occurs within
        // the valid boundaries calculated during the painting phase.
        let elapsed_stack = unsafe { self.cpu_utils.read_stack_peak(sp) };
        let elapsed_cycles = end_cycles.saturating_sub(start_cycles);
        let elapsed_time_us = end_time_ns.saturating_sub(start_time_ns) / 1000;

        TestMetrics {
            cycles: elapsed_cycles,
            stack_peak: elapsed_stack,
            time_us: elapsed_time_us,
        }
    }

    /// Reports an error message via Log telemetry.
    ///
    /// # Errors
    /// Returns a transport error if writing or flushing fails.
    pub fn report_error_log(
        &mut self,
        suite_id: u16,
        test_id: u16,
        msg: &'static str,
    ) -> ServerResult<C::Error> {
        let timestamp_us = self.cpu_utils.get_nanos() / 1000;
        let _ = self.send_telemetry_and_flush_locked(&Telemetry::Log(
            crate::comms::LogMessage {
                payload: msg,
                suite_id,
                test_id,
                timestamp_us,
            },
        ))?;
        Ok(())
    }

    /// Reports a setting set failure via Log telemetry.
    ///
    /// # Errors
    /// Returns a transport error if sending fails.
    pub fn report_setting_error(
        &mut self,
        suite_id: u16,
        setting_name: &str,
        err: &'static str,
    ) -> ServerResult<C::Error> {
        let timestamp_us = self.cpu_utils.get_nanos() / 1000;
        let mut err_buf = [0u8; 128];
        let pos = {
            let mut writer = crate::util::FailureBufWriter {
                buf: &mut err_buf,
                pos: 0,
            };
            let _ = core::fmt::write(
                &mut writer,
                format_args!("Failed to set setting '{setting_name}': {err}"),
            );
            writer.pos
        };
        if let Some(Ok(msg)) = err_buf.get(..pos).map(core::str::from_utf8) {
            let _ = self.send_telemetry_locked(&Telemetry::Log(
                crate::comms::LogMessage {
                    payload: msg,
                    suite_id,
                    test_id: 0,
                    timestamp_us,
                },
            ))?;
        }
        Ok(())
    }
}

#[allow(clippy::type_complexity)]
impl<'a, C, P> Server<'a, C, P>
where
    C: HostComms,
    P: crate::profiler::CPUProfiler,
{
    /// Exits the server, using target-specific exit mechanisms.
    ///
    /// Attempts to cleanly close the communication channel before invoking the target exit routine.
    ///
    /// # Returns
    /// * `!` - This function never returns.
    pub fn exit(&mut self) -> ! {
        if self.context.comms_lock.try_lock() {
            self.context.comms.close();
            self.context.comms_lock.unlock();
        } else {
            self.context.comms.close();
        }
        self.context.cpu_utils.exit()
    }

    /// Creates a new `Server` instance with the given context.
    ///
    /// # Arguments
    /// * `context` - Communication and hardware environment context.
    /// * `suites` - Slice of static test suite references.
    ///
    /// # Returns
    /// * `Self` - Server instance.
    pub const fn new(
        context: Context<C, P>,
        suites: &'a [&'static SuiteDescriptor],
    ) -> Self {
        Self {
            context,
            tasks: &[],
            suites,
        }
    }

    /// Registers the tasks of the image, replacing the empty default.
    ///
    /// # Arguments
    /// * `tasks` - Slice of static task descriptor references.
    ///
    /// # Returns
    /// * `Self` - Server instance holding `tasks`.
    #[must_use]
    pub const fn with_tasks(
        mut self,
        tasks: &'a [&'static TaskDescriptor],
    ) -> Self {
        self.tasks = tasks;
        self
    }

    /// Runs the interactive server event loop.
    ///
    /// This function polls for incoming host commands, executes requested tests,
    /// and streams telemetry and metrics back to the host.
    ///
    /// # Returns
    /// * `ServerResult<C::Error>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error `C::Error` propagated from the underlying communication interface if polling,
    /// transmitting telemetry or flushing fails.
    pub fn run(&mut self) -> ServerResult<C::Error> {
        loop {
            let cmd = self.context.poll_command_locked()?;

            if let Some(cmd) = cmd {
                match cmd {
                    Command::ListSuites => {
                        self.stream_discovery()?;
                    }
                    Command::RunExecutable { suite_id, test_id } => {
                        self.run_test(suite_id, test_id)?;
                    }
                    Command::SetSetting {
                        suite_id,
                        setting_id,
                        value,
                    } => {
                        self.set_setting(suite_id, setting_id, value)?;
                    }
                    Command::StartTask {
                        suite_id,
                        test_id,
                        max_steps,
                        lockstep,
                    } => {
                        self.start_task(StartRequest {
                            lockstep,
                            max_steps,
                            suite_id,
                            test_id,
                        })?;
                    }
                    Command::TryReset
                    | Command::StopNow { .. }
                    | Command::Heartbeat
                    | Command::TaskInput { .. } => {}
                }
            }

            let _ = self.context.flush_locked()?;
        }
    }

    fn run_test(
        &mut self,
        suite_id: u16,
        test_id: u16,
    ) -> ServerResult<C::Error> {
        let suite_idx = suite_id as usize;
        let test_idx = test_id as usize;

        let Some(&suite) = self.suites.get(suite_idx) else {
            self.context.report_error_log(
                suite_id,
                test_id,
                "Error: RunExecutable suite_id out of range",
            )?;
            return Ok(());
        };
        let Some(exec) = suite.executables.get(test_idx) else {
            self.context.report_error_log(
                suite_id,
                test_id,
                "Error: RunExecutable test_id out of range",
            )?;
            return Ok(());
        };

        // Update state to Running
        let _ = self.context.send_telemetry_and_flush_locked(
            &Telemetry::TestStateChange {
                suite_id,
                test_id,
                state: TestState::Running,
            },
        )?;

        // Track globally in case of panic during test execution
        CURRENT_SUITE.set_active(suite_id as usize);
        CURRENT_TEST.set_active(test_id as usize);

        // Profile the test
        let metrics = self.context.profile_test(exec.test_fn);

        // Clear global trackers on success
        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();

        // Update state to Passed & Send metric report
        let _ = self.context.send_telemetry_locked(
            &Telemetry::TestStateChange {
                suite_id,
                test_id,
                state: TestState::Passed,
            },
        )?;

        let _ = self.context.send_telemetry_and_flush_locked(
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

    fn set_setting(
        &mut self,
        suite_id: u16,
        setting_id: u16,
        value: SettingValue,
    ) -> ServerResult<C::Error> {
        self.apply_setting(suite_id, setting_id, value).map(|_| ())
    }

    /// Stores a setting value and confirms it to the host.
    ///
    /// Returns whether the value was stored.
    fn apply_setting(
        &mut self,
        suite_id: u16,
        setting_id: u16,
        value: SettingValue,
    ) -> Result<bool, C::Error> {
        let suite_idx = suite_id as usize;
        let setting_idx = setting_id as usize;

        let Some(&suite) = self.suites.get(suite_idx) else {
            self.context.report_error_log(
                suite_id,
                0,
                "Error: SetSetting suite_id out of range",
            )?;
            return Ok(false);
        };
        let Some(&setting) = suite.settings.get(setting_idx) else {
            self.context.report_error_log(
                suite_id,
                0,
                "Error: SetSetting setting_id out of range",
            )?;
            return Ok(false);
        };

        let stored = setting.set(value);
        if let Err(err) = stored {
            self.context
                .report_setting_error(suite_id, setting.name(), err)?;
        }

        // Stream back the updated value to confirm
        let _ = self.context.send_telemetry_and_flush_locked(
            &Telemetry::SettingInfo {
                suite_id,
                setting_id,
                name: setting.name(),
                description: setting.description(),
                value: setting.get(),
            },
        )?;

        Ok(stored.is_ok())
    }

    fn stream_discovery(&mut self) -> ServerResult<C::Error> {
        let cpu = &self.context.cpu_utils;
        let info = Telemetry::TargetInfo {
            protocol_version: PROTOCOL_VERSION,
            board_id: cpu.board_id(),
            core_clock_hz: cpu.core_clock_hz(),
            fpu_flags: cpu.fpu_flags(),
        };
        let _ = self.context.send_telemetry_locked(&info)?;
        for (suite_id, &suite) in (0_u16..).zip(self.suites.iter()) {
            let _ =
                self.context.send_telemetry_locked(&Telemetry::SuiteInfo {
                    suite_id,
                    name: suite.name,
                    description: suite.description,
                    test_count: suite
                        .executables
                        .len()
                        .try_into()
                        .unwrap_or(u16::MAX),
                    setting_count: suite
                        .settings
                        .len()
                        .try_into()
                        .unwrap_or(u16::MAX),
                })?;

            for (test_id, exec) in (0_u16..).zip(suite.executables.iter()) {
                let _ = self.context.send_telemetry_locked(
                    &Telemetry::TestInfo {
                        suite_id,
                        test_id,
                        name: exec.name,
                        description: exec.description,
                    },
                )?;
            }

            for (setting_id, setting) in (0_u16..).zip(suite.settings.iter()) {
                let _ = self.context.send_telemetry_locked(
                    &Telemetry::SettingInfo {
                        suite_id,
                        setting_id,
                        name: setting.name(),
                        description: setting.description(),
                        value: setting.get(),
                    },
                )?;
            }

            self.stream_suite_tasks(suite_id, suite)?;
        }
        self.report_orphan_tasks()?;

        let _ = self
            .context
            .send_telemetry_and_flush_locked(&Telemetry::DiscoveryComplete)?;

        Ok(())
    }

    /// Sends `LifecycleSuite` and the `TaskInfo` of the task of `suite`, if any.
    ///
    /// Tasks beyond the first for one suite are skipped with an error log.
    fn stream_suite_tasks(
        &mut self,
        suite_id: u16,
        suite: &'static SuiteDescriptor,
    ) -> ServerResult<C::Error> {
        let tasks = self.tasks;
        let mut matching = tasks
            .iter()
            .copied()
            .filter(|l| core::ptr::eq(l.suite, suite));
        let first = matching.next();
        let extras = matching.count();

        if let Some(desc) = first {
            let _ = self.context.send_telemetry_locked(
                &Telemetry::LifecycleSuite {
                    suite_id,
                    task_count: 1,
                },
            )?;
            let _ =
                self.context.send_telemetry_locked(&Telemetry::TaskInfo {
                    suite_id,
                    test_id: u16::try_from(suite.executables.len())
                        .unwrap_or(u16::MAX),
                    name: desc.name,
                    description: desc.description,
                    input_type: desc.input_type,
                    output_type: desc.output_type,
                })?;
        }
        for _ in 0..extras {
            self.context.report_error_log(
                suite_id,
                0,
                "Error: second lifecycle task for suite skipped",
            )?;
        }
        Ok(())
    }

    /// Logs each task whose suite is not registered.
    fn report_orphan_tasks(&mut self) -> ServerResult<C::Error> {
        let (suites, tasks) = (self.suites, self.tasks);
        for desc in tasks {
            if !suites.iter().any(|s| core::ptr::eq(*s, desc.suite)) {
                self.context.report_error_log(
                    0,
                    0,
                    "Error: lifecycle task's suite not registered, skipped",
                )?;
            }
        }
        Ok(())
    }

    /// Resolves the task addressed by `StartTask` and runs it.
    fn start_task(&mut self, req: StartRequest) -> ServerResult<C::Error> {
        let (suite_id, test_id) = (req.suite_id, req.test_id);
        let Some(&suite) = self.suites.get(usize::from(suite_id)) else {
            self.context.report_error_log(
                suite_id,
                test_id,
                "Error: StartLifecycle suite_id out of range",
            )?;
            return Ok(());
        };
        let tasks = self.tasks;
        let found = tasks
            .iter()
            .copied()
            .find(|l| core::ptr::eq(l.suite, suite))
            .filter(|_| usize::from(test_id) == suite.executables.len());
        let Some(desc) = found else {
            self.context.report_error_log(
                suite_id,
                test_id,
                "Error: StartTask does not address a lifecycle task",
            )?;
            return Ok(());
        };
        self.run_task(desc, req)
    }

    /// Runs setup, the steps and teardown of a task until the run ends.
    ///
    /// After setup has been invoked, teardown always runs (FR-5), including
    /// when a later transport error aborts the step loop.
    fn run_task(
        &mut self,
        desc: &'static TaskDescriptor,
        req: StartRequest,
    ) -> ServerResult<C::Error> {
        let started_ns = self.context.cpu_utils.get_nanos();
        let mut run = TaskRun::new(req, started_ns, desc.link_timeout_ms);

        CURRENT_SUITE.set_active(usize::from(req.suite_id));
        CURRENT_TEST.set_active(usize::from(req.test_id));
        TEARDOWN_STARTED.store(false, Ordering::Release);
        ACTIVE_TASK
            .store(core::ptr::from_ref(desc).cast_mut(), Ordering::Release);

        if let Err(e) = self.context.send_telemetry_and_flush_locked(
            &Telemetry::TaskState {
                suite_id: req.suite_id,
                test_id: req.test_id,
                state: TaskRunState::Running,
                message: None,
            },
        ) {
            // Setup was not invoked; clear the active-run indicator and exit.
            Self::clear_active_run();
            return Err(e);
        }
        if let Err(msg) = (desc.setup)() {
            run.verdict = Some((TaskRunState::Error, Some(msg)));
        }

        let steps = self.run_steps(desc, &mut run, req.max_steps);
        if steps.is_err() && run.verdict.is_none() {
            run.verdict = Some((TaskRunState::Error, Some("link lost")));
        }
        // Teardown after setup even when the link is already dead (FR-5).
        let finished = self.finish_run(desc, &run);
        match steps {
            Err(e) => Err(e),
            Ok(()) => finished,
        }
    }

    /// Clears the active lifecycle-run indicators without calling teardown.
    fn clear_active_run() {
        ACTIVE_TASK.store(core::ptr::null_mut(), Ordering::Release);
        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();
    }

    /// Calls the task's steps until a verdict is set.
    fn run_steps(
        &mut self,
        desc: &'static TaskDescriptor,
        run: &mut TaskRun,
        max_steps: u64,
    ) -> ServerResult<C::Error> {
        let mut output = [0u8; MAX_PACKET_SIZE];
        while run.verdict.is_none() {
            while run.lockstep
                && run.verdict.is_none()
                && run.input_seq != Some(run.k)
            {
                self.task_boundary(desc, run)?;
            }
            if run.verdict.is_some() {
                break;
            }

            let mut io = TaskIo {
                input: run
                    .input_seq
                    .and_then(|_| run.input.get(..run.input_len)),
                input_seq: run.input_seq,
                output: &mut output,
                output_len: 0,
                step: run.k,
            };
            let outcome = (desc.step)(&mut io);
            let output_len = io.output_len;
            run.steps = run.steps.saturating_add(1);

            let packet = output.get(..output_len).unwrap_or(&[]);
            self.report_step(run, outcome, packet)?;
            run.k = run.k.saturating_add(1);

            if run.verdict.is_none() && max_steps != 0 && run.steps >= max_steps
            {
                run.verdict = Some((TaskRunState::Bounded, None));
            }
            if run.verdict.is_some() {
                break;
            }
            self.task_boundary(desc, run)?;
        }
        Ok(())
    }

    /// Calls teardown and sends the run's closing reports.
    ///
    /// Teardown always runs and the active-run indicator is always cleared,
    /// even when a closing telemetry send fails on a dead link.
    fn finish_run(
        &mut self,
        desc: &'static TaskDescriptor,
        run: &TaskRun,
    ) -> ServerResult<C::Error> {
        let (suite_id, test_id) = (run.suite_id, run.test_id);
        let (state, message) = run.verdict.unwrap_or((
            TaskRunState::Error,
            Some("run ended without verdict"),
        ));
        TEARDOWN_STARTED.store(true, Ordering::Release);
        let ended_ns = self.context.cpu_utils.get_nanos();
        let teardown = (desc.teardown)();

        let teardown_send =
            self.context
                .send_telemetry_locked(&Telemetry::TeardownReport {
                    suite_id,
                    test_id,
                    ok: teardown.is_ok(),
                    message: teardown.err().map(truncate_message),
                });
        let stats_send =
            self.context.send_telemetry_locked(&Telemetry::TaskStats {
                suite_id,
                test_id,
                steps: run.steps,
                time_us: ended_ns.saturating_sub(run.started_ns) / 1000,
            });
        let verdict_send = self.context.send_telemetry_and_flush_locked(
            &Telemetry::TaskState {
                suite_id,
                test_id,
                state,
                message: message.map(truncate_message),
            },
        );

        Self::clear_active_run();
        teardown_send?;
        stats_send?;
        verdict_send?;
        Ok(())
    }

    /// Sends the output packet and any state change of a finished step, or
    /// records its terminal verdict.
    fn report_step(
        &mut self,
        run: &mut TaskRun,
        outcome: crate::TaskOutcome,
        packet: &[u8],
    ) -> ServerResult<C::Error> {
        match outcome.status {
            TaskRunState::Running | TaskRunState::Warn => {
                if run.lockstep || !packet.is_empty() {
                    let _ = self.context.send_telemetry_locked(
                        &Telemetry::TaskSample {
                            suite_id: run.suite_id,
                            test_id: run.test_id,
                            seq: run.k,
                            payload: packet,
                        },
                    )?;
                }
                let sent = (outcome.status, outcome.message);
                if sent != run.last_sent {
                    let _ = self.context.send_telemetry_locked(
                        &Telemetry::TaskState {
                            suite_id: run.suite_id,
                            test_id: run.test_id,
                            state: outcome.status,
                            message: outcome.message.map(truncate_message),
                        },
                    )?;
                    run.last_sent = sent;
                }
            }
            state => run.verdict = Some((state, outcome.message)),
        }
        Ok(())
    }

    /// The server's work between two steps: poll one command, act on it,
    /// supervise the host link and flush.
    fn task_boundary(
        &mut self,
        desc: &'static TaskDescriptor,
        run: &mut TaskRun,
    ) -> ServerResult<C::Error> {
        let (frame_seen, action) = self
            .context
            .poll_command_locked()?
            .map_or((false, Boundary::Nothing), |cmd| (true, run.accept(&cmd)));

        match action {
            Boundary::Nothing => {}
            Boundary::Log(msg) => {
                self.context.report_error_log(
                    run.suite_id,
                    run.test_id,
                    msg,
                )?;
            }
            Boundary::Setting { setting_id, value } => {
                let stored =
                    self.apply_setting(run.suite_id, setting_id, value)?;
                if stored
                    && run.verdict.is_none()
                    && let Err(msg) = (desc.reset)()
                {
                    run.verdict = Some((TaskRunState::Error, Some(msg)));
                }
            }
        }

        if run.timeout_ns != 0 {
            let now = self.context.cpu_utils.get_nanos();
            if frame_seen {
                run.deadline_ns = now.saturating_add(run.timeout_ns);
            } else if run.verdict.is_none() && now > run.deadline_ns {
                run.verdict = Some((TaskRunState::TimedOut, None));
            }
        }

        let _ = self.context.flush_locked()?;
        Ok(())
    }
}

impl TaskRun {
    /// A fresh run for `req` that started at `started_ns`.
    const fn new(
        req: StartRequest,
        started_ns: u64,
        link_timeout_ms: u32,
    ) -> Self {
        let timeout_ns = (link_timeout_ms as u64).saturating_mul(1_000_000);
        Self {
            deadline_ns: started_ns.saturating_add(timeout_ns),
            input: [0u8; MAX_PACKET_SIZE],
            input_len: 0,
            input_seq: None,
            k: 0,
            last_sent: (TaskRunState::Running, None),
            lockstep: req.lockstep,
            started_ns,
            steps: 0,
            suite_id: req.suite_id,
            test_id: req.test_id,
            timeout_ns,
            verdict: None,
        }
    }

    /// Applies a polled command to the run and returns what the server does
    /// next.
    fn accept(&mut self, cmd: &Command<'_>) -> Boundary {
        match *cmd {
            Command::StopNow { suite_id, test_id }
                if self.is_active(suite_id, test_id) =>
            {
                if self.verdict.is_none() {
                    self.verdict = Some((TaskRunState::Aborted, None));
                }
                Boundary::Nothing
            }
            Command::StopNow { .. } => {
                Boundary::Log("Error: StopNow addresses another lifecycle task")
            }
            Command::TaskInput {
                suite_id,
                test_id,
                seq,
                payload,
            } if self.is_active(suite_id, test_id) => {
                self.store_input(seq, payload)
            }
            Command::TaskInput { .. } => Boundary::Log(
                "Error: TaskInput addresses another lifecycle task",
            ),
            Command::SetSetting {
                suite_id,
                setting_id,
                value,
            } if suite_id == self.suite_id => {
                Boundary::Setting { setting_id, value }
            }
            Command::SetSetting { .. } => Boundary::Log(
                "Error: SetSetting for another suite rejected during a lifecycle run",
            ),
            Command::Heartbeat | Command::TryReset => Boundary::Nothing,
            Command::ListSuites
            | Command::RunExecutable { .. }
            | Command::StartTask { .. } => {
                Boundary::Log("Error: command rejected during a lifecycle run")
            }
        }
    }

    const fn is_active(&self, suite_id: u16, test_id: u16) -> bool {
        suite_id == self.suite_id && test_id == self.test_id
    }

    /// Stores an input packet when its index is current.
    fn store_input(&mut self, seq: u64, payload: &[u8]) -> Boundary {
        let Some(dst) = self.input.get_mut(..payload.len()) else {
            return Boundary::Log("Error: TaskInput payload too large");
        };
        if self.lockstep {
            if seq > self.k {
                self.verdict =
                    Some((TaskRunState::Error, Some("input sequence gap")));
                return Boundary::Nothing;
            }
            if seq < self.k {
                return Boundary::Log("Error: stale TaskInput ignored");
            }
        } else if self.input_seq.is_some_and(|stored| seq < stored) {
            return Boundary::Log("Error: stale TaskInput ignored");
        }
        dst.copy_from_slice(payload);
        self.input_len = payload.len();
        self.input_seq = Some(seq);
        Boundary::Nothing
    }
}

impl Default for TestIndexIndicator {
    fn default() -> Self {
        Self::new()
    }
}

impl TestIndexIndicator {
    /// Sentinel value representing the Idle state.
    const IDLE_STATE: isize = -1;

    /// Retrieves the current state, returning it as a safe Option.
    ///
    /// # Returns
    /// * `Option<usize>`
    ///     * `Some(idx)` - The active suite or test index.
    ///     * `None` - If currently idle.
    pub fn get(&self) -> Option<usize> {
        let current_state = self.state.load(Ordering::Acquire);

        if current_state < 0 {
            None
        } else {
            usize::try_from(current_state).ok()
        }
    }

    /// Creates a new indicator in the Idle state.
    ///
    /// # Returns
    /// * `Self` - An idle `TestIndexIndicator`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            state: AtomicIsize::new(Self::IDLE_STATE),
        }
    }

    /// Sets the indicator to a specific active test index.
    ///
    /// # Arguments
    /// * `index` - The active execution index.
    pub fn set_active(&self, index: usize) {
        // Explicitly pattern match on the Result
        let safe_index = isize::try_from(index).unwrap_or(Self::IDLE_STATE);

        self.state.store(safe_index, Ordering::Release);
    }

    /// Sets the indicator back to Idle.
    pub fn set_idle(&self) {
        self.state.store(Self::IDLE_STATE, Ordering::Release);
    }
}

/// Truncates a message to [`MAX_MESSAGE_SIZE`] bytes at a `char` boundary.
fn truncate_message(msg: &str) -> &str {
    let mut end = msg.len().min(MAX_MESSAGE_SIZE);
    while !msg.is_char_boundary(end) {
        end = end.saturating_sub(1);
    }
    msg.get(..end).unwrap_or("")
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::comms::{
        Command, HostComms, TaskRunState, Telemetry, TestState,
    };
    use crate::profiler::CPUProfiler;
    use crate::settings::{
        AtomicU8Setting, AtomicU32Setting, Setting, SettingValue,
    };
    use crate::{ExecDescriptor, SuiteDescriptor};
    use lifecycle_support::{
        Config, Counts, Event, RUN, Run, TASKS_PLAIN, TASKS_TIMEOUT,
        TASKS_TWINS, begin, count_events, count_frames, counts, final_state,
        input, log_count, run_tasks, samples, seen_inputs, sent, set_u8, start,
        states, stop, within_deadline,
    };
    use std::string::ToString;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::vec::Vec;

    mod lifecycle_support {
        include!("server_lifecycle_support.rs");
    }

    // --- Statics ---
    /// Held by the tests that read or write `TEST_U8_SETTING`, which every
    /// test over `SUITES` shares.
    static SETTING_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static SUITES: &[&SuiteDescriptor] = &[&SUITE_DESC];

    static SUITE_DESC: SuiteDescriptor = SuiteDescriptor {
        description: "mock_suite_desc",
        executables: SUITE_EXECUTABLES,
        name: "mock_suite",
        settings: SUITE_SETTINGS,
    };

    static SUITE_EXECUTABLES: &[ExecDescriptor] = &[ExecDescriptor {
        description: "dummy_desc",
        name: "dummy_test",
        test_fn: dummy_test_fn,
    }];

    static SUITE_SETTINGS: SettingsSlice = &[&TEST_U8_SETTING];

    static TEST_CALLED: AtomicBool = AtomicBool::new(false);

    static TEST_U8_SETTING: AtomicU8Setting =
        AtomicU8Setting::new("test_u8", "test_u8_desc", 42);

    // --- Types & Structs ---
    pub struct HostCPUProfiler;

    struct MockComms {
        commands: Vec<Command<'static>>,
        fail_on_poll: bool,
        flush_count: usize,
        payloads: RawPayloads,
    }

    /// A profiler whose clocks replay fixed readings.
    struct ClockProfiler {
        cycles: [u64; 2],
        nanos: [u64; 2],
        cycle_reads: std::cell::Cell<usize>,
        nano_reads: std::cell::Cell<usize>,
    }

    type RawPayloads = Vec<Vec<u8>>;
    /// Borrowed run of encoded telemetry frames.
    type Frames<'a> = &'a [Vec<u8>];
    type SettingsSlice = &'static [&'static dyn Setting];
    /// A server over the mock link.
    type MockServer = Server<'static, MockComms, HostCPUProfiler>;
    /// Outcome of a bounded server run: the loop result and the server.
    type RunOutcome = (ServerResult<&'static str>, MockServer);

    impl CPUProfiler for HostCPUProfiler {
        fn exit(&self) -> ! {
            panic!("exit called in tests");
        }

        fn get_cycles(&self) -> u64 {
            0
        }

        fn get_nanos(&self) -> u64 {
            0
        }

        fn get_sp(&self) -> usize {
            0
        }

        fn get_stack_end(&self) -> usize {
            0
        }

        fn reset(&self) -> ! {
            panic!("reset called in tests");
        }
    }

    impl HostComms for MockComms {
        type Error = &'static str;

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.flush_count = self.flush_count.saturating_add(1);
            Ok(())
        }

        fn poll_command(
            &mut self,
        ) -> Result<Option<Command<'static>>, Self::Error> {
            if self.fail_on_poll {
                return Err("Poll failed");
            }
            if self.commands.is_empty() {
                // Return an error to break the infinite runner loop
                return Err("Exit loop");
            }
            Ok(Some(self.commands.remove(0)))
        }

        fn send_telemetry(
            &mut self,
            telemetry: &Telemetry<'_>,
        ) -> Result<(), Self::Error> {
            let mut buf = [0u8; 512];
            let size = crate::comms::frame_telemetry(telemetry, &mut buf)
                .map_err(|_| "Failed to frame telemetry")?;

            let mut reader = crate::comms::FrameReader::new();
            let mut payload = None;
            for &b in buf.get(..size).ok_or("Buffer slice out of bounds")? {
                if let Some(p) = reader.handle_byte(b) {
                    payload = Some(p.to_vec());
                    break;
                }
            }

            let payload = payload.ok_or("No payload decoded")?;
            self.payloads.push(payload);
            Ok(())
        }
    }

    impl ClockProfiler {
        const fn new(cycles: [u64; 2], nanos: [u64; 2]) -> Self {
            Self {
                cycles,
                nanos,
                cycle_reads: std::cell::Cell::new(0),
                nano_reads: std::cell::Cell::new(0),
            }
        }

        fn next(reads: &std::cell::Cell<usize>, values: [u64; 2]) -> u64 {
            let n = reads.get();
            reads.set(n.saturating_add(1));
            values.get(n).copied().unwrap_or_else(|| values[1])
        }
    }

    impl CPUProfiler for ClockProfiler {
        fn get_cycles(&self) -> u64 {
            Self::next(&self.cycle_reads, self.cycles)
        }

        fn get_nanos(&self) -> u64 {
            Self::next(&self.nano_reads, self.nanos)
        }

        fn get_sp(&self) -> usize {
            0
        }

        fn get_stack_end(&self) -> usize {
            0
        }
    }

    // --- Helper Functions ---
    fn dummy_test_fn() {
        TEST_CALLED.store(true, Ordering::SeqCst);
    }

    /// Runs `server` until its loop fails, then hands it back with the result.
    ///
    /// The loop only ends when the link errors, so a server that stops
    /// polling would spin forever; the run is bounded to fail the test
    /// instead.
    fn lock_setting() -> std::sync::MutexGuard<'static, ()> {
        SETTING_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn run_bounded(server: MockServer) -> RunOutcome {
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let mut server = server;
            let result = server.run();
            let _ = tx.send((result, server));
        });
        rx.recv_timeout(std::time::Duration::from_secs(5))
            .expect("the server loop did not terminate")
    }

    // --- Tests ---
    #[test]
    fn test_atomic_settings_u8_u32() {
        let u8_setting = AtomicU8Setting::new("u8_set", "u8_desc", 10);
        assert_eq!(u8_setting.name(), "u8_set");
        assert_eq!(u8_setting.description(), "u8_desc");
        assert_eq!(
            u8_setting.expected_type(),
            crate::settings::SettingType::U8
        );
        assert_eq!(u8_setting.get(), SettingValue::U8(10));
        assert!(u8_setting.set(SettingValue::U8(20)).is_ok());
        assert_eq!(u8_setting.get(), SettingValue::U8(20));
        assert!(u8_setting.set(SettingValue::U32(20)).is_err());

        let u32_setting = AtomicU32Setting::new("u32_set", "u32_desc", 100);
        assert_eq!(u32_setting.name(), "u32_set");
        assert_eq!(u32_setting.description(), "u32_desc");
        assert_eq!(
            u32_setting.expected_type(),
            crate::settings::SettingType::U32
        );
        assert_eq!(u32_setting.get(), SettingValue::U32(100));
        assert!(u32_setting.set(SettingValue::U32(200)).is_ok());
        assert_eq!(u32_setting.get(), SettingValue::U32(200));
        assert!(u32_setting.set(SettingValue::U8(200)).is_err());
    }

    #[test]
    fn test_atomic_settings_bool_f32() {
        use crate::settings::{
            AtomicBoolSetting, AtomicF32Setting, SettingType,
        };

        let bool_set = AtomicBoolSetting::new("bool_set", "bool_desc", true);
        assert_eq!(bool_set.expected_type(), SettingType::Bool);
        assert_eq!(bool_set.get(), SettingValue::Bool(true));
        assert!(bool_set.set(SettingValue::Bool(false)).is_ok());
        assert_eq!(bool_set.get(), SettingValue::Bool(false));
        assert!(bool_set.set(SettingValue::U8(0)).is_err());

        let f32_set = AtomicF32Setting::new("f32_set", "f32_desc", 1.5);
        assert_eq!(f32_set.expected_type(), SettingType::F32);
        assert_eq!(f32_set.get(), SettingValue::F32(1.5));
        assert!(f32_set.set(SettingValue::F32(-2.5)).is_ok());
        assert_eq!(f32_set.get(), SettingValue::F32(-2.5));
        assert!(f32_set.set(SettingValue::Bool(true)).is_err());
    }

    #[test]
    fn test_atomic_settings_ints() {
        use crate::settings::{
            AtomicI8Setting, AtomicI32Setting, AtomicU16Setting, SettingType,
        };

        let i32_set = AtomicI32Setting::new("i32_set", "i32_desc", -10);
        assert_eq!(i32_set.expected_type(), SettingType::I32);
        assert!(i32_set.set(SettingValue::I32(10)).is_ok());
        assert_eq!(i32_set.get(), SettingValue::I32(10));
        assert!(i32_set.set(SettingValue::Bool(true)).is_err());

        let i8_set = AtomicI8Setting::new("i8_set", "i8_desc", -5);
        assert_eq!(i8_set.expected_type(), SettingType::I8);
        assert!(i8_set.set(SettingValue::I8(5)).is_ok());
        assert_eq!(i8_set.get(), SettingValue::I8(5));
        assert!(i8_set.set(SettingValue::Bool(true)).is_err());

        let u16_set = AtomicU16Setting::new("u16_set", "u16_desc", 20);
        assert_eq!(u16_set.expected_type(), SettingType::U16);
        assert!(u16_set.set(SettingValue::U16(40)).is_ok());
        assert_eq!(u16_set.get(), SettingValue::U16(40));
        assert!(u16_set.set(SettingValue::Bool(true)).is_err());
    }

    #[test]
    #[cfg(target_has_atomic = "64")]
    fn test_atomic_settings_u64() {
        use crate::settings::{AtomicU64Setting, SettingType};

        let u64_set = AtomicU64Setting::new("u64_set", "u64_desc", 100);
        assert_eq!(u64_set.expected_type(), SettingType::U64);
        assert!(u64_set.set(SettingValue::U64(200)).is_ok());
        assert_eq!(u64_set.get(), SettingValue::U64(200));
        assert!(u64_set.set(SettingValue::Bool(true)).is_err());
    }

    #[test]
    fn test_setting_value_partial_eq() {
        assert_eq!(SettingValue::Bool(true), SettingValue::Bool(true));
        assert_ne!(SettingValue::Bool(true), SettingValue::Bool(false));

        assert_eq!(SettingValue::F32(1.5), SettingValue::F32(1.5));
        assert_ne!(SettingValue::F32(1.5), SettingValue::F32(2.5));

        assert_eq!(SettingValue::I32(-10), SettingValue::I32(-10));
        assert_ne!(SettingValue::I32(-10), SettingValue::I32(10));

        assert_eq!(SettingValue::I8(-5), SettingValue::I8(-5));
        assert_ne!(SettingValue::I8(-5), SettingValue::I8(5));

        assert_eq!(SettingValue::U16(20), SettingValue::U16(20));
        assert_ne!(SettingValue::U16(20), SettingValue::U16(40));

        assert_eq!(SettingValue::U32(100), SettingValue::U32(100));
        assert_ne!(SettingValue::U32(100), SettingValue::U32(200));

        assert_eq!(SettingValue::U64(1000), SettingValue::U64(1000));
        assert_ne!(SettingValue::U64(1000), SettingValue::U64(2000));

        assert_eq!(SettingValue::U8(42), SettingValue::U8(42));
        assert_ne!(SettingValue::U8(42), SettingValue::U8(24));

        // Mismatched types
        assert_ne!(SettingValue::Bool(true), SettingValue::U8(1));
        assert_ne!(SettingValue::F32(1.0), SettingValue::I32(1));
    }

    #[test]
    fn test_context_locked_noop() {
        let comms = MockComms {
            commands: Vec::new(),
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let mut context = Context::new(comms, HostCPUProfiler);
        assert!(context.comms_lock.try_lock());

        assert!(!context.flush_locked().unwrap());
        assert!(context.poll_command_locked().unwrap().is_none());
        assert!(
            !context
                .send_telemetry_and_flush_locked(&Telemetry::DiscoveryComplete)
                .unwrap()
        );
        assert!(
            !context
                .send_telemetry_locked(&Telemetry::DiscoveryComplete)
                .unwrap()
        );

        assert_eq!(context.comms.payloads, Vec::<Vec<u8>>::new());
        assert_eq!(context.comms.flush_count, 0);

        context.comms_lock.unlock();
        assert!(
            context
                .send_telemetry_and_flush_locked(&Telemetry::DiscoveryComplete)
                .unwrap()
        );
        assert_eq!(context.comms.payloads.len(), 1);
        assert_eq!(context.comms.flush_count, 1);
    }

    #[test]
    fn test_server_exit() {
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut s = Server::new(
                    Context::new(
                        MockComms {
                            commands: Vec::new(),
                            payloads: Vec::new(),
                            flush_count: 0,
                            fail_on_poll: false,
                        },
                        HostCPUProfiler,
                    ),
                    SUITES,
                );
                s.exit();
            }));
        assert!(res.is_err());

        let res2 =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut s = Server::new(
                    Context::new(
                        MockComms {
                            commands: Vec::new(),
                            payloads: Vec::new(),
                            flush_count: 0,
                            fail_on_poll: false,
                        },
                        HostCPUProfiler,
                    ),
                    SUITES,
                );
                assert!(s.context.comms_lock.try_lock());
                s.exit();
            }));
        assert!(res2.is_err());
    }

    #[test]
    fn test_index_indicator() {
        let indicator = TestIndexIndicator::new();
        assert!(indicator.get().is_none());

        indicator.set_active(5);
        assert_eq!(indicator.get(), Some(5));

        indicator.set_idle();
        assert!(indicator.get().is_none());

        let default_indicator = TestIndexIndicator::default();
        assert!(default_indicator.get().is_none());
    }

    #[test]
    fn test_host_cpu_profile_utils() {
        let utils = HostCPUProfiler;
        assert_eq!(utils.get_cycles(), 0);
        assert_eq!(utils.get_nanos(), 0);
        assert_eq!(utils.get_sp(), 0);
    }

    /// Asserts that `payloads` opens with a matching `TargetInfo` and
    /// returns the frames after it.
    fn after_target_info(payloads: &RawPayloads) -> Frames<'_> {
        assert!(payloads.len() >= 5);
        let info: Telemetry<'_> =
            postcard::from_bytes(payloads.first().unwrap()).unwrap();
        assert!(matches!(
            info,
            Telemetry::TargetInfo {
                protocol_version: crate::comms::PROTOCOL_VERSION,
                board_id: 0,
                core_clock_hz: 0,
                ..
            }
        ));
        payloads.get(1..).unwrap()
    }

    #[test]
    fn test_server_discovery() {
        let _guard = lock_setting();
        let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
        let comms = MockComms {
            commands: std::vec![Command::ListSuites],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        // TargetInfo leads every discovery stream.
        let p = after_target_info(&server.context.comms.payloads);

        let t0: Telemetry<'_> =
            postcard::from_bytes(p.first().unwrap()).unwrap();
        assert!(matches!(
            t0,
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "mock_suite",
                ..
            }
        ));

        let t1: Telemetry<'_> =
            postcard::from_bytes(p.get(1).unwrap()).unwrap();
        assert!(matches!(
            t1,
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "dummy_test",
                ..
            }
        ));

        let t2: Telemetry<'_> =
            postcard::from_bytes(p.get(2).unwrap()).unwrap();
        assert!(matches!(
            t2,
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                name: "test_u8",
                value: SettingValue::U8(42),
                ..
            }
        ));

        let t3: Telemetry<'_> =
            postcard::from_bytes(p.get(3).unwrap()).unwrap();
        assert!(matches!(t3, Telemetry::DiscoveryComplete));
    }

    #[test]
    fn test_server_ok_to_reset() {
        let comms = MockComms {
            commands: std::vec![Command::TryReset],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));
        assert_eq!(server.context.comms.payloads, Vec::<Vec<u8>>::new());
    }

    #[test]
    fn test_server_out_of_bounds() {
        let comms = MockComms {
            commands: std::vec![
                Command::RunExecutable {
                    suite_id: 99,
                    test_id: 0
                },
                Command::RunExecutable {
                    suite_id: 0,
                    test_id: 99
                },
                Command::SetSetting {
                    suite_id: 99,
                    setting_id: 0,
                    value: SettingValue::U8(0)
                },
                Command::SetSetting {
                    suite_id: 0,
                    setting_id: 99,
                    value: SettingValue::U8(0)
                },
            ],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.comms.payloads;
        assert_eq!(p.len(), 4);
        for item in p {
            let t: Telemetry<'_> = postcard::from_bytes(item).unwrap();
            assert!(matches!(t, Telemetry::Log(_)));
        }
    }

    #[test]
    fn test_server_poll_command_error() {
        let comms = MockComms {
            commands: Vec::new(),
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: true,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, _server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Poll failed"));
    }

    #[test]
    fn test_server_run_test() {
        TEST_CALLED.store(false, Ordering::SeqCst);
        let comms = MockComms {
            commands: std::vec![Command::RunExecutable {
                suite_id: 0,
                test_id: 0
            }],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        assert!(TEST_CALLED.load(Ordering::SeqCst));

        let p = &server.context.comms.payloads;
        assert_eq!(p.len(), 3);

        let t0: Telemetry<'_> =
            postcard::from_bytes(p.first().unwrap()).unwrap();
        assert!(matches!(
            t0,
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Running
            }
        ));

        let t1: Telemetry<'_> =
            postcard::from_bytes(p.get(1).unwrap()).unwrap();
        assert!(matches!(
            t1,
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Passed
            }
        ));

        let t2: Telemetry<'_> =
            postcard::from_bytes(p.get(2).unwrap()).unwrap();
        assert!(matches!(
            t2,
            Telemetry::MetricReport {
                suite_id: 0,
                test_id: 0,
                ..
            }
        ));
    }

    #[test]
    fn test_server_set_setting() {
        let _guard = lock_setting();
        let comms = MockComms {
            commands: std::vec![Command::SetSetting {
                suite_id: 0,
                setting_id: 0,
                value: SettingValue::U8(100)
            }],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.comms.payloads;
        assert_eq!(p.len(), 1);
        let t0: Telemetry<'_> =
            postcard::from_bytes(p.first().unwrap()).unwrap();
        assert!(matches!(
            t0,
            Telemetry::SettingInfo {
                suite_id: 0,
                setting_id: 0,
                value: SettingValue::U8(100),
                ..
            }
        ));
        let _ = TEST_U8_SETTING.set(SettingValue::U8(42));
    }

    #[test]
    fn test_server_set_setting_type_mismatch() {
        let _guard = lock_setting();
        let comms = MockComms {
            commands: std::vec![Command::SetSetting {
                suite_id: 0,
                setting_id: 0,
                value: SettingValue::U32(999)
            }],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.comms.payloads;
        assert_eq!(p.len(), 2);
        let t0: Telemetry<'_> =
            postcard::from_bytes(p.first().unwrap()).unwrap();
        assert!(matches!(t0, Telemetry::Log(_)));

        let t1: Telemetry<'_> =
            postcard::from_bytes(p.get(1).unwrap()).unwrap();
        if let Telemetry::SettingInfo { value, .. } = t1 {
            assert!(matches!(value, SettingValue::U8(_)));
        } else {
            panic!("Expected SettingInfo");
        }
    }

    fn quiet_comms() -> MockComms {
        MockComms {
            commands: Vec::new(),
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        }
    }

    #[test]
    fn an_unlocked_flush_reaches_the_link() {
        let mut context = Context::new(quiet_comms(), HostCPUProfiler);
        assert_eq!(context.flush_locked(), Ok(true));
        assert_eq!(context.comms.flush_count, 1);
        assert!(context.comms_lock.try_lock(), "the lock was released");
    }

    #[test]
    fn profiling_reports_elapsed_cycles_and_microseconds() {
        let context = Context::new(
            quiet_comms(),
            ClockProfiler::new([100, 350], [2_000, 7_500]),
        );
        let metrics = context.profile_test(|| {});
        assert_eq!(metrics.cycles, 250);
        assert_eq!(metrics.time_us, 5, "5500 ns is 5 whole microseconds");
    }

    #[test]
    fn log_timestamps_are_in_microseconds() {
        let mut context = Context::new(
            quiet_comms(),
            ClockProfiler::new([0, 0], [4_321_000, 4_321_000]),
        );
        context.report_error_log(1, 2, "boom").unwrap();
        context
            .report_setting_error(3, "gain", "out of range")
            .unwrap();
        let stamps: Vec<u64> = context
            .comms
            .payloads
            .iter()
            .map(|payload| {
                match postcard::from_bytes::<Telemetry<'_>>(payload).unwrap() {
                    Telemetry::Log(log) => log.timestamp_us,
                    other => panic!("expected a log, got {other:?}"),
                }
            })
            .collect();
        assert_eq!(stamps, [4_321, 4_321]);
    }

    #[test]
    fn the_index_indicator_holds_zero() {
        let indicator = TestIndexIndicator::new();
        indicator.set_active(0);
        assert_eq!(indicator.get(), Some(0));
        indicator.set_active(7);
        assert_eq!(indicator.get(), Some(7));
    }

    include!("server_lifecycle_tests.rs");
}
