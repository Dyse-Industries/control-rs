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
//! - **HIL Server**: The central coordinator (`Server`) executing suites, managing settings updates,
//!   and reporting telemetry outcomes back to the host machine.
//! - **Execution Context**: `Context` wraps host communication mechanisms and CPU profiling metrics
//!   using thread-safe mutex-free locking (`CommsLock`).
//! - **Test Index Indicator**: Thread-safe atomic indicator (`TestIndexIndicator`) tracking the active suite
//!   and test indexes to let exception/panic handlers report where a crash happened.

use core::sync::atomic::{AtomicIsize, Ordering};

use crate::SuiteDescriptor;
use crate::comms::{Command, CommsLock, HostComms, Telemetry, TestState};
use crate::settings::SettingValue;

// --- Static variables ---

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
/// use control_rs_hil::server::Context;
/// # use control_rs_hil::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
/// # use control_rs_hil::profiler::CPUProfiler;
///
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

/// Interactive HIL Server.
///
/// The `Server` coordinates the execution of HIL tests on the target. It manages the boot discovery phase,
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
/// use control_rs_hil::server::{Server, Context};
/// # use control_rs_hil::comms::{HostComms, Telemetry, Command, SendResult, PollResult};
/// # use control_rs_hil::profiler::CPUProfiler;
///
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
///
/// let ctx = Context::new(MockComms, MockProfiler);
/// let server = Server::new(ctx, &[]);
/// assert_eq!(server.suites.len(), 0);
/// ```
pub struct Server<'a, C, P> {
    /// The global context containing comms, `cpu_utils` and the comms lock.
    pub context: Context<C, P>,
    /// The registered test suites.
    pub suites: &'a [&'static SuiteDescriptor],
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
/// use control_rs_hil::server::TestIndexIndicator;
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
    /// If the lock is already held (e.g., by a panic handler or an interrupt), this method returns `Ok(false)`
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
    pub fn poll_command_locked(&mut self) -> Result<Option<Command>, C::Error> {
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
        // SAFETY: We retrieve the active stack pointer `sp` immediately before painting.
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

        // SAFETY: We query the peak stack usage relative to the same stack pointer `sp` used
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
        Self { context, suites }
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
                    Command::TryReset => {}
                }
            }

            let _ = self.context.flush_locked()?;
        }
    }

    #[allow(clippy::arithmetic_side_effects)]
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
        let suite_idx = suite_id as usize;
        let setting_idx = setting_id as usize;

        let Some(&suite) = self.suites.get(suite_idx) else {
            self.context.report_error_log(
                suite_id,
                0,
                "Error: SetSetting suite_id out of range",
            )?;
            return Ok(());
        };
        let Some(&setting) = suite.settings.get(setting_idx) else {
            self.context.report_error_log(
                suite_id,
                0,
                "Error: SetSetting setting_id out of range",
            )?;
            return Ok(());
        };

        if let Err(err) = setting.set(value) {
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

        Ok(())
    }

    fn stream_discovery(&mut self) -> ServerResult<C::Error> {
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
        }

        let _ = self
            .context
            .send_telemetry_and_flush_locked(&Telemetry::DiscoveryComplete)?;

        Ok(())
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

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::comms::{Command, HostComms, Telemetry, TestState};
    use crate::profiler::CPUProfiler;
    use crate::settings::{
        AtomicU8Setting, AtomicU32Setting, Setting, SettingValue,
    };
    use crate::{ExecDescriptor, SuiteDescriptor};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::vec::Vec;

    // --- Statics ---
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
        commands: Vec<Command>,
        fail_on_poll: bool,
        flush_count: usize,
        payloads: RawPayloads,
    }

    type RawPayloads = Vec<Vec<u8>>;
    type SettingsSlice = &'static [&'static dyn Setting];

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

        #[allow(clippy::arithmetic_side_effects)]
        fn flush(&mut self) -> Result<(), Self::Error> {
            self.flush_count += 1;
            Ok(())
        }

        fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
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

    // --- Helper Functions ---
    fn dummy_test_fn() {
        TEST_CALLED.store(true, Ordering::SeqCst);
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

    #[test]
    fn test_server_discovery() {
        let comms = MockComms {
            commands: std::vec![Command::ListSuites],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let mut server = Server::new(context, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        // Check telemetry sent
        let p = &server.context.comms.payloads;
        assert!(p.len() >= 4);

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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
    }

    #[test]
    fn test_server_set_setting_type_mismatch() {
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
        let mut server = Server::new(context, SUITES);
        let res = server.run();
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
}
