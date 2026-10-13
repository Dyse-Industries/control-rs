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

use core::sync::atomic::{AtomicIsize, Ordering};

use crate::SuiteDescriptor;
use crate::comms::{Command, PROTOCOL_VERSION, Telemetry};
pub use crate::context::{Context, TestMetrics};
use crate::link::{Link, UnitOf};
use crate::run::{Atomic, Runner};
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
/// # use control_rs_ets::link::{Link, RecvOf, SendOf};
/// # use control_rs_ets::profiler::CPUProfiler;
///
/// # struct MockLink;
/// # impl Link for MockLink {
/// #     type Error = &'static str;
/// #     fn recv(&mut self, _: &mut [u8]) -> RecvOf<Self> { Ok(None) }
/// #     fn send(&mut self, _: &[u8]) -> SendOf<Self> { Ok(true) }
/// # }
/// # struct MockProfiler;
/// # impl CPUProfiler for MockProfiler {
/// #     fn get_cycles(&self) -> u64 { 0 }
/// #     fn get_nanos(&self) -> u64 { 0 }
/// #     fn get_sp(&self) -> usize { 0 }
/// #     fn get_stack_end(&self) -> usize { 0 }
/// # }
///
/// let ctx = Context::new(MockLink, MockProfiler);
/// let server = Server::new(ctx, &[]);
/// assert_eq!(server.suites.len(), 0);
/// ```
pub struct Server<'a, L, P> {
    /// The global context containing comms, `cpu_utils` and the comms lock.
    pub context: Context<L, P>,
    /// The registered test suites.
    pub suites: &'a [&'static SuiteDescriptor],
}

/// Result of server operations.
///
/// Indicates success or contains transport error `E`.
pub type ServerResult<E> = Result<(), E>;

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

impl<'a, L, P> Server<'a, L, P>
where
    L: Link,
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
            self.context.link.close();
            self.context.comms_lock.unlock();
        } else {
            self.context.link.close();
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
        context: Context<L, P>,
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
    /// * `UnitOf<L>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error `L::Error` propagated from the underlying communication interface if polling,
    /// or transmitting telemetry fails.
    pub fn run(&mut self) -> UnitOf<L> {
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
        }
    }

    fn run_test(&mut self, suite_id: u16, test_id: u16) -> UnitOf<L> {
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

        Runner::<L, P, Atomic>::new(&mut self.context, suite_id, test_id, exec)
            .run()
    }

    fn set_setting(
        &mut self,
        suite_id: u16,
        setting_id: u16,
        value: SettingValue,
    ) -> UnitOf<L> {
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
        let _ =
            self.context
                .send_telemetry_locked(&Telemetry::SettingInfo {
                    suite_id,
                    setting_id,
                    name: setting.name(),
                    description: setting.description(),
                    value: setting.get(),
                })?;

        Ok(())
    }

    fn stream_discovery(&mut self) -> UnitOf<L> {
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
        }

        let _ = self
            .context
            .send_telemetry_locked(&Telemetry::DiscoveryComplete)?;

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
    use crate::comms::{Command, Telemetry, TestState};
    use crate::link::{Link, RecvOf, SendOf};
    use crate::profiler::CPUProfiler;
    use crate::settings::{
        AtomicU8Setting, AtomicU32Setting, Setting, SettingValue,
    };
    use crate::{ExecDescriptor, SuiteDescriptor};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::vec::Vec;

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
        commands: Vec<Command>,
        fail_on_poll: bool,
        payloads: RawPayloads,
    }

    /// A link that reports busy for the first `busy` sends.
    struct BusyLink {
        busy: usize,
        offered: usize,
        sent: usize,
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

    impl Link for MockComms {
        type Error = &'static str;

        fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
            if self.fail_on_poll {
                return Err("Poll failed");
            }
            if self.commands.is_empty() {
                // Return an error to break the infinite runner loop
                return Err("Exit loop");
            }
            let cmd = self.commands.remove(0);
            let len = postcard::to_slice(&cmd, buf)
                .map_err(|_| "Failed to encode command")?
                .len();
            Ok(Some(len))
        }

        fn send(&mut self, msg: &[u8]) -> SendOf<Self> {
            self.payloads.push(msg.to_vec());
            Ok(true)
        }
    }

    impl Link for BusyLink {
        type Error = &'static str;

        fn recv(&mut self, _buf: &mut [u8]) -> RecvOf<Self> {
            Ok(None)
        }

        fn send(&mut self, _msg: &[u8]) -> SendOf<Self> {
            self.offered = self.offered.saturating_add(1);
            if self.offered <= self.busy {
                return Ok(false);
            }
            self.sent = self.sent.saturating_add(1);
            Ok(true)
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
            fail_on_poll: false,
        };
        let mut context = Context::new(comms, HostCPUProfiler);
        assert!(context.comms_lock.try_lock());

        assert!(context.poll_command_locked().unwrap().is_none());
        assert!(
            !context
                .send_telemetry_locked(&Telemetry::DiscoveryComplete)
                .unwrap()
        );

        assert_eq!(context.link.payloads, Vec::<Vec<u8>>::new());

        context.comms_lock.unlock();
        assert!(
            context
                .send_telemetry_locked(&Telemetry::DiscoveryComplete)
                .unwrap()
        );
        assert_eq!(context.link.payloads.len(), 1);
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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        // TargetInfo leads every discovery stream.
        let p = after_target_info(&server.context.link.payloads);

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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));
        assert_eq!(server.context.link.payloads, Vec::<Vec<u8>>::new());
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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.link.payloads;
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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        assert!(TEST_CALLED.load(Ordering::SeqCst));

        let p = &server.context.link.payloads;
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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.link.payloads;
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
            fail_on_poll: false,
        };
        let context = Context::new(comms, HostCPUProfiler);
        let (res, server) = run_bounded(Server::new(context, SUITES));
        assert_eq!(res, Err("Exit loop"));

        let p = &server.context.link.payloads;
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
            fail_on_poll: false,
        }
    }

    #[test]
    fn an_unlocked_send_reaches_the_link_and_releases_the_lock() {
        let mut context = Context::new(quiet_comms(), HostCPUProfiler);
        assert_eq!(
            context.send_telemetry_locked(&Telemetry::DiscoveryComplete),
            Ok(true)
        );
        assert_eq!(context.link.payloads.len(), 1);
        assert!(context.comms_lock.try_lock(), "the lock was released");
    }

    #[test]
    fn a_busy_link_is_retried_until_it_accepts_the_whole_message() {
        let link = BusyLink {
            busy: 5,
            offered: 0,
            sent: 0,
        };
        let mut context = Context::new(link, HostCPUProfiler);
        assert_eq!(
            context.send_telemetry_locked(&Telemetry::DiscoveryComplete),
            Ok(true)
        );
        assert_eq!(context.link.offered, 6);
        assert_eq!(context.link.sent, 1);
    }

    #[test]
    fn the_unlocked_send_offers_once_and_reports_busy() {
        let link = BusyLink {
            busy: 1,
            offered: 0,
            sent: 0,
        };
        let mut context = Context::new(link, HostCPUProfiler);
        assert_eq!(
            context.try_send_telemetry(&Telemetry::DiscoveryComplete),
            Ok(crate::context::SendOutcome::Busy)
        );
        assert_eq!(context.link.offered, 1);
        assert_eq!(context.link.sent, 0);
    }

    #[test]
    fn a_message_over_the_size_limit_is_dropped_not_sent() {
        let long = "x".repeat(600);
        let mut context = Context::new(quiet_comms(), HostCPUProfiler);
        assert_eq!(
            context.try_send_telemetry(&Telemetry::TargetPanic {
                message: &long,
                file: "f.rs",
                line: 1,
            }),
            Ok(crate::context::SendOutcome::Dropped)
        );
        assert_eq!(context.link.payloads, Vec::<Vec<u8>>::new());
    }

    #[test]
    fn an_undecodable_message_is_discarded() {
        struct GarbageLink;
        impl Link for GarbageLink {
            type Error = &'static str;
            fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
                for (dst, src) in buf.iter_mut().zip([0xFF_u8, 0xFF]) {
                    *dst = src;
                }
                Ok(Some(2))
            }
            fn send(&mut self, _msg: &[u8]) -> SendOf<Self> {
                Ok(true)
            }
        }
        let mut context = Context::new(GarbageLink, HostCPUProfiler);
        assert!(context.poll_command_locked().unwrap().is_none());
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
            .link
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
}
