//! On-target Server event loop.
//!
//! # Description
//!
//! This module implements the interactive Server, its execution context and the global run state
//! indicators. The server polls whole messages from its [`Link`], decodes `Command`s, edits
//! parameters, runs cases and answers with `Telemetry`. It holds no framing code.
//!
//! # Core Concepts
//!
//! - **Server**: The dispatcher (`Server`) that executes suites, manages parameters and reports
//!   outcomes to the host.
//! - **Execution Context**: `Context` owns the link, the CPU profiler and a lock (`CommsLock`)
//!   that serializes the main loop and the fault path on the link without blocking.
//! - **Test Index Indicator**: Atomic indicators (`TestIndexIndicator`) tracking the active suite
//!   and item so the fault path can report where a crash happened.
//! - **Unsupported commands**: A command the server cannot execute yet is answered with an error
//!   log and changes no state.

use core::sync::atomic::{AtomicBool, AtomicIsize, Ordering};

use crate::link::{Link, RecvOf, SendOf, UnitOf};
use crate::wire::{
    Command, ExecMode, ItemState, LogMessage, MAX_MESSAGE_SIZE,
    PROTOCOL_VERSION, PacketInfo, Param, ParamKey, PayloadEncoding, Telemetry,
    Verdict,
};
use crate::{ParamTable, SuiteDescriptor};

// --- Static variables ---

/// Global tracker for the currently executing suite ID.
/// Used by the panic handler to report test failures.
///
/// # Safety
/// Accessing or updating this indicator is thread-safe and atomic.
pub static CURRENT_SUITE: TestIndexIndicator = TestIndexIndicator::new();

/// Global tracker for the currently executing item ID.
/// Used by the panic handler to report test failures.
///
/// # Safety
/// Accessing or updating this indicator is thread-safe and atomic.
pub static CURRENT_TEST: TestIndexIndicator = TestIndexIndicator::new();

/// Log level of an error.
const LEVEL_ERROR: u8 = 0;
/// Origin tag of log messages the server produces.
const LOG_TARGET: &str = "server";

// --- Type definitions ---

/// Non-blocking lock that serializes access to the link between the main loop and the fault path.
///
/// `try_lock` never waits, so a handler running with interrupts disabled cannot deadlock on it.
///
/// # Safety
/// This struct does not use `unsafe` code.
///
/// # Panics
/// Lock operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::server::CommsLock;
///
/// let lock = CommsLock::new();
/// assert!(lock.try_lock());
/// assert!(!lock.try_lock());
/// lock.unlock();
/// assert!(lock.try_lock());
/// ```
pub struct CommsLock {
    locked: AtomicBool,
}

/// Context object that encapsulates the link, CPU profiling utilities and the communication lock.
///
/// To avoid races between the main event loop and the fault handlers that also send telemetry,
/// the `Context` holds a `CommsLock`. The `_locked` methods take the lock and return `Ok(false)`
/// without touching the link when it is held, so they never block on it.
///
/// # Safety
/// Access to the link is protected by the internal `CommsLock`. This struct does not use `unsafe`
/// code directly.
///
/// # Panics
/// Context operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::server::Context;
/// # use control_rs_ets::link::{Link, RecvOf, SendOf};
/// # use control_rs_ets::wire::Telemetry;
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
/// let mut ctx = Context::new(MockLink, MockProfiler);
/// assert_eq!(ctx.send_telemetry_locked(&Telemetry::DiscoveryComplete), Ok(true));
/// ```
pub struct Context<L, P> {
    /// Message link to the host.
    pub link: L,
    /// Thread-safe communications lock to protect the link.
    pub comms_lock: CommsLock,
    /// CPU profiling and execution utilities.
    pub cpu_utils: P,
    tx: [u8; MAX_MESSAGE_SIZE],
}

/// Exclusive borrow of a [`Context`].
pub type ContextMut<'a, L, P> = &'a mut Context<L, P>;

/// What happened to one send attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SendOutcome {
    /// The link accepted the whole message.
    Sent,
    /// The link is busy; nothing was transmitted.
    Busy,
    /// The message does not encode within the maximum message size and was not transmitted.
    Dropped,
}

/// Result of one unlocked send attempt on link `L`.
pub type TrySendOf<L> = Result<SendOutcome, <L as Link>::Error>;

/// Interactive Embedded Test Server.
///
/// The `Server` coordinates the execution of ETS cases on the target. It answers discovery,
/// processes parameter updates, runs cases inside critical sections (interrupts disabled) and
/// returns telemetry and cycle/stack usage metrics.
///
/// # Safety
/// This structure does not use `unsafe` code.
///
/// # Panics
/// Server operations do not panic.
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
    /// The global context containing the link, `cpu_utils` and the comms lock.
    pub context: Context<L, P>,
    /// The registered test suites.
    pub suites: &'a [&'static SuiteDescriptor],
    rx: [u8; MAX_MESSAGE_SIZE],
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

/// Borrows of the server that a command handler needs, disjoint from the receive buffer.
struct Exec<'e, L, P> {
    context: ContextMut<'e, L, P>,
    suites: &'e [&'static SuiteDescriptor],
}

impl Default for CommsLock {
    fn default() -> Self {
        Self::new()
    }
}

impl CommsLock {
    /// Creates a new unlocked `CommsLock`.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            locked: AtomicBool::new(false),
        }
    }

    /// Attempts to take the lock without waiting; returns `true` if it was taken.
    pub fn try_lock(&self) -> bool {
        self.locked
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_ok()
    }

    /// Releases the lock.
    pub fn unlock(&self) {
        self.locked.store(false, Ordering::Release);
    }
}

impl<L: Link, P> Context<L, P> {
    /// Creates a new `Context`.
    ///
    /// # Arguments
    /// * `link` - Message link to the host.
    /// * `cpu_utils` - CPU execution/profiler.
    ///
    /// # Returns
    /// * `Self` - Context instance.
    pub const fn new(link: L, cpu_utils: P) -> Self {
        Self {
            link,
            comms_lock: CommsLock::new(),
            cpu_utils,
            tx: [0; MAX_MESSAGE_SIZE],
        }
    }

    /// Receives one message into `buf` without taking the lock.
    ///
    /// The caller must hold the communications lock; the fault path uses this after it took the
    /// lock itself.
    ///
    /// # Errors
    /// Returns a transport error if the link fails.
    pub fn recv(&mut self, buf: &mut [u8]) -> RecvOf<L> {
        self.link.recv(buf)
    }

    /// Receives one message into `buf` safely by acquiring the comms lock first.
    ///
    /// If the lock is already held, this returns `Ok(None)` immediately to prevent nested or
    /// concurrent receives.
    ///
    /// # Errors
    /// Returns a transport error if receiving fails.
    pub fn recv_locked(&mut self, buf: &mut [u8]) -> RecvOf<L> {
        if !self.comms_lock.try_lock() {
            return Ok(None);
        }
        let res = self.link.recv(buf);
        self.comms_lock.unlock();
        res
    }

    /// Encodes `telemetry` and offers it to the link once, without taking the lock.
    ///
    /// The caller must hold the communications lock. The fault path uses this so a busy link
    /// drops the report instead of waiting.
    ///
    /// # Errors
    /// Returns a transport error if the link fails.
    pub fn try_send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> TrySendOf<L> {
        let Ok(encoded) = postcard::to_slice(telemetry, &mut self.tx) else {
            return Ok(SendOutcome::Dropped);
        };
        if self.link.send(encoded)? {
            Ok(SendOutcome::Sent)
        } else {
            Ok(SendOutcome::Busy)
        }
    }

    /// Sends telemetry safely by acquiring the comms lock first, retrying while the link is busy.
    ///
    /// If the lock is already held, this returns `Ok(false)` immediately. This is the primary safe
    /// interface for regular telemetry.
    ///
    /// # Arguments
    /// * `telemetry` - Reference to the telemetry data structure.
    ///
    /// # Returns
    /// * `SendOf<L>` - `true` if the link accepted the message; `false` if the lock was busy or
    ///   the message did not encode within the maximum message size.
    ///
    /// # Errors
    /// Returns a transport error if writing fails.
    pub fn send_telemetry_locked(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> SendOf<L> {
        if !self.comms_lock.try_lock() {
            return Ok(false);
        }
        let res = loop {
            match self.try_send_telemetry(telemetry) {
                Ok(SendOutcome::Busy) => core::hint::spin_loop(),
                Ok(outcome) => break Ok(outcome == SendOutcome::Sent),
                Err(err) => break Err(err),
            }
        };
        self.comms_lock.unlock();
        res
    }
}

impl<L: Link, P: crate::profiler::CPUProfiler> Context<L, P> {
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
    /// Returns a transport error if writing fails.
    pub fn report_error_log(&mut self, msg: &str) -> UnitOf<L> {
        let _ = self.send_telemetry_locked(&Telemetry::Log(LogMessage {
            level: LEVEL_ERROR,
            target: LOG_TARGET,
            message: msg,
        }))?;
        Ok(())
    }

    /// Reports a refused parameter write via Log telemetry.
    ///
    /// # Errors
    /// Returns a transport error if sending fails.
    pub fn report_param_error(
        &mut self,
        param_name: &str,
        err: &'static str,
    ) -> UnitOf<L> {
        let mut err_buf = [0u8; 128];
        let pos = {
            let mut writer = crate::util::FailureBufWriter {
                buf: &mut err_buf,
                pos: 0,
            };
            let _ = core::fmt::write(
                &mut writer,
                format_args!("Failed to set parameter '{param_name}': {err}"),
            );
            writer.pos
        };
        if let Some(Ok(msg)) = err_buf.get(..pos).map(core::str::from_utf8) {
            self.report_error_log(msg)?;
        }
        Ok(())
    }
}

impl<'a, L, P> Server<'a, L, P>
where
    L: Link,
    P: crate::profiler::CPUProfiler,
{
    /// Exits the server, using target-specific exit mechanisms.
    ///
    /// The link is dropped first, which closes it (for example, a semihosting link ends the
    /// session in its `Drop`), and then the target exit routine runs.
    ///
    /// # Returns
    /// * `!` - This function never returns.
    pub fn exit(self) -> ! {
        let Self { context, .. } = self;
        let Context {
            link, cpu_utils, ..
        } = context;
        drop(link);
        cpu_utils.exit()
    }

    /// Creates a new `Server` instance with the given context.
    ///
    /// # Arguments
    /// * `context` - Link and hardware environment context.
    /// * `suites` - Slice of static test suite references.
    ///
    /// # Returns
    /// * `Self` - Server instance.
    pub const fn new(
        context: Context<L, P>,
        suites: &'a [&'static SuiteDescriptor],
    ) -> Self {
        Self {
            context,
            suites,
            rx: [0; MAX_MESSAGE_SIZE],
        }
    }

    /// Runs the interactive server event loop.
    ///
    /// This function polls for incoming host commands, executes requested cases,
    /// and streams telemetry and metrics back to the host.
    ///
    /// # Returns
    /// * `ServerResult<L::Error>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error `L::Error` propagated from the link if receiving or
    /// transmitting telemetry fails.
    pub fn run(&mut self) -> UnitOf<L> {
        loop {
            let received = self.context.recv_locked(&mut self.rx)?;
            let Some(len) = received else {
                continue;
            };
            let Some(Ok(cmd)) =
                self.rx.get(..len).map(postcard::from_bytes::<Command<'_>>)
            else {
                continue;
            };
            Exec {
                context: &mut self.context,
                suites: self.suites,
            }
            .handle(cmd)?;
        }
    }
}

impl<L, P> Exec<'_, L, P>
where
    L: Link,
    P: crate::profiler::CPUProfiler,
{
    fn handle(&mut self, cmd: Command<'_>) -> UnitOf<L> {
        match cmd {
            Command::ListSuites => self.stream_discovery(),
            Command::RunExecutable {
                suite, item, mode, ..
            } => self.run_executable(suite, item, mode),
            Command::Param { suite, key, value } => {
                self.param(suite, key, value)
            }
            Command::StopNow { .. } => self
                .context
                .report_error_log("Error: StopNow is not supported"),
            Command::TaskInput { .. } => self
                .context
                .report_error_log("Error: TaskInput is not supported"),
            Command::Heartbeat | Command::TryReset => Ok(()),
        }
    }

    fn run_executable(
        &mut self,
        suite_id: u16,
        item_id: u16,
        mode: ExecMode,
    ) -> UnitOf<L> {
        if mode != ExecMode::Atomic {
            return self.context.report_error_log(
                "Error: RunExecutable mode is not supported",
            );
        }
        let Some(&suite) = self.suites.get(usize::from(suite_id)) else {
            return self
                .context
                .report_error_log("Error: RunExecutable suite out of range");
        };
        let Some(exec) = suite.executables.get(usize::from(item_id)) else {
            return self
                .context
                .report_error_log("Error: RunExecutable item out of range");
        };

        self.send_state(suite_id, item_id, ItemState::Running)?;

        // Track globally in case of panic during test execution
        CURRENT_SUITE.set_active(usize::from(suite_id));
        CURRENT_TEST.set_active(usize::from(item_id));

        let metrics = self.context.profile_test(exec.test_fn);

        // Clear global trackers on success
        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();

        self.send_state(suite_id, item_id, ItemState::Done(Verdict::Pass))?;

        let time_ns = metrics.time_us.saturating_mul(1000);
        let step_ns = u32::try_from(time_ns).unwrap_or(u32::MAX);
        let _ = self.context.send_telemetry_locked(&Telemetry::RunStats {
            suite: suite_id,
            item: item_id,
            steps: 1,
            time_ns,
            step_min_ns: step_ns,
            step_max_ns: step_ns,
            cycles: metrics.cycles,
            stack_peak: metrics.stack_peak,
            jitter_max_ns: 0,
            missed_releases: 0,
        })?;
        Ok(())
    }

    fn send_state(
        &mut self,
        suite: u16,
        item: u16,
        state: ItemState,
    ) -> UnitOf<L> {
        let _ = self.context.send_telemetry_locked(&Telemetry::ItemState {
            suite,
            item,
            state,
            message: None,
        })?;
        Ok(())
    }

    fn param(
        &mut self,
        suite_id: u16,
        key: ParamKey<'_>,
        value: Option<Param>,
    ) -> UnitOf<L> {
        let Some(&suite) = self.suites.get(usize::from(suite_id)) else {
            return self
                .context
                .report_error_log("Error: Param suite out of range");
        };
        let found = match key {
            ParamKey::Name(name) => {
                suite.params.iter().position(|p| p.name() == name)
            }
            ParamKey::Index(index) => Some(usize::from(index)),
        };
        let Some((index, entry)) =
            found.and_then(|i| suite.params.get(i).map(|&entry| (i, entry)))
        else {
            return self.context.report_error_log("Error: Param is unknown");
        };
        if let Some(new) = value
            && let Err(err) = entry.set(new)
        {
            self.context.report_param_error(entry.name(), err)?;
        }
        // Answer with the stored value, also when a write was refused
        self.send_param(suite_id, index, suite.params)
    }

    fn send_param(
        &mut self,
        suite_id: u16,
        index: usize,
        params: ParamTable,
    ) -> UnitOf<L> {
        let Some(&entry) = params.get(index) else {
            return Ok(());
        };
        let _ = self.context.send_telemetry_locked(&Telemetry::ParamValue {
            suite: suite_id,
            index: u16::try_from(index).unwrap_or(u16::MAX),
            count: u16::try_from(params.len()).unwrap_or(u16::MAX),
            name: entry.name(),
            description: entry.description(),
            value: entry.get(),
            min: entry.min(),
            max: entry.max(),
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
            release_source: false,
        };
        let _ = self.context.send_telemetry_locked(&info)?;
        for (suite_id, &suite) in (0_u16..).zip(self.suites.iter()) {
            self.stream_suite(suite_id, suite)?;
        }
        let _ = self
            .context
            .send_telemetry_locked(&Telemetry::DiscoveryComplete)?;
        Ok(())
    }

    fn stream_suite(
        &mut self,
        suite_id: u16,
        suite: &'static SuiteDescriptor,
    ) -> UnitOf<L> {
        let _ = self.context.send_telemetry_locked(&Telemetry::SuiteInfo {
            suite: suite_id,
            name: suite.name,
            description: suite.description,
            executable_count: u16::try_from(suite.executables.len())
                .unwrap_or(u16::MAX),
            param_count: u16::try_from(suite.params.len()).unwrap_or(u16::MAX),
        })?;
        for (item_id, exec) in (0_u16..).zip(suite.executables.iter()) {
            let _ = self.context.send_telemetry_locked(
                &Telemetry::ExecutableInfo {
                    suite: suite_id,
                    item: item_id,
                    name: exec.name,
                    description: exec.description,
                    modes: ExecMode::Atomic.bit(),
                    input: PacketInfo {
                        type_name: "",
                        encoding: PayloadEncoding::Postcard,
                    },
                    output: PacketInfo {
                        type_name: "",
                        encoding: PayloadEncoding::Postcard,
                    },
                    link_timeout_ms: 0,
                },
            )?;
        }
        for index in 0..suite.params.len() {
            self.send_param(suite_id, index, suite.params)?;
        }
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
    use crate::link::{Link, RecvOf, SendOf};
    use crate::params::ParamEntry;
    use crate::profiler::CPUProfiler;
    use crate::wire::{
        Command, ExecMode, ItemState, ParamKey, Telemetry, Verdict,
    };
    use crate::{ExecDescriptor, SuiteDescriptor};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::vec::Vec;

    // --- Statics ---
    /// Held by the tests that read or write the shared statics.
    static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

    static SUITES: &[&SuiteDescriptor] = &[&SUITE_DESC];

    static SUITE_DESC: SuiteDescriptor = SuiteDescriptor {
        description: "mock_suite_desc",
        executables: SUITE_EXECUTABLES,
        name: "mock_suite",
        params: SUITE_PARAMS,
    };

    static SUITE_EXECUTABLES: &[ExecDescriptor] = &[ExecDescriptor {
        description: "dummy_desc",
        name: "dummy_test",
        test_fn: dummy_test_fn,
    }];

    static SUITE_PARAMS: ParamTable = &[&TEST_U8_PARAM, &TEST_BOUNDED_PARAM];

    static TEST_BOUNDED_PARAM: ParamEntry =
        ParamEntry::u16("test_bounded", "bounded", 5)
            .bounded(Param::U16(1), Param::U16(10));

    static TEST_CALLED: AtomicBool = AtomicBool::new(false);

    static TEST_U8_PARAM: ParamEntry =
        ParamEntry::u8("test_u8", "test_u8_desc", 42);

    // --- Types & Structs ---
    struct HostCPUProfiler;

    struct MockLink {
        commands: Vec<Command<'static>>,
        fail_on_poll: bool,
        payloads: Payloads,
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

    type Payloads = Vec<Vec<u8>>;
    type Telemetries<'a> = Vec<Telemetry<'a>>;
    type TestServer = Server<'static, MockLink, HostCPUProfiler>;
    type RunOutcome = (ServerResult<&'static str>, TestServer);

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
            let at = reads.get();
            reads.set(at.saturating_add(1));
            values.get(at).copied().unwrap_or(0)
        }
    }

    impl Link for MockLink {
        type Error = &'static str;

        fn recv(&mut self, buf: &mut [u8]) -> RecvOf<Self> {
            if self.fail_on_poll {
                return Err("Poll failed");
            }
            if self.commands.is_empty() {
                // Return an error to break the infinite server loop
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

    fn dummy_test_fn() {
        TEST_CALLED.store(true, Ordering::SeqCst);
    }

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn link_of(commands: Vec<Command<'static>>) -> MockLink {
        MockLink {
            commands,
            fail_on_poll: false,
            payloads: Vec::new(),
        }
    }

    fn run_bounded(commands: Vec<Command<'static>>) -> RunOutcome {
        let context = Context::new(link_of(commands), HostCPUProfiler);
        let mut server = Server::new(context, SUITES);
        let res = server.run();
        (res, server)
    }

    fn decoded(server: &TestServer) -> Telemetries<'_> {
        server
            .context
            .link
            .payloads
            .iter()
            .map(|p| postcard::from_bytes(p).unwrap())
            .collect()
    }

    fn read_param(key: ParamKey<'static>) -> Command<'static> {
        Command::Param {
            suite: 0,
            key,
            value: None,
        }
    }

    fn write_param(key: ParamKey<'static>, value: Param) -> Command<'static> {
        Command::Param {
            suite: 0,
            key,
            value: Some(value),
        }
    }

    fn run_atomic() -> Command<'static> {
        Command::RunExecutable {
            suite: 0,
            item: 0,
            mode: ExecMode::Atomic,
            max_steps: 0,
            period_ns: 0,
        }
    }

    fn param_value(t: &Telemetry<'_>) -> Option<Param> {
        match *t {
            Telemetry::ParamValue { value, .. } => Some(value),
            _ => None,
        }
    }

    #[test]
    fn fake_link_round_trip() {
        let _g = serial();
        let (res, server) = run_bounded(std::vec![Command::ListSuites]);
        assert_eq!(res, Err("Exit loop"));
        let t = decoded(&server);
        assert!(matches!(t.first(), Some(Telemetry::TargetInfo { .. })));
        assert!(matches!(
            t.get(1),
            Some(Telemetry::SuiteInfo {
                suite: 0,
                executable_count: 1,
                param_count: 2,
                ..
            })
        ));
        assert!(matches!(
            t.get(2),
            Some(Telemetry::ExecutableInfo {
                suite: 0,
                item: 0,
                modes: 1,
                ..
            })
        ));
        assert!(matches!(
            t.get(3),
            Some(Telemetry::ParamValue {
                suite: 0,
                index: 0,
                count: 2,
                name: "test_u8",
                min: Param::U8(0),
                max: Param::U8(255),
                ..
            })
        ));
        assert!(matches!(
            t.get(4),
            Some(Telemetry::ParamValue {
                index: 1,
                name: "test_bounded",
                ..
            })
        ));
        assert!(matches!(t.get(5), Some(Telemetry::DiscoveryComplete)));
        assert_eq!(t.len(), 6);
    }

    #[test]
    fn target_info_sent_at_boot() {
        let _g = serial();
        let (_, server) = run_bounded(std::vec![Command::ListSuites]);
        assert!(matches!(
            decoded(&server).first(),
            Some(Telemetry::TargetInfo {
                protocol_version: 2,
                release_source: false,
                ..
            })
        ));
    }

    #[test]
    fn atomic_run_reports_state_and_stats() {
        let _g = serial();
        TEST_CALLED.store(false, Ordering::SeqCst);
        let (res, server) = run_bounded(std::vec![run_atomic()]);
        assert_eq!(res, Err("Exit loop"));
        assert!(TEST_CALLED.load(Ordering::SeqCst));
        let t = decoded(&server);
        assert_eq!(t.len(), 3);
        assert!(matches!(
            t.first(),
            Some(Telemetry::ItemState {
                suite: 0,
                item: 0,
                state: ItemState::Running,
                ..
            })
        ));
        assert!(matches!(
            t.get(1),
            Some(Telemetry::ItemState {
                state: ItemState::Done(Verdict::Pass),
                ..
            })
        ));
        assert!(matches!(
            t.get(2),
            Some(Telemetry::RunStats {
                steps: 1,
                jitter_max_ns: 0,
                missed_releases: 0,
                ..
            })
        ));
        assert!(CURRENT_SUITE.get().is_none());
        assert!(CURRENT_TEST.get().is_none());
    }

    #[test]
    fn requests_the_server_cannot_serve_are_answered_with_a_log() {
        let _g = serial();
        let (_, server) = run_bounded(std::vec![
            Command::RunExecutable {
                suite: 0,
                item: 0,
                mode: ExecMode::Lockstep,
                max_steps: 0,
                period_ns: 0,
            },
            Command::RunExecutable {
                suite: 9,
                item: 0,
                mode: ExecMode::Atomic,
                max_steps: 0,
                period_ns: 0,
            },
            Command::RunExecutable {
                suite: 0,
                item: 9,
                mode: ExecMode::Atomic,
                max_steps: 0,
                period_ns: 0,
            },
            Command::StopNow { suite: 0, item: 0 },
            Command::TaskInput {
                suite: 0,
                item: 0,
                seq: 0,
                payload: &[1],
            },
        ]);
        let t = decoded(&server);
        assert_eq!(t.len(), 5);
        assert!(
            t.iter()
                .all(|m| matches!(m, Telemetry::Log(l) if l.level == 0))
        );
    }

    #[test]
    fn heartbeat_try_reset_and_undecodable_messages_are_silent() {
        let _g = serial();
        let (res, server) =
            run_bounded(std::vec![Command::Heartbeat, Command::TryReset]);
        assert_eq!(res, Err("Exit loop"));
        assert_eq!(server.context.link.payloads, Payloads::new());
    }

    #[test]
    fn param_table_get_set_echo() {
        let _g = serial();
        let _ = TEST_U8_PARAM.set(Param::U8(42));
        let (_, server) = run_bounded(std::vec![
            read_param(ParamKey::Name("test_u8")),
            read_param(ParamKey::Index(0)),
            write_param(ParamKey::Name("test_u8"), Param::U8(7)),
            write_param(ParamKey::Index(0), Param::U8(8)),
            write_param(ParamKey::Name("test_u8"), Param::U32(1)),
            write_param(ParamKey::Name("test_bounded"), Param::U16(99)),
            write_param(ParamKey::Name("missing"), Param::U8(1)),
            read_param(ParamKey::Index(5)),
        ]);
        let t = decoded(&server);
        assert_eq!(t.first(), t.get(1), "name and index agree");
        assert_eq!(t.first().and_then(param_value), Some(Param::U8(42)));
        assert_eq!(t.get(2).and_then(param_value), Some(Param::U8(7)));
        assert_eq!(t.get(3).and_then(param_value), Some(Param::U8(8)));
        // A type mismatch logs, then echoes the unchanged stored value.
        assert!(matches!(t.get(4), Some(Telemetry::Log(_))));
        assert_eq!(t.get(5).and_then(param_value), Some(Param::U8(8)));
        // An out-of-bounds write logs, then echoes the unchanged stored value and the bounds.
        assert!(matches!(t.get(6), Some(Telemetry::Log(_))));
        assert!(matches!(
            t.get(7),
            Some(Telemetry::ParamValue {
                index: 1,
                value: Param::U16(5),
                min: Param::U16(1),
                max: Param::U16(10),
                ..
            })
        ));
        // An unknown name and an unknown index log and change nothing.
        assert!(matches!(t.get(8), Some(Telemetry::Log(_))));
        assert!(matches!(t.get(9), Some(Telemetry::Log(_))));
        assert_eq!(t.len(), 10);
        assert_eq!(TEST_U8_PARAM.get(), Param::U8(8));
        let _ = TEST_U8_PARAM.set(Param::U8(42));
    }

    #[test]
    fn a_param_request_for_an_unknown_suite_is_logged() {
        let _g = serial();
        let (_, server) = run_bounded(std::vec![Command::Param {
            suite: 4,
            key: ParamKey::Index(0),
            value: None,
        }]);
        assert!(matches!(
            decoded(&server).as_slice(),
            [Telemetry::Log(l)] if l.level == 0
        ));
    }

    #[test]
    fn a_transport_error_ends_the_loop() {
        let _g = serial();
        let context = Context::new(
            MockLink {
                commands: Vec::new(),
                fail_on_poll: true,
                payloads: Vec::new(),
            },
            HostCPUProfiler,
        );
        let mut server = Server::new(context, SUITES);
        assert_eq!(server.run(), Err("Poll failed"));
    }

    #[test]
    fn server_exit_drops_the_link_then_exits() {
        let context = Context::new(link_of(Vec::new()), HostCPUProfiler);
        let server = Server::new(context, SUITES);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                server.exit()
            }));
        assert!(res.is_err());
    }

    #[test]
    fn locked_context_never_touches_the_link() {
        let mut context = Context::new(link_of(Vec::new()), HostCPUProfiler);
        assert!(context.comms_lock.try_lock());
        let mut buf = [0u8; 8];
        assert_eq!(context.recv_locked(&mut buf), Ok(None));
        assert_eq!(
            context.send_telemetry_locked(&Telemetry::DiscoveryComplete),
            Ok(false)
        );
        assert_eq!(context.link.payloads, Payloads::new());
        context.comms_lock.unlock();
        assert_eq!(
            context.send_telemetry_locked(&Telemetry::DiscoveryComplete),
            Ok(true)
        );
        assert_eq!(context.link.payloads.len(), 1);
    }

    #[test]
    fn a_busy_link_is_retried_until_it_accepts() {
        let mut context = Context::new(
            BusyLink {
                busy: 3,
                offered: 0,
                sent: 0,
            },
            HostCPUProfiler,
        );
        assert_eq!(
            context.send_telemetry_locked(&Telemetry::DiscoveryComplete),
            Ok(true)
        );
        assert_eq!(context.link.offered, 4);
        assert_eq!(context.link.sent, 1);
        assert_eq!(
            context.try_send_telemetry(&Telemetry::DiscoveryComplete),
            Ok(SendOutcome::Sent)
        );
    }

    #[test]
    fn try_send_reports_busy_and_dropped() {
        let mut busy = Context::new(
            BusyLink {
                busy: 1,
                offered: 0,
                sent: 0,
            },
            HostCPUProfiler,
        );
        assert_eq!(
            busy.try_send_telemetry(&Telemetry::DiscoveryComplete),
            Ok(SendOutcome::Busy)
        );
        let long = "x".repeat(600);
        let mut ctx = Context::new(link_of(Vec::new()), HostCPUProfiler);
        let outcome =
            ctx.try_send_telemetry(&Telemetry::Log(crate::wire::LogMessage {
                level: 0,
                target: "t",
                message: &long,
            }));
        assert_eq!(outcome, Ok(SendOutcome::Dropped));
        assert_eq!(ctx.link.payloads, Payloads::new());
    }

    #[test]
    fn profiling_reports_elapsed_cycles_and_microseconds() {
        let profiler = ClockProfiler::new([100, 1_100], [5_000, 9_000]);
        let context = Context::new(link_of(Vec::new()), profiler);
        let metrics = context.profile_test(|| {});
        assert_eq!(metrics.cycles, 1_000);
        assert_eq!(metrics.time_us, 4);
    }

    #[test]
    fn report_param_error_names_the_parameter() {
        let mut context = Context::new(link_of(Vec::new()), HostCPUProfiler);
        context.report_param_error("kp", "Type mismatch").unwrap();
        let t: Telemetry<'_> =
            postcard::from_bytes(context.link.payloads.first().unwrap())
                .unwrap();
        assert!(matches!(
            t,
            Telemetry::Log(l) if l.message == "Failed to set parameter 'kp': Type mismatch"
        ));
    }

    #[test]
    fn test_index_indicator() {
        let indicator = TestIndexIndicator::default();
        assert!(indicator.get().is_none());
        indicator.set_active(0);
        assert_eq!(indicator.get(), Some(0), "zero is a valid index");
        indicator.set_active(usize::MAX);
        assert!(
            indicator.get().is_none(),
            "an unrepresentable index is idle"
        );
        indicator.set_idle();
        assert!(indicator.get().is_none());
    }

    #[test]
    fn comms_lock_excludes_a_second_holder() {
        let lock = CommsLock::default();
        assert!(lock.try_lock());
        assert!(!lock.try_lock());
        lock.unlock();
        assert!(lock.try_lock());
    }
}
