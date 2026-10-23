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
//! - **Test Index Indicator**: Single-writer indicators (`TestIndexIndicator`) tracking the active suite
//!   and item so the fault path can report where a crash happened.
//! - **Unsupported commands**: A command the server cannot execute yet is answered with an error
//!   log and changes no state.

use crate::board::{Board, NoBoard};
use crate::cell::StateCell;
use crate::exec::HookOf;
use crate::link::{Link, RecvOf, SendOf, UnitOf};
use crate::run::{
    Atomic, FreeRunning, Job, Lockstep, Periodic, Runner, Triggered,
};
use crate::wire::{
    Command, ExecMode, LogMessage, MAX_MESSAGE_SIZE, PROTOCOL_VERSION, Param,
    ParamKey, Telemetry,
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
    locked: StateCell<bool>,
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
pub struct Server<'a, L, P, B = NoBoard> {
    /// The board binding.
    pub board: B,
    /// The global context containing the link, `cpu_utils` and the comms lock.
    pub context: Context<L, P>,
    /// The registered test suites.
    pub suites: &'a [&'static SuiteDescriptor],
    rx: [u8; MAX_MESSAGE_SIZE],
    run_rx: [u8; MAX_MESSAGE_SIZE],
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
    state: StateCell<isize>,
}

/// Borrows of the server that a command handler needs, disjoint from the receive buffer.
struct Exec<'e, L, P, B> {
    board: &'e mut B,
    context: ContextMut<'e, L, P>,
    rx: &'e mut [u8; MAX_MESSAGE_SIZE],
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
            locked: StateCell::new(false),
        }
    }

    /// Attempts to take the lock without waiting; returns `true` if it was taken.
    ///
    /// The fault path, the only other taker, never returns control to the code it interrupted,
    /// so a check followed by a store needs no read-modify-write.
    pub fn try_lock(&self) -> bool {
        if self.locked.get() {
            return false;
        }
        self.locked.set(true);
        true
    }

    /// Releases the lock.
    pub fn unlock(&self) {
        self.locked.set(false);
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

    /// Answers a parameter request: reads the entry, or writes it first when `value` is set.
    ///
    /// The stored value is sent back whether or not a write succeeded. Returns whether a write
    /// stored a value.
    ///
    /// # Errors
    /// Returns a transport error if sending fails.
    pub fn param_request(
        &mut self,
        (suite_id, params): (u16, ParamTable),
        key: ParamKey<'_>,
        value: Option<Param>,
    ) -> SendOf<L> {
        let found = match key {
            ParamKey::Name(name) => {
                params.iter().position(|p| p.name() == name)
            }
            ParamKey::Index(index) => Some(usize::from(index)),
        };
        let Some((index, entry)) =
            found.and_then(|i| params.get(i).map(|&entry| (i, entry)))
        else {
            self.report_error_log("Error: Param is unknown")?;
            return Ok(false);
        };
        let stored = match value.map(|new| entry.set(new)) {
            Some(Err(err)) => {
                self.report_param_error(entry.name(), err)?;
                false
            }
            Some(Ok(())) => true,
            None => false,
        };
        self.send_param(suite_id, index, params)?;
        Ok(stored)
    }

    /// Sends the `ParamValue` of entry `index`.
    ///
    /// # Errors
    /// Returns a transport error if sending fails.
    pub fn send_param(
        &mut self,
        suite_id: u16,
        index: usize,
        params: ParamTable,
    ) -> UnitOf<L> {
        let Some(&entry) = params.get(index) else {
            return Ok(());
        };
        let _ = self.send_telemetry_locked(&Telemetry::ParamValue {
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

impl<'a, L, P> Server<'a, L, P, NoBoard>
where
    L: Link,
    P: crate::profiler::CPUProfiler,
{
    /// Creates a new `Server` without a board binding.
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
        Self::with_board(context, suites, NoBoard)
    }
}

impl<'a, L, P, B> Server<'a, L, P, B>
where
    L: Link,
    P: crate::profiler::CPUProfiler,
    B: Board,
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

    /// Creates a new `Server` with a board binding.
    ///
    /// # Arguments
    /// * `context` - Link and hardware environment context.
    /// * `suites` - Slice of static test suite references.
    /// * `board` - Board binding: watchdog feed, wait and release source.
    ///
    /// # Returns
    /// * `Self` - Server instance.
    pub const fn with_board(
        context: Context<L, P>,
        suites: &'a [&'static SuiteDescriptor],
        board: B,
    ) -> Self {
        Self {
            board,
            context,
            suites,
            rx: [0; MAX_MESSAGE_SIZE],
            run_rx: [0; MAX_MESSAGE_SIZE],
        }
    }

    /// Runs the interactive server event loop.
    ///
    /// This function polls for incoming host commands, executes requested executables,
    /// and streams telemetry and metrics back to the host.
    ///
    /// # Returns
    /// * `UnitOf<L>` - Success or transport error status.
    ///
    /// # Errors
    /// Returns a transport error `L::Error` propagated from the link if receiving or
    /// transmitting telemetry fails.
    pub fn run(&mut self) -> UnitOf<L> {
        loop {
            self.board.feed();
            let received = self.context.recv_locked(&mut self.rx)?;
            let Some(len) = received else {
                self.board.wait();
                continue;
            };
            let Some(Ok(cmd)) =
                self.rx.get(..len).map(postcard::from_bytes::<Command<'_>>)
            else {
                continue;
            };
            Exec {
                board: &mut self.board,
                context: &mut self.context,
                rx: &mut self.run_rx,
                suites: self.suites,
            }
            .handle(cmd)?;
        }
    }
}

impl<L, P, B> Exec<'_, L, P, B>
where
    L: Link,
    P: crate::profiler::CPUProfiler,
    B: Board,
{
    fn handle(&mut self, cmd: Command<'_>) -> UnitOf<L> {
        match cmd {
            Command::ListSuites => self.stream_discovery(),
            Command::RunExecutable {
                suite,
                item,
                mode,
                max_steps,
                period_ns,
            } => {
                self.run_executable((suite, item), mode, (max_steps, period_ns))
            }
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
        (suite_id, item_id): (u16, u16),
        mode: ExecMode,
        (max_steps, period_ns): (u64, u32),
    ) -> UnitOf<L> {
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
        if let Err(message) = self.admit(exec.modes, mode, period_ns) {
            return self.context.report_error_log(message);
        }
        let job = Job {
            exec,
            item: item_id,
            max_steps,
            params: suite.params,
            period_ns,
            suite: suite_id,
        };
        let (board, context, rx) =
            (&mut *self.board, &mut *self.context, &mut *self.rx);
        match mode {
            ExecMode::Atomic => {
                Runner::<_, _, _, Atomic>::new(board, context, rx, job).run()
            }
            ExecMode::Lockstep => {
                Runner::<_, _, _, Lockstep>::new(board, context, rx, job).run()
            }
            ExecMode::FreeRunning => {
                Runner::<_, _, _, FreeRunning>::new(board, context, rx, job)
                    .run()
            }
            ExecMode::Periodic => {
                Runner::<_, _, _, Periodic>::new(board, context, rx, job).run()
            }
            ExecMode::Triggered => {
                Runner::<_, _, _, Triggered>::new(board, context, rx, job).run()
            }
        }
    }

    /// Checks that `mode` is allowed and that the board can pace it.
    fn admit(&self, allowed: u8, mode: ExecMode, period_ns: u32) -> HookOf {
        if allowed & mode.bit() == 0 {
            return Err("Error: RunExecutable mode is not allowed");
        }
        let source = self.board.has_release_source();
        match mode {
            ExecMode::Triggered if !source => {
                Err("Error: Triggered needs a board release source")
            }
            ExecMode::Periodic
                if !source
                    && (period_ns == 0
                        || !self.context.cpu_utils.has_clock()) =>
            {
                Err(
                    "Error: Periodic needs a period and a release source or clock",
                )
            }
            _ => Ok(()),
        }
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
        self.context
            .param_request((suite_id, suite.params), key, value)
            .map(|_| ())
    }

    fn stream_discovery(&mut self) -> UnitOf<L> {
        let cpu = &self.context.cpu_utils;
        let info = Telemetry::TargetInfo {
            protocol_version: PROTOCOL_VERSION,
            board_id: cpu.board_id(),
            core_clock_hz: cpu.core_clock_hz(),
            fpu_flags: cpu.fpu_flags(),
            release_source: self.board.has_release_source(),
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
                    modes: exec.modes,
                    input: exec.input,
                    output: exec.output,
                    link_timeout_ms: exec.link_timeout_ms,
                },
            )?;
        }
        for index in 0..suite.params.len() {
            self.context.send_param(suite_id, index, suite.params)?;
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
        let current_state = self.state.get();

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
            state: StateCell::new(Self::IDLE_STATE),
        }
    }

    /// Sets the indicator to a specific active test index.
    ///
    /// # Arguments
    /// * `index` - The active execution index.
    pub fn set_active(&self, index: usize) {
        // Explicitly pattern match on the Result
        let safe_index = isize::try_from(index).unwrap_or(Self::IDLE_STATE);

        self.state.set(safe_index);
    }

    /// Sets the indicator back to Idle.
    pub fn set_idle(&self) {
        self.state.set(Self::IDLE_STATE);
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::exec::{TaskIo, TaskOutcome};
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

    static SUITE_EXECUTABLES: &[ExecDescriptor] = &[ExecDescriptor::case(
        "dummy_test",
        "dummy_desc",
        dummy_test_fn,
    )];

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
        idle_polls: usize,
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
        fn has_clock(&self) -> bool {
            false
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
                // Return an error after a number of idle polls to end the server loop
                self.idle_polls =
                    self.idle_polls.checked_sub(1).ok_or("Exit loop")?;
                return Ok(None);
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

    fn dummy_test_fn(_: &mut TaskIo<'_>) -> TaskOutcome {
        TEST_CALLED.store(true, Ordering::SeqCst);
        TaskOutcome::Continue
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
            idle_polls: 20,
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

    /// Runs `f` on an `Exec` over `profiler` with the default board.
    fn with_exec<P: CPUProfiler, R>(
        profiler: P,
        f: impl FnOnce(&Exec<'_, MockLink, P, NoBoard>) -> R,
    ) -> R {
        let mut context = Context::new(link_of(Vec::new()), profiler);
        let mut board = NoBoard;
        let mut rx = [0u8; MAX_MESSAGE_SIZE];
        f(&Exec {
            board: &mut board,
            context: &mut context,
            rx: &mut rx,
            suites: SUITES,
        })
    }

    #[test]
    fn discovery_reports_cases_tasks_and_params_with_metadata() {
        let _g = serial();
        let (_, server) = run_bounded(std::vec![Command::ListSuites]);
        let t = decoded(&server);
        assert!(matches!(
            t.as_slice(),
            [
                Telemetry::TargetInfo {
                    release_source: false,
                    ..
                },
                Telemetry::SuiteInfo {
                    name: "mock_suite",
                    description: "mock_suite_desc",
                    executable_count: 1,
                    param_count: 2,
                    ..
                },
                Telemetry::ExecutableInfo {
                    name: "dummy_test",
                    description: "dummy_desc",
                    modes: 1,
                    ..
                },
                Telemetry::ParamValue {
                    name: "test_u8",
                    index: 0,
                    count: 2,
                    ..
                },
                Telemetry::ParamValue {
                    name: "test_bounded",
                    min: Param::U16(1),
                    max: Param::U16(10),
                    ..
                },
                Telemetry::DiscoveryComplete,
            ]
        ));
    }

    #[test]
    fn periodic_without_clock_or_source_is_rejected() {
        let periodic = ExecMode::Periodic.bit();
        with_exec(HostCPUProfiler, |e| {
            assert!(e.admit(periodic, ExecMode::Periodic, 1000).is_err());
        });
        with_exec(ClockProfiler::new([0; 2], [0; 2]), |e| {
            assert!(e.admit(periodic, ExecMode::Periodic, 1000).is_ok());
            assert!(e.admit(periodic, ExecMode::Periodic, 0).is_err());
        });
    }

    #[test]
    fn a_release_source_admits_triggered_and_unpaced_periodic() {
        struct Sourced;
        impl Board for Sourced {
            fn has_release_source(&self) -> bool {
                true
            }
        }
        let mut context = Context::new(link_of(Vec::new()), HostCPUProfiler);
        let mut board = Sourced;
        let mut rx = [0u8; MAX_MESSAGE_SIZE];
        let e = Exec {
            board: &mut board,
            context: &mut context,
            rx: &mut rx,
            suites: SUITES,
        };
        let all = 0b1_1111;
        assert!(e.admit(all, ExecMode::Triggered, 0).is_ok());
        assert!(e.admit(all, ExecMode::Periodic, 0).is_ok());
    }

    #[test]
    fn default_binding_runs_all_but_triggered() {
        with_exec(ClockProfiler::new([0; 2], [0; 2]), |e| {
            let all = 0b1_1111;
            for mode in [
                ExecMode::Atomic,
                ExecMode::Lockstep,
                ExecMode::FreeRunning,
                ExecMode::Periodic,
            ] {
                assert!(e.admit(all, mode, 1000).is_ok());
            }
            assert!(e.admit(all, ExecMode::Triggered, 0).is_err());
            assert!(
                e.admit(ExecMode::Atomic.bit(), ExecMode::Lockstep, 0)
                    .is_err()
            );
        });
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
                idle_polls: 0,
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
