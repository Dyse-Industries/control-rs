//! Execution context shared by the server and its runners.
//!
//! # Description
//!
//! `Context` owns the message [`Link`], the CPU profiler and the communications lock. It encodes
//! telemetry and decodes commands (Postcard), so the link carries whole messages and the server
//! holds no framing code. The server constructs one context and lends it to the runner of each
//! run, so the dispatcher and the runners share a single writer for the link.

use crate::comms::{Command, CommsLock, MAX_PAYLOAD_SIZE, Telemetry};
use crate::link::{Link, SendOf, UnitOf};

/// Context object that encapsulates the link, CPU profiling utilities and the communication lock.
///
/// This type coordinates accesses to low-level target hardware. To ensure thread safety and avoid race
/// conditions (for example, between the main event loop and interrupt/exception/panic handlers that
/// also need to send telemetry), the `Context` holds a `CommsLock`. The `_locked` methods take the lock
/// and return `Ok(false)` without touching the link when it is held, so they never block on it.
///
/// # Safety
/// Access to the link is protected by the internal `CommsLock`. This struct does not use `unsafe` code
/// directly.
///
/// # Panics
/// Context operations do not panic.
///
/// # Example
/// ```
/// use control_rs_ets::server::Context;
/// # use control_rs_ets::link::{Link, RecvOf, SendOf};
/// # use control_rs_ets::comms::Telemetry;
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
    rx: [u8; MAX_PAYLOAD_SIZE],
    tx: [u8; MAX_PAYLOAD_SIZE],
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

/// Result of a poll on link `L`: a decoded command, nothing ready, or a transport error.
pub type PollOf<L> = Result<Option<Command>, <L as Link>::Error>;

/// Performance metrics measured during a test execution.
pub struct TestMetrics {
    /// Number of elapsed CPU cycles.
    pub cycles: u64,
    /// Peak stack usage in bytes.
    pub stack_peak: u32,
    /// Elapsed wall-clock time in microseconds.
    pub time_us: u64,
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
            rx: [0; MAX_PAYLOAD_SIZE],
            tx: [0; MAX_PAYLOAD_SIZE],
        }
    }

    /// Receives and decodes one command without taking the lock.
    ///
    /// A message that does not decode as a [`Command`] is discarded. The caller must hold the
    /// communications lock; the fault path uses this after it took the lock itself.
    ///
    /// # Errors
    /// Returns a transport error if the link fails.
    pub fn poll_command(&mut self) -> PollOf<L> {
        let Some(len) = self.link.recv(&mut self.rx)? else {
            return Ok(None);
        };
        Ok(self
            .rx
            .get(..len)
            .and_then(|msg| postcard::from_bytes(msg).ok()))
    }

    /// Polls a command safely by acquiring the comms lock first.
    ///
    /// If the lock is already held, this returns `Ok(None)` immediately to prevent nested/concurrent polls.
    ///
    /// # Errors
    /// Returns a transport error if polling fails.
    pub fn poll_command_locked(&mut self) -> PollOf<L> {
        if self.comms_lock.try_lock() {
            let res = self.poll_command();
            self.comms_lock.unlock();
            res
        } else {
            Ok(None)
        }
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
    /// * `SendOf<L>` - `true` if the link accepted the message; `false` if the lock was
    ///   busy or the message did not encode within the maximum message size.
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
    pub fn report_error_log(
        &mut self,
        suite_id: u16,
        test_id: u16,
        msg: &'static str,
    ) -> UnitOf<L> {
        let timestamp_us = self.cpu_utils.get_nanos() / 1000;
        let _ = self.send_telemetry_locked(&Telemetry::Log(
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
    ) -> UnitOf<L> {
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
