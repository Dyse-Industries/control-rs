//! Execution context shared by the server and its runners.
//!
//! # Description
//!
//! `Context` owns the host link, the CPU profiler and the communications lock. The server
//! constructs one context and lends it to the runner of each run, so the dispatcher and the
//! runners share a single writer for the link.

use crate::comms::{Command, CommsLock, HostComms, Telemetry};
use crate::server::ServerResult;

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
/// Performance metrics measured during a test execution.
pub struct TestMetrics {
    /// Number of elapsed CPU cycles.
    pub cycles: u64,
    /// Peak stack usage in bytes.
    pub stack_peak: u32,
    /// Elapsed wall-clock time in microseconds.
    pub time_us: u64,
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
