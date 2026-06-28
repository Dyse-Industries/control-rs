//! Utility module for target-agnostic server operations.
//! Includes functions for test suite discovery and panic/exception handlers.

#[cfg(target_os = "none")]
use {
    crate::SuiteDescriptor, core::fmt::Write, core::str,
    core::sync::atomic::Ordering,
};

/// Reconstructs the slice of static test suites from the raw boundary pointers provided by the entrypoint.
///
/// This safely aggregates all registered suites compiled into the `.hil_test_suites` custom ELF/binary section.
///
/// # Safety
///
/// The caller must ensure that:
/// 1. Both `start` and `end` point to valid, aligned references of `SuiteDescriptor` located within the same
///    contiguous read-only memory allocation.
/// 2. `start` is less than or equal to `end`.
/// 3. The memory range `[start, end)` is populated with valid, initialized static references to `SuiteDescriptor`s,
///    and no other writes or modifications occur within this region.
#[cfg(target_os = "none")]
pub unsafe fn get_suites(
    start: *const &'static SuiteDescriptor,
    end: *const &'static SuiteDescriptor,
) -> &'static [&'static SuiteDescriptor] {
    let len = (end as usize - start as usize)
        / ::core::mem::size_of::<&SuiteDescriptor>();
    // SAFETY: The safety invariants of the function guarantee that `start` and `end` enclose a valid,
    // contiguous, initialized array of references to `SuiteDescriptor` instances in static memory.
    // Length is computed based on size of pointer offsets, which is safe to convert to a slice.
    unsafe { ::core::slice::from_raw_parts(start, len) }
}

/// Buffer writer that implements `core::fmt::Write` to format failure and panic messages into a static buffer.
///
/// To prevent unsafe operations like buffer overflows during formatting of panic telemetry, this helper
/// performs explicit bounds checks on every write operation, returning a formatting error instead of
/// writing past the end of the buffer.
#[cfg(target_os = "none")]
pub struct FailureBufWriter<'a> {
    /// Reference to the mutable backing byte slice.
    pub buf: &'a mut [u8],
    /// Current write position.
    pub pos: usize,
}

#[cfg(target_os = "none")]
impl<'a> Write for FailureBufWriter<'a> {
    fn write_str(&mut self, s: &str) -> ::core::fmt::Result {
        let bytes = s.as_bytes();
        let len = bytes.len();
        // Safe check to avoid writing past the end of `buf`.
        if self.pos + len > self.buf.len() {
            return Err(::core::fmt::Error);
        }
        self.buf[self.pos..self.pos + len].copy_from_slice(bytes);
        self.pos += len;
        Ok(())
    }
}

/// Target-agnostic logic for handling server failure or panics.
/// Disables interrupts, broadcasts target panic and failure telemetry, polls for host reset permission,
/// and executes target reset.
///
/// # Safety
///
/// This function is unsafe because:
/// 1. It operates during a failure/panic state where the target hardware/software state might be corrupted.
/// 2. It bypasses regular context locking/concurrency controls, permanently disabling processor interrupts
///    and directly driving raw peripheral I/O to broadcast panic reports.
/// 3. It triggers a hardware CPU reset at completion, causing sudden stack and state termination.
///
/// The caller must ensure that `context` is a valid, reference-stable reference to the HIL server context.
#[cfg(target_os = "none")]
pub unsafe fn handle_failure<
    C: crate::comms::HostComms,
    P: crate::profiler::CPUProfiler,
>(
    context: &mut crate::server::Context<C, P>,
    msg: &str,
    file: &str,
    line: u32,
    comms_ok: bool,
) -> ! {
    context.cpu_utils.disable_interrupts_permanently();

    if comms_ok {
        let suite = crate::server::CURRENT_SUITE.load(Ordering::SeqCst);
        let test = crate::server::CURRENT_TEST.load(Ordering::SeqCst);

        if suite >= 0 && test >= 0 {
            let _ = context.comms.send_telemetry(
                &crate::comms::Telemetry::TestStateChange {
                    suite_id: suite as u16,
                    test_id: test as u16,
                    state: crate::comms::TestState::Failed,
                },
            );
        }

        let _ = context.comms.send_telemetry(
            &crate::comms::Telemetry::TargetPanic {
                message: msg,
                file,
                line,
            },
        );
        let _ = context.comms.flush();

        // Wait for TryReset command from host
        loop {
            let command = context.comms.poll_command().ok().flatten();

            if let Some(crate::comms::Command::TryReset) = command {
                break;
            }

            let _ = context.comms.flush();

            // Small delay to prevent pegging the CPU too hard
            for _ in 0..1000 {
                ::core::hint::spin_loop();
            }
        }

        context.comms.close_on_failure();
    }

    context.cpu_utils.reset();
}

/// Target-agnostic logic for handling exceptions.
/// Formats exception information and delegates to `handle_failure`.
///
/// # Safety
///
/// This inherits all safety requirements of `handle_failure`. It operates in an exception handler
/// context (e.g. HardFault, PageFault, etc.) where execution registers and hardware state are unstable.
#[cfg(target_os = "none")]
pub unsafe fn handle_exception<
    C: crate::comms::HostComms,
    P: crate::profiler::CPUProfiler,
>(
    context: &mut crate::server::Context<C, P>,
    msg: &str,
    comms_ok: bool,
) -> ! {
    // SAFETY: We propagate the safety context to `handle_failure` using the current exception information.
    unsafe {
        handle_failure(context, msg, "exception_handler", 0, comms_ok);
    }
}
