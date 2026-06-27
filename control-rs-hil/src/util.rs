//! Utility module for target-agnostic server operations.
//! Includes functions for test suite discovery and panic/exception handlers.

#[cfg(target_os = "none")]
use {
    crate::SuiteDescriptor, core::fmt::Write, core::str,
    core::sync::atomic::Ordering,
};

/// Reconstructs the slice of static test suites from the raw boundary pointers provided by the entrypoint.
#[cfg(target_os = "none")]
pub unsafe fn get_suites(
    start: *const &'static SuiteDescriptor,
    end: *const &'static SuiteDescriptor,
) -> &'static [&'static SuiteDescriptor] {
    let len = (end as usize - start as usize)
        / ::core::mem::size_of::<&SuiteDescriptor>();
    unsafe { ::core::slice::from_raw_parts(start, len) }
}

/// Buffer writer that implements `core::fmt::Write` to format failure and panic messages into a static buffer.
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

        // Wait for OkToReset command from host
        loop {
            let command = context.comms.poll_command().ok().flatten();

            if let Some(crate::comms::Command::OkToReset) = command {
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
#[cfg(target_os = "none")]
pub unsafe fn handle_exception<
    C: crate::comms::HostComms,
    P: crate::profiler::CPUProfiler,
>(
    context: &mut crate::server::Context<C, P>,
    msg: &str,
    comms_ok: bool,
) -> ! {
    unsafe {
        handle_failure(context, msg, "exception_handler", 0, comms_ok);
    }
}
