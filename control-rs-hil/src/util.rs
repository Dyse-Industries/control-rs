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
pub unsafe fn handle_failure(
    msg: &str,
    file: &str,
    line: u32,
    disable_interrupts: impl FnOnce(),
    reset: impl FnOnce(),
) -> ! {
    disable_interrupts();

    let suite = crate::server::CURRENT_SUITE.load(Ordering::SeqCst);
    let test = crate::server::CURRENT_TEST.load(Ordering::SeqCst);

    unsafe {
        if let (Some(sender), ptr) = (
            crate::server::PANIC_TELEMETRY_SENDER,
            crate::server::ACTIVE_COMMS_PTR,
        ) {
            if !ptr.is_null() {
                if suite >= 0 && test >= 0 {
                    sender(
                        ptr,
                        &crate::comms::Telemetry::TestStateChange {
                            suite_id: suite as u16,
                            test_id: test as u16,
                            state: crate::comms::TestState::Failed,
                        },
                    );
                }
                sender(
                    ptr,
                    &crate::comms::Telemetry::TargetPanic {
                        message: msg,
                        file,
                        line,
                    },
                );
            }
        }
    }

    // Wait for OkToReset command from host
    loop {
        let command = unsafe {
            if let (Some(poller), ptr) = (
                crate::server::PANIC_CMD_POLLER,
                crate::server::ACTIVE_COMMS_PTR,
            ) {
                if !ptr.is_null() { poller(ptr) } else { None }
            } else {
                None
            }
        };

        if let Some(crate::comms::Command::OkToReset) = command {
            break;
        }

        unsafe {
            if let (Some(flusher), ptr) = (
                crate::server::PANIC_COMMS_FLUSHER,
                crate::server::ACTIVE_COMMS_PTR,
            ) {
                if !ptr.is_null() {
                    flusher(ptr);
                } else {
                    ::core::hint::spin_loop();
                }
            } else {
                ::core::hint::spin_loop();
            }
        }

        // Small delay to prevent pegging the CPU too hard
        for _ in 0..1000 {
            ::core::hint::spin_loop();
        }
    }

    reset();

    // Fallback loop to satisfy -> ! return type
    loop {
        ::core::hint::spin_loop();
    }
}

/// Target-agnostic logic for handling exceptions.
/// Formats exception information and delegates to `handle_failure`.
#[cfg(target_os = "none")]
pub unsafe fn handle_exception(
    msg: &str,
    disable_interrupts: impl FnOnce(),
    reset: impl FnOnce(),
) -> ! {
    unsafe {
        handle_failure(msg, "exception_handler", 0, disable_interrupts, reset);
    }
}
