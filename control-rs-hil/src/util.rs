//! Utility module for target-agnostic server operations.
//! Includes functions for test suite discovery and panic/exception handlers.

#[cfg(any(target_os = "none", test))]
use {crate::SuiteDescriptor, core::fmt::Write, core::str};

/// Buffer writer that implements `core::fmt::Write` to format failure and panic messages into a static buffer.
///
/// To prevent unsafe operations like buffer overflows during formatting of panic telemetry, this helper
/// performs explicit bounds checks on every write operation, returning a formatting error instead of
/// writing past the end of the buffer.
#[cfg(any(target_os = "none", test))]
pub struct FailureBufWriter<'a> {
    /// Reference to the mutable backing byte slice.
    pub buf: &'a mut [u8],
    /// Current write position.
    pub pos: usize,
}

#[cfg(any(target_os = "none", test))]
impl Write for FailureBufWriter<'_> {
    #[allow(clippy::arithmetic_side_effects)]
    fn write_str(&mut self, s: &str) -> ::core::fmt::Result {
        let bytes = s.as_bytes();
        let len = bytes.len();
        // Safe check to avoid writing past the end of `buf`.
        if self.pos + len > self.buf.len() {
            return Err(::core::fmt::Error);
        }
        let dest = self
            .buf
            .get_mut(self.pos..self.pos + len)
            .ok_or(::core::fmt::Error)?;
        dest.copy_from_slice(bytes);
        self.pos += len;
        Ok(())
    }
}

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
#[cfg(any(target_os = "none", test))]
#[must_use]
#[allow(clippy::arithmetic_side_effects)]
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
#[cfg(any(target_os = "none", test))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
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
        let suite = crate::server::CURRENT_SUITE.get();
        let test = crate::server::CURRENT_TEST.get();

        if let (Some(suite_id), Some(test_id)) = (
            suite.and_then(|s| u16::try_from(s).ok()),
            test.and_then(|t| u16::try_from(t).ok()),
        ) {
            let _ = context.comms.send_telemetry(
                &crate::comms::Telemetry::TestStateChange {
                    suite_id,
                    test_id,
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

            if matches!(command, Some(crate::comms::Command::TryReset)) {
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
/// context (e.g. `HardFault`, `PageFault`, etc.) where execution registers and hardware state are unstable.
#[cfg(any(target_os = "none", test))]
#[allow(clippy::type_complexity)]
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

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::comms::{Command, HostComms, Telemetry, TestState};
    use crate::profiler::CPUProfiler;

    type PayloadsList = std::vec::Vec<std::vec::Vec<u8>>;

    pub struct HostCPUProfiler;

    struct MockComms {
        commands: std::vec::Vec<Command>,
        payloads: PayloadsList,
    }

    impl CPUProfiler for HostCPUProfiler {
        fn exit(&self) -> ! {
            panic!("exit");
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
            panic!("reset");
        }
    }

    impl HostComms for MockComms {
        type Error = &'static str;
        fn flush(&mut self) -> Result<(), Self::Error> {
            Ok(())
        }
        fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
            if self.commands.is_empty() {
                Ok(None)
            } else {
                Ok(Some(self.commands.remove(0)))
            }
        }
        fn send_telemetry(
            &mut self,
            telemetry: &Telemetry<'_>,
        ) -> Result<(), Self::Error> {
            let payload = postcard::to_allocvec(telemetry)
                .map_err(|_| "Failed to serialize")?;
            self.payloads.push(payload);
            Ok(())
        }
    }

    #[test]
    fn test_get_suites_fn() {
        static S1: SuiteDescriptor = SuiteDescriptor {
            name: "s1",
            description: "d1",
            executables: &[],
            settings: &[],
        };
        static S2: SuiteDescriptor = SuiteDescriptor {
            name: "s2",
            description: "d2",
            executables: &[],
            settings: &[],
        };
        static SUITES_ARR: &[&SuiteDescriptor] = &[&S1, &S2];
        let start = SUITES_ARR.as_ptr();
        let end = unsafe { start.add(2) };
        let suites = unsafe { get_suites(start, end) };
        assert_eq!(suites.len(), 2);
        assert_eq!(suites.first().unwrap().name, "s1");
        assert_eq!(suites.get(1).unwrap().name, "s2");
    }

    #[test]
    fn test_failure_buf_writer_fn() {
        let mut buf = [0u8; 10];
        let mut writer = FailureBufWriter {
            buf: &mut buf,
            pos: 0,
        };
        assert!(write!(writer, "hello").is_ok());
        assert_eq!(writer.pos, 5);
        assert_eq!(writer.buf.get(..5).unwrap(), b"hello");

        assert!(write!(writer, "world!").is_err());
    }

    #[test]
    fn test_handle_failure_comms_disabled() {
        let comms = MockComms {
            commands: std::vec![],
            payloads: std::vec![],
        };
        let mut context = crate::server::Context::new(comms, HostCPUProfiler);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                handle_failure(&mut context, "panic msg", "file.rs", 42, false);
            }));
        assert!(res.is_err());
    }

    #[test]
    fn test_handle_failure_comms_idle() {
        use crate::server::{CURRENT_SUITE, CURRENT_TEST};

        CURRENT_SUITE.set_idle();
        CURRENT_TEST.set_idle();
        let comms = MockComms {
            commands: std::vec![Command::TryReset],
            payloads: std::vec![],
        };
        let mut context = crate::server::Context::new(comms, HostCPUProfiler);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                handle_failure(&mut context, "panic msg", "file.rs", 42, true);
            }));
        assert!(res.is_err());

        let payloads = &context.comms.payloads;
        assert_eq!(payloads.len(), 1);
        let t: Telemetry<'_> =
            postcard::from_bytes(payloads.first().unwrap()).unwrap();
        assert!(matches!(
            t,
            Telemetry::TargetPanic {
                message: "panic msg",
                file: "file.rs",
                line: 42
            }
        ));
    }

    #[test]
    fn test_handle_failure_comms_active() {
        use crate::server::{CURRENT_SUITE, CURRENT_TEST};

        CURRENT_SUITE.set_active(1);
        CURRENT_TEST.set_active(2);
        let comms = MockComms {
            commands: std::vec![Command::TryReset],
            payloads: std::vec![],
        };
        let mut context = crate::server::Context::new(comms, HostCPUProfiler);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                handle_failure(&mut context, "panic msg", "file.rs", 42, true);
            }));
        assert!(res.is_err());

        let payloads = &context.comms.payloads;
        assert_eq!(payloads.len(), 2);
        let t0: Telemetry<'_> =
            postcard::from_bytes(payloads.first().unwrap()).unwrap();
        assert!(matches!(
            t0,
            Telemetry::TestStateChange {
                suite_id: 1,
                test_id: 2,
                state: TestState::Failed
            }
        ));
        let t1: Telemetry<'_> =
            postcard::from_bytes(payloads.get(1).unwrap()).unwrap();
        assert!(matches!(
            t1,
            Telemetry::TargetPanic {
                message: "panic msg",
                file: "file.rs",
                line: 42
            }
        ));
    }

    #[test]
    fn test_handle_exception_fn() {
        let comms = MockComms {
            commands: std::vec![],
            payloads: std::vec![],
        };
        let mut context = crate::server::Context::new(comms, HostCPUProfiler);
        let res =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe {
                handle_exception(&mut context, "fault msg", false);
            }));
        assert!(res.is_err());
    }
}
