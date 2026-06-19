//! On-target test runner server loop.

use core::sync::atomic::{AtomicI16, Ordering};

use crate::SuiteDescriptor;
use crate::comms::{Command, HostComms, Telemetry, TestState};
use crate::settings::SettingValue;
use crate::time::ClientClock;

// --- Static variables (UPPER_SNAKE_CASE) ---

/// Raw pointer to the active communication device.
pub static mut ACTIVE_COMMS_PTR: *mut core::ffi::c_void = core::ptr::null_mut();

/// Global tracker for the currently executing suite ID.
/// Used by the panic handler to report test failures.
pub static CURRENT_SUITE: AtomicI16 = AtomicI16::new(-1);

/// Global tracker for the currently executing test ID.
/// Used by the panic handler to report test failures.
pub static CURRENT_TEST: AtomicI16 = AtomicI16::new(-1);

/// Static function pointer used to command poller during a panic.
pub static mut PANIC_CMD_POLLER: Option<PanicCmdPoller> = None;

/// Static function pointer used to flush communications during a panic.
pub static mut PANIC_COMMS_FLUSHER: Option<PanicCommsFlusher> = None;

/// Static function pointer used to transmit telemetry during a panic.
pub static mut PANIC_TELEMETRY_SENDER: Option<PanicTelemetrySender> = None;

// --- Type aliases and Structs (PascalCase) ---

/// Context object that encapsulates communication and timekeeper peripherals.
pub struct Context<C, T, E = crate::executor::DummyExecutor> {
    /// Host communication channel.
    pub comms: C,
    /// Test execution mechanism.
    pub executor: E,
    /// Hardware timekeeper clock.
    pub timer: T,
}

/// Function signature for command poller during a panic.
pub type PanicCmdPoller = unsafe fn(*mut core::ffi::c_void) -> Option<Command>;

/// Function signature for communication flusher during a panic.
pub type PanicCommsFlusher = unsafe fn(*mut core::ffi::c_void);

/// Function signature for telemetry sender during a panic.
pub type PanicTelemetrySender =
    unsafe fn(*mut core::ffi::c_void, &Telemetry<'_>);

/// Interactive test runner server.
pub struct Server<'a, C, T, E = crate::executor::DummyExecutor> {
    clock: T,
    comms: C,
    executor: E,
    suites: &'a [&'static SuiteDescriptor],
}

/// Result of server operations.
pub type ServerResult<E> = Result<(), E>;

impl<'a, C, T> Server<'a, C, T, crate::executor::DummyExecutor>
where
    C: HostComms,
    T: ClientClock,
{
    /// Creates a new `Server` instance.
    pub const fn new(
        comms: C,
        clock: T,
        suites: &'a [&'static SuiteDescriptor],
    ) -> Self {
        Self {
            clock,
            comms,
            executor: crate::executor::DummyExecutor,
            suites,
        }
    }
}

#[allow(clippy::type_complexity)]
impl<'a, C, T, E> Server<'a, C, T, E>
where
    C: HostComms,
    T: ClientClock,
    E: crate::executor::TestExecutor,
{
    /// Creates a new `Server` instance with a target-specific test executor.
    pub const fn new_with_executor(
        comms: C,
        clock: T,
        executor: E,
        suites: &'a [&'static SuiteDescriptor],
    ) -> Self {
        Self {
            clock,
            comms,
            executor,
            suites,
        }
    }

    /// Runs the interactive server event loop.
    ///
    /// This function polls for incoming host commands, executes requested tests,
    /// and streams telemetry and metrics back to the host.
    ///
    /// # Errors
    ///
    /// Returns a transport error `C::Error` propagated from the underlying communication interface if:
    /// * `poll_command()` fails: An error occurs when reading or de-framing incoming bytes from the
    ///   host, such as physical transport issues or serial port failures.
    /// * `stream_discovery()` fails: Transmission of suite, test, or setting metadata fails while
    ///   writing telemetry packets or flushing them to the transport interface during discovery.
    /// * `run_test()` fails: An error occurs while communicating test state transitions (e.g. `Running` or
    ///   `Passed`), sending performance metric reports, or flushing the transport buffer.
    /// * `set_setting()` fails: Broadcasting the confirmation of the updated setting telemetry fails,
    ///   or flushing the communication buffer fails.
    /// * `flush()` fails: Flushing the pending buffered telemetry data at the end of the command loop
    ///   iteration fails.
    pub fn run(&mut self) -> ServerResult<C::Error> {
        unsafe {
            ACTIVE_COMMS_PTR =
                core::ptr::addr_of_mut!(self.comms).cast::<core::ffi::c_void>();
            PANIC_TELEMETRY_SENDER = Some(send_telemetry_via_ptr::<C>);
            PANIC_COMMS_FLUSHER = Some(flush_comms_via_ptr::<C>);
            PANIC_CMD_POLLER = Some(poll_command_via_ptr::<C>);
        }

        let res = (|| {
            loop {
                if let Some(cmd) = self.comms.poll_command()? {
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
                        Command::OkToReset => {}
                    }
                }
                self.comms.flush()?;
            }
        })();

        unsafe {
            ACTIVE_COMMS_PTR = core::ptr::null_mut();
            PANIC_TELEMETRY_SENDER = None;
            PANIC_COMMS_FLUSHER = None;
            PANIC_CMD_POLLER = None;
        }

        res
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
            return Ok(());
        };
        let Some(exec) = suite.executables.get(test_idx) else {
            return Ok(());
        };

        // Update state to Running
        self.comms.send_telemetry(&Telemetry::TestStateChange {
            suite_id,
            test_id,
            state: TestState::Running,
        })?;
        self.comms.flush()?;

        // Track globally in case of panic during test execution
        CURRENT_SUITE
            .store(suite_id.try_into().unwrap_or(-1), Ordering::SeqCst);
        CURRENT_TEST.store(test_id.try_into().unwrap_or(-1), Ordering::SeqCst);

        let start_time_us = self.clock.now_us();
        let (elapsed_cycles, elapsed_stack) =
            self.executor.execute(exec.test_fn);
        let end_time_us = self.clock.now_us();
        let elapsed_time_us = end_time_us.saturating_sub(start_time_us);

        // Clear global trackers on success
        CURRENT_SUITE.store(-1, Ordering::SeqCst);
        CURRENT_TEST.store(-1, Ordering::SeqCst);

        // Update state to Passed
        self.comms.send_telemetry(&Telemetry::TestStateChange {
            suite_id,
            test_id,
            state: TestState::Passed,
        })?;

        // Send metric report
        self.comms.send_telemetry(&Telemetry::MetricReport {
            suite_id,
            test_id,
            cycles: elapsed_cycles,
            time_us: elapsed_time_us,
            stack_peak: elapsed_stack,
        })?;

        self.comms.flush()?;
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
            return Ok(());
        };
        let Some(&setting) = suite.settings.get(setting_idx) else {
            return Ok(());
        };

        let _ = setting.set(value);

        // Stream back the updated value to confirm
        self.comms.send_telemetry(&Telemetry::SettingInfo {
            suite_id,
            setting_id,
            name: setting.name(),
            description: setting.description(),
            value: setting.get(),
        })?;

        self.comms.flush()?;
        Ok(())
    }

    fn stream_discovery(&mut self) -> ServerResult<C::Error> {
        for (suite_idx, &suite) in self.suites.iter().enumerate() {
            let suite_id: u16 = match suite_idx.try_into() {
                Ok(id) => id,
                Err(_) => {
                    return Ok(());
                }
            };
            self.comms.send_telemetry(&Telemetry::SuiteInfo {
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

            for (test_idx, exec) in suite.executables.iter().enumerate() {
                let test_id: u16 = test_idx.try_into().unwrap_or(u16::MAX);
                self.comms.send_telemetry(&Telemetry::TestInfo {
                    suite_id,
                    test_id,
                    name: exec.name,
                    description: exec.description,
                })?;
            }

            for (setting_idx, setting) in suite.settings.iter().enumerate() {
                let setting_id: u16 =
                    setting_idx.try_into().unwrap_or(u16::MAX);
                self.comms.send_telemetry(&Telemetry::SettingInfo {
                    suite_id,
                    setting_id,
                    name: setting.name(),
                    description: setting.description(),
                    value: setting.get(),
                })?;
            }
        }

        self.comms.send_telemetry(&Telemetry::DiscoveryComplete)?;
        self.comms.flush()?;
        Ok(())
    }
}

/// Helper function to flush communications via type-erased pointer.
unsafe fn flush_comms_via_ptr<C: HostComms>(comms_ptr: *mut core::ffi::c_void) {
    if !comms_ptr.is_null() {
        let comms = unsafe { &mut *comms_ptr.cast::<C>() };
        let _ = comms.flush();
    }
}

/// Helper function to poll commands via type-erased pointer.
unsafe fn poll_command_via_ptr<C: HostComms>(
    comms_ptr: *mut core::ffi::c_void,
) -> Option<Command> {
    if comms_ptr.is_null() {
        None
    } else {
        let comms = unsafe { &mut *comms_ptr.cast::<C>() };
        comms.poll_command().ok().flatten()
    }
}

/// Helper function to transmit telemetry via type-erased pointer.
unsafe fn send_telemetry_via_ptr<C: HostComms>(
    comms_ptr: *mut core::ffi::c_void,
    telemetry: &Telemetry<'_>,
) {
    if !comms_ptr.is_null() {
        let comms = unsafe { &mut *comms_ptr.cast::<C>() };
        let _ = comms.send_telemetry(telemetry);
        let _ = comms.flush();
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::comms::{Command, HostComms, Telemetry, TestState};
    use crate::settings::{
        AtomicU8Setting, AtomicU32Setting, Setting, SettingValue,
    };
    use crate::time::DummyClock;
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
    type RawPayloads = Vec<Vec<u8>>;
    type SettingsSlice = &'static [&'static dyn Setting];

    struct MockComms {
        commands: Vec<Command>,
        fail_on_poll: bool,
        flush_count: usize,
        payloads: RawPayloads,
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
    fn test_atomic_settings() {
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
    fn test_dummy_clock() {
        let clock = DummyClock;
        assert_eq!(clock.now_ms(), 0);
        assert_eq!(clock.now_us(), 0);
    }

    #[test]
    fn test_server_discovery() {
        let comms = MockComms {
            commands: std::vec![Command::ListSuites],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        // Check telemetry sent
        let p = &server.comms.payloads;
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
            commands: std::vec![Command::OkToReset],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));
        assert!(server.comms.payloads.is_empty());
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
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        assert!(server.comms.payloads.is_empty());
    }

    #[test]
    fn test_server_poll_command_error() {
        let comms = MockComms {
            commands: Vec::new(),
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: true,
        };
        let mut server = Server::new(comms, DummyClock, SUITES);
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
        let mut server = Server::new_with_executor(
            comms,
            DummyClock,
            crate::executor::DummyExecutor,
            SUITES,
        );
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        assert!(TEST_CALLED.load(Ordering::SeqCst));

        let p = &server.comms.payloads;
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
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        let p = &server.comms.payloads;
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
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        let p = &server.comms.payloads;
        assert_eq!(p.len(), 1);
        let t0: Telemetry<'_> =
            postcard::from_bytes(p.first().unwrap()).unwrap();
        if let Telemetry::SettingInfo { value, .. } = t0 {
            assert!(matches!(value, SettingValue::U8(_)));
        } else {
            panic!("Expected SettingInfo");
        }
    }

    #[test]
    fn test_unsafe_ptr_helpers() {
        let mut comms = MockComms {
            commands: std::vec![Command::OkToReset],
            payloads: Vec::new(),
            flush_count: 0,
            fail_on_poll: false,
        };

        unsafe {
            send_telemetry_via_ptr::<MockComms>(
                core::ptr::null_mut(),
                &Telemetry::DiscoveryComplete,
            );
            flush_comms_via_ptr::<MockComms>(core::ptr::null_mut());
            let cmd = poll_command_via_ptr::<MockComms>(core::ptr::null_mut());
            assert!(cmd.is_none());
        }

        let comms_ptr =
            core::ptr::addr_of_mut!(comms).cast::<core::ffi::c_void>();
        unsafe {
            send_telemetry_via_ptr::<MockComms>(
                comms_ptr,
                &Telemetry::DiscoveryComplete,
            );
            assert_eq!(comms.payloads.len(), 1);
            let t0: Telemetry<'_> =
                postcard::from_bytes(comms.payloads.first().unwrap()).unwrap();
            assert!(matches!(t0, Telemetry::DiscoveryComplete));

            assert_eq!(comms.flush_count, 1);
            flush_comms_via_ptr::<MockComms>(comms_ptr);
            assert_eq!(comms.flush_count, 2);

            let cmd = poll_command_via_ptr::<MockComms>(comms_ptr);
            assert!(matches!(cmd, Some(Command::OkToReset)));
            assert_eq!(comms.commands.len(), 0);
        }
    }
}
