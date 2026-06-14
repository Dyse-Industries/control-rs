//! On-target test runner server loop.

use crate::SuiteDescriptor;
use crate::comms::{Command, HostComms, Telemetry, TestState};
use crate::settings::SettingValue;
use crate::time::ClientClock;
use core::sync::atomic::{AtomicI16, Ordering};

/// Context object that encapsulates communication and timekeeper peripherals.
pub struct Context<C, T> {
    /// Host communication channel.
    pub comms: C,
    /// Hardware timekeeper clock.
    pub timer: T,
}

/// Global tracker for the currently executing suite ID.
/// Used by the panic handler to report test failures.
pub static CURRENT_SUITE: AtomicI16 = AtomicI16::new(-1);

/// Global tracker for the currently executing test ID.
/// Used by the panic handler to report test failures.
pub static CURRENT_TEST: AtomicI16 = AtomicI16::new(-1);

/// Raw pointer to the active communication device.
pub static mut ACTIVE_COMMS_PTR: *mut core::ffi::c_void = core::ptr::null_mut();

/// Function signature for telemetry sender during a panic.
pub type PanicTelemetrySender =
    unsafe fn(*mut core::ffi::c_void, &Telemetry<'_>);

/// Function signature for communication flusher during a panic.
pub type PanicCommsFlusher = unsafe fn(*mut core::ffi::c_void);

/// Function signature for command poller during a panic.
pub type PanicCmdPoller = unsafe fn(*mut core::ffi::c_void) -> Option<Command>;

/// Static function pointer used to transmit telemetry during a panic.
pub static mut PANIC_TELEMETRY_SENDER: Option<PanicTelemetrySender> = None;

/// Static function pointer used to flush communications during a panic.
pub static mut PANIC_COMMS_FLUSHER: Option<PanicCommsFlusher> = None;

/// Static function pointer used to poll commands during a panic.
pub static mut PANIC_CMD_POLLER: Option<PanicCmdPoller> = None;

/// Helper function to transmit telemetry via type-erased pointer.
unsafe fn send_telemetry_via_ptr<C: HostComms>(
    comms_ptr: *mut core::ffi::c_void,
    telemetry: &Telemetry<'_>,
) {
    if !comms_ptr.is_null() {
        let comms = unsafe { &mut *(comms_ptr as *mut C) };
        let _ = comms.send_telemetry(telemetry);
        let _ = comms.flush();
    }
}

/// Helper function to flush communications via type-erased pointer.
unsafe fn flush_comms_via_ptr<C: HostComms>(comms_ptr: *mut core::ffi::c_void) {
    if !comms_ptr.is_null() {
        let comms = unsafe { &mut *(comms_ptr as *mut C) };
        let _ = comms.flush();
    }
}

/// Helper function to poll commands via type-erased pointer.
unsafe fn poll_command_via_ptr<C: HostComms>(
    comms_ptr: *mut core::ffi::c_void,
) -> Option<Command> {
    if !comms_ptr.is_null() {
        let comms = unsafe { &mut *(comms_ptr as *mut C) };
        comms.poll_command().ok().flatten()
    } else {
        None
    }
}

/// Result of server operations.
pub type ServerResult<E> = Result<(), E>;

/// Interactive test runner server.
pub struct Server<'a, C, T> {
    comms: C,
    clock: T,
    suites: &'a [&'static SuiteDescriptor],
}

#[allow(clippy::type_complexity)]
impl<'a, C, T> Server<'a, C, T>
where
    C: HostComms,
    T: ClientClock,
{
    /// Creates a new `Server` instance.
    pub fn new(
        comms: C,
        clock: T,
        suites: &'a [&'static SuiteDescriptor],
    ) -> Self {
        // Enable DWT cycle counter if on compatible ARM target
        enable_dwt_cycle_counter();

        Self {
            comms,
            clock,
            suites,
        }
    }

    /// Runs the interactive server event loop.
    ///
    /// This function polls for incoming host commands, executes requested tests,
    /// and streams telemetry and metrics back to the host.
    pub fn run(&mut self) -> ServerResult<C::Error> {
        unsafe {
            ACTIVE_COMMS_PTR =
                &mut self.comms as *mut C as *mut core::ffi::c_void;
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

    fn stream_discovery(&mut self) -> ServerResult<C::Error> {
        for (suite_idx, &suite) in self.suites.iter().enumerate() {
            let suite_id = suite_idx as u16;
            self.comms.send_telemetry(&Telemetry::SuiteInfo {
                suite_id,
                name: suite.name,
                description: suite.description,
                test_count: suite.executables.len() as u16,
                setting_count: suite.settings.len() as u16,
            })?;

            for (test_idx, exec) in suite.executables.iter().enumerate() {
                self.comms.send_telemetry(&Telemetry::TestInfo {
                    suite_id,
                    test_id: test_idx as u16,
                    name: exec.name,
                    description: exec.description,
                })?;
            }

            for (setting_idx, setting) in suite.settings.iter().enumerate() {
                self.comms.send_telemetry(&Telemetry::SettingInfo {
                    suite_id,
                    setting_id: setting_idx as u16,
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

    fn run_test(
        &mut self,
        suite_id: u16,
        test_id: u16,
    ) -> ServerResult<C::Error> {
        let suite_idx = suite_id as usize;
        let test_idx = test_id as usize;

        if suite_idx >= self.suites.len() {
            return Ok(());
        }
        let suite = self.suites[suite_idx];
        if test_idx >= suite.executables.len() {
            return Ok(());
        }
        let exec = &suite.executables[test_idx];

        // Update state to Running
        self.comms.send_telemetry(&Telemetry::TestStateChange {
            suite_id,
            test_id,
            state: TestState::Running,
        })?;
        self.comms.flush()?;

        // Track globally in case of panic during test execution
        CURRENT_SUITE.store(suite_id as i16, Ordering::SeqCst);
        CURRENT_TEST.store(test_id as i16, Ordering::SeqCst);

        // Record start time before painting the stack to clean any stack footprint from the clock call
        let start_time_us = self.clock.now_us();

        #[cfg(all(target_os = "none", feature = "stack-paint"))]
        let (elapsed_cycles, elapsed_stack) = cortex_m::interrupt::free(|_| {
            let sp_before = crate::util::get_sp();
            unsafe {
                crate::util::paint_stack();
            }
            let start_cycles = read_cycle_counter();
            (exec.test_fn)();
            let end_cycles = read_cycle_counter();
            let elapsed_stack = unsafe { crate::util::scan_stack(sp_before) };
            (end_cycles.saturating_sub(start_cycles), elapsed_stack)
        });

        #[cfg(not(all(target_os = "none", feature = "stack-paint")))]
        let (elapsed_cycles, elapsed_stack) = {
            let start_cycles = read_cycle_counter();
            (exec.test_fn)();
            let end_cycles = read_cycle_counter();
            (end_cycles.saturating_sub(start_cycles), 0)
        };

        // Record end time after stack scanning
        let end_time_us = self.clock.now_us();

        // Clear global trackers on success
        CURRENT_SUITE.store(-1, Ordering::SeqCst);
        CURRENT_TEST.store(-1, Ordering::SeqCst);

        let elapsed_time_us = end_time_us.saturating_sub(start_time_us);

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

        if suite_idx >= self.suites.len() {
            return Ok(());
        }
        let suite = self.suites[suite_idx];
        if setting_idx >= suite.settings.len() {
            return Ok(());
        }

        let setting = suite.settings[setting_idx];
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
}

// --- DWT Cycle Counter helpers ---

#[cfg(all(target_arch = "arm", target_os = "none"))]
fn enable_dwt_cycle_counter() {
    unsafe {
        let core_debug_demcr = 0xE000_EDFC as *mut u32;
        let dwt_ctrl = 0xE000_1000 as *mut u32;
        let dwt_cyccnt = 0xE000_1004 as *mut u32;

        // Enable DWT tracing
        core_debug_demcr
            .write_volatile(core_debug_demcr.read_volatile() | 0x0100_0000);
        // Reset cycle counter
        dwt_cyccnt.write_volatile(0);
        // Enable cycle counter in control register
        dwt_ctrl.write_volatile(dwt_ctrl.read_volatile() | 1);
    }
}

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn enable_dwt_cycle_counter() {}

#[cfg(all(target_arch = "arm", target_os = "none"))]
fn read_cycle_counter() -> u64 {
    unsafe {
        let dwt_cyccnt = 0xE000_1004 as *const u32;
        core::ptr::read_volatile(dwt_cyccnt) as u64
    }
}

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn read_cycle_counter() -> u64 {
    0
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

    type RawPayloads = Vec<Vec<u8>>;

    struct MockComms {
        commands: Vec<Command>,
        payloads: RawPayloads,
        flush_count: usize,
        fail_on_poll: bool,
    }

    impl HostComms for MockComms {
        type Error = &'static str;

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
            for &b in &buf[..size] {
                if let Some(p) = reader.handle_byte(b) {
                    payload = Some(p.to_vec());
                    break;
                }
            }

            let payload = payload.ok_or("No payload decoded")?;
            self.payloads.push(payload);
            Ok(())
        }

        fn flush(&mut self) -> Result<(), Self::Error> {
            self.flush_count += 1;
            Ok(())
        }
    }

    static TEST_CALLED: AtomicBool = AtomicBool::new(false);

    fn dummy_test_fn() {
        TEST_CALLED.store(true, Ordering::SeqCst);
    }

    static TEST_U8_SETTING: AtomicU8Setting =
        AtomicU8Setting::new("test_u8", "test_u8_desc", 42);

    type SettingsSlice = &'static [&'static dyn Setting];

    static SUITE_SETTINGS: SettingsSlice = &[&TEST_U8_SETTING];

    static SUITE_EXECUTABLES: &[ExecDescriptor] = &[ExecDescriptor {
        name: "dummy_test",
        description: "dummy_desc",
        test_fn: dummy_test_fn,
    }];

    static SUITE_DESC: SuiteDescriptor = SuiteDescriptor {
        name: "mock_suite",
        description: "mock_suite_desc",
        executables: SUITE_EXECUTABLES,
        settings: SUITE_SETTINGS,
    };

    static SUITES: &[&SuiteDescriptor] = &[&SUITE_DESC];

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

        let t0: Telemetry<'_> = postcard::from_bytes(&p[0]).unwrap();
        assert!(matches!(
            t0,
            Telemetry::SuiteInfo {
                suite_id: 0,
                name: "mock_suite",
                ..
            }
        ));

        let t1: Telemetry<'_> = postcard::from_bytes(&p[1]).unwrap();
        assert!(matches!(
            t1,
            Telemetry::TestInfo {
                suite_id: 0,
                test_id: 0,
                name: "dummy_test",
                ..
            }
        ));

        let t2: Telemetry<'_> = postcard::from_bytes(&p[2]).unwrap();
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

        let t3: Telemetry<'_> = postcard::from_bytes(&p[3]).unwrap();
        assert!(matches!(t3, Telemetry::DiscoveryComplete));
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
        let mut server = Server::new(comms, DummyClock, SUITES);
        let res = server.run();
        assert_eq!(res, Err("Exit loop"));

        assert!(TEST_CALLED.load(Ordering::SeqCst));

        let p = &server.comms.payloads;
        assert_eq!(p.len(), 3);

        let t0: Telemetry<'_> = postcard::from_bytes(&p[0]).unwrap();
        assert!(matches!(
            t0,
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Running
            }
        ));

        let t1: Telemetry<'_> = postcard::from_bytes(&p[1]).unwrap();
        assert!(matches!(
            t1,
            Telemetry::TestStateChange {
                suite_id: 0,
                test_id: 0,
                state: TestState::Passed
            }
        ));

        let t2: Telemetry<'_> = postcard::from_bytes(&p[2]).unwrap();
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
        let t0: Telemetry<'_> = postcard::from_bytes(&p[0]).unwrap();
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
        let t0: Telemetry<'_> = postcard::from_bytes(&p[0]).unwrap();
        if let Telemetry::SettingInfo { value, .. } = t0 {
            assert!(matches!(value, SettingValue::U8(_)));
        } else {
            panic!("Expected SettingInfo");
        }
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

        let comms_ptr = &mut comms as *mut MockComms as *mut core::ffi::c_void;
        unsafe {
            send_telemetry_via_ptr::<MockComms>(
                comms_ptr,
                &Telemetry::DiscoveryComplete,
            );
            assert_eq!(comms.payloads.len(), 1);
            let t0: Telemetry<'_> =
                postcard::from_bytes(&comms.payloads[0]).unwrap();
            assert!(matches!(t0, Telemetry::DiscoveryComplete));

            assert_eq!(comms.flush_count, 1);
            flush_comms_via_ptr::<MockComms>(comms_ptr);
            assert_eq!(comms.flush_count, 2);

            let cmd = poll_command_via_ptr::<MockComms>(comms_ptr);
            assert!(matches!(cmd, Some(Command::OkToReset)));
            assert_eq!(comms.commands.len(), 0);
        }
    }

    #[test]
    fn test_dummy_clock() {
        let clock = DummyClock;
        assert_eq!(clock.now_ms(), 0);
        assert_eq!(clock.now_us(), 0);
    }

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
}
