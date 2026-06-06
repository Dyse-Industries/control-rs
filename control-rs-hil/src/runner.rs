//! On-target test runner server loop.

use crate::comms::{Command, HostComms, Telemetry, TestState};
use crate::hil_test::{SettingValue, SuiteDescriptor};
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

/// Static function pointer used to transmit telemetry during a panic.
pub static mut PANIC_TELEMETRY_SENDER: Option<
    unsafe fn(*mut core::ffi::c_void, &Telemetry<'_>),
> = None;

/// Static function pointer used to flush communications during a panic.
pub static mut PANIC_COMMS_FLUSHER: Option<unsafe fn(*mut core::ffi::c_void)> =
    None;

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

/// Interactive test runner server.
pub struct Server<'a, C, T> {
    comms: C,
    clock: T,
    suites: &'a [&'static SuiteDescriptor],
}

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
    pub fn run(&mut self) -> Result<(), C::Error> {
        unsafe {
            ACTIVE_COMMS_PTR =
                &mut self.comms as *mut C as *mut core::ffi::c_void;
            PANIC_TELEMETRY_SENDER = Some(send_telemetry_via_ptr::<C>);
            PANIC_COMMS_FLUSHER = Some(flush_comms_via_ptr::<C>);
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
                    }
                }
                self.comms.flush()?;
            }
        })();

        unsafe {
            ACTIVE_COMMS_PTR = core::ptr::null_mut();
            PANIC_TELEMETRY_SENDER = None;
            PANIC_COMMS_FLUSHER = None;
        }

        res
    }

    fn stream_discovery(&mut self) -> Result<(), C::Error> {
        for (suite_idx, &suite) in self.suites.iter().enumerate() {
            let suite_id = suite_idx as u16;
            self.comms.send_telemetry(&Telemetry::SuiteInfo {
                suite_id,
                name: suite.name,
                test_count: suite.executables.len() as u16,
                setting_count: suite.settings.len() as u16,
            })?;

            for (test_idx, exec) in suite.executables.iter().enumerate() {
                self.comms.send_telemetry(&Telemetry::TestInfo {
                    suite_id,
                    test_id: test_idx as u16,
                    name: exec.name,
                })?;
            }

            for (setting_idx, setting) in suite.settings.iter().enumerate() {
                self.comms.send_telemetry(&Telemetry::SettingInfo {
                    suite_id,
                    setting_id: setting_idx as u16,
                    name: setting.name(),
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
    ) -> Result<(), C::Error> {
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

        // Record start metrics
        let start_time_us = self.clock.now_us();
        let start_cycles = read_cycle_counter();

        // Run the test
        (exec.test_fn)();

        // Record end metrics
        let end_cycles = read_cycle_counter();
        let end_time_us = self.clock.now_us();

        // Clear global trackers on success
        CURRENT_SUITE.store(-1, Ordering::SeqCst);
        CURRENT_TEST.store(-1, Ordering::SeqCst);

        let elapsed_time_us = end_time_us.saturating_sub(start_time_us);
        let elapsed_cycles = end_cycles.saturating_sub(start_cycles);

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
        })?;

        self.comms.flush()?;
        Ok(())
    }

    fn set_setting(
        &mut self,
        suite_id: u16,
        setting_id: u16,
        value: SettingValue,
    ) -> Result<(), C::Error> {
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
