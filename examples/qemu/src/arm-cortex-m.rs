#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    frame_telemetry, Command, FrameReader, HostComms, Telemetry,
};
use control_rs_hil::server::Context;
use control_rs_macros::hil_setup;
// --- Communication Implementation via Direct Semihosting Syscalls ---
//
// The HIL testing framework uses the `HostComms` trait to define how the target MCU 
// exchanges bytes with the host test runner/TUI.
// Here we implement `HostComms` using ARM Semihosting, which allows the QEMU-emulated 
// target to make syscalls directly to the host machine for standard input/output.
//
// We use a `FrameReader` state machine (from the HIL crate) to decode raw incoming bytes 
// from the host into parsed `Command` packets (like "run test X" or "list suites").

use core::sync::atomic::Ordering;

#[allow(dead_code)]
struct SemihostingComms {
    /// Helper state machine to reassemble packets from incoming byte stream.
    reader: FrameReader,
}

impl HostComms for SemihostingComms {
    type Error = ();

    #[allow(clippy::collapsible_if)]
    fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
        // Read a single character from host terminal using semihosting READC syscall.
        let c = unsafe {
            cortex_m_semihosting::syscall1(cortex_m_semihosting::nr::READC, 0)
        } as u8;

        if let Some(payload) = self.reader.handle_byte(c) {
            if let Ok(cmd) = postcard::from_bytes(payload) {
                return Ok(Some(cmd));
            }
        }
        Ok(None)
    }

    fn send_telemetry(
        &mut self,
        telemetry: &Telemetry<'_>,
    ) -> Result<(), Self::Error> {
        let mut buf = [0u8; 512];
        if let Ok(len) = frame_telemetry(telemetry, &mut buf) {
            for &b in &buf[..len] {
                unsafe {
                    cortex_m_semihosting::syscall(
                        cortex_m_semihosting::nr::WRITEC,
                        &b,
                    );
                }
            }
            Ok(())
        } else {
            Err(())
        }
    }

    fn close(&mut self) {
        cortex_m_semihosting::debug::exit(cortex_m_semihosting::debug::EXIT_SUCCESS);
    }

    fn flush(&mut self) -> Result<(), Self::Error> {
        Ok(())
    }
}

// Force linking of the math test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::math::tests::complex_num_tests::{
    test_arithmetic::SUITE_DESCRIPTOR_PTR as _,
    test_axioms::SUITE_DESCRIPTOR_PTR as _,
    test_basics::SUITE_DESCRIPTOR_PTR as _,
    test_core_math::SUITE_DESCRIPTOR_PTR as _,
    test_dsp_patterns::SUITE_DESCRIPTOR_PTR as _,
    test_ffi_layout::SUITE_DESCRIPTOR_PTR as _,
    test_limitations::SUITE_DESCRIPTOR_PTR as _,
    test_transcendental::SUITE_DESCRIPTOR_PTR as _,
};

// --- Profiler Implementation for ARM Cortex-M ---
//
// The HIL testing framework uses the `CPUProfiler` trait to fetch hardware metrics 
// in a target-agnostic manner. 
// We use the HIL crate's built-in `CortexMProfiler` to track CPU cycles consumed during 
// tests (via the Data Watchpoint and Trace (DWT) peripheral) and elapsed time (using 
// the SysTick timer, which fires an interrupt incrementing our atomic millisecond counter).
// It also provides a way to read and paint the stack memory to measure peak stack usage.

use control_rs_hil::CortexMProfiler;
use core::sync::atomic::AtomicU32;
use cortex_m::peripheral::syst::SystClkSource;

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

// --- Test Execution Implementation for ARM Cortex-M ---

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

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<SemihostingComms, CortexMProfiler> {
    let p = cortex_m::Peripherals::take().unwrap();

    // Enable DWT cycle counter
    enable_dwt_cycle_counter();

    let mut syst = p.SYST;
    syst.set_clock_source(SystClkSource::Core);
    const CLOCK_FREQ: u32 = 25_000_000;
    let reload = CLOCK_FREQ / 1000;
    syst.set_reload(reload - 1);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();

    let comms = SemihostingComms {
        reader: FrameReader::new(),
    };
    let cpu_utils = CortexMProfiler::new(25_000_000, &MILLISECONDS);
    Context::new(comms, cpu_utils)
}