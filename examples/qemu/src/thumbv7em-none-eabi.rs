#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_ets::comms::{
    frame_telemetry, Command, FrameReader, HostComms, Telemetry,
};
use control_rs_ets::server::Context;
use control_rs_macros::ets_setup;
// --- Communication Implementation via Direct Semihosting Syscalls ---

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
pub use control_rs::math::tests::suites::*;
// Force linking of the matrix test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::matrix::tests::suites::*;

use control_rs_ets::CortexMProfiler;
use core::sync::atomic::AtomicU32;
use cortex_m::peripheral::syst::SystClkSource;

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

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

#[ets_setup]
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
