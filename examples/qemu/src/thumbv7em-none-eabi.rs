#![no_std]
#![no_main]

extern crate control_rs;

mod gain_task;

use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_macros::ets_setup;
// --- Communication Implementation via Direct Semihosting Syscalls ---

use core::sync::atomic::Ordering;

/// Byte stream over the semihosting console: `READC` blocks until the host sends a byte.
struct SemihostingStream;

impl ByteStream for SemihostingStream {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        let Some(slot) = buf.first_mut() else {
            return Ok(0);
        };
        *slot = unsafe {
            cortex_m_semihosting::syscall1(cortex_m_semihosting::nr::READC, 0)
        } as u8;
        Ok(1)
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        for b in buf {
            unsafe {
                cortex_m_semihosting::syscall(
                    cortex_m_semihosting::nr::WRITEC,
                    b,
                );
            }
        }
        Ok(buf.len())
    }
}

impl Drop for SemihostingStream {
    /// Ends the semihosting session when the server drops its link.
    fn drop(&mut self) {
        cortex_m_semihosting::debug::exit(
            cortex_m_semihosting::debug::EXIT_SUCCESS,
        );
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
fn setup() -> Context<StreamLink<SemihostingStream>, CortexMProfiler> {
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

    let link = StreamLink::new(SemihostingStream);
    let cpu_utils = CortexMProfiler::new(25_000_000, &MILLISECONDS);
    Context::new(link, cpu_utils)
}
