#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    Command, FrameReader, HostComms, Telemetry, frame_telemetry,
};
use control_rs_hil::server::Context;
use control_rs_macros::hil_setup;

// --- Communication Implementation via Direct Semihosting Syscalls ---

#[allow(dead_code)]
struct SemihostingComms {
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

use control_rs_hil::time::ClientClock;
use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::peripheral::syst::SystClkSource;

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

struct QemuClock;

impl ClientClock for QemuClock {
    fn now_ms(&self) -> u32 {
        MILLISECONDS.load(Ordering::Relaxed)
    }

    fn now_us(&self) -> u64 {
        let ms = MILLISECONDS.load(Ordering::Relaxed) as u64;
        let current = cortex_m::peripheral::SYST::get_current();
        const CLOCK_FREQ: u32 = 25_000_000;
        let reload = CLOCK_FREQ / 1000 - 1;
        let cycles = reload.saturating_sub(current);
        ms * 1000 + (cycles as u64) / (CLOCK_FREQ as u64 / 1_000_000)
    }
}

// --- Test Execution Implementation for ARM Cortex-M ---

#[inline(always)]
fn get_sp() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    sp
}

unsafe fn paint_stack(sp: usize) {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;
    // Leave a 32-byte safety margin below the current stack pointer to avoid
    // overwriting active stack frames (like paint_stack's own frame and return address).
    let limit = sp.saturating_sub(32);

    if limit > stack_end_ptr {
        let mut ptr = stack_end_ptr as *mut u32;
        let limit_ptr = limit as *mut u32;

        while ptr < limit_ptr {
            unsafe {
                core::ptr::write_volatile(ptr, 0xCDCD_CDCD);
                ptr = ptr.add(1);
            }
        }
    }
}

unsafe fn scan_stack(sp: usize) -> u32 {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;

    let mut ptr = stack_end_ptr as *const u32;
    let limit_ptr = sp as *const u32;

    while ptr < limit_ptr {
        let is_sentinel =
            unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
        if !is_sentinel {
            break;
        }
        unsafe {
            ptr = ptr.add(1);
        }
    }

    let lowest_address = ptr as usize;
    if lowest_address < sp {
        (sp - lowest_address) as u32
    } else {
        0
    }
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

fn read_cycle_counter() -> u64 {
    unsafe {
        let dwt_cyccnt = 0xE000_1004 as *const u32;
        core::ptr::read_volatile(dwt_cyccnt) as u64
    }
}

struct CortexMExecutor;

impl ::control_rs_hil::executor::TestExecutor for CortexMExecutor {
    fn execute(&self, test_fn: fn()) -> (u64, u32) {
        cortex_m::interrupt::free(|_| {
            let sp_before = get_sp();
            unsafe {
                paint_stack(sp_before);
            }
            let start_cycles = read_cycle_counter();
            test_fn();
            let end_cycles = read_cycle_counter();
            let elapsed_stack = unsafe { scan_stack(sp_before) };
            let elapsed_cycles = end_cycles.saturating_sub(start_cycles);
            (elapsed_cycles, elapsed_stack)
        })
    }
}

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<SemihostingComms, QemuClock, CortexMExecutor> {
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
    let timer = QemuClock;
    let executor = CortexMExecutor;
    Context { comms, timer, executor }
}
