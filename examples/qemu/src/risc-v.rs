#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    Command, FrameReader, HostComms, Telemetry, frame_telemetry,
};
use control_rs_hil::server::Context;
use control_rs_hil::time::ClientClock;
use control_rs_macros::hil_setup;

use semihosting::io::Write;

// --- Communication Implementation via Direct Semihosting ---

struct RiscvSemihostingComms {
    reader: FrameReader,
}

impl HostComms for RiscvSemihostingComms {
    type Error = ();

    fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
        let c = unsafe {
            riscv_semihosting::syscall1(riscv_semihosting::nr::READC, 0)
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
            if let Ok(mut stdout) = semihosting::io::stdout() {
                let _ = stdout.write_all(&buf[..len]);
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

// --- Test Execution Implementation for RISC-V ---

#[inline(always)]
fn get_sp() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mv {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    sp
}

unsafe fn paint_stack(sp: usize) {
    unsafe extern "C" {
        static _stack_start: u32;
        static _hart_stack_size: u32;
    }

    let stack_start_ptr = core::ptr::addr_of!(_stack_start) as usize;
    let stack_size = core::ptr::addr_of!(_hart_stack_size) as usize;
    let stack_end_ptr = stack_start_ptr - stack_size;

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
        static _stack_start: u32;
        static _hart_stack_size: u32;
    }

    let stack_start_ptr = core::ptr::addr_of!(_stack_start) as usize;
    let stack_size = core::ptr::addr_of!(_hart_stack_size) as usize;
    let stack_end_ptr = stack_start_ptr - stack_size;

    let mut ptr = stack_end_ptr as *const u32;
    let limit_ptr = sp as *const u32;

    while ptr < limit_ptr {
        let is_sentinel = unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
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

struct RiscvExecutor;

impl ::control_rs_hil::executor::TestExecutor for RiscvExecutor {
    fn execute(&self, test_fn: fn()) -> (u64, u32) {
        // Disable interrupts for duration of the test
        unsafe {
            ::riscv::interrupt::disable();
        }
        let sp_before = get_sp();
        unsafe {
            paint_stack(sp_before);
        }
        let start_cycles = ::riscv::register::mcycle::read() as u64;
        test_fn();
        let end_cycles = ::riscv::register::mcycle::read() as u64;
        let elapsed_stack = unsafe { scan_stack(sp_before) };
        let elapsed_cycles = end_cycles.saturating_sub(start_cycles);
        (elapsed_cycles, elapsed_stack)
    }
}

struct RiscvClock;

impl ClientClock for RiscvClock {
    fn now_ms(&self) -> u32 {
        let ticks = ::riscv::register::time::read64();
        (ticks / 10_000) as u32
    }

    fn now_us(&self) -> u64 {
        let ticks = ::riscv::register::time::read64();
        ticks / 10
    }
}

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<RiscvSemihostingComms, RiscvClock, RiscvExecutor> {
    let comms = RiscvSemihostingComms {
        reader: FrameReader::new(),
    };
    let timer = RiscvClock;
    let executor = RiscvExecutor;
    Context { comms, timer, executor }
}
