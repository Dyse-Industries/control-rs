#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    Command, FrameReader, HostComms, Telemetry, frame_telemetry,
};
use control_rs_hil::server::Context;
use control_rs_hil::time::DummyClock;
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

struct RiscvExecutor;

impl ::control_rs_hil::executor::TestExecutor for RiscvExecutor {
    fn execute(&self, test_fn: fn()) -> (u64, u32) {
        // Disable interrupts for duration of the test
        unsafe {
            ::riscv::interrupt::disable();
        }
        let start_cycles = ::riscv::register::mcycle::read() as u64;
        test_fn();
        let end_cycles = ::riscv::register::mcycle::read() as u64;
        let elapsed_cycles = end_cycles.saturating_sub(start_cycles);
        (elapsed_cycles, 0)
    }
}

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<RiscvSemihostingComms, DummyClock, RiscvExecutor> {
    let comms = RiscvSemihostingComms {
        reader: FrameReader::new(),
    };
    let timer = DummyClock;
    let executor = RiscvExecutor;
    Context { comms, timer, executor }
}
