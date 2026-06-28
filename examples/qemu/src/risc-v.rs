#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    frame_telemetry, Command, FrameReader, HostComms, Telemetry,
};
use control_rs_hil::server::Context;
use control_rs_hil::RiscvProfiler;
use control_rs_macros::hil_setup;

use semihosting::io::Write;

// The HIL testing framework uses the `HostComms` trait to define target-to-host
// communication. On RISC-V QEMU, we implement it using RISC-V Semihosting, which allows
// the emulator to pass standard stream data directly to the host process.
// We decode incoming host data into `Command`s using the HIL `FrameReader` utility.

struct RiscvSemihostingComms {
    /// State machine to decode incoming byte stream into Commands.
    reader: FrameReader,
}

impl HostComms for RiscvSemihostingComms {
    type Error = ();

    fn poll_command(&mut self) -> Result<Option<Command>, Self::Error> {
        // Read a single character from the host terminal via RISC-V semihosting syscall.
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

    fn close(&mut self) {
        semihosting::process::exit(0);
    }

    fn close_on_failure(&mut self) {
        semihosting::process::exit(1);
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

#[allow(unused_imports)]
pub use control_rs::polynomial::test::test_polynomial::SUITE_DESCRIPTOR_PTR as _;
#[allow(unused_imports)]
pub use control_rs::transfer_function::test::test_transfer_function::SUITE_DESCRIPTOR_PTR as _;
#[allow(unused_imports)]
pub use control_rs::state_space::test::test_state_space::SUITE_DESCRIPTOR_PTR as _;
#[allow(unused_imports)]
pub use control_rs::integrators::test::test_integrators::SUITE_DESCRIPTOR_PTR as _;


// --- Profiler Implementation for RISC-V ---
//
// We use the HIL crate's built-in `RiscvProfiler` to implement the `CPUProfiler` trait.
// This wraps target-specific instructions and CSR registers to read performance counter
// registers like `mcycle` for clock cycles and `time` for nanoseconds, enabling
// target-agnostic HIL testing.

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<RiscvSemihostingComms, RiscvProfiler> {
    let comms = RiscvSemihostingComms {
        reader: FrameReader::new(),
    };
    let cpu_utils = RiscvProfiler::new(10_000_000);
    Context::new(comms, cpu_utils)
}