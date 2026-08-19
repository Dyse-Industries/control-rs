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

struct RiscvSemihostingComms {
    /// State machine to decode incoming byte stream into Commands.
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
pub use control_rs::math::tests::suites::*;
// Force linking of the matrix test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::matrix::tests::suites::*;

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<RiscvSemihostingComms, RiscvProfiler> {
    let comms = RiscvSemihostingComms {
        reader: FrameReader::new(),
    };
    let cpu_utils = RiscvProfiler::new(10_000_000);
    Context::new(comms, cpu_utils)
}
