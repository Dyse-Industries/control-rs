#![no_std]
#![no_main]

extern crate control_rs;

mod gain_task;

use control_rs_ets::RiscvProfiler;
use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_macros::ets_setup;

use semihosting::io::Write;

/// Byte stream over the semihosting console: `READC` blocks until the host sends a byte.
struct RiscvSemihostingStream;

impl ByteStream for RiscvSemihostingStream {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        let Some(slot) = buf.first_mut() else {
            return Ok(0);
        };
        *slot = unsafe {
            riscv_semihosting::syscall1(riscv_semihosting::nr::READC, 0)
        } as u8;
        Ok(1)
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        if let Ok(mut stdout) = semihosting::io::stdout() {
            let _ = stdout.write_all(buf);
        }
        Ok(buf.len())
    }

    fn close_on_failure(&mut self) {
        semihosting::process::exit(1);
    }
}

impl Drop for RiscvSemihostingStream {
    /// Ends the semihosting session when the server drops its link.
    fn drop(&mut self) {
        semihosting::process::exit(0);
    }
}

// Force linking of the math test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::math::tests::suites::*;
// Force linking of the matrix test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::matrix::tests::suites::*;
// Force linking of the classical control test suites by referencing them
#[allow(unused_imports)]
pub use control_rs::classical_control::suites::*;

#[ets_setup]
#[allow(dead_code)]
fn setup() -> Context<StreamLink<RiscvSemihostingStream>, RiscvProfiler> {
    let link = StreamLink::new(RiscvSemihostingStream);
    let cpu_utils = RiscvProfiler::new(10_000_000);
    Context::new(link, cpu_utils)
}
