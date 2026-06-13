#![no_std]
#![no_main]

extern crate control_rs;

use control_rs_hil::comms::{
    Command, FrameReader, HostComms, Telemetry, frame_telemetry,
};
use control_rs_hil::server::Context;
use control_rs_hil::time::DummyClock;
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

// --- Main Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<SemihostingComms, DummyClock> {
    let comms = SemihostingComms {
        reader: FrameReader::new(),
    };
    let timer = DummyClock;
    Context { comms, timer }
}
