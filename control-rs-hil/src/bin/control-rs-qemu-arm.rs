#![no_std]
#![no_main]

use control_rs_hil::comms::{
    Command, FrameReader, HostComms, Telemetry, frame_telemetry,
};
use control_rs_hil::runner::Context;
use control_rs_hil::time::DummyClock;
use control_rs_macros::{hil_setup, hil_suite};

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

// --- Test Definitions ---

#[hil_suite]
pub mod qemu_math_suite {
    pub static CONNECTION_TIMEOUT_MS: u32 = 1000;

    #[allow(clippy::eq_op)]
    fn math_addition() {
        assert_eq!(2 + 2, 4);
    }

    #[allow(clippy::eq_op)]
    fn math_subtraction() {
        assert_eq!(5 - 3, 2);
    }
}

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
