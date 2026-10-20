#![no_std]
#![no_main]

extern crate control_rs;

#[path = "../../qemu/src/gain_task.rs"]
mod gain_task;

use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::peripheral::syst::SystClkSource;

use arduino_uno_r4_hal::serial::Serial;
use control_rs_ets::CortexMProfiler;
use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_macros::{ets_setup, ets_suite};

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

/// Byte stream over the SCI2 UART (Arduino `Serial1`).
struct UnoR4Uart {
    serial: Serial,
}

impl ByteStream for UnoR4Uart {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        let mut count = 0_usize;
        for slot in buf.iter_mut() {
            let Ok(byte) = self.serial.read_byte() else {
                break;
            };
            *slot = byte;
            count += 1;
        }
        Ok(count)
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        for &byte in buf {
            let _ = nb::block!(self.serial.write_byte(byte));
        }
        Ok(buf.len())
    }
}

// Example ETS test suite showcasing on-target computation on ARM Cortex-M4 (RA4M1)
#[ets_suite]
/// Control checks on the Arduino Uno R4 Minima.
pub mod uno_r4_control_suite {
    use control_rs::matrix::Owned;
    use control_rs_ets::wire::Param;

    /// Proportional gain setting.
    pub static KP: u32 = 1500;
    /// Integral gain setting.
    pub static KI: u32 = 250;

    fn test_gain_constraints() {
        let Param::U32(kp) = KP.get() else {
            panic!("KP is a u32")
        };
        let Param::U32(ki) = KI.get() else {
            panic!("KI is a u32")
        };
        assert!(kp > ki, "KP must exceed KI");
    }

    fn test_matrix_state_transition() {
        // Discrete state-space dynamic step: x[k+1] = A * x[k]
        // `from_array` is column-major; use `from_row_arrays` for row-major literals.
        let a =
            Owned::<f64, 2, 2>::from_row_arrays([[0.95, 0.0], [0.05, 0.90]]);
        let x = Owned::<f64, 2, 1>::from_array([[1.0, 0.0]]);
        let mut x_next = Owned::<f64, 2, 1>::zero();
        a.mul_into(&x, &mut x_next);
        assert_eq!(x_next.get(0, 0), Some(&0.95));
        assert_eq!(x_next.get(1, 0), Some(&0.05));
    }
}

#[ets_setup]
#[allow(dead_code)]
fn setup() -> Context<StreamLink<UnoR4Uart>, CortexMProfiler> {
    let p = arduino_uno_r4_hal::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    // Configure SysTick for 1 ms tick rate (48 MHz core clock)
    let mut syst = cp.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(48_000 - 1);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();

    // Arduino Serial1 on pins D1 (TX = P302) and D0 (RX = P301)
    let serial =
        Serial::new(p.sci2, p.pins.d1, p.pins.d0, 115_200, &p.clocks).unwrap();

    Context::new(
        StreamLink::new(UnoR4Uart { serial }),
        CortexMProfiler::new(48_000_000, &MILLISECONDS),
    )
}
