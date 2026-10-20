#![no_std]
#![no_main]

extern crate control_rs;

#[path = "../../qemu/src/gain_task.rs"]
mod gain_task;

esp_bootloader_esp_idf::esp_app_desc!();

use control_rs_ets::RiscvProfiler;
use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_macros::{ets_setup, ets_suite};
use esp_hal::usb::usb_serial_jtag::UsbSerialJtag;

/// Byte stream over the native USB Serial/JTAG controller.
struct Esp32c6Usb {
    usb: UsbSerialJtag<'static, esp_hal::Blocking>,
}

impl ByteStream for Esp32c6Usb {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        let mut count = 0_usize;
        for slot in buf.iter_mut() {
            let Ok(byte) = self.usb.read_byte() else {
                break;
            };
            *slot = byte;
            count += 1;
        }
        Ok(count)
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        self.usb.write(buf).map_err(|_| ())?;
        self.usb.flush_tx().map_err(|_| ())?;
        Ok(buf.len())
    }
}

// Example ETS test suite showcasing on-target computation & profiling on RISC-V
#[ets_suite]
/// Control checks on the ESP32-C6.
pub mod esp32c6_control_suite {
    use control_rs::matrix::Owned;
    use control_rs_ets::wire::Param;

    /// Proportional gain setting.
    pub static KP: u32 = 1200;
    /// Derivative gain setting.
    pub static KD: u32 = 50;

    fn test_gain_sanity() {
        let Param::U32(kp) = KP.get() else {
            panic!("KP is a u32")
        };
        let Param::U32(kd) = KD.get() else {
            panic!("KD is a u32")
        };
        assert!(kp > kd, "KP must exceed KD");
    }

    fn test_matrix_multiplication() {
        let a = Owned::<f64, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let b = Owned::<f64, 2, 2>::from_array([[2.0, 1.0], [0.0, 2.0]]);
        let mut c = Owned::<f64, 2, 2>::zero();
        a.mul_into(&b, &mut c);
        assert_eq!(c.get(0, 0), Some(&4.0));
        assert_eq!(c.get(0, 1), Some(&4.0));
        assert_eq!(c.get(1, 0), Some(&10.0));
        assert_eq!(c.get(1, 1), Some(&8.0));
    }
}

#[ets_setup]
#[allow(dead_code)]
fn setup() -> Context<StreamLink<Esp32c6Usb>, RiscvProfiler> {
    let peripherals = esp_hal::init(esp_hal::Config::default());
    let usb = UsbSerialJtag::new(peripherals.USB_DEVICE);

    // ESP32-C6 HP core operates at 160 MHz
    Context::new(
        StreamLink::new(Esp32c6Usb { usb }),
        RiscvProfiler::new(160_000_000),
    )
}
