#![no_std]
#![no_main]

extern crate control_rs;

#[path = "../../qemu/src/gain_task.rs"]
mod gain_task;

use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::peripheral::syst::SystClkSource;

use control_rs_ets::CortexMProfiler;
use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_macros::{ets_setup, ets_suite};
use nrf52840_hal::Clocks;
use nrf52840_hal::usbd::{UsbPeripheral, Usbd};
use usb_device::bus::UsbBusAllocator;
use usb_device::device::{
    UsbDevice, UsbDeviceBuilder, UsbDeviceState, UsbVidPid,
};
use usbd_serial::SerialPort;

type UsbBus = UsbBusAllocator<Usbd<UsbPeripheral<'static>>>;
type UsbSerialPort = SerialPort<'static, Usbd<UsbPeripheral<'static>>>;
type UsbDev = UsbDevice<'static, Usbd<UsbPeripheral<'static>>>;

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

/// Non-blocking byte stream over USBD CDC-ACM: every read and write first drives the device.
struct XiaoNrfUsb {
    usb_class: UsbSerialPort,
    usb_device: UsbDev,
    configured: bool,
}

impl XiaoNrfUsb {
    fn service(&mut self) {
        if self.usb_device.poll(&mut [&mut self.usb_class]) {
            self.configured =
                self.usb_device.state() == UsbDeviceState::Configured;
        }
    }
}

impl ByteStream for XiaoNrfUsb {
    type Error = ();

    fn read(&mut self, buf: &mut [u8]) -> CountOf<Self> {
        self.service();
        if !self.configured {
            return Ok(0);
        }
        Ok(self.usb_class.read(buf).unwrap_or(0))
    }

    fn write(&mut self, buf: &[u8]) -> CountOf<Self> {
        self.service();
        if !self.configured {
            // Discard output while no host is connected.
            return Ok(buf.len());
        }
        match self.usb_class.write(buf) {
            Ok(written) => Ok(written),
            Err(usb_device::UsbError::WouldBlock) => Ok(0),
            Err(_) => Err(()),
        }
    }
}

// Example ETS test suite showcasing on-target computation on ARM Cortex-M4F (nRF52840)
#[ets_suite]
/// Signal checks on the Seeed XIAO nRF52840.
pub mod xiao_nrf52840_suite {
    use control_rs::matrix::Owned;
    use control_rs_ets::wire::Param;

    /// High cutoff frequency setting.
    pub static CUTOFF_FREQ_HZ: u32 = 2500;
    /// Sample rate setting.
    pub static SAMPLE_RATE_HZ: u32 = 10000;

    fn test_frequency_nyquist() {
        let Param::U32(cutoff) = CUTOFF_FREQ_HZ.get() else {
            panic!("CUTOFF_FREQ_HZ is a u32")
        };
        let Param::U32(rate) = SAMPLE_RATE_HZ.get() else {
            panic!("SAMPLE_RATE_HZ is a u32")
        };
        assert!(rate >= 2 * cutoff, "Nyquist criterion violated");
    }

    fn test_kalman_state_estimate() {
        // Discrete state prediction: x = F * x
        let f = Owned::<f64, 2, 2>::from_row_arrays([[1.0, 0.0], [0.01, 1.0]]);
        let x = Owned::<f64, 2, 1>::from_array([[10.0, 0.5]]);
        let mut x_pred = Owned::<f64, 2, 1>::zero();
        f.mul_into(&x, &mut x_pred);
        assert_eq!(x_pred.get(0, 0), Some(&10.0));
        assert_eq!(x_pred.get(1, 0), Some(&0.6));
    }
}

struct SyncUnsafeCell<T>(core::cell::UnsafeCell<T>);
unsafe impl<T> Sync for SyncUnsafeCell<T> {}

// Global USB bus allocator instance
type BusCell = SyncUnsafeCell<Option<UsbBus>>;

static USB_BUS: BusCell = SyncUnsafeCell(core::cell::UnsafeCell::new(None));

type NrfClocks = Clocks<
    nrf52840_hal::clocks::ExternalOscillator,
    nrf52840_hal::clocks::Internal,
    nrf52840_hal::clocks::LfOscStopped,
>;

type ClockCell = SyncUnsafeCell<Option<NrfClocks>>;

static CLOCKS_STORAGE: ClockCell =
    SyncUnsafeCell(core::cell::UnsafeCell::new(None));

#[ets_setup]
#[allow(dead_code)]
fn setup() -> Context<StreamLink<XiaoNrfUsb>, CortexMProfiler> {
    let p = nrf52840_hal::pac::Peripherals::take().unwrap();
    let cp = cortex_m::Peripherals::take().unwrap();

    // Enable high-frequency external oscillator for USB clock accuracy
    let clocks = Clocks::new(p.CLOCK).enable_ext_hfosc();

    // Configure SysTick for 1 ms tick rate (64 MHz core clock)
    let mut syst = cp.SYST;
    syst.set_clock_source(SystClkSource::Core);
    syst.set_reload(64_000 - 1);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();

    // Initialize USB stack
    // SAFETY: Single-threaded initialization during boot before interrupts start
    let usb_peripheral = unsafe {
        *CLOCKS_STORAGE.0.get() = Some(clocks);
        let clocks_ref = (*CLOCKS_STORAGE.0.get()).as_ref().unwrap();
        UsbPeripheral::new(p.USBD, clocks_ref)
    };

    let usb_bus = UsbBusAllocator::new(Usbd::new(usb_peripheral));
    unsafe {
        *USB_BUS.0.get() = Some(usb_bus);
    }

    let bus_ref = unsafe { (*USB_BUS.0.get()).as_ref().unwrap() };
    let usb_class = SerialPort::new(bus_ref);
    let usb_device =
        UsbDeviceBuilder::new(bus_ref, UsbVidPid(0x2886, 0x8044)) // Seeed XIAO nRF52840 VID/PID
            .strings(&[usb_device::device::StringDescriptors::default()
                .manufacturer("Seeed Studio")
                .product("XIAO nRF52840 ETS")])
            .unwrap()
            .device_class(usbd_serial::USB_CLASS_CDC)
            .build();

    Context::new(
        StreamLink::new(XiaoNrfUsb {
            usb_class,
            usb_device,
            configured: false,
        }),
        CortexMProfiler::new(64_000_000, &MILLISECONDS),
    )
}
