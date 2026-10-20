#![no_std]
#![no_main]

extern crate control_rs;

use bsp::board;
use bsp::hal::usbd::{BusAdapter, EndpointMemory, EndpointState, Speed};
use teensy4_bsp as bsp;

use control_rs_ets::link::{ByteStream, CountOf, StreamLink};
use control_rs_ets::server::Context;
use control_rs_ets::CortexMProfiler;
use control_rs_macros::{ets_setup, ets_suite};
#[allow(unused_imports)]
use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::peripheral::syst::SystClkSource;

use usb_device::bus::UsbBusAllocator;
use usb_device::device::{
    UsbDevice, UsbDeviceBuilder, UsbDeviceState, UsbVidPid,
};
use usbd_serial::SerialPort;

type UsbBus = UsbBusAllocator<BusAdapter>;
/// ETS link: `StreamLink` framing over the USB CDC port.
type UsbLink = StreamLink<TeensyUsb>;

// --- Millisecond Counter & SysTick Interrupt Handler ---

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

// --- Communication Implementation ---
//
// The ETS server exchanges whole messages over a `Link`. On the Teensy this is a
// `StreamLink` (sync, length and CRC-16 frame) over a USB CDC virtual serial port.
// `TeensyUsb` is the non-blocking byte stream: every read and write first drives the
// USB device state machine.

struct TeensyUsb {
    /// USB CDC class representing the virtual serial port.
    usb_class: SerialPort<'static, BusAdapter>,
    /// USB device manager driving the overall USB descriptor and state.
    usb_device: UsbDevice<'static, BusAdapter>,
    /// Flag indicating whether the host has configured the USB connection.
    configured: bool,
}

impl TeensyUsb {
    /// Polls the USB device stack and tracks the configured state.
    fn service(&mut self) {
        if self.usb_device.poll(&mut [&mut self.usb_class]) {
            if self.usb_device.state() == UsbDeviceState::Configured {
                if !self.configured {
                    self.usb_device.bus().configure();
                    self.configured = true;
                }
            } else {
                self.configured = false;
            }
        }
    }
}

impl ByteStream for TeensyUsb {
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
            // Back-pressure until CDC is configured; a full count would drain StreamLink TX.
            return Ok(0);
        }
        match self.usb_class.write(buf) {
            Ok(written) => Ok(written),
            Err(usb_device::UsbError::WouldBlock) => Ok(0),
            Err(_) => Err(()),
        }
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

// --- Test Suite Definition ---

#[ets_suite]
/// PID controller hardware-in-the-loop test suite.
pub mod teensy_pid_suite {
    use control_rs_ets::wire::Param;

    /// Proportional gain setting. Scales the magnitude of controller response to system error.
    pub static PROPORTIONAL_GAIN: u32 = 1500;
    /// Integral gain setting. Minimizes steady-state tracking error over time.
    pub static INTEGRAL_GAIN: u32 = 400;
    /// Derivative gain setting. Dampens transient response and overshoot.
    pub static DERIVATIVE_GAIN: u32 = 80;

    /// Verifies that Proportional > Integral > Derivative gains constraint holds.
    fn test_gain_inequality() {
        let kp = match PROPORTIONAL_GAIN.get() {
            Param::U32(v) => v,
            _ => 0,
        };
        let ki = match INTEGRAL_GAIN.get() {
            Param::U32(v) => v,
            _ => 0,
        };
        let kd = match DERIVATIVE_GAIN.get() {
            Param::U32(v) => v,
            _ => 0,
        };

        assert!(kp > ki);
        assert!(ki > kd);
    }

    fn test_kp_bounds() {
        let kp = match PROPORTIONAL_GAIN.get() {
            Param::U32(v) => v,
            _ => 0,
        };
        assert!((500..=5000).contains(&kp));
    }

    fn test_intentional_failure() {
        panic!("Intentionally failed to demonstrate TUI state retention");
    }
}

// --- Profiler Implementation for ARM Cortex-M ---
//
// The ETS crate's built-in `CortexMProfiler` implements the target-agnostic
// `CPUProfiler` trait. It reads clock cycles from the ARM DWT cycle counter and tracks
// real-time duration using the ARM SysTick timer. It also paints/profiles stack space.

fn enable_dwt_cycle_counter() {
    unsafe {
        let core_debug_demcr = 0xE000_EDFC as *mut u32;
        let dwt_ctrl = 0xE000_1000 as *mut u32;
        let dwt_cyccnt = 0xE000_1004 as *mut u32;

        // Enable DWT tracing
        core_debug_demcr
            .write_volatile(core_debug_demcr.read_volatile() | 0x0100_0000);
        // Reset cycle counter
        dwt_cyccnt.write_volatile(0);
        // Enable cycle counter in control register
        dwt_ctrl.write_volatile(dwt_ctrl.read_volatile() | 1);
    }
}

// --- Main Setup Entrypoint ---

#[ets_setup]
#[allow(dead_code)]
fn setup() -> Context<UsbLink, CortexMProfiler> {
    let p = cortex_m::Peripherals::take().unwrap();
    let d = board::instances();

    // Enable DWT cycle counter
    enable_dwt_cycle_counter();

    // 1. Initialize Board Resources
    let board::Resources {
        usb, pins, gpio2, ..
    } = board::t40(d);

    // 2. Set up the status LED (pin 13) to indicate ETS status
    let mut gpio2 = gpio2;
    let led = board::led(&mut gpio2, pins.p13);
    led.set();

    // 3. Set up the USB device stack statically
    static mut EP_MEMORY: EndpointMemory<1024> = EndpointMemory::new();
    static mut EP_STATE: EndpointState = EndpointState::max_endpoints();
    static mut USB_BUS: Option<UsbBus> = None;

    let ep_memory = unsafe { &mut *core::ptr::addr_of_mut!(EP_MEMORY) };
    let ep_state = unsafe { &mut *core::ptr::addr_of_mut!(EP_STATE) };
    let usb_bus_opt = unsafe { &mut *core::ptr::addr_of_mut!(USB_BUS) };

    const SPEED: Speed = Speed::LowFull;
    let bus_adapter = BusAdapter::with_speed(usb, ep_memory, ep_state, SPEED);

    // Disable interrupts during manual polling
    bus_adapter.set_interrupts(false);

    let usb_bus = usb_bus_opt.insert(UsbBusAllocator::new(bus_adapter));
    let usb_class = SerialPort::new(usb_bus);

    const VID_PID: UsbVidPid = UsbVidPid(0x16c0, 0x0413);
    const PRODUCT: &str = "teensy4";
    let usb_device = UsbDeviceBuilder::new(usb_bus, VID_PID)
        .product(PRODUCT)
        .device_class(usbd_serial::USB_CLASS_CDC)
        .build();

    // 4. Configure SysTick for 1 millisecond ticks
    let mut syst = p.SYST;
    syst.set_clock_source(SystClkSource::Core);
    let reload = board::ARM_FREQUENCY / 1000;
    syst.set_reload(reload - 1);
    syst.clear_current();
    syst.enable_counter();
    syst.enable_interrupt();

    let link = StreamLink::new(TeensyUsb {
        usb_class,
        usb_device,
        configured: false,
    });
    let cpu_utils = CortexMProfiler::new(board::ARM_FREQUENCY, &MILLISECONDS);
    Context::new(link, cpu_utils)
}
