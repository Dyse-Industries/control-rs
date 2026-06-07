#![no_std]
#![no_main]

use bsp::board;
use bsp::hal::usbd::{BusAdapter, EndpointMemory, EndpointState, Speed};
use teensy4_bsp as bsp;

use control_rs::hil::comms::{
    frame_telemetry, Command, FrameReader, HostComms, Telemetry,
};
use control_rs::hil::runner::Context;
use control_rs::hil::time::ClientClock;
use control_rs::hil::{hil_setup, hil_suite};
use core::sync::atomic::{AtomicU32, Ordering};
use cortex_m::peripheral::syst::SystClkSource;

use usb_device::bus::UsbBusAllocator;
use usb_device::device::{
    UsbDevice, UsbDeviceBuilder, UsbDeviceState, UsbVidPid,
};
use usbd_serial::SerialPort;

// --- Millisecond Counter & SysTick Interrupt Handler ---

static MILLISECONDS: AtomicU32 = AtomicU32::new(0);

#[cortex_m_rt::exception]
fn SysTick() {
    MILLISECONDS.fetch_add(1, Ordering::Relaxed);
}

// --- Timing Implementation ---

struct TeensyClock;

impl ClientClock for TeensyClock {
    fn now_ms(&self) -> u32 {
        MILLISECONDS.load(Ordering::Relaxed)
    }

    fn now_us(&self) -> u64 {
        let ms = MILLISECONDS.load(Ordering::Relaxed) as u64;
        // SysTick counts down from reload to 0
        let current = cortex_m::peripheral::SYST::get_current();
        let reload = board::ARM_FREQUENCY / 1000 - 1;
        let cycles = reload.saturating_sub(current);
        ms * 1000 + (cycles as u64) / (board::ARM_FREQUENCY as u64 / 1_000_000)
    }
}

// --- Communication Implementation ---

struct TeensyComms {
    usb_class: SerialPort<'static, BusAdapter>,
    usb_device: UsbDevice<'static, BusAdapter>,
    reader: FrameReader,
    configured: bool,
}

impl HostComms for TeensyComms {
    type Error = ();

    fn poll_command(&mut self) -> Result<Option<Command>, ()> {
        // 1. Poll the USB device stack to process CDC events
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

        // 2. Read bytes from the CDC virtual serial port
        if self.configured {
            let mut buf = [0u8; 64];
            match self.usb_class.read(&mut buf) {
                Ok(count) if count > 0 => {
                    for &byte in &buf[..count] {
                        if let Some(payload) = self.reader.handle_byte(byte) {
                            if let Ok(cmd) = postcard::from_bytes(payload) {
                                return Ok(Some(cmd));
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(None)
    }

    fn send_telemetry(&mut self, telemetry: &Telemetry<'_>) -> Result<(), ()> {
        let mut buf = [0u8; 512];
        if let Ok(len) = frame_telemetry(telemetry, &mut buf) {
            if self.configured {
                let mut data = &buf[..len];
                while !data.is_empty() {
                    // Drive the USB CDC state machine
                    self.usb_device.poll(&mut [&mut self.usb_class]);
                    match self.usb_class.write(data) {
                        Ok(written) => {
                            data = &data[written..];
                        }
                        Err(usb_device::UsbError::WouldBlock) => {
                            core::hint::spin_loop();
                        }
                        Err(_) => return Err(()),
                    }
                }
                Ok(())
            } else {
                Ok(()) // Gracefully do nothing if host hasn't connected
            }
        } else {
            Err(())
        }
    }

    fn flush(&mut self) -> Result<(), ()> {
        if self.configured {
            for _ in 0..10_000 {
                self.usb_device.poll(&mut [&mut self.usb_class]);
                core::hint::spin_loop();
            }
        }
        Ok(())
    }
}

// --- Test Suite Definition ---

#[hil_suite]
pub mod teensy_pid_suite {
    use control_rs::hil::hil_test::{Setting, SettingValue};

    // Statics representing configurable controller parameters
    pub static PROPORTIONAL_GAIN: u32 = 1500;
    pub static INTEGRAL_GAIN: u32 = 400;
    pub static DERIVATIVE_GAIN: u32 = 80;

    fn test_gain_inequality() {
        let kp = match PROPORTIONAL_GAIN.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        let ki = match INTEGRAL_GAIN.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        let kd = match DERIVATIVE_GAIN.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };

        assert!(kp > ki);
        assert!(ki > kd);
    }

    fn test_kp_bounds() {
        let kp = match PROPORTIONAL_GAIN.get() {
            SettingValue::U32(v) => v,
            _ => 0,
        };
        assert!(kp >= 500 && kp <= 5000);
    }

    fn test_intentional_failure() {
        assert!(
            false,
            "Intentionally failed to demonstrate TUI state retention"
        );
    }
}

// --- Main Setup Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<TeensyComms, TeensyClock> {
    let p = cortex_m::Peripherals::take().unwrap();
    let d = board::instances();

    // 1. Initialize Board Resources
    let board::Resources {
        usb, pins, gpio2, ..
    } = board::t40(d);

    // 2. Set up the status LED (pin 13) to indicate HIL server status
    let mut gpio2 = gpio2;
    let led = board::led(&mut gpio2, pins.p13);
    led.set();

    // 3. Set up the USB device stack statically
    static mut EP_MEMORY: EndpointMemory<1024> = EndpointMemory::new();
    static mut EP_STATE: EndpointState = EndpointState::max_endpoints();
    static mut USB_BUS: Option<UsbBusAllocator<BusAdapter>> = None;

    let ep_memory = unsafe { &mut *core::ptr::addr_of_mut!(EP_MEMORY) };
    let ep_state = unsafe { &mut *core::ptr::addr_of_mut!(EP_STATE) };
    let usb_bus_opt = unsafe { &mut *core::ptr::addr_of_mut!(USB_BUS) };

    const SPEED: Speed = Speed::LowFull;
    let bus_adapter = BusAdapter::with_speed(usb, ep_memory, ep_state, SPEED);

    // Disable interrupts since we are polling manually
    bus_adapter.set_interrupts(false);

    let usb_bus = usb_bus_opt.insert(UsbBusAllocator::new(bus_adapter));
    let usb_class = SerialPort::new(usb_bus);

    const VID_PID: UsbVidPid = UsbVidPid(0x16c0, 0x0413);
    const PRODUCT: &str = "teensy4-bsp-example";
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

    let comms = TeensyComms {
        usb_class,
        usb_device,
        reader: FrameReader::new(),
        configured: false,
    };
    let timer = TeensyClock;

    Context { comms, timer }
}