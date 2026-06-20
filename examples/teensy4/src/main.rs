#![no_std]
#![no_main]

extern crate control_rs;

use bsp::board;
use bsp::hal::usbd::{BusAdapter, EndpointMemory, EndpointState, Speed};
use teensy4_bsp as bsp;

use control_rs_hil::comms::{
    frame_telemetry, Command, FrameReader, HostComms, Telemetry,
};
use control_rs_hil::server::Context;
use control_rs_hil::time::ClientClock;
use control_rs_macros::{hil_setup, hil_suite};
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

// --- Test Suite Definition ---

#[hil_suite]
/// PID controller hardware-in-the-loop test suite.
pub mod teensy_pid_suite {
    use control_rs_hil::settings::{Setting, SettingValue};

    /// Proportional gain setting. Scales the magnitude of controller response to system error.
    pub static PROPORTIONAL_GAIN: u32 = 1500;
    /// Integral gain setting. Minimizes steady-state tracking error over time.
    pub static INTEGRAL_GAIN: u32 = 400;
    /// Derivative gain setting. Dampens transient response and overshoot.
    pub static DERIVATIVE_GAIN: u32 = 80;

    /// Verifies that Proportional > Integral > Derivative gains constraint holds.
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

// --- Test Execution Implementation for ARM Cortex-M ---

#[inline(always)]
fn get_sp() -> usize {
    let sp: usize;
    unsafe {
        core::arch::asm!("mov {}, sp", out(reg) sp, options(nomem, nostack, preserves_flags));
    }
    sp
}

unsafe fn paint_stack(sp: usize) {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;
    // Leave a 32-byte safety margin below the current stack pointer to avoid
    // overwriting active stack frames (like paint_stack's own frame and return address).
    let limit = sp.saturating_sub(32);

    if limit > stack_end_ptr {
        let mut ptr = stack_end_ptr as *mut u32;
        let limit_ptr = limit as *mut u32;

        while ptr < limit_ptr {
            unsafe {
                core::ptr::write_volatile(ptr, 0xCDCD_CDCD);
                ptr = ptr.add(1);
            }
        }
    }
}

unsafe fn scan_stack(sp: usize) -> u32 {
    unsafe extern "C" {
        static mut _stack_end: u32;
    }

    let stack_end_ptr = core::ptr::addr_of!(_stack_end) as usize;

    let mut ptr = stack_end_ptr as *const u32;
    let limit_ptr = sp as *const u32;

    while ptr < limit_ptr {
        let is_sentinel =
            unsafe { core::ptr::read_volatile(ptr) == 0xCDCD_CDCD };
        if !is_sentinel {
            break;
        }
        unsafe {
            ptr = ptr.add(1);
        }
    }

    let lowest_address = ptr as usize;
    if lowest_address < sp {
        (sp - lowest_address) as u32
    } else {
        0
    }
}

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

fn read_cycle_counter() -> u64 {
    unsafe {
        let dwt_cyccnt = 0xE000_1004 as *const u32;
        core::ptr::read_volatile(dwt_cyccnt) as u64
    }
}

struct CortexMExecutor;

impl ::control_rs_hil::executor::TestExecutor for CortexMExecutor {
    fn execute(&self, test_fn: fn()) -> (u64, u32) {
        cortex_m::interrupt::free(|_| {
            let sp_before = get_sp();
            unsafe {
                paint_stack(sp_before);
            }
            let start_cycles = read_cycle_counter();
            test_fn();
            let end_cycles = read_cycle_counter();
            let elapsed_stack = unsafe { scan_stack(sp_before) };
            let elapsed_cycles = end_cycles.saturating_sub(start_cycles);
            (elapsed_cycles, elapsed_stack)
        })
    }
}

// --- Main Setup Entrypoint ---

#[hil_setup]
#[allow(dead_code)]
fn setup() -> Context<TeensyComms, TeensyClock, CortexMExecutor> {
    let p = cortex_m::Peripherals::take().unwrap();
    let d = board::instances();

    // Enable DWT cycle counter
    enable_dwt_cycle_counter();

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

    let comms = TeensyComms {
        usb_class,
        usb_device,
        reader: FrameReader::new(),
        configured: false,
    };
    let timer = TeensyClock;
    let executor = CortexMExecutor;

    Context { comms, timer, executor }
}