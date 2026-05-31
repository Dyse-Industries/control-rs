//! The starter code slowly blinks the LED and sets up
//! USB logging. It periodically logs messages over USB.
//!
//! Despite targeting the Teensy 4.0, this starter code
//! should also work on the Teensy 4.1 and Teensy MicroMod.
//! You should eventually target your board! See inline notes.
//!
//! This template uses [RTIC v2](https://rtic.rs/2/book/en/)
//! for structuring the application.
//!
//! Verify the hil suite is in the binary with:
//! `cargo objdump --release -p teensy4-examples --target thumbv7em-none-eabihf -- -h`

#![no_std]
#![no_main]

use teensy4_panic as _;

// Import test suite components from control-rs
use control_rs::hil_test::{
    AtomicU32Setting, ExecDescriptor, Setting, SuiteDescriptor,
};

// Define a test setting
static CONNECTION_TIMEOUT_MS: AtomicU32Setting =
    AtomicU32Setting::new("connection_timeout_ms", 1000);

// Define a test function
fn dummy_test() {
    log::info!("Running dummy test...");
}

// Define the test executables
static EXECUTABLES: &[ExecDescriptor] = &[ExecDescriptor {
    name: "dummy_test",
    test_fn: dummy_test,
}];

// Define the settings
static SETTINGS: &[&dyn Setting] = &[&CONNECTION_TIMEOUT_MS];

// Define the suite descriptor and place it in the custom linker section
#[link_section = ".hil_test_suites"]
#[used]
static SUITE_DESCRIPTOR: SuiteDescriptor = SuiteDescriptor {
    name: "teensy4_example_suite",
    executables: EXECUTABLES,
    settings: SETTINGS,
};

extern "Rust" {
    static __hil_test_suites_start: u8;
    static __hil_test_suites_end: u8;
}

#[rtic::app(device = teensy4_bsp, peripherals = true, dispatchers = [KPP])]
mod app {
    use super::__hil_test_suites_end;
    use super::__hil_test_suites_start;
    use bsp::board;
    use control_rs::hil_test::SuiteDescriptor;
    use teensy4_bsp as bsp;

    use imxrt_log as logging;

    // If you're using a Teensy 4.1 or MicroMod, you should eventually
    // change 't40' to 't41' or micromod, respectively.
    use board::t40 as my_board;

    use rtic_monotonics::systick::{Systick, *};

    /// There are no resources shared across tasks.
    #[shared]
    struct Shared {}

    /// These resources are local to individual tasks.
    #[local]
    struct Local {
        /// The LED on pin 13.
        led: board::Led,
        /// A poller to control USB logging.
        poller: logging::Poller,
    }

    #[init]
    fn init(cx: init::Context) -> (Shared, Local) {
        let board::Resources {
            mut gpio2,
            pins,
            usb,
            ..
        } = my_board(cx.device);

        let led = board::led(&mut gpio2, pins.p13);
        let poller =
            logging::log::usbd(usb, logging::Interrupts::Enabled).unwrap();

        Systick::start(
            cx.core.SYST,
            board::ARM_FREQUENCY,
            rtic_monotonics::create_systick_token!(),
        );

        // Verify the test suite is in the binary
        unsafe {
            let start =
                &__hil_test_suites_start as *const u8 as *const SuiteDescriptor;
            let end =
                &__hil_test_suites_end as *const u8 as *const SuiteDescriptor;

            let mut current = start;
            while current < end {
                let suite = &*current;
                log::info!("Found test suite: {}", suite.name);
                current = current.add(1);
            }
        }

        blink::spawn().unwrap();
        (Shared {}, Local { led, poller })
    }

    #[task(local = [led])]
    async fn blink(cx: blink::Context) {
        let mut count = 0u32;
        loop {
            cx.local.led.toggle();
            Systick::delay(500.millis()).await;

            log::info!("Hello from your Teensy 4! The count is {count}");
            if count % 7 == 0 {
                log::warn!("Here's a warning at count {count}");
            }
            if count % 23 == 0 {
                log::error!("Here's an error at count {count}");
            }

            count = count.wrapping_add(1);
        }
    }

    #[task(binds = USB_OTG1, local = [poller])]
    fn log_over_usb(cx: log_over_usb::Context) {
        cx.local.poller.poll();
    }
}