#![no_std]
#![no_main]

use cortex_m_rt::entry;
use cortex_m_semihosting::{debug, hprintln};

use panic_semihosting as _;

use control_rs::hil_test::{
    AtomicU32Setting, ExecDescriptor, Setting, SuiteDescriptor,
};

// --- Test Definitions ---

static CONNECTION_TIMEOUT_MS: AtomicU32Setting =
    AtomicU32Setting::new("connection_timeout_ms", 1000);

fn math_addition_test() {
    let _ = hprintln!("Executing math addition test...");
    assert_eq!(2 + 2, 4);
}

fn math_subtraction_test() {
    let _ = hprintln!("Executing math subtraction test...");
    assert_eq!(5 - 3, 2);
}

// While defining these manually works for the demo, wrapping this boilerplate
// in a procedural macro will ultimately provide a much cleaner developer
// experience for the HIL test harness setup moving forward.
static EXECUTABLES: &[ExecDescriptor] = &[
    ExecDescriptor {
        name: "math_addition",
        test_fn: math_addition_test,
    },
    ExecDescriptor {
        name: "math_subtraction",
        test_fn: math_subtraction_test,
    },
];

static SETTINGS: &[&dyn Setting] = &[&CONNECTION_TIMEOUT_MS];

#[unsafe(link_section = ".hil_test_suites")]
#[used]
static SUITE_DESCRIPTOR: SuiteDescriptor = SuiteDescriptor {
    name: "qemu_math_suite",
    executables: EXECUTABLES,
    settings: SETTINGS,
};

unsafe extern "Rust" {
    static __hil_test_suites_start: u8;
    static __hil_test_suites_end: u8;
}

// --- Main Runner ---

#[entry]
fn main() -> ! {
    let _ = hprintln!("===============================================");
    let _ = hprintln!("  QEMU HIL Test Runner Started");
    let _ = hprintln!("===============================================");

    unsafe {
        let start =
            &__hil_test_suites_start as *const u8 as *const SuiteDescriptor;
        let end = &__hil_test_suites_end as *const u8 as *const SuiteDescriptor;

        let mut current = start;
        let mut total_suites = 0;
        let mut total_tests = 0;

        while current < end {
            let suite = &*current;
            let _ = hprintln!("--- Suite: {} ---", suite.name);

            for exec in suite.executables {
                let _ = hprintln!("  [RUNNING] {}", exec.name);
                (exec.test_fn)();
                let _ = hprintln!("  [PASSED]  {}", exec.name);
                total_tests += 1;
            }

            total_suites += 1;
            current = current.add(1);
        }

        let _ = hprintln!("===============================================");
        let _ = hprintln!("  HIL Test Runner Completed");
        let _ = hprintln!(
            "  Ran {} tests across {} suites.",
            total_tests,
            total_suites
        );
        let _ = hprintln!("===============================================");
    }

    // Tell QEMU to exit cleanly with a success code.
    // This allows your xtask CI script to detect when the run finishes.
    debug::exit(debug::EXIT_SUCCESS);
    loop {}
}
