//! Build script for the Teensy 4 HIL target example.
//!
//! This script prepares the microcontroller's flash memory layout:
//! 1. **Stack Boundary Registration**: It provides the symbol `_stack_end` mapped to the
//!    linker-defined `__estack` variable, which the Cortex-M stack painting/profiling logic
//!    uses to determine the bottom boundary of stack space.
//! 2. **HIL Test Suite Registry**: It inserts a custom `.hil_test_suites` section into the
//!    flash region to collect all HIL test suites defined with the `#[hil_suite]` macro.
//! 3. **Linker Directives**: It tells rustc/cargo to load this generated script during linking.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate the linker script containing our HIL test suites custom section.
    // This allows the HIL test runner to dynamically iterate over all registered
    // tests at boot-up by scanning between `__hil_test_suites_start` and `__hil_test_suites_end`.
    let mut hil_file = File::create(out.join("hil_suites.x")).unwrap();
    hil_file
        .write_all(
            b"
PROVIDE(_stack_end = __estack);

SECTIONS
{
  .hil_test_suites :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__hil_test_suites_start = .);
    KEEP (*(.hil_test_suites));
    . = ALIGN(4);
    PROVIDE_HIDDEN (__hil_test_suites_end = .);
  } > FLASH
}
",
        )
        .unwrap();

    // Advertise search directory containing generated scripts
    println!("cargo:rustc-link-search={}", out.display());
    // Direct the linker to append the hil_suites.x layout configuration
    println!("cargo:rustc-link-arg=-Thil_suites.x");
    // Re-run build.rs if it has been modified
    println!("cargo:rerun-if-changed=build.rs");
}
