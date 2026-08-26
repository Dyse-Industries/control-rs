//! Build script for the QEMU ETS example.
//!
//! This script plays a crucial role in preparing the target firmware for ETS testing:
//! 1. **Target Memory Mapping**: It dynamically generates a `memory.x` linker script
//!    specifying the flash and RAM boundaries depending on the target architecture (ARM Cortex-M vs RISC-V).
//! 2. **ETS Test Suite Registry**: It generates a custom linker script `ets_suites.x` containing
//!    the `.ets_test_suites` section. Test suites registered via `#[ets_suite]` place their
//!    `SuiteDescriptor` pointers in this section, enabling the server to discover and execute them at runtime.
//! 3. **Linker Configuration**: It registers the search directory for these generated scripts
//!    and instructs cargo/rustc to pass them to the linker.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    // Generate memory.x depending on the target architecture to define hardware boundaries.
    if target.starts_with("thumbv") {
        let mut memory_file = File::create(out.join("memory.x")).unwrap();
        memory_file
            .write_all(
                b"
MEMORY
{
  /* mps2-an500: ZBT SSRAM1 at 0x00000000 (4M), ZBT SSRAM 2&3 at 0x20000000 (4M) */
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM : ORIGIN = 0x20000000, LENGTH = 4M
}
",
            )
            .unwrap();
    } else if target.starts_with("risc") {
        let mut memory_file = File::create(out.join("memory.x")).unwrap();
        memory_file
            .write_all(
                b"
MEMORY
{
  /* QEMU virt board RAM starts at 0x80000000. We partition it into FLASH (rom) and RAM (ram) */
  FLASH : ORIGIN = 0x80000000, LENGTH = 16M
  RAM : ORIGIN = 0x81000000, LENGTH = 16M
}
REGION_ALIAS(\"REGION_TEXT\", FLASH);
REGION_ALIAS(\"REGION_RODATA\", FLASH);
REGION_ALIAS(\"REGION_DATA\", RAM);
REGION_ALIAS(\"REGION_BSS\", RAM);
REGION_ALIAS(\"REGION_HEAP\", RAM);
REGION_ALIAS(\"REGION_STACK\", RAM);

_hart_stack_size = 32K;

SECTIONS
{
  /DISCARD/ :
  {
    *(.eh_frame);
    *(.eh_frame_hdr);
  }
}
",
            )
            .unwrap();
    }

    // Generate the linker script containing our ETS test suites custom section.
    // This defines the symbols `__ets_test_suites_start` and `__ets_test_suites_end`
    // surrounding the `.ets_test_suites` section in flash memory.
    // The target-side ETS runner reads between these boundaries to dynamically discover
    // all test suites registered across the application.
    let mut ets_file = File::create(out.join("ets_suites.x")).unwrap();
    ets_file
        .write_all(
            b"
SECTIONS
{
  .ets_test_suites :
  {
    . = ALIGN(4);
    PROVIDE_HIDDEN (__ets_test_suites_start = .);
    KEEP (*(.ets_test_suites));
    . = ALIGN(4);
    PROVIDE_HIDDEN (__ets_test_suites_end = .);
  } > FLASH
}

/* Calculate the absolute RAM address of the stack start */
PROVIDE(_profiler_stack_start = _stack_start - _hart_stack_size);

",
        )
        .unwrap();

    // Advertise search directory containing memory.x and ets_suites.x
    println!("cargo:rustc-link-search={}", out.display());
    // Direct the linker to use the generated ets_suites.x script
    println!("cargo:rustc-link-arg=-Tets_suites.x");
    // Ensure build.rs is re-run if this script changes
    println!("cargo:rerun-if-changed=build.rs");
}
