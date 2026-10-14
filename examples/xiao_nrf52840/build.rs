//! Build script for Seeed Studio XIAO nRF52840 ETS target example.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate memory.x matching the Adafruit / Seeed UF2 bootloader memory map
    let mut mem_file = File::create(out.join("memory.x")).unwrap();
    mem_file
        .write_all(
            b"
MEMORY
{
  /* nRF52840 with Adafruit / Seeed UF2 Bootloader offset */
  FLASH : ORIGIN = 0x00001000, LENGTH = 1020K
  RAM   : ORIGIN = 0x20000008, LENGTH = 255K
}
",
        )
        .unwrap();

    // Generate ets_suites.x linker script containing the .ets_test_suites custom section
    let mut ets_file = File::create(out.join("ets_suites.x")).unwrap();
    ets_file
        .write_all(
            b"
PROVIDE(_stack_end = __estack);

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
",
        )
        .unwrap();

    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Tets_suites.x");
    println!("cargo:rerun-if-changed=build.rs");
}
