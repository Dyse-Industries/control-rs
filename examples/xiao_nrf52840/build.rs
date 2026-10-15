//! Build script for Seeed Studio XIAO nRF52840 ETS target example.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // SoftDevice S140 7.3.0 + Adafruit/Seeed UF2 bootloader (stock XIAO nRF52840).
    // App starts after SoftDevice; RAM leaves the SoftDevice arena below 0x20006000.
    // A 0x1000/0x20000008 map overlaps SoftDevice flash/RAM and bricks bring-up.
    let mut mem_file = File::create(out.join("memory.x")).unwrap();
    mem_file
        .write_all(
            b"
MEMORY
{
  /* S140 7.3.0: app 0x27000..0xED000; UF2 bootloader owns 0xF4000.. */
  FLASH : ORIGIN = 0x00027000, LENGTH = 0xC6000
  RAM   : ORIGIN = 0x20006000, LENGTH = 0x3A000
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
