//! Build script for control-rs-hil crate.
//! Generates custom linker scripts for the HIL test suite execution.

#![allow(clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 2. Custom Linker Section for HIL Test Suites
    let mut hil_file = File::create(out_dir.join("hil_suites.x")).unwrap();
    hil_file
        .write_all(
            b"
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

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
