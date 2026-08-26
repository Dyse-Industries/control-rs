//! Build script for control-rs-ets crate.
//! Generates custom linker scripts for the ETS test suite execution.

#![allow(clippy::unwrap_used)]

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 2. Custom Linker Section for ETS Test Suites
    let mut ets_file = File::create(out_dir.join("ets_suites.x")).unwrap();
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

    println!("cargo:rustc-link-search={}", out_dir.display());
    println!("cargo:rerun-if-changed=build.rs");
}
