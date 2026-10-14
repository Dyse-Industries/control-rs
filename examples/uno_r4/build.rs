//! Build script for the Arduino Uno R4 ETS target example.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

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
