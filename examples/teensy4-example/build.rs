use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate the linker script containing our HIL test suites custom section
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

    // Advertise search directory and tell the linker to use hil_suites.x
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Thil_suites.x");
    println!("cargo:rerun-if-changed=build.rs");
}
