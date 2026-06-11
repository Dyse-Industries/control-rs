use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // 1. QEMU LM3S6965EVB Memory Map
    let mut memory_file = File::create(out_dir.join("memory.x")).unwrap();
    memory_file
        .write_all(
            b"
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 256K
  RAM : ORIGIN = 0x20000000, LENGTH = 64K
}
",
        )
        .unwrap();

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
