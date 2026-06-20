use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let target = env::var("TARGET").unwrap();

    // Generate memory.x depending on the target architecture
    if target.starts_with("thumbv") {
        let mut memory_file = File::create(out.join("memory.x")).unwrap();
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
    } else if target.starts_with("riscv32") {
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

    // Generate the linker script containing our HIL test suites custom section
    let mut hil_file = File::create(out.join("hil_suites.x")).unwrap();
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

    // Advertise search directory and tell the linker to use hil_suites.x
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Thil_suites.x");
    println!("cargo:rerun-if-changed=build.rs");
}
