use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate memory.x for QEMU virt board
    let mut memory_file = File::create(out.join("memory.x")).unwrap();
    memory_file
        .write_all(
            b"
MEMORY
{
  FLASH : ORIGIN = 0x80000000, LENGTH = 16M
  RAM : ORIGIN = 0x81000000, LENGTH = 16M
}
REGION_ALIAS(\"REGION_TEXT\", FLASH);
REGION_ALIAS(\"REGION_RODATA\", FLASH);
REGION_ALIAS(\"REGION_DATA\", RAM);
REGION_ALIAS(\"REGION_BSS\", RAM);
REGION_ALIAS(\"REGION_HEAP\", RAM);
REGION_ALIAS(\"REGION_STACK\", RAM);

_hart_stack_size = 256K;

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

    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Tmemory.x");
    println!("cargo:rustc-link-arg=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=c_src");

    // Optional: try compiling C sources if a RISC-V C compiler is present on PATH
    let compilers = ["riscv32-unknown-elf-gcc", "riscv-none-elf-gcc", "riscv64-unknown-elf-gcc"];
    let mut found_compiler = None;
    for cc in &compilers {
        if Command::new(cc).arg("--version").output().is_ok() {
            found_compiler = Some(*cc);
            break;
        }
    }

    if let Some(cc) = found_compiler {
        let c_files = [
            "c_src/Source/riscv_mat_mult_f32.c",
            "c_src/Source/riscv_mat_vec_mult_f32.c",
            "c_src/Source/riscv_dot_prod_f32.c",
            "c_src/Source/riscv_cmplx_dot_prod_f32.c",
            "c_src/Source/riscv_scale_f32.c",
            "c_src/Source/riscv_mat_cholesky_f32.c",
            "c_src/Source/riscv_mat_solve_upper_triangular_f32.c",
            "c_src/Source/riscv_mat_init_f32.c",
        ];

        for c_file in &c_files {
            let stem = std::path::Path::new(c_file).file_stem().unwrap().to_str().unwrap();
            let obj = out.join(format!("{}.o", stem));
            let status = Command::new(cc)
                .arg("-march=rv32imac")
                .arg("-mabi=ilp32")
                .arg("-ffreestanding")
                .arg("-I")
                .arg("c_src/Include")
                .arg("-c")
                .arg(c_file)
                .arg("-o")
                .arg(&obj)
                .status();
            if let Ok(s) = status {
                if s.success() {
                    println!("cargo:rustc-link-arg={}", obj.display());
                }
            }
        }
    }
}
