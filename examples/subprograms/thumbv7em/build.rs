use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Generate memory.x for QEMU mps2-an500 target
    let mut memory_file = File::create(out.join("memory.x")).unwrap();
    memory_file
        .write_all(
            b"
MEMORY
{
  FLASH : ORIGIN = 0x00000000, LENGTH = 4M
  RAM : ORIGIN = 0x20000000, LENGTH = 4M
}
",
        )
        .unwrap();

    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rustc-link-arg=-Tlink.x");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=c_src");

    let c_files = [
        "c_src/Source/arm_mat_mult_f32.c",
        "c_src/Source/arm_mat_vec_mult_f32.c",
        "c_src/Source/arm_dot_prod_f32.c",
        "c_src/Source/arm_cmplx_dot_prod_f32.c",
        "c_src/Source/arm_scale_f32.c",
        "c_src/Source/arm_mat_cholesky_f32.c",
        "c_src/Source/arm_mat_solve_upper_triangular_f32.c",
        "c_src/Source/arm_mat_init_f32.c",
    ];

    for c_file in &c_files {
        let stem = std::path::Path::new(c_file).file_stem().unwrap().to_str().unwrap();
        let obj = out.join(format!("{}.o", stem));
        let status = Command::new("clang")
            .arg("--target=thumbv7em-none-eabihf")
            .arg("-ffreestanding")
            .arg("-mthumb")
            .arg("-mcpu=cortex-m7")
            .arg("-mfpu=fpv5-d16")
            .arg("-mfloat-abi=hard")
            .arg("-I")
            .arg("c_src/Include")
            .arg("-c")
            .arg(c_file)
            .arg("-o")
            .arg(&obj)
            .status()
            .expect("Failed to run clang");
        assert!(status.success(), "Clang failed to compile {}", c_file);
        println!("cargo:rustc-link-arg={}", obj.display());
    }
}
