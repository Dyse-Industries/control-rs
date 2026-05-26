use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    // Only generate linker script if auto-linker feature is enabled (which could be the default later)
    // or we are just generating it unconditionally for now based on the spec
    if env::var("CARGO_FEATURE_AUTO_LINKER").is_err() {
        return;
    }

    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());

    // Write our supplementary linker script
    let mut file = File::create(out.join("control_rs.x")).unwrap();
    file.write_all(b"
        SECTIONS {
            .control_rs_tests : ALIGN(4) {
                __control_rs_tests_start = .;
                KEEP(*(.control_rs_tests .control_rs_tests.*));
                __control_rs_tests_end = .;
            } > FLASH

            .control_rs_benchmarks : ALIGN(4) {
                __control_rs_benchmarks_start = .;
                KEEP(*(.control_rs_benchmarks .control_rs_benchmarks.*));
                __control_rs_benchmarks_end = .;
            } > FLASH
        }
    ").unwrap();

    // Tell Cargo to tell rustc where to find the file
    println!("cargo:rustc-link-search={}", out.display());

    // Tell Cargo to pass our script to the linker
    println!("cargo:rustc-link-arg=-Tcontrol_rs.x");
}