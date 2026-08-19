// Compiles csrc/gemv_c.c into a staticlib and links it, only when the
// "cffi" feature is on. A no-op build script (default: feature off) so
// `measure`'s cross-target staticlib build -- which always runs with
// --no-default-features and never enables "cffi" -- never shells out to a
// host C compiler for a bare-metal target it can't build for.
use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    if env::var_os("CARGO_FEATURE_CFFI").is_none() {
        return;
    }

    println!("cargo:rerun-if-changed=csrc/gemv_c.c");

    let out_dir = PathBuf::from(
        env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"),
    );
    let cc = env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let obj = out_dir.join("gemv_c.o");

    let status = Command::new(&cc)
        .args(["-O2", "-c", "csrc/gemv_c.c", "-o"])
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke C compiler {cc:?}: {e}"));
    assert!(status.success(), "{cc:?} failed to compile csrc/gemv_c.c");

    let lib = out_dir.join("libgemv_c.a");
    let ar = env::var("AR").unwrap_or_else(|_| "ar".to_string());
    let status = Command::new(&ar)
        .arg("crs")
        .arg(&lib)
        .arg(&obj)
        .status()
        .unwrap_or_else(|e| panic!("failed to invoke {ar:?}: {e}"));
    assert!(status.success(), "{ar:?} failed to archive gemv_c.o");

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=gemv_c");
}
