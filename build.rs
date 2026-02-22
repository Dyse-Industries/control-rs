fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    let is_formatting = std::env::var("RUSTC_WORKSPACE_WRAPPER")
        .map(|val| {
            val.contains("clippy")
                || val.contains("rust-analyzer")
                || val.contains("rustfmt")
        })
        .unwrap_or(false);

    // If Clippy, rust-analyzer, or rust fmt is running, only generate up to 4 to keep it lightning fast.
    // Otherwise, generate the full 64 for compilation.
    let max_dim = if is_formatting { 4 } else { 64 };

    control_rs_codegen::generate_const_math(&out_dir, max_dim);

    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC_WORKSPACE_WRAPPER");
}
