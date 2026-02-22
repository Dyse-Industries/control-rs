fn main() {
    let out_dir = std::env::var("OUT_DIR").unwrap();
    codegen::generate_const_math(&out_dir, 64);
    println!("cargo:rerun-if-changed=build.rs");
}
