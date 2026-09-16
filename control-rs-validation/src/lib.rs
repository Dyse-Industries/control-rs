//! Cross-language oracle validation suites for `control-rs`.
//!
//! Each suite builds a JSON payload from deliberately ill-conditioned inputs,
//! writes it to `results/<suite>.rust.h5`, and leaves the comparison against
//! the NumPy/SciPy (or ngspice) oracle to `control-rs-ci`. Suites assert
//! cross-implementation *consistency* on problems where floating-point error
//! is observable; they do not restate the pedagogical examples in
//! `examples/`, and they do not measure latency. Latency lives in `benches/`.
//!
//! Layout:
//! - `src/<suite>.rs` builds the payload and writes the container.
//! - `src/bin/<suite>.rs` is the emitter the validate runner invokes.
//! - `tests/<suite>.rs` asserts the Rust-side invariants of the same inputs.
//! - `python3/<suite>_oracle.py` writes the peer container.
//! - `tolerances/*.toml` declares the write-time acceptance bounds.

// Validation is a harness, not library code: it aborts loudly on a bad input,
// converts indices to `f64` for reporting, and performs dense floating-point
// arithmetic throughout.
#![allow(
    clippy::arbitrary_source_item_ordering,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::expect_used,
    clippy::missing_panics_doc,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::doc_markdown,
    clippy::float_cmp,
    clippy::imprecise_flops,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::map_unwrap_or,
    clippy::missing_const_for_fn,
    clippy::multiple_crate_versions,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::or_fun_call,
    clippy::panic,
    clippy::similar_names,
    clippy::suboptimal_flops,
    clippy::suspicious_operation_groupings,
    clippy::too_many_lines,
    clippy::uninlined_format_args,
    clippy::unreadable_literal,
    clippy::unwrap_used
)]

pub mod buck_converter;
pub mod dc_motor;
pub mod matrix;
pub mod polynomial;
pub mod state_space;
pub mod tensor;
pub mod transfer_function;

use std::path::PathBuf;

/// Absolute path of this crate's `results/` directory, created if absent.
///
/// # Panics
/// Panics when the directory cannot be created.
#[must_use]
pub fn results_dir() -> PathBuf {
    let dir = std::env::var("CARGO_MANIFEST_DIR")
        .map_or_else(|_| PathBuf::from("."), PathBuf::from)
        .join("results");
    std::fs::create_dir_all(&dir).expect("create results directory");
    dir
}

/// Write `payload` to `results/<suite>.rust.h5`, gating `gated_paths`.
///
/// # Panics
/// Panics when the container cannot be written, so that the validate runner
/// observes a non-zero exit rather than a missing peer file.
pub fn write_rust_container(
    suite: &str,
    payload: &serde_json::Value,
    gated_paths: &[&str],
) {
    let path = results_dir().join(format!("{suite}.rust.h5"));
    control_rs_ci::H5Container::write_variant_file(&path, payload, gated_paths)
        .unwrap_or_else(|e| {
            panic!("failed to write {}: {e}", path.display());
        });
    println!("wrote {}", path.display());
}
