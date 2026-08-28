//! JSON artifact emitters for numerical-model V&V.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde_json::Value;

/// Crate root (`examples/numerical-models/`).
pub fn manifest_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// Write a JSON artifact under the crate root (creates parent dirs).
pub fn write_artifact(rel: impl AsRef<Path>, doc: &Value) {
    let path = manifest_dir().join(rel.as_ref());
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).expect("create results dir");
    }
    let text = serde_json::to_string_pretty(doc).expect("serialize json");
    std::fs::write(&path, format!("{text}\n")).expect("write artifact");
    eprintln!("wrote {}", path.display());
}

/// Run a Python oracle script from `python/src/`.
pub fn run_python(script: &str) {
    let script_path = manifest_dir().join("python/src").join(script);
    let status = Command::new("python3")
        .arg(&script_path)
        .current_dir(manifest_dir())
        .status()
        .unwrap_or_else(|e| panic!("failed to run python3 {script_path:?}: {e}"));
    if !status.success() {
        panic!(
            "python oracle failed: {}\nensure: pip install -r python/requirements.txt",
            script_path.display()
        );
    }
}

const PY_ORACLES: &[&str] = &[
    "matrix.py",
    "polynomial.py",
    "state_space.py",
    "transfer_function.py",
    "tensor.py",
    "host_scale.py",
];

/// Run all Python oracle scripts.
pub fn emit_python_all() {
    for script in PY_ORACLES {
        run_python(script);
    }
}

/// Write native JSON artifacts for the five model slugs.
pub fn emit_native_slugs() {
    use crate::{
        matrix, polynomial, state_space, tensor, transfer_function,
    };

    write_artifact("results/matrix/native.json", &matrix::native_artifact());
    write_artifact(
        "results/polynomial/native.json",
        &polynomial::native_artifact(),
    );
    write_artifact(
        "results/state_space/native.json",
        &state_space::native_artifact(),
    );
    write_artifact(
        "results/transfer_function/native.json",
        &transfer_function::native_artifact(),
    );
    write_artifact("results/tensor/native.json", &tensor::native_artifact());
}

/// Emit every Python and native JSON artifact (host-scale included).
pub fn emit_all() {
    emit_python_all();
    emit_native_slugs();
    crate::host_scale_emit::emit_all();
}
