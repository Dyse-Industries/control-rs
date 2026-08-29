//! Compare helpers. File-based V&V is `cargo run --bin validate -- suites/`.

use control_rs_numerical_model_examples::compare::compare_slug;
use serde_json::json;

#[test]
fn missing_python_is_an_error() {
    let rust = json!({
        "slug": "matrix",
        "source": "rust",
        "values": {},
        "metrics": {}
    });
    let err = compare_slug("matrix", &[rust]).unwrap_err();
    assert!(err.iter().any(|e| e.contains("python")));
}

#[test]
fn unknown_slug_is_an_error() {
    let py = json!({"slug": "nope", "source": "python", "values": {}, "metrics": {}});
    let rs =
        json!({"slug": "nope", "source": "rust", "values": {}, "metrics": {}});
    let err = compare_slug("nope", &[py, rs]).unwrap_err();
    assert!(err.iter().any(|e| e.contains("unknown slug")));
}
