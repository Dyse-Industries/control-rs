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

#[test]
fn gemm_frob_mismatch_is_an_error() {
    let py = json!({
        "slug": "matrix",
        "source": "python",
        "values": { "GEMM00": 1.0 },
        "metrics": {
            "gemm_frob": 1.0,
            "residual_ratio": 0.0,
            "residual_ratio_hilbert": 0.0,
            "kappa_hilbert": 1.0
        }
    });
    let rs = json!({
        "slug": "matrix",
        "source": "rust",
        "values": { "GEMM00": 1.0 },
        "metrics": {
            "gemm_frob": 2.0,
            "residual_ratio": 0.0,
            "residual_ratio_hilbert": 0.0,
            "kappa_hilbert": 1.0
        }
    });
    let err = compare_slug("matrix", &[py, rs]).unwrap_err();
    assert!(err.iter().any(|e| e.contains("gemm_frob")));
}
