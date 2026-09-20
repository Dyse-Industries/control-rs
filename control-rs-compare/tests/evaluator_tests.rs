//! Unit tests for typed numerical comparison algorithms.

#![allow(clippy::unwrap_used)]

use control_rs_compare::compare::{ToleranceSpec, compare_float_arrays};

#[test]
fn test_abs_comparison() {
    let o = vec![1.0, 2.0, 3.0];
    let p_pass = vec![1.0001, 1.9999, 3.0005];
    let p_fail = vec![1.002, 2.0, 3.0];

    let tol = ToleranceSpec {
        method: "abs".to_string(),
        bound: 1e-3,
    };

    let res_pass = compare_float_arrays(&o, &p_pass, &tol);
    assert_eq!(res_pass.verdict, "pass");
    assert!(res_pass.observed <= 1e-3);

    let res_fail = compare_float_arrays(&o, &p_fail, &tol);
    assert_eq!(res_fail.verdict, "fail");
    assert!(res_fail.observed > 1e-3);
}

#[test]
fn test_rel_comparison() {
    let o = vec![100.0, 200.0];
    let p_pass = vec![100.5, 199.0];
    let p_fail = vec![105.0, 200.0];

    let tol = ToleranceSpec {
        method: "rel".to_string(),
        bound: 1e-2,
    };

    assert_eq!(compare_float_arrays(&o, &p_pass, &tol).verdict, "pass");
    assert_eq!(compare_float_arrays(&o, &p_fail, &tol).verdict, "fail");
}

#[test]
fn test_rms_comparison() {
    let o = vec![1.0, 2.0, 3.0, 4.0];
    let p = vec![1.0001, 2.0001, 3.0001, 4.0001];

    let tol = ToleranceSpec {
        method: "rms".to_string(),
        bound: 1e-3,
    };

    let res = compare_float_arrays(&o, &p, &tol);
    assert_eq!(res.verdict, "pass");
    assert!(res.observed < 1e-3);
}

#[test]
fn test_matrix_norm_comparison() {
    let o = vec![1.0, 0.0, 0.0, 1.0];
    let p_pass = vec![1.01, 0.01, 0.0, 0.99];
    let p_fail = vec![1.1, 0.0, 0.0, 1.0];

    let tol = ToleranceSpec {
        method: "matrix_norm".to_string(),
        bound: 0.05,
    };

    assert_eq!(compare_float_arrays(&o, &p_pass, &tol).verdict, "pass");
    assert_eq!(compare_float_arrays(&o, &p_fail, &tol).verdict, "fail");
}

#[test]
fn test_nan_detection() {
    let o = vec![1.0, f64::NAN, 3.0];
    let p = vec![1.0, 2.0, 3.0];

    let tol = ToleranceSpec::default();
    let res = compare_float_arrays(&o, &p, &tol);
    assert_eq!(res.verdict, "fail");
    assert!(res.details.as_deref().unwrap().contains("NaN"));
}
