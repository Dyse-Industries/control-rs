//! File-based Python vs native JSON equivalence. Generators write; this reads.

use std::path::Path;

use control_rs_numerical_model_examples::{
    ABS_F32, ABS_F64, SOLVE_RESIDUAL_TAU, ZOH_AD_TAU,
};
use serde_json::Value;

fn load_artifact(rel: &str, hint: &str) -> Value {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
    let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
        panic!("missing artifact: {}\nrun: {}", path.display(), hint);
    });
    serde_json::from_str(&text).expect("parse json")
}

fn assert_num_close(py: f64, rs: f64, what: &str, abs: f64) {
    control_rs::assert_almost_eq!(py, rs, abs, "{}", what);
}

fn assert_values_close(python: &Value, native: &Value, path: &str, abs: f64) {
    match (python, native) {
        (Value::Number(a), Value::Number(b)) => {
            assert_num_close(
                a.as_f64().unwrap(),
                b.as_f64().unwrap(),
                path,
                abs,
            );
        }
        (Value::Array(a), Value::Array(b)) => {
            assert_eq!(a.len(), b.len(), "{path} length");
            for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
                assert_values_close(pa, pb, &format!("{path}[{i}]"), abs);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for (k, pv) in a {
                let rv = b
                    .get(k)
                    .unwrap_or_else(|| panic!("{path} missing rust {k}"));
                assert_values_close(pv, rv, &format!("{path}.{k}"), abs);
            }
        }
        (a, b) => panic!("{path}: type mismatch {a:?} vs {b:?}"),
    }
}

fn inf_norm_rows<const N: usize>(rows: &[[f64; N]; N]) -> f64 {
    let mut best = 0.0_f64;
    for row in rows {
        let mut s = 0.0_f64;
        for &v in row {
            s += v.abs();
        }
        if s > best {
            best = s;
        }
    }
    best
}

fn pair(slug: &str) -> (Value, Value) {
    let python = load_artifact(
        &format!("results/{slug}/python.json"),
        &format!("python3 python3/{slug}.py"),
    );
    let native = load_artifact(
        &format!("results/{slug}/native.json"),
        &format!("cargo run --bin {slug}"),
    );
    (python, native)
}

#[test]
fn matrix_equiv() {
    let (python, native) = pair("matrix");
    assert_values_close(
        &python["values"],
        &native["values"],
        "matrix",
        ABS_F64,
    );
    let rr = native["values"]["residual_ratio"].as_f64().unwrap();
    assert!(
        rr < SOLVE_RESIDUAL_TAU,
        "solve residual ratio {rr} exceeds {SOLVE_RESIDUAL_TAU}"
    );
}

#[test]
fn polynomial_equiv() {
    let (python, native) = pair("polynomial");
    assert_values_close(
        &python["values"],
        &native["values"],
        "polynomial",
        ABS_F64,
    );
}

#[test]
fn state_space_equiv() {
    let (python, native) = pair("state_space");
    assert_values_close(
        &python["values"],
        &native["values"],
        "state_space",
        ABS_F64,
    );
    let py = &python["values"];
    let rs = &native["values"];
    let mut diff_ad = [[0.0_f64; 2]; 2];
    let mut gold = [[0.0_f64; 2]; 2];
    for i in 0..2 {
        for j in 0..2 {
            let rust = rs["AD"][i][j].as_f64().unwrap();
            let scipy = py["AD"][i][j].as_f64().unwrap();
            diff_ad[i][j] = rust - scipy;
            gold[i][j] = scipy;
        }
    }
    let den = inf_norm_rows(&gold) * f64::EPSILON;
    let ratio = if den == 0.0 {
        0.0
    } else {
        inf_norm_rows(&diff_ad) / den
    };
    assert!(
        ratio < ZOH_AD_TAU,
        "ZOH A_d residual ratio {ratio} exceeds {ZOH_AD_TAU}"
    );
}

#[test]
fn transfer_function_equiv() {
    let (python, native) = pair("transfer_function");
    assert_values_close(
        &python["values"],
        &native["values"],
        "transfer_function",
        ABS_F64,
    );
}

#[test]
fn tensor_equiv() {
    let (python, native) = pair("tensor");
    assert_values_close(
        &python["values"],
        &native["values"],
        "tensor",
        f64::from(ABS_F32),
    );
}
