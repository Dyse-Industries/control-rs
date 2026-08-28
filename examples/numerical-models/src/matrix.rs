//! Matrix native artifact and JSON equivalence test.

use control_rs::matrix::{LuDecomposition, Owned};
use serde_json::{Value, json};

use crate::solve_residual_ratio;

/// Native matrix scenario matching `python/src/matrix.py`.
pub fn native_values() -> Value {
    let m1: Owned<f64, 2, 2> = Owned::from_array([[1.0, 3.0], [2.0, 4.0]]);
    let m2: Owned<f64, 2, 2> = Owned::from_array([[5.0, 7.0], [6.0, 8.0]]);
    let a: Owned<f64, 3, 3> = Owned::from_array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 0.5],
        [-1.0, 4.0, -1.0],
    ]);
    let b: Owned<f64, 3, 1> = Owned::from_array([[1.0, -2.0, 0.0]]);

    let lu = LuDecomposition::decompose(a).expect("LU decompose");
    let mut x = b;
    lu.solve_mut(&mut x).expect("LU solve");
    let a_inv = lu.inverse().expect("inverse");

    let a_ref: Owned<f64, 3, 3> = Owned::from_array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 0.5],
        [-1.0, 4.0, -1.0],
    ]);
    let b_ref: Owned<f64, 3, 1> = Owned::from_array([[1.0, -2.0, 0.0]]);
    let residual_ratio = solve_residual_ratio(&a_ref, &x, &b_ref);

    json!({
        "SUM": owned_to_rows(&(&m1 + &m2)),
        "DIFF": owned_to_rows(&(&m2 - &m1)),
        "PROD": owned_to_rows(&(&m1 * &m2)),
        "TRANSPOSE": owned_to_rows(&m1.transpose()),
        "X": owned_to_rows(&x),
        "A_INV": owned_to_rows(&a_inv),
        "residual_ratio": residual_ratio,
    })
}

pub fn native_artifact() -> Value {
    json!({
        "slug": "matrix",
        "source": "rust",
        "values": native_values(),
        "series": {},
    })
}

fn owned_to_rows<const R: usize, const C: usize>(m: &Owned<f64, R, C>) -> Value
where
    control_rs::math::num_types::Const<R>: control_rs::math::num_types::Dim,
    control_rs::math::num_types::Const<C>: control_rs::math::num_types::Dim,
{
    let mut rows = Vec::with_capacity(R);
    for i in 0..R {
        let mut row = Vec::with_capacity(C);
        for j in 0..C {
            row.push(m.get(i, j).copied().unwrap_or(0.0));
        }
        rows.push(Value::Array(row.into_iter().map(Value::from).collect()));
    }
    Value::Array(rows)
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use crate::{SOLVE_RESIDUAL_TAU, assert_f64};

    const PYTHON_JSON: &str = "results/matrix/python.json";
    const NATIVE_JSON: &str = "results/matrix/native.json";

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    fn assert_matrix_close(python: &Value, native: &Value, key: &str) {
        let py = python.get(key).unwrap_or_else(|| panic!("python missing {key}"));
        let rs = native.get(key).unwrap_or_else(|| panic!("rust missing {key}"));
        let py_rows = py.as_array().expect("matrix rows");
        let rs_rows = rs.as_array().expect("matrix rows");
        assert_eq!(py_rows.len(), rs_rows.len(), "{key} row count");
        for (i, (pr, rr)) in py_rows.iter().zip(rs_rows.iter()).enumerate() {
            let pr = pr.as_array().expect("row");
            let rr = rr.as_array().expect("row");
            for (j, (pv, rv)) in pr.iter().zip(rr.iter()).enumerate() {
                assert_f64(
                    pv.as_f64().expect("f64"),
                    rv.as_f64().expect("f64"),
                    &format!("{key}[{i}][{j}]"),
                );
            }
        }
    }

    #[test]
    fn matrix_equiv() {
        let python = load_artifact(PYTHON_JSON, "python3 python/src/matrix.py");
        let native = load_artifact(NATIVE_JSON, "cargo run --example matrix");
        let py_vals = &python["values"];
        let rs_vals = &native["values"];
        for key in ["SUM", "DIFF", "PROD", "TRANSPOSE", "X", "A_INV"] {
            assert_matrix_close(py_vals, rs_vals, key);
        }
        let rr_py = py_vals["residual_ratio"].as_f64().expect("residual_ratio");
        let rr_rs = rs_vals["residual_ratio"].as_f64().expect("residual_ratio");
        assert_f64(rr_py, rr_rs, "residual_ratio");
        assert!(
            rr_rs < SOLVE_RESIDUAL_TAU,
            "solve residual ratio {rr_rs} exceeds {SOLVE_RESIDUAL_TAU}"
        );
    }
}
