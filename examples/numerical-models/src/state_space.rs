//! State-space native artifact and JSON equivalence test.

use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;
use serde_json::{Value, json};

/// Native state-space scenario matching `python/src/state_space.py`.
pub fn native_values() -> Value {
    let a_c: Owned<f64, 2, 2> = Owned::from_array([[0.0, -4.0], [1.0, -0.8]]);
    let b_c: Owned<f64, 2, 1> = Owned::from_array([[0.0, 1.0]]);
    let c_c: Owned<f64, 1, 2> = Owned::from_array([[1.0], [0.0]]);
    let d_c: Owned<f64, 1, 1> = Owned::zero();
    let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);

    let x_test: Owned<f64, 2, 1> = Owned::from_array([[1.0, 0.5]]);
    let u_test: Owned<f64, 1, 1> = Owned::zero();
    let (x_dot, y_test) = sys_c.derivative(&x_test, &u_test);

    let dt = 0.05;
    let sys_d = sys_c.to_discrete_zoh(dt);
    let mut x_k: Owned<f64, 2, 1> = Owned::zero();
    let u_step: Owned<f64, 1, 1> = Owned::from_fn(|_, _| 1.0);
    let mut step_x1 = [0.0_f64; 20];
    let mut step_x2 = [0.0_f64; 20];
    let mut step_y = [0.0_f64; 20];
    for k in 0..20 {
        step_x1[k] = x_k.get(0, 0).copied().unwrap_or(0.0);
        step_x2[k] = x_k.get(1, 0).copied().unwrap_or(0.0);
        let (x_next, y_k) = sys_d.step(&x_k, &u_step);
        step_y[k] = y_k.get(0, 0).copied().unwrap_or(0.0);
        x_k = x_next;
    }

    let t: Owned<f64, 2, 2> = Owned::from_array([[1.0, 0.0], [1.0, 1.0]]);
    let sys_transformed = sys_d.similarity_transform(&t).expect("similarity");

    json!({
        "X_DOT": owned_col_to_rows(&x_dot),
        "Y_TEST": y_test.get(0, 0).copied().unwrap_or(0.0),
        "AD": owned_to_rows(&sys_d.a()),
        "BD": owned_col_to_rows(&sys_d.b()),
        "STEP_X1": step_x1,
        "STEP_X2": step_x2,
        "STEP_Y": step_y,
        "A_TILDE": owned_to_rows(&sys_transformed.a()),
        "B_TILDE": owned_col_to_rows(&sys_transformed.b()),
        "C_TILDE": owned_row_to_cols(&sys_transformed.c()),
    })
}

pub fn native_series(values: &Value) -> Value {
    let dt = 0.05;
    let t: Vec<f64> = (0..20).map(|k| f64::from(k) * dt).collect();
    json!({
        "step_y": { "x": t, "y": values["STEP_Y"] },
        "step_x1": { "x": t, "y": values["STEP_X1"] },
        "step_x2": { "x": t, "y": values["STEP_X2"] },
    })
}

pub fn native_artifact() -> Value {
    let values = native_values();
    json!({
        "slug": "state_space",
        "source": "rust",
        "values": values,
        "series": native_series(&values),
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

fn owned_col_to_rows<const N: usize>(m: &Owned<f64, N, 1>) -> Value
where
    control_rs::math::num_types::Const<N>: control_rs::math::num_types::Dim,
{
    let mut rows = Vec::with_capacity(N);
    for i in 0..N {
        rows.push(Value::from(vec![m.get(i, 0).copied().unwrap_or(0.0)]));
    }
    Value::Array(rows)
}

fn owned_row_to_cols<const N: usize>(m: &Owned<f64, 1, N>) -> Value
where
    control_rs::math::num_types::Const<N>: control_rs::math::num_types::Dim,
{
    let mut row = Vec::with_capacity(N);
    for j in 0..N {
        row.push(m.get(0, j).copied().unwrap_or(0.0));
    }
    Value::Array(vec![Value::Array(row.into_iter().map(Value::from).collect())])
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use crate::{ZOH_AD_TAU, assert_f64, inf_norm_rows};

    const PYTHON_JSON: &str = "results/state_space/python.json";
    const NATIVE_JSON: &str = "results/state_space/native.json";

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    fn assert_matrix_key(py: &Value, rs: &Value, key: &str) {
        let py_m = py[key].as_array().unwrap();
        let rs_m = rs[key].as_array().unwrap();
        for (i, (pr, rr)) in py_m.iter().zip(rs_m.iter()).enumerate() {
            let pr = pr.as_array().unwrap();
            let rr = rr.as_array().unwrap();
            for (j, (pv, rv)) in pr.iter().zip(rr.iter()).enumerate() {
                assert_f64(
                    pv.as_f64().unwrap(),
                    rv.as_f64().unwrap(),
                    &format!("{key}[{i}][{j}]"),
                );
            }
        }
    }

    #[test]
    fn state_space_equiv() {
        let python = load_artifact(PYTHON_JSON, "python3 python/src/state_space.py");
        let native = load_artifact(NATIVE_JSON, "cargo run --example state_space");
        let py = &python["values"];
        let rs = &native["values"];
        assert_matrix_key(py, rs, "X_DOT");
        assert_f64(
            py["Y_TEST"].as_f64().unwrap(),
            rs["Y_TEST"].as_f64().unwrap(),
            "Y_TEST",
        );
        assert_matrix_key(py, rs, "AD");
        assert_matrix_key(py, rs, "BD");
        for key in ["STEP_X1", "STEP_X2", "STEP_Y"] {
            let py_a = py[key].as_array().unwrap();
            let rs_a = rs[key].as_array().unwrap();
            for (k, (pv, rv)) in py_a.iter().zip(rs_a.iter()).enumerate() {
                assert_f64(
                    pv.as_f64().unwrap(),
                    rv.as_f64().unwrap(),
                    &format!("{key}[{k}]"),
                );
            }
        }
        assert_matrix_key(py, rs, "A_TILDE");
        assert_matrix_key(py, rs, "B_TILDE");
        assert_matrix_key(py, rs, "C_TILDE");

        let mut diff_ad = [[0.0_f64; 2]; 2];
        let ad_py = py["AD"].as_array().unwrap();
        let ad_rs = rs["AD"].as_array().unwrap();
        for i in 0..2 {
            for j in 0..2 {
                let rust = ad_rs[i][j].as_f64().unwrap();
                diff_ad[i][j] = rust - ad_py[i][j].as_f64().unwrap();
            }
        }
        let num = inf_norm_rows(&diff_ad);
        let mut gold = [[0.0_f64; 2]; 2];
        for i in 0..2 {
            for j in 0..2 {
                gold[i][j] = ad_py[i][j].as_f64().unwrap();
            }
        }
        let den = inf_norm_rows(&gold) * f64::EPSILON;
        let ratio = if den == 0.0 { 0.0 } else { num / den };
        assert!(
            ratio < ZOH_AD_TAU,
            "ZOH A_d residual ratio {ratio} exceeds {ZOH_AD_TAU}"
        );
    }
}
