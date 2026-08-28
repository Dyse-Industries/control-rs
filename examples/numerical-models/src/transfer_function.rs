//! Transfer-function native artifact and JSON equivalence test.

use control_rs::transfer_function::ArrayTransferFunction;
use serde_json::{Value, json};

/// Native transfer-function scenario matching `python/src/transfer_function.py`.
pub fn native_values() -> Value {
    let omega_c = 10.0;
    let w_c2 = omega_c * omega_c;
    let sqrt2_wc = core::f64::consts::SQRT_2 * omega_c;
    let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
        [w_c2],
        [w_c2, sqrt2_wc, 1.0],
    );
    let test_freqs = [0.1, 1.0, 10.0, 100.0];
    let mut h_re = [0.0_f64; 4];
    let mut h_im = [0.0_f64; 4];
    for (idx, &w) in test_freqs.iter().enumerate() {
        let resp = tf.eval_frequency(w);
        h_re[idx] = resp.re;
        h_im[idx] = resp.im;
    }

    let h1 = ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);
    let h2 = ArrayTransferFunction::<f64, 1, 2>::continuous([5.0], [5.0, 1.0]);
    let h_series = h1.series::<1, 2, 1, 3>(&h2);
    let mut num_ser = [0.0_f64; 1];
    let mut den_ser = [0.0_f64; 3];
    num_ser[0] = h_series.num_slice()[0];
    for i in 0..3 {
        den_ser[i] = h_series.den_slice()[i];
    }

    let tf_realize =
        ArrayTransferFunction::<f64, 2, 3>::continuous([2.0, 3.0], [4.0, 5.0, 1.0]);
    let ss = tf_realize
        .to_controllable_canonical_form::<2>()
        .expect("CCF");

    json!({
        "H_RE": h_re,
        "H_IM": h_im,
        "NUM_SER": num_ser,
        "DEN_SER": den_ser,
        "CCF_A": matrix_owned_to_rows(&ss.a()),
        "CCF_B": col_owned_to_rows(&ss.b()),
        "CCF_C": row_owned_to_cols(&ss.c()),
        "CCF_D": [[ss.d().get(0, 0).copied().unwrap()]],
    })
}

pub fn native_series_from_freqs(freqs: [f64; 4], mag: [f64; 4]) -> Value {
    json!({
        "bode_mag": {
            "x": freqs,
            "y": mag,
        }
    })
}

pub fn native_artifact() -> Value {
    let omega_c = 10.0;
    let w_c2 = omega_c * omega_c;
    let sqrt2_wc = core::f64::consts::SQRT_2 * omega_c;
    let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
        [w_c2],
        [w_c2, sqrt2_wc, 1.0],
    );
    let test_freqs = [0.1, 1.0, 10.0, 100.0];
    let mut mag = [0.0_f64; 4];
    for (idx, &w) in test_freqs.iter().enumerate() {
        let resp = tf.eval_frequency(w);
        mag[idx] = (resp.re * resp.re + resp.im * resp.im).sqrt();
    }
    let values = native_values();
    json!({
        "slug": "transfer_function",
        "source": "rust",
        "values": values,
        "series": native_series_from_freqs(test_freqs, mag),
    })
}

fn matrix_owned_to_rows(
    m: &control_rs::matrix::Owned<f64, 2, 2>,
) -> Value {
    let mut rows = Vec::with_capacity(2);
    for i in 0..2 {
        let mut row = Vec::with_capacity(2);
        for j in 0..2 {
            row.push(m.get(i, j).copied().unwrap_or(0.0));
        }
        rows.push(Value::Array(row.into_iter().map(Value::from).collect()));
    }
    Value::Array(rows)
}

fn col_owned_to_rows(m: &control_rs::matrix::Owned<f64, 2, 1>) -> Value {
    let mut rows = Vec::with_capacity(2);
    for i in 0..2 {
        rows.push(Value::from(vec![m.get(i, 0).copied().unwrap_or(0.0)]));
    }
    Value::Array(rows)
}

fn row_owned_to_cols(m: &control_rs::matrix::Owned<f64, 1, 2>) -> Value {
    let mut row = Vec::with_capacity(2);
    for j in 0..2 {
        row.push(m.get(0, j).copied().unwrap_or(0.0));
    }
    Value::Array(vec![Value::Array(row.into_iter().map(Value::from).collect())])
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use crate::assert_f64;

    const PYTHON_JSON: &str = "results/transfer_function/python.json";
    const NATIVE_JSON: &str = "results/transfer_function/native.json";

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    #[test]
    fn transfer_function_equiv() {
        let python = load_artifact(PYTHON_JSON, "python3 python/src/transfer_function.py");
        let native = load_artifact(NATIVE_JSON, "cargo run --example transfer_function");
        let py = &python["values"];
        let rs = &native["values"];
        for i in 0..4 {
            assert_f64(py["H_RE"][i].as_f64().unwrap(), rs["H_RE"][i].as_f64().unwrap(), "H.re");
            assert_f64(py["H_IM"][i].as_f64().unwrap(), rs["H_IM"][i].as_f64().unwrap(), "H.im");
        }
        for key in ["NUM_SER", "DEN_SER"] {
            let py_a = py[key].as_array().unwrap();
            let rs_a = rs[key].as_array().unwrap();
            for (j, (pv, rv)) in py_a.iter().zip(rs_a.iter()).enumerate() {
                assert_f64(pv.as_f64().unwrap(), rv.as_f64().unwrap(), &format!("{key}[{j}]"));
            }
        }
        for key in ["CCF_A", "CCF_B", "CCF_C", "CCF_D"] {
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
    }
}
