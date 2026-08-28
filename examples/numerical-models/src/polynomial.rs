//! Polynomial native artifact and JSON equivalence test.

use control_rs::math::complex_num::Complex;
use control_rs::polynomial::ArrayPolynomial;
use serde_json::{Value, json};

/// Native polynomial scenario matching `python/src/polynomial.py`.
pub fn native_values() -> Value {
    let p = ArrayPolynomial::<f64, 5>::from_coefficients([2.0, -3.0, 4.0, 1.0, 0.0]);
    let x_test = 2.5;
    let val_complex = p.evaluate_complex(Complex::new(1.0, 2.0));
    let dp = p.derivative();
    let integ = p.integral(5.0);
    let p1 = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 2.0]);
    let p2 = ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 4.0]);
    let prod = p1.mul_poly::<2, 3>(&p2);
    let (quot, rem) = prod.div_rem::<2, 2, 1>(&p1).expect("div_rem");
    let p_monic = ArrayPolynomial::<f64, 3>::from_coefficients([-6.0, -5.0, 1.0]);
    let companion = p_monic.companion_matrix::<2>().expect("companion");

    let mut deriv = [0.0_f64; 5];
    let mut integ_c = [0.0_f64; 5];
    let mut prod_c = [0.0_f64; 3];
    let mut quot_c = [0.0_f64; 2];
    for i in 0..5 {
        deriv[i] = dp.get(i).copied().unwrap();
        integ_c[i] = integ.get(i).copied().unwrap();
    }
    for i in 0..3 {
        prod_c[i] = prod.get(i).copied().unwrap();
    }
    for i in 0..2 {
        quot_c[i] = quot.get(i).copied().unwrap();
    }

    let mut companion_rows = Vec::with_capacity(2);
    for i in 0..2 {
        let mut row = Vec::with_capacity(2);
        for j in 0..2 {
            row.push(companion.get(i, j).copied().unwrap_or(0.0));
        }
        companion_rows.push(Value::Array(row.into_iter().map(Value::from).collect()));
    }

    json!({
        "P_REAL": p.evaluate(x_test),
        "P_C_RE": val_complex.re,
        "P_C_IM": val_complex.im,
        "DERIV": deriv,
        "P_DERIV": dp.evaluate(x_test),
        "INTEG": integ_c,
        "P_INTEG": integ.evaluate(x_test),
        "PROD": prod_c,
        "QUOT": quot_c,
        "REM": rem.get(0).copied().unwrap(),
        "COMPANION": companion_rows,
    })
}

pub fn native_artifact() -> Value {
    json!({
        "slug": "polynomial",
        "source": "rust",
        "values": native_values(),
        "series": {},
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use crate::assert_f64;

    const PYTHON_JSON: &str = "results/polynomial/python.json";
    const NATIVE_JSON: &str = "results/polynomial/native.json";

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    #[test]
    fn polynomial_equiv() {
        let python = load_artifact(PYTHON_JSON, "python3 python/src/polynomial.py");
        let native = load_artifact(NATIVE_JSON, "cargo run --example polynomial");
        let py = &python["values"];
        let rs = &native["values"];
        assert_f64(py["P_REAL"].as_f64().unwrap(), rs["P_REAL"].as_f64().unwrap(), "P_REAL");
        assert_f64(py["P_C_RE"].as_f64().unwrap(), rs["P_C_RE"].as_f64().unwrap(), "P_C_RE");
        assert_f64(py["P_C_IM"].as_f64().unwrap(), rs["P_C_IM"].as_f64().unwrap(), "P_C_IM");
        assert_f64(py["P_DERIV"].as_f64().unwrap(), rs["P_DERIV"].as_f64().unwrap(), "P_DERIV");
        assert_f64(py["P_INTEG"].as_f64().unwrap(), rs["P_INTEG"].as_f64().unwrap(), "P_INTEG");
        assert_f64(py["REM"].as_f64().unwrap(), rs["REM"].as_f64().unwrap(), "REM");
        for (i, key) in ["DERIV", "INTEG", "PROD", "QUOT"].into_iter().enumerate() {
            let py_arr = py[key].as_array().unwrap();
            let rs_arr = rs[key].as_array().unwrap();
            for (j, (pv, rv)) in py_arr.iter().zip(rs_arr.iter()).enumerate() {
                assert_f64(pv.as_f64().unwrap(), rv.as_f64().unwrap(), &format!("{key}[{j}]"));
            }
            let _ = i;
        }
        let py_c = py["COMPANION"].as_array().unwrap();
        let rs_c = rs["COMPANION"].as_array().unwrap();
        for (i, (pr, rr)) in py_c.iter().zip(rs_c.iter()).enumerate() {
            for (j, (pv, rv)) in pr.as_array().unwrap().iter().zip(rr.as_array().unwrap().iter()).enumerate() {
                assert_f64(pv.as_f64().unwrap(), rv.as_f64().unwrap(), &format!("COMPANION[{i}][{j}]"));
            }
        }
    }
}
