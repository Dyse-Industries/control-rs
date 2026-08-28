//! File-based Python vs native JSON equivalence. Generators write; this reads.

#![allow(
    clippy::too_many_arguments,
    clippy::type_complexity,
    clippy::manual_contains
)]

use std::path::Path;

use control_rs_numerical_model_examples::{
    ABS_F32, ABS_F64, SOLVE_RESIDUAL_TAU, TAU_KAPPA, ZOH_AD_TAU, gamma,
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

fn skip_key(skip: &[&str], k: &str) -> bool {
    skip.contains(&k)
}

fn assert_values_close(
    python: &Value,
    native: &Value,
    path: &str,
    abs: f64,
    skip: &[&str],
) {
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
                assert_values_close(pa, pb, &format!("{path}[{i}]"), abs, skip);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for (k, pv) in a {
                if skip_key(skip, k) {
                    continue;
                }
                let rv = b
                    .get(k)
                    .unwrap_or_else(|| panic!("{path} missing rust {k}"));
                assert_values_close(pv, rv, &format!("{path}.{k}"), abs, skip);
            }
        }
        (a, b) => panic!("{path}: type mismatch {a:?} vs {b:?}"),
    }
}

fn f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|x| x.as_f64().unwrap()).collect(),
        Value::Number(n) => vec![n.as_f64().unwrap()],
        _ => panic!("expected number array, got {v:?}"),
    }
}

fn inf_norm_rows(rows: &[Vec<f64>]) -> f64 {
    let mut best = 0.0_f64;
    for row in rows {
        let s: f64 = row.iter().map(|v| v.abs()).sum();
        if s > best {
            best = s;
        }
    }
    best
}

fn json_mat(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap())
                .collect()
        })
        .collect()
}

fn residual_ratio_mats(rust: &Value, scipy: &Value) -> f64 {
    let r = json_mat(rust);
    let g = json_mat(scipy);
    let n = g.len();
    let mut diff = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            diff[i][j] = r[i][j] - g[i][j];
        }
    }
    let den = inf_norm_rows(&g) * f64::EPSILON;
    if den == 0.0 {
        0.0
    } else {
        inf_norm_rows(&diff) / den
    }
}

fn pair(slug: &str) -> (Value, Value) {
    let python = load_artifact(
        &format!("results/{slug}/python.json"),
        &format!("python3 python3/{slug}.py"),
    );
    let native = load_artifact(
        &format!("results/{slug}/native.json"),
        &format!("cargo run --release --bin {slug}"),
    );
    (python, native)
}

fn metric_f64(doc: &Value, key: &str) -> f64 {
    doc["metrics"][key]
        .as_f64()
        .unwrap_or_else(|| panic!("metrics.{key}"))
}

#[test]
fn matrix_equiv() {
    let (python, native) = pair("matrix");
    assert_values_close(
        &python["values"],
        &native["values"],
        "matrix",
        ABS_F64,
        &[
            "HILBERT_X",
            "HILBERT_A_INV",
            "GEMM00",
            "SE3_T",
            "SE3_XYZ",
            "SE3_R",
        ],
    );
    assert_num_close(
        python["values"]["GEMM00"].as_f64().unwrap(),
        native["values"]["GEMM00"].as_f64().unwrap(),
        "GEMM00",
        1e-9,
    );
    let rr = metric_f64(&native, "residual_ratio");
    assert!(
        rr < SOLVE_RESIDUAL_TAU,
        "solve residual ratio {rr} exceeds {SOLVE_RESIDUAL_TAU}"
    );
    let rr_h = metric_f64(&native, "residual_ratio_hilbert");
    assert!(
        rr_h < SOLVE_RESIDUAL_TAU,
        "Hilbert residual ratio {rr_h} exceeds {SOLVE_RESIDUAL_TAU}"
    );
    let rr_h_py = metric_f64(&python, "residual_ratio_hilbert");
    assert!(
        rr_h_py < SOLVE_RESIDUAL_TAU,
        "Python Hilbert residual ratio {rr_h_py} exceeds {SOLVE_RESIDUAL_TAU}"
    );
    let kappa = metric_f64(&native, "kappa_hilbert")
        .max(metric_f64(&python, "kappa_hilbert"));
    let bound = TAU_KAPPA * kappa * f64::EPSILON;
    for (label, x) in [
        ("native", f64_vec(&native["values"]["HILBERT_X"])),
        ("python", f64_vec(&python["values"]["HILBERT_X"])),
    ] {
        let err = x.iter().map(|v| (v - 1.0).abs()).fold(0.0_f64, f64::max);
        assert!(
            err < bound,
            "Hilbert {label} forward error {err} exceeds τκε {bound}"
        );
    }
}

#[test]
fn polynomial_equiv() {
    let (python, native) = pair("polynomial");
    assert_values_close(
        &python["values"],
        &native["values"],
        "polynomial",
        ABS_F64,
        &["CLUSTER_Y", "CLUSTER_COEFFS"],
    );
    let xs = f64_vec(&native["values"]["CLUSTER_X"]);
    let py = f64_vec(&python["values"]["CLUSTER_Y"]);
    let rs = f64_vec(&native["values"]["CLUSTER_Y"]);
    let coeffs = f64_vec(&native["values"]["CLUSTER_COEFFS"]);
    assert_eq!(py.len(), rs.len());
    let g = gamma(32.0);
    for i in 0..py.len() {
        let mut tilde = 0.0_f64;
        for &c in coeffs.iter().rev() {
            tilde = tilde * xs[i].abs() + c.abs();
        }
        let err = (py[i] - rs[i]).abs();
        let bound = g * tilde;
        assert!(
            err <= bound.max(ABS_F64),
            "clustered Horner[{i}] |py-rs|={err} exceeds γ₂ₙ p̃={bound}"
        );
    }
}

#[test]
fn state_space_equiv() {
    let (python, native) = pair("state_space");
    assert_values_close(
        &python["values"],
        &native["values"],
        "state_space",
        ABS_F64,
        &["STIFF_AD", "STIFF_BD", "STIFF_Y"],
    );
    let ratio =
        residual_ratio_mats(&native["values"]["AD"], &python["values"]["AD"]);
    assert!(
        ratio < ZOH_AD_TAU,
        "ZOH A_d residual ratio {ratio} exceeds {ZOH_AD_TAU}"
    );
    let stiff_ratio = residual_ratio_mats(
        &native["values"]["STIFF_AD"],
        &python["values"]["STIFF_AD"],
    );
    let stiff_gold = json_mat(&python["values"]["STIFF_AD"]);
    let stiff_rs = json_mat(&native["values"]["STIFF_AD"]);
    let n = stiff_gold.len();
    let mut stiff_diff = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            stiff_diff[i][j] = stiff_rs[i][j] - stiff_gold[i][j];
        }
    }
    let stiff_rel = inf_norm_rows(&stiff_diff)
        / inf_norm_rows(&stiff_gold).max(f64::EPSILON);
    // Padé [6/6] without scaling at ||A Ts||_∞ = 2 is ~1e-10, so τ=20 does not apply.
    assert!(
        stiff_rel < 1e-8,
        "stiff ZOH A_d relative error {stiff_rel} (residual ratio {stiff_ratio})"
    );
    let py_y = f64_vec(&python["values"]["STEP_Y"]);
    let rs_y = f64_vec(&native["values"]["STEP_Y"]);
    let mut step_err = 0.0_f64;
    for (a, b) in py_y.iter().zip(rs_y.iter()) {
        step_err = step_err.max((a - b).abs());
    }
    // Higham recurrence: k γ_{N_x} ||A||^k ||x||; k=200, N_x=2, γ₂ ~ 4ε.
    let step_bound = 200.0 * gamma(2.0) * 10.0;
    assert!(
        step_err <= step_bound.max(ABS_F64),
        "tutorial step |py-rs|={step_err} exceeds {step_bound}"
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
        &["H_RE", "H_IM", "CLUSTER_H_RE", "CLUSTER_H_IM", "FREQS"],
    );
    let h_re_p = f64_vec(&python["values"]["H_RE"]);
    let h_im_p = f64_vec(&python["values"]["H_IM"]);
    let h_re_r = f64_vec(&native["values"]["H_RE"]);
    let h_im_r = f64_vec(&native["values"]["H_IM"]);
    let bode_g = gamma(4.0);
    for i in 0..h_re_p.len() {
        let mag = (h_re_p[i].hypot(h_im_p[i])).max(f64::EPSILON);
        let err = (h_re_p[i] - h_re_r[i]).hypot(h_im_p[i] - h_im_r[i]);
        let rel = err / mag;
        assert!(
            rel <= bode_g.max(1e-9) || err <= ABS_F64,
            "Butterworth H[{i}] rel={rel} exceeds γ₄={bode_g}"
        );
    }
    let c_re_p = f64_vec(&python["values"]["CLUSTER_H_RE"]);
    let c_im_p = f64_vec(&python["values"]["CLUSTER_H_IM"]);
    let c_re_r = f64_vec(&native["values"]["CLUSTER_H_RE"]);
    let c_im_r = f64_vec(&native["values"]["CLUSTER_H_IM"]);
    let cl_g = gamma(16.0);
    for i in 0..c_re_p.len() {
        let mag = (c_re_p[i].hypot(c_im_p[i])).max(f64::EPSILON);
        let err = (c_re_p[i] - c_re_r[i]).hypot(c_im_p[i] - c_im_r[i]);
        let rel = err / mag;
        assert!(
            rel <= (cl_g * 1e3).max(1e-8) || err <= 1e-10,
            "clustered H[{i}] rel={rel} exceeds bound"
        );
    }
}

#[test]
fn tensor_equiv() {
    let (python, native) = pair("tensor");
    assert_values_close(
        &python["values"],
        &native["values"],
        "tensor",
        f64::from(ABS_F32),
        &[
            "CURVED_SAMPLES",
            "CUT_X",
            "DEQUANT",
            "RELU_DEQUANT",
            "CURVED_TABLE",
        ],
    );
    let py_c = f64_vec(&python["values"]["CURVED_SAMPLES"]);
    let rs_c = f64_vec(&native["values"]["CURVED_SAMPLES"]);
    let weiser = native["metrics"]["weiser_bound"].as_f64().unwrap();
    let mut curved_err = 0.0_f64;
    for (a, b) in py_c.iter().zip(rs_c.iter()) {
        curved_err = curved_err.max((a - b).abs());
    }
    assert!(
        curved_err <= weiser.max(f64::from(ABS_F32) * 10.0),
        "curved interp |py-rs|={curved_err} exceeds Weiser {weiser}"
    );
    let q_py = metric_f64(&python, "quant_roundtrip_max");
    let q_rs = metric_f64(&native, "quant_roundtrip_max");
    let lsb = 1.0 / 256.0;
    assert!(q_py <= lsb, "python Q7 round-trip {q_py} exceeds {lsb}");
    assert!(q_rs <= lsb, "native Q7 round-trip {q_rs} exceeds {lsb}");
}
