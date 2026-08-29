//! §6.3 Python-vs-other-source comparison of result JSON documents.

use serde_json::Value;

use crate::{
    ABS_F32, ABS_F64, SOLVE_RESIDUAL_TAU, TAU_KAPPA, ZOH_AD_TAU, gamma,
};

/// Compare every non-python artifact against the python oracle for `slug`.
pub fn compare_slug(
    slug: &str,
    artifacts: &[Value],
) -> Result<(), Vec<String>> {
    let Some(python) = artifacts
        .iter()
        .find(|a| a["source"].as_str() == Some("python"))
    else {
        return Err(vec![format!("{slug}: missing python artifact")]);
    };

    let mut errs = Vec::new();
    for other in artifacts {
        let src = other["source"].as_str().unwrap_or("?");
        if src == "python" {
            continue;
        }
        if let Err(mut e) = compare_pair(slug, python, other, src) {
            errs.append(&mut e);
        }
    }

    if artifacts.len() < 2 {
        errs.push(format!("{slug}: need at least two artifacts to compare"));
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

fn compare_pair(
    slug: &str,
    python: &Value,
    native: &Value,
    src: &str,
) -> Result<(), Vec<String>> {
    let mut errs = Vec::new();
    match slug {
        "matrix" => matrix_equiv(python, native, src, &mut errs),
        "polynomial" => polynomial_equiv(python, native, src, &mut errs),
        "state_space" => state_space_equiv(python, native, src, &mut errs),
        "transfer_function" => {
            transfer_function_equiv(python, native, src, &mut errs)
        }
        "tensor" => tensor_equiv(python, native, src, &mut errs),
        other => errs.push(format!("unknown slug `{other}`")),
    }

    if errs.is_empty() { Ok(()) } else { Err(errs) }
}

fn check_tolerance(
    py: f64,
    rs: f64,
    what: &str,
    abs: f64,
    errs: &mut Vec<String>,
) {
    if (py - rs).abs() > abs {
        errs.push(format!("{what}: |{py} - {rs}| > {abs}"));
    }
}

fn values_close(
    python: &Value,
    native: &Value,
    path: &str,
    abs: f64,
    skip: &[&str],
    errs: &mut Vec<String>,
) {
    match (python, native) {
        (Value::Number(a), Value::Number(b)) => {
            if let (Some(fa), Some(fb)) = (a.as_f64(), b.as_f64()) {
                check_tolerance(fa, fb, path, abs, errs);
            }
        }
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                errs.push(format!("{path} length {} vs {}", a.len(), b.len()));
                return;
            }
            for (i, (pa, pb)) in a.iter().zip(b.iter()).enumerate() {
                values_close(pa, pb, &format!("{path}[{i}]"), abs, skip, errs);
            }
        }
        (Value::Object(a), Value::Object(b)) => {
            for (k, pv) in a.iter().filter(|(k, _)| !skip.contains(&k.as_str()))
            {
                match b.get(k) {
                    Some(rv) => values_close(
                        pv,
                        rv,
                        &format!("{path}.{k}"),
                        abs,
                        skip,
                        errs,
                    ),
                    None => errs.push(format!("{path} missing {k}")),
                }
            }
        }
        (a, b) => errs.push(format!("{path}: type mismatch {a:?} vs {b:?}")),
    }
}

fn f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().filter_map(Value::as_f64).collect(),
        Value::Number(n) => n.as_f64().map(|f| vec![f]).unwrap_or_default(),
        _ => Vec::new(),
    }
}

fn inf_norm_rows(rows: &[Vec<f64>]) -> f64 {
    rows.iter()
        .map(|row| row.iter().map(|v| v.abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn json_mat(v: &Value) -> Vec<Vec<f64>> {
    v.as_array()
        .map(|rows| rows.iter().map(f64_vec).collect())
        .unwrap_or_default()
}

fn residual_ratio_mats(rust: &Value, scipy: &Value) -> f64 {
    let r = json_mat(rust);
    let g = json_mat(scipy);
    if r.is_empty() || g.is_empty() || r.len() != g.len() {
        return f64::INFINITY;
    }

    let diff: Vec<Vec<f64>> = g
        .iter()
        .zip(r.iter())
        .map(|(grow, rrow)| {
            grow.iter()
                .zip(rrow.iter())
                .map(|(gv, rv)| rv - gv)
                .collect()
        })
        .collect();

    let den = inf_norm_rows(&g) * f64::EPSILON;
    if den == 0.0 {
        0.0
    } else {
        inf_norm_rows(&diff) / den
    }
}

fn metric_f64(doc: &Value, key: &str) -> Option<f64> {
    doc.get("metrics")?.get(key)?.as_f64()
}

fn matrix_equiv(
    python: &Value,
    native: &Value,
    src: &str,
    errs: &mut Vec<String>,
) {
    values_close(
        &python["values"],
        &native["values"],
        &format!("matrix/{src}"),
        ABS_F64,
        &[
            "HILBERT_X",
            "HILBERT_A_INV",
            "GEMM00",
            "SE3_T",
            "SE3_XYZ",
            "SE3_R",
        ],
        errs,
    );

    if let (Some(p), Some(r)) = (
        python["values"]["GEMM00"].as_f64(),
        native["values"]["GEMM00"].as_f64(),
    ) {
        check_tolerance(p, r, &format!("matrix/{src} GEMM00"), 1e-9, errs);
    }

    for (label, doc) in [("native", native), ("python", python)] {
        if let Some(rr) = metric_f64(doc, "residual_ratio")
            && rr >= SOLVE_RESIDUAL_TAU
        {
            errs.push(format!("{src} {label} solve residual ratio {rr}"));
        }
        if let Some(rr) = metric_f64(doc, "residual_ratio_hilbert")
            && rr >= SOLVE_RESIDUAL_TAU
        {
            errs.push(format!("{src} {label} Hilbert residual ratio {rr}"));
        }
    }

    let kappa = metric_f64(native, "kappa_hilbert")
        .unwrap_or(0.0)
        .max(metric_f64(python, "kappa_hilbert").unwrap_or(0.0));
    let bound = TAU_KAPPA * kappa * f64::EPSILON;

    for (label, doc) in [(src, native), ("python", python)] {
        let x = f64_vec(&doc["values"]["HILBERT_X"]);
        let err = x.iter().map(|v| (v - 1.0).abs()).fold(0.0_f64, f64::max);
        if err >= bound {
            errs.push(format!(
                "Hilbert {label} forward error {err} exceeds {bound}"
            ));
        }
    }
}

fn polynomial_equiv(
    python: &Value,
    native: &Value,
    src: &str,
    errs: &mut Vec<String>,
) {
    values_close(
        &python["values"],
        &native["values"],
        &format!("polynomial/{src}"),
        ABS_F64,
        &["CLUSTER_Y", "CLUSTER_COEFFS"],
        errs,
    );

    let xs = f64_vec(&native["values"]["CLUSTER_X"]);
    let py = f64_vec(&python["values"]["CLUSTER_Y"]);
    let rs = f64_vec(&native["values"]["CLUSTER_Y"]);
    let coeffs = f64_vec(&native["values"]["CLUSTER_COEFFS"]);

    if py.len() != rs.len() {
        errs.push(format!("polynomial/{src} CLUSTER_Y length"));
        return;
    }

    let g = gamma(32.0);
    for i in 0..py.len() {
        let tilde = coeffs.iter().rev().fold(0.0_f64, |acc, &c| {
            acc * xs.get(i).copied().unwrap_or(0.0).abs() + c.abs()
        });

        let err = (py[i] - rs[i]).abs();
        let bound = g * tilde;
        if err > bound.max(ABS_F64) {
            errs.push(format!(
                "clustered Horner[{i}] |py-{src}|={err} exceeds {bound}"
            ));
        }
    }
}

fn state_space_equiv(
    python: &Value,
    native: &Value,
    src: &str,
    errs: &mut Vec<String>,
) {
    values_close(
        &python["values"],
        &native["values"],
        &format!("state_space/{src}"),
        ABS_F64,
        &["STIFF_AD", "STIFF_BD", "STIFF_Y"],
        errs,
    );

    let ratio =
        residual_ratio_mats(&native["values"]["AD"], &python["values"]["AD"]);
    if ratio >= ZOH_AD_TAU {
        errs.push(format!("{src} ZOH A_d residual ratio {ratio}"));
    }

    let stiff_gold = json_mat(&python["values"]["STIFF_AD"]);
    let stiff_rs = json_mat(&native["values"]["STIFF_AD"]);

    if !stiff_gold.is_empty() && stiff_gold.len() == stiff_rs.len() {
        let stiff_diff: Vec<Vec<f64>> = stiff_gold
            .iter()
            .zip(stiff_rs.iter())
            .map(|(grow, rrow)| {
                grow.iter()
                    .zip(rrow.iter())
                    .map(|(gv, rv)| rv - gv)
                    .collect()
            })
            .collect();

        let stiff_rel = inf_norm_rows(&stiff_diff)
            / inf_norm_rows(&stiff_gold).max(f64::EPSILON);
        if stiff_rel >= 1e-8 {
            errs.push(format!(
                "{src} stiff ZOH A_d relative error {stiff_rel}"
            ));
        }
    }

    let py_y = f64_vec(&python["values"]["STEP_Y"]);
    let rs_y = f64_vec(&native["values"]["STEP_Y"]);
    let step_err = py_y
        .iter()
        .zip(rs_y.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    let step_bound = 200.0 * gamma(2.0) * 10.0;
    if step_err > step_bound.max(ABS_F64) {
        errs.push(format!("{src} tutorial step |py-rs|={step_err}"));
    }
}

fn transfer_function_equiv(
    python: &Value,
    native: &Value,
    src: &str,
    errs: &mut Vec<String>,
) {
    values_close(
        &python["values"],
        &native["values"],
        &format!("transfer_function/{src}"),
        ABS_F64,
        &["H_RE", "H_IM", "CLUSTER_H_RE", "CLUSTER_H_IM", "FREQS"],
        errs,
    );

    let bode_g = gamma(4.0);
    let check_complex = |re_p: &[f64],
                         im_p: &[f64],
                         re_r: &[f64],
                         im_r: &[f64],
                         label: &str,
                         g_tol: f64,
                         abs_tol: f64,
                         errs: &mut Vec<String>| {
        let len = re_p.len().min(re_r.len());
        for i in 0..len {
            let mag = re_p[i].hypot(im_p[i]).max(f64::EPSILON);
            let err = (re_p[i] - re_r[i]).hypot(im_p[i] - im_r[i]);
            let rel = err / mag;
            if rel > g_tol && err > abs_tol {
                errs.push(format!("{src} {label}[{i}] rel={rel}"));
            }
        }
    };

    check_complex(
        &f64_vec(&python["values"]["H_RE"]),
        &f64_vec(&python["values"]["H_IM"]),
        &f64_vec(&native["values"]["H_RE"]),
        &f64_vec(&native["values"]["H_IM"]),
        "complex-pair H",
        bode_g.max(1e-9),
        ABS_F64,
        errs,
    );

    check_complex(
        &f64_vec(&python["values"]["CLUSTER_H_RE"]),
        &f64_vec(&python["values"]["CLUSTER_H_IM"]),
        &f64_vec(&native["values"]["CLUSTER_H_RE"]),
        &f64_vec(&native["values"]["CLUSTER_H_IM"]),
        "clustered H",
        (gamma(16.0) * 1e3).max(1e-8),
        1e-10,
        errs,
    );
}

fn tensor_equiv(
    python: &Value,
    native: &Value,
    src: &str,
    errs: &mut Vec<String>,
) {
    values_close(
        &python["values"],
        &native["values"],
        &format!("tensor/{src}"),
        f64::from(ABS_F32),
        &[
            "CURVED_SAMPLES",
            "CUT_X",
            "DEQUANT",
            "RELU_DEQUANT",
            "CURVED_TABLE",
        ],
        errs,
    );

    let py_c = f64_vec(&python["values"]["CURVED_SAMPLES"]);
    let rs_c = f64_vec(&native["values"]["CURVED_SAMPLES"]);
    let weiser = metric_f64(native, "weiser_bound").unwrap_or(0.0);

    let curved_err = py_c
        .iter()
        .zip(rs_c.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    if curved_err > weiser.max(f64::from(ABS_F32) * 10.0) {
        errs.push(format!("{src} curved interp |py-rs|={curved_err}"));
    }

    let lsb = 1.0 / 256.0;
    for (label, doc) in [("python", python), (src, native)] {
        if let Some(q) = metric_f64(doc, "quant_roundtrip_max")
            && q > lsb
        {
            errs.push(format!("{label} Q7 round-trip {q} exceeds {lsb}"));
        }
    }
}
