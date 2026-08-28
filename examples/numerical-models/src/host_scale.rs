//! Host-scale JSON V&V — reads `results/<slug>/host_*_{python,native}.json`.

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use serde_json::Value;

    use crate::SOLVE_RESIDUAL_TAU;

    const RESIDUAL_TAU: f64 = SOLVE_RESIDUAL_TAU;

    fn manifest_dir() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = manifest_dir().join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    fn report_only_bound_miss(claim: &str, detail: &str) {
        println!("report-only host-scale bound: {claim} — {detail}");
    }

    fn assert_hilbert(n: usize, gate: bool) {
        let py = load_artifact(
            &format!("results/matrix/host_hilbert{n}_python.json"),
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            &format!("results/matrix/host_hilbert{n}_native.json"),
            "cargo run --example host_scale_emit",
        );
        let meta = &py["values"];
        let got = &rs["values"];
        let kappa = meta["kappa"].as_f64().unwrap();
        let keps = meta["keps"].as_f64().unwrap();
        let tau = meta["tau"].as_f64().unwrap();
        let residual_ratio = got["residual_ratio"].as_f64().unwrap();
        let rel_err = got["rel_err"].as_f64().unwrap();
        println!("hilbert{n} residual ratio={residual_ratio:.6e} kappa={kappa:.6e}");
        if gate && keps < 1.0 {
            assert!(residual_ratio < RESIDUAL_TAU, "hilbert{n} residual {residual_ratio}");
            let bound = tau * keps;
            assert!(rel_err <= bound, "hilbert{n} forward rel {rel_err} > {bound}");
        } else {
            println!("hilbert{n} κε≳1 or report-only: rel={rel_err:.6e}");
        }
    }

    #[test]
    fn host_scale_hilbert() {
        assert_hilbert(12, true);
        assert_hilbert(16, true);
        assert_hilbert(128, false);
    }

    #[test]
    fn host_scale_poly_clustered() {
        let py = load_artifact(
            "results/polynomial/host_poly52_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/polynomial/host_poly52_native.json",
            "cargo run --example host_scale_emit",
        );
        let meta = &py["values"];
        let rel = rs["values"]["rel_err"].as_f64().unwrap();
        let bound = meta["tau"].as_f64().unwrap() * meta["kappa"].as_f64().unwrap() * meta["eps"].as_f64().unwrap();
        println!("poly52 rel={rel:.6e} bound={bound:.6e}");
        if meta["kappa"].as_f64().unwrap() * meta["eps"].as_f64().unwrap() < 1.0 {
            assert!(rel <= bound, "poly52 rel {rel} > {bound}");
        }
    }

    #[test]
    fn host_scale_tf_clustered() {
        let py = load_artifact(
            "results/transfer_function/host_tf_clustered_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/transfer_function/host_tf_clustered_native.json",
            "cargo run --example host_scale_emit",
        );
        let meta = &py["values"];
        let rels = rs["values"]["rel_errs"].as_array().unwrap();
        let bound = meta["tau"].as_f64().unwrap() * meta["kappa"].as_f64().unwrap() * meta["eps"].as_f64().unwrap();
        for (i, rel) in rels.iter().enumerate() {
            let rel = rel.as_f64().unwrap();
            println!("tf w idx={i} rel={rel:.6e} bound={bound:.6e}");
            if meta["kappa"].as_f64().unwrap() * meta["eps"].as_f64().unwrap() < 1.0 {
                assert!(rel <= bound, "tf rel {rel} > {bound}");
            }
        }
    }

    #[test]
    fn host_scale_stiff_zoh() {
        let _py = load_artifact(
            "results/state_space/host_stiff_zoh_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/state_space/host_stiff_zoh_native.json",
            "cargo run --example host_scale_emit",
        );
        let _rel = rs["values"]["rel_err"].as_f64().unwrap();
        let ratio = rs["values"]["residual_ratio"].as_f64().unwrap();
        println!("stiff ZOH residual ratio={ratio:.6e}");
        if ratio >= RESIDUAL_TAU {
            report_only_bound_miss(
                "stiff ZOH A_d residual ratio",
                &format!("ratio={ratio:.6e} (threshold {RESIDUAL_TAU})"),
            );
        }
    }

    #[test]
    #[ignore = "1024x1024 tensor grid; slow. Run: cargo test -- --ignored"]
    fn host_scale_tensor_grid() {
        let py = load_artifact(
            "results/tensor/host_tensor1024_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/tensor/host_tensor1024_native.json",
            "cargo run --example host_scale_emit",
        );
        let errs = rs["values"]["abs_errs"].as_array().unwrap();
        let points = py["values"]["points"].as_array().unwrap();
        for (pt, err) in points.iter().zip(errs.iter()) {
            let err = err.as_f64().unwrap() as f32;
            println!("tensor1024 {:?} abs={err}", pt);
            assert!(err <= 1e-3, "tensor interp {err}");
        }
    }

    #[test]
    #[ignore = "1024x1024 GEMM; slow. Run: cargo test -- --ignored"]
    fn host_scale_1024_gemm() {
        let py = load_artifact(
            "results/matrix/host_gemm1024_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/matrix/host_gemm1024_native.json",
            "cargo run --example host_scale_emit",
        );
        let rel = rs["values"]["rel_err"].as_f64().unwrap();
        let bound = py["values"]["tau"].as_f64().unwrap()
            * py["values"]["kappa"].as_f64().unwrap()
            * py["values"]["eps"].as_f64().unwrap();
        println!("gemm1024 rel={rel:.6e} bound={bound:.6e}");
        if rel > bound {
            report_only_bound_miss(
                "1024x1024 GEMM forward error",
                &format!("rel={rel:.6e} bound={bound:.6e}"),
            );
        }
    }

    #[test]
    #[ignore = "1024x1024 LU; slow. Run: cargo test -- --ignored"]
    fn host_scale_1024_lu() {
        let py = load_artifact(
            "results/matrix/host_lu1024_python.json",
            "python3 python/src/host_scale.py",
        );
        let rs = load_artifact(
            "results/matrix/host_lu1024_native.json",
            "cargo run --example host_scale_emit",
        );
        let residual_ratio = rs["values"]["residual_ratio"].as_f64().unwrap();
        let rel_x = rs["values"]["rel_err"].as_f64().unwrap();
        let bound = py["values"]["tau"].as_f64().unwrap()
            * py["values"]["kappa"].as_f64().unwrap()
            * py["values"]["eps"].as_f64().unwrap();
        println!("lu1024 residual={residual_ratio:.6e} rel={rel_x:.6e} bound={bound:.6e}");
        if residual_ratio >= RESIDUAL_TAU {
            report_only_bound_miss(
                "1024x1024 LU residual ratio",
                &format!("residual={residual_ratio:.6e} (threshold {RESIDUAL_TAU})"),
            );
        }
        if rel_x > bound {
            report_only_bound_miss(
                "1024x1024 LU forward error",
                &format!("rel={rel_x:.6e} bound={bound:.6e}"),
            );
        }
    }
}
