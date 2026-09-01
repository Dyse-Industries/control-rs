//! src/main.rs
//!
//! Orchestrates cross-language numerical model validation by calling each
//! model validator's `run()` function directly in-process.

mod matrix_validation;
mod polynomial_validation;
mod state_space_validation;
mod tensor_validation;
mod transfer_function_validation;

fn main() {
    println!();
    println!("--- [1/5] Matrix Validation ---");
    matrix_validation::run();
    println!();

    println!("--- [2/5] Polynomial Validation ---");
    polynomial_validation::run();
    println!();

    println!("--- [3/5] State-Space Validation ---");
    state_space_validation::run();
    println!();

    println!("--- [4/5] Transfer Function Validation ---");
    transfer_function_validation::run();
    println!();

    println!("--- [5/5] Tensor Validation ---");
    tensor_validation::run();
    println!();
}

#[cfg(test)]
mod cross_validate_gate_tests {
    use super::matrix_validation::ValidationResult;
    use serde_json::json;

    type GateTestCase<'a> = (&'a str, ValidationResult);

    fn assert_err_contains(result: ValidationResult, needle: &str) {
        match result {
            Ok(()) => panic!("expected error containing {needle:?}, got Ok"),
            Err(errs) => {
                assert!(
                    errs.iter().any(|e| e.contains(needle)),
                    "expected {needle:?} in {errs:?}"
                );
            }
        }
    }

    #[test]
    fn empty_oracle_object_fails_every_gate() {
        let rust = json!({"placeholder": true});
        let empty = json!({});
        let cases: [GateTestCase; 9] = [
            (
                "matrix",
                super::matrix_validation::cross_validate(&rust, &empty),
            ),
            (
                "matrix_jax",
                super::matrix_validation::cross_validate_jax(&rust, &empty),
            ),
            (
                "polynomial",
                super::polynomial_validation::cross_validate(&rust, &empty),
            ),
            (
                "polynomial_flint",
                super::polynomial_validation::cross_validate_flint(
                    &rust, &empty,
                ),
            ),
            (
                "state_space",
                super::state_space_validation::cross_validate(&rust, &empty),
            ),
            (
                "state_space_harold",
                super::state_space_validation::cross_validate_harold(
                    &rust, &empty,
                ),
            ),
            (
                "transfer_function",
                super::transfer_function_validation::cross_validate(
                    &rust, &empty,
                ),
            ),
            (
                "transfer_function_harold",
                super::transfer_function_validation::cross_validate_harold(
                    &rust, &empty,
                ),
            ),
            (
                "tensor",
                super::tensor_validation::cross_validate(&rust, &empty),
            ),
        ];
        for (name, result) in cases {
            match result {
                Ok(()) => panic!("{name}: empty payload must fail, got Ok"),
                Err(errs) => assert!(
                    errs.iter().any(|e| e.contains("empty payload")),
                    "{name}: expected empty payload error, got {errs:?}"
                ),
            }
        }
    }

    #[test]
    fn polynomial_requires_tutorial_keys_and_python_wilkinson() {
        let zeros20 = vec![0.0_f64; 20];
        let ones20 = vec![1.0_f64; 20];
        let rust = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": [1, 2, 3]},
            "wilkinson_residual": {
                "residual_f64": zeros20,
                "residual_f32": ones20
            }
        });
        let python = json!({
            "tutorial": {"p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": [1, 2, 3]}
        });
        let errs = super::polynomial_validation::cross_validate(&rust, &python)
            .expect_err("missing python keys must fail");
        assert!(
            errs.iter().any(|e| e.contains("Missing tutorial p_real")),
            "{errs:?}"
        );
        assert!(
            errs.iter()
                .any(|e| e.contains("Missing wilkinson_residual")),
            "{errs:?}"
        );
    }

    #[test]
    fn tensor_missing_interp_mesh_mat_c_q_raw_fails() {
        let rust = json!({
            "manifold": {"interp_mesh": [[1.0]]},
            "contraction": {"mat_c": [[1.0]]},
            "boundaries": {"q_raw": [1], "act_outputs": [0.0]}
        });
        let python = json!({
            "manifold": {},
            "contraction": {},
            "boundaries": {}
        });
        let errs = super::tensor_validation::cross_validate(&rust, &python)
            .expect_err("missing tensor keys must fail");
        assert!(errs.iter().any(|e| e.contains("interp_mesh")), "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("mat_c")), "{errs:?}");
        assert!(errs.iter().any(|e| e.contains("q_raw")), "{errs:?}");
    }

    #[test]
    fn transfer_function_missing_bode_phase_fails() {
        let mag = vec![0.0_f64; 4];
        let rust = json!({
            "tutorial": {"a00": 0.0, "a01": 0.0, "a10": 0.0, "a11": 0.0},
            "discretization_error": {
                "cont_mag_db": mag,
                "cont_phase_deg": mag,
                "tustin_mag_db": mag,
                "tustin_phase_deg": mag,
                "zoh_mag_db": mag,
                "zoh_phase_deg": mag
            },
            "nyquist_criterion": {
                "h_re": mag,
                "h_im": mag,
                "phase_margin_deg": 0.0,
                "gain_margin_db": 0.0
            }
        });
        let python = json!({
            "tutorial": {"a00": 0.0, "a01": 0.0, "a10": 0.0, "a11": 0.0},
            "discretization_error": {
                "cont_mag_db": mag,
                "tustin_mag_db": mag,
                "zoh_mag_db": mag
            },
            "nyquist_criterion": {
                "h_re": mag,
                "h_im": mag,
                "phase_margin_deg": 0.0,
                "gain_margin_db": 0.0
            }
        });
        let errs =
            super::transfer_function_validation::cross_validate(&rust, &python)
                .expect_err("mag-only Bode payload must fail");
        assert!(
            errs.iter().any(|e| e.contains("cont_phase_deg")),
            "{errs:?}"
        );
        assert!(
            errs.iter().any(|e| e.contains("tustin_phase_deg")),
            "{errs:?}"
        );
        assert!(errs.iter().any(|e| e.contains("zoh_phase_deg")), "{errs:?}");
    }

    #[test]
    fn state_space_length_mismatch_fails() {
        let rust = json!({
            "phase_portrait": {
                "theta": [1.0, 2.0, 3.0],
                "theta_dot": [0.0, 0.0, 0.0]
            },
            "jitter": {"step_data": [0.0, 0.0, 0.0]}
        });
        let python = json!({
            "phase_portrait": {
                "theta": [1.0, 2.0],
                "theta_dot": [0.0, 0.0, 0.0]
            },
            "jitter": {"step_data": [0.0, 0.0, 0.0]}
        });
        assert_err_contains(
            super::state_space_validation::cross_validate(&rust, &python),
            "phase_portrait.theta length mismatch",
        );
    }

    #[test]
    fn matrix_and_jax_column_length_mismatch_fails() {
        let rust = json!({
            "covariance_heatmap": {"matrix": [[1.0, 2.0, 3.0]]},
            "scaling": {"inversion_time_ns": [1, 2, 3, 4, 5, 6]},
            "jitter": {"hilbert_solve_times_ns": vec![0; 1000]}
        });
        let short = json!({
            "covariance_heatmap": {"matrix": [[1.0, 2.0]]},
            "scaling": {"inversion_time_ns": [1, 2, 3, 4, 5, 6]},
            "jitter": {"hilbert_solve_times_ns": vec![0; 1000]}
        });
        assert_err_contains(
            super::matrix_validation::cross_validate(&rust, &short),
            "col count mismatch",
        );
        assert_err_contains(
            super::matrix_validation::cross_validate_jax(&rust, &short),
            "col count mismatch",
        );
    }

    #[test]
    fn polynomial_wilkinson_higham_bound_violation_fails() {
        let zeros20 = vec![0.0_f64; 20];
        let mut huge_f64 = vec![0.0_f64; 20];
        huge_f64[0] = 1e25; // Exceeds Higham bound for k=1 (~4.5e5)
        let rust = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": huge_f64,
                "residual_f32": vec![1e26_f64; 20]
            }
        });
        let python = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": zeros20,
                "residual_f32": vec![1e26_f64; 20]
            }
        });
        assert_err_contains(
            super::polynomial_validation::cross_validate(&rust, &python),
            "exceeds Higham backward error bound",
        );
    }

    #[test]
    fn polynomial_wilkinson_f32_precision_inversion_fails() {
        let rust = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": vec![1000.0_f64; 20],
                "residual_f32": vec![10.0_f64; 20] // f32 < f64 precision inversion
            }
        });
        let python = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": vec![1000.0_f64; 20],
                "residual_f32": vec![10.0_f64; 20]
            }
        });
        assert_err_contains(
            super::polynomial_validation::cross_validate(&rust, &python),
            "expected precision loss in f32",
        );
    }

    #[test]
    fn polynomial_wilkinson_log_scale_mismatch_fails() {
        let mut rust_f64 = vec![1e8_f64; 20];
        let py_f64 = vec![1e8_f64; 20];
        rust_f64[15] = 1e12; // 4 orders of magnitude larger than python at k=16
        let rust = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": rust_f64,
                "residual_f32": vec![1e20_f64; 20]
            }
        });
        let python = json!({
            "tutorial": {"p_real": 1.0, "p_c_re": 0.0, "p_c_im": 0.0},
            "root_convergence": {"iterations": vec![1; 11]},
            "wilkinson_residual": {
                "residual_f64": py_f64,
                "residual_f32": vec![1e20_f64; 20]
            }
        });
        assert_err_contains(
            super::polynomial_validation::cross_validate(&rust, &python),
            "scale mismatch",
        );
    }
}
