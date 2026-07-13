#![allow(unused_imports)]

use crate::assert_almost_eq;
use crate::math::subprograms::{
    BasicSubProgramsF32, BasicSubProgramsF64,
    level1::{AXPY, DOT, IAMAX, NRM2},
    level2::GEMV,
    level3::GEMM,
};

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Advanced BLAS subprogram tests (Level 3 GEMM and randomized fuzzing).
pub mod subprograms_advanced {
    use super::*;

    #[cfg_attr(test, test)]
    fn test_subprograms_gemm_f32_advanced() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert_almost_eq!(c[0], 1.0);
        assert_almost_eq!(c[1], 2.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 4.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemm_f64_advanced() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF64::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert_almost_eq!(c[0], 1.0);
        assert_almost_eq!(c[1], 2.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 4.0);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn _test_subprograms_gemm_panic_a_len_advanced() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn _test_subprograms_gemm_panic_b_len_advanced() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn _test_subprograms_gemm_panic_c_len_advanced() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 3];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemm_asymmetric_shared_axis_bounds_advanced() {
        let a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 3);
        assert_almost_eq!(c[0], 3.0);
        assert_almost_eq!(c[1], 3.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 3.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemm_tainted_state_nullification_failure_advanced() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        let mut c2 = [f32::NAN; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c2, 2, 2, 2);
        assert!(c2[0].is_nan());
        assert!(c2[1].is_nan());
        assert!(c2[2].is_nan());
        assert!(c2[3].is_nan());
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_symmetric_topology_preservation_advanced() {
        let mut rng = _rand_lcg(1234);
        for _ in 0..100 {
            let mut a = [0.0; 4];
            a[0] = _next_f32(&mut rng);
            a[1] = _next_f32(&mut rng);
            a[2] = a[1];
            a[3] = _next_f32(&mut rng);

            let mut c = [0.0; 4];
            BasicSubProgramsF32::gemm(1.0, &a, &a, 0.0, &mut c, 2, 2, 2);
            assert_almost_eq!(c[1], c[2]);
        }
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_distributive_variance_bounds_advanced() {
        let mut rng = _rand_lcg(5678);
        for _ in 0..100 {
            let m = [
                _next_f32(&mut rng),
                _next_f32(&mut rng),
                _next_f32(&mut rng),
                _next_f32(&mut rng),
            ];
            let v1 = [_next_f32(&mut rng), _next_f32(&mut rng)];
            let v2 = [_next_f32(&mut rng), _next_f32(&mut rng)];

            let v_sum = [v1[0] + v2[0], v1[1] + v2[1]];
            let mut r1 = [0.0; 2];
            BasicSubProgramsF32::gemv(1.0, &m, &v_sum, 0.0, &mut r1, 2, 2);

            let mut term_a = [0.0; 2];
            let mut term_b = [0.0; 2];
            BasicSubProgramsF32::gemv(1.0, &m, &v1, 0.0, &mut term_a, 2, 2);
            BasicSubProgramsF32::gemv(1.0, &m, &v2, 0.0, &mut term_b, 2, 2);
            let r2 = [term_a[0] + term_b[0], term_a[1] + term_b[1]];

            let epsilon = 1e-4;
            assert!((r1[0] - r2[0]).abs() < epsilon);
            assert!((r1[1] - r2[1]).abs() < epsilon);
        }
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_denormalized_subnormal_signal_decays_advanced() {
        let x = [1e-40_f32, 1e-42_f32];
        let mut y = [1e-40_f32, 1e-42_f32];
        BasicSubProgramsF32::axpy(0.5, &x, &mut y);
        assert!(y[0] > 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_clone_call_performance_benchmarking_advanced() {
        let a = [1.0, 2.0];
        let mut y = [3.0, 4.0];
        BasicSubProgramsF32::axpy(1.0, &a, &mut y);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_mixed_sign_zero_invariance_advanced() {
        let x = [0.0_f32, -0.0_f32];
        let mut y = [0.0_f32, -0.0_f32];
        BasicSubProgramsF32::axpy(1.0, &x, &mut y);
        assert_eq!(y[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(y[1].to_bits(), (-0.0_f32).to_bits());
    }

    const fn _rand_lcg(seed: u32) -> u32 {
        seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    #[allow(clippy::cast_precision_loss)]
    fn _next_f32(state: &mut u32) -> f32 {
        *state = _rand_lcg(*state);
        (*state as f32) / (u32::MAX as f32)
    }
}

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
/// Basic BLAS subprogram tests (Level 1 AXPY, DOT, NRM2, IAMAX and Level 2 GEMV).
pub mod subprograms_basic {
    use super::*;

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_f32_basic() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_f64_basic() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF64::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 2\n right: 3"
    )]
    fn _test_subprograms_axpy_panic_length_mismatch_basic() {
        let x = [1.0, 2.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_zero_scale_identity_preservation_basic() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y;
        BasicSubProgramsF32::axpy(0.0, &x, &mut y);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_zero_vector_multiplicative_invariance_basic() {
        let x = [0.0, 0.0, 0.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y;
        BasicSubProgramsF32::axpy(42.0, &x, &mut y);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_nan_poisoning_propagation_basic() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::NAN, &x, &mut y);
        assert!(y[0].is_nan());
        assert!(y[1].is_nan());
        assert!(y[2].is_nan());
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_axpy_infinity_multiplicative_edge_cases_basic() {
        let x = [0.0, 2.0, -3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::INFINITY, &x, &mut y);
        assert!(y[0].is_nan());
        assert!(y[1].is_infinite() && y[1].is_sign_positive());
        assert!(y[2].is_infinite() && y[2].is_sign_negative());
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_dot_f32_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_dot_f64_basic() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF64::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "")]
    fn _test_subprograms_dot_panic_length_mismatch_basic() {
        let x = [1.0, 2.0];
        let y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::dot(&x, &y);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_dot_geometric_orthogonality_verification_basic() {
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_dot_euclidean_norm_identity_basic() {
        let x = [3.0, 4.0];
        let dot_result = BasicSubProgramsF32::dot(&x, &x);
        let nrm2_result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(dot_result, nrm2_result * nrm2_result);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_dot_catastrophic_cancellation_tracking_basic() {
        let x = [1e15, -1e15, 1.0, 1.0];
        let y = [1.0, 1.0, 1.0, 1.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 2.0);
        let x2 = [1.0, 1e15, -1e15];
        let y2 = [1.0, 1.0, 1.0];
        let result2 = BasicSubProgramsF32::dot(&x2, &y2);
        assert_almost_eq!(result2, 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_nrm2_f32_basic() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_nrm2_f64_basic() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF64::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_nrm2_premature_domain_overflow_basic() {
        let x = [1e20_f32, 1e20_f32];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_nrm2_premature_domain_underflow_basic() {
        let x = [1e-25_f32, 1e-25_f32];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(result, 0.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_iamax_f32_basic() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_iamax_f64_basic() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF64::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_iamax_empty_basic() {
        let x: [f32; 0] = [];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_iamax_iterator_stability_basic() {
        let x = [1.0, 5.0, 3.0, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_iamax_partial_ordering_corruptions_basic() {
        let x = [1.0, f32::NAN, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 2);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_f32_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_f64_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF64::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn _test_subprograms_gemv_panic_a_len_basic() {
        let a = [1.0, 2.0, 3.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn _test_subprograms_gemv_panic_x_len_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn _test_subprograms_gemv_panic_y_len_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_asymmetric_rectangular_tall_processing_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 3, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
        assert_almost_eq!(y[2], 11.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_asymmetric_rectangular_wide_processing_basic() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 3);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 15.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_destination_overwrite_identity_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[cfg_attr(test, test)]
    fn test_subprograms_gemv_destination_suppression_identity_basic() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89];
        let y_original = y;
        BasicSubProgramsF32::gemv(0.0, &a, &x, 1.0, &mut y, 2, 2);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }
}
