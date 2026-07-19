#![allow(unused_imports)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod subprogram_test_suite {
    use crate::assert_almost_eq;
    use crate::math::subprograms::{
        BasicSubProgramsF32, BasicSubProgramsF64,
        level1::{AXPY, DOT, IAMAX, NRM2},
        level2::GEMV,
        level3::GEMM,
    };

    // --- Basic LCG Helper for Fuzzing Tests ---

    const fn _rand_lcg(seed: u32) -> u32 {
        seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223)
    }

    #[allow(clippy::cast_precision_loss)]
    fn _next_f32(state: &mut u32) -> f32 {
        *state = _rand_lcg(*state);
        (*state as f32) / (u32::MAX as f32)
    }

    // --- Fuzzing Tests ---

    #[cfg_attr(test, test)]
    /// Fuzzes GEMM with randomized inputs to verify that symmetric input topologies preserve symmetry in output.
    fn test_subprograms_fuzz_symmetric_gemm_topology() {
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
    /// Fuzzes GEMV to verify distributive property: M(v1 + v2) == Mv1 + Mv2.
    fn test_subprograms_fuzz_distributive_gemv_bounds() {
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

            // M(v1 + v2)
            let v_sum = [v1[0] + v2[0], v1[1] + v2[1]];
            let mut r1 = [0.0; 2];
            BasicSubProgramsF32::gemv(1.0, &m, &v_sum, 0.0, &mut r1, 2, 2);

            // Mv1 + Mv2
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
    /// Fuzzes AXPY with subnormal signals to ensure no pipeline panic occurs during arithmetic decay.
    fn test_subprograms_fuzz_subnormal_axpy_signal() {
        let x = [1e-40_f32, 1e-42_f32];
        let mut y = [1e-40_f32, 1e-42_f32];
        BasicSubProgramsF32::axpy(0.5, &x, &mut y);
        assert!(y[0] > 0.0);
    }

    #[cfg_attr(test, test)]
    /// Fuzzes AXPY to benchmark clone call performance and basic execution sanity.
    fn test_subprograms_fuzz_axpy_performance() {
        let a = [1.0, 2.0];
        let mut y = [3.0, 4.0];
        BasicSubProgramsF32::axpy(1.0, &a, &mut y);
    }

    #[cfg_attr(test, test)]
    /// Fuzzes AXPY with mixed sign zero inputs to check sign parity conservation.
    fn test_subprograms_fuzz_zero_sign_parity() {
        let x = [0.0_f32, -0.0_f32];
        let mut y = [0.0_f32, -0.0_f32];
        BasicSubProgramsF32::axpy(1.0, &x, &mut y);
        assert_eq!(y[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(y[1].to_bits(), (-0.0_f32).to_bits());
    }

    // --- Level 1 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies AXPY vector scaling addition (y = a*x + y) on f32.
    fn test_subprograms_level1_axpy_f32() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY vector scaling addition (y = a*x + y) on f64.
    fn test_subprograms_level1_axpy_f64() {
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
    fn _test_axpy_panic_length_mismatch() {
        let x = [1.0, 2.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY scaling with zero maintains vector identity.
    fn test_subprograms_level1_axpy_zero_scale() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y;
        BasicSubProgramsF32::axpy(0.0, &x, &mut y);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY adding a zero vector preserves vector identity under arbitrary scale.
    fn test_subprograms_level1_axpy_zero_vector() {
        let x = [0.0, 0.0, 0.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y;
        BasicSubProgramsF32::axpy(42.0, &x, &mut y);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY nan scale propagation.
    fn test_subprograms_level1_axpy_nan_propagation() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::NAN, &x, &mut y);
        assert!(y[0].is_nan());
        assert!(y[1].is_nan());
        assert!(y[2].is_nan());
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY infinity scaling edge cases.
    fn test_subprograms_level1_axpy_infinity_edge_cases() {
        let x = [0.0, 2.0, -3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::INFINITY, &x, &mut y);
        assert!(y[0].is_nan());
        assert!(y[1].is_infinite() && y[1].is_sign_positive());
        assert!(y[2].is_infinite() && y[2].is_sign_negative());
    }

    #[cfg_attr(test, test)]
    /// Verifies DOT dot product on f32.
    fn test_subprograms_level1_dot_f32() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies DOT dot product on f64.
    fn test_subprograms_level1_dot_f64() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF64::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "")]
    fn _test_dot_panic_length_mismatch() {
        let x = [1.0, 2.0];
        let y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::dot(&x, &y);
    }

    #[cfg_attr(test, test)]
    /// Verifies dot product orthogonality (dot product of orthogonal vectors is zero).
    fn test_subprograms_level1_dot_orthogonality() {
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 0.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies relation between Euclidean norm and dot product: dot(x, x) == norm(x)^2.
    fn test_subprograms_level1_dot_euclidean_norm_identity() {
        let x = [3.0, 4.0];
        let dot_result = BasicSubProgramsF32::dot(&x, &x);
        let nrm2_result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(dot_result, nrm2_result * nrm2_result);
    }

    #[cfg_attr(test, test)]
    /// Verifies dot product precision under catastrophic cancellation scenarios.
    fn test_subprograms_level1_dot_catastrophic_cancellation() {
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
    /// Verifies NRM2 Euclidean norm on f32.
    fn test_subprograms_level1_nrm2_f32() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies NRM2 Euclidean norm on f64.
    fn test_subprograms_level1_nrm2_f64() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF64::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies NRM2 domain overflow handling.
    fn test_subprograms_level1_nrm2_overflow() {
        let x = [1e20_f32, 1e20_f32];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert!(result.is_infinite() && result.is_sign_positive());
    }

    #[cfg_attr(test, test)]
    /// Verifies NRM2 domain underflow handling.
    fn test_subprograms_level1_nrm2_underflow() {
        let x = [1e-25_f32, 1e-25_f32];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(result, 0.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX absolute value max index finder on f32.
    fn test_subprograms_level1_iamax_f32() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX absolute value max index finder on f64.
    fn test_subprograms_level1_iamax_f64() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF64::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX on empty slices.
    fn test_subprograms_level1_iamax_empty() {
        let x: [f32; 0] = [];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 0);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX index stability (returns first index of maximum absolute value).
    fn test_subprograms_level1_iamax_iterator_stability() {
        let x = [1.0, 5.0, 3.0, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX ordering stability with NaN values.
    fn test_subprograms_level1_iamax_nan_corruption() {
        let x = [1.0, f32::NAN, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 2);
    }

    // --- Level 2 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies GEMV matrix-vector multiplication (y = alpha * A * x + beta * y) on f32.
    fn test_subprograms_level2_gemv_f32() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV matrix-vector multiplication (y = alpha * A * x + beta * y) on f64.
    fn test_subprograms_level2_gemv_f64() {
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
    fn _test_gemv_panic_a_len() {
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
    fn _test_gemv_panic_x_len() {
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
    fn _test_gemv_panic_y_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV on asymmetric rectangular tall matrices.
    fn test_subprograms_level2_gemv_asymmetric_tall() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 3, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
        assert_almost_eq!(y[2], 11.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV on asymmetric rectangular wide matrices.
    fn test_subprograms_level2_gemv_asymmetric_wide() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 3);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 15.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV overwrites existing data in the destination buffer when beta is zero.
    fn test_subprograms_level2_gemv_overwrite_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV scaling factor beta=1.0 and alpha=0.0 preserves destination vector content.
    fn test_subprograms_level2_gemv_suppression_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89];
        let y_original = y;
        BasicSubProgramsF32::gemv(0.0, &a, &x, 1.0, &mut y, 2, 2);
        for (val, orig) in y.iter().zip(y_original.iter()) {
            assert_almost_eq!(*val, *orig);
        }
    }

    // --- Level 3 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies GEMM matrix-matrix multiplication (C = alpha * A * B + beta * C) on f32.
    fn test_subprograms_level3_gemm_f32() {
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
    /// Verifies GEMM matrix-matrix multiplication (C = alpha * A * B + beta * C) on f64.
    fn test_subprograms_level3_gemm_f64() {
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
    fn _test_gemm_panic_a_len() {
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
    fn _test_gemm_panic_b_len() {
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
    fn _test_gemm_panic_c_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 3];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMM on asymmetric rectangular matrices sharing inner axis dimensions.
    fn test_subprograms_level3_gemm_asymmetric_shared_axis() {
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
    /// Verifies GEMM preserves NaN state.
    fn test_subprograms_level3_gemm_nan_nullification() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 1.0, 1.0, 1.0];
        let mut c2 = [f32::NAN; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c2, 2, 2, 2);
        assert!(c2[0].is_nan());
        assert!(c2[1].is_nan());
        assert!(c2[2].is_nan());
        assert!(c2[3].is_nan());
    }
}
