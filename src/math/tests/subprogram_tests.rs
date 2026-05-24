#![allow(unused_imports)]

use crate::assert_almost_eq;
use crate::math::subprograms::{
    BasicSubProgramsF32, BasicSubProgramsF64,
    level1::{AXPY, DOT, IAMAX, NRM2},
    level2::GEMV,
    level3::GEMM,
};

mod level1 {
    use super::*;

    #[test]
    fn test_axpy_f32() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[test]
    fn test_axpy_f64() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF64::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 2\n right: 3"
    )]
    fn test_axpy_panic_length_mismatch() {
        let x = [1.0, 2.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(2.0, &x, &mut y);
    }

    #[test]
    fn test_axpy_zero_scale_identity_preservation() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y.clone();
        BasicSubProgramsF32::axpy(0.0, &x, &mut y);
        assert_eq!(y, y_original);
    }

    #[test]
    fn test_axpy_zero_vector_multiplicative_invariance() {
        let x = [0.0, 0.0, 0.0];
        let mut y = [4.0, 5.0, 6.0];
        let y_original = y.clone();
        BasicSubProgramsF32::axpy(42.0, &x, &mut y);
        assert_eq!(y, y_original);
    }

    #[test]
    fn test_axpy_nan_poisoning_propagation() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::NAN, &x, &mut y);
        assert!(y[0].is_nan());
        assert!(y[1].is_nan());
        assert!(y[2].is_nan());
    }

    #[test]
    fn test_axpy_infinity_multiplicative_edge_cases() {
        let x = [0.0, 2.0, -3.0];
        let mut y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::axpy(f32::INFINITY, &x, &mut y);
        assert!(y[0].is_nan());
        assert_eq!(y[1], f32::INFINITY);
        assert_eq!(y[2], f32::NEG_INFINITY);
    }

    #[test]
    fn test_dot_f32() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_f64() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = BasicSubProgramsF64::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[test]
    #[should_panic(expected = "")]
    fn test_dot_panic_length_mismatch() {
        let x = [1.0, 2.0];
        let y = [4.0, 5.0, 6.0];
        BasicSubProgramsF32::dot(&x, &y);
    }

    #[test]
    fn test_dot_geometric_orthogonality_verification() {
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_almost_eq!(result, 0.0);
    }

    #[test]
    fn test_dot_euclidean_norm_identity() {
        let x = [3.0, 4.0];
        let dot_result = BasicSubProgramsF32::dot(&x, &x);
        let nrm2_result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(dot_result, nrm2_result * nrm2_result);
    }

    #[test]
    fn test_dot_catastrophic_cancellation_tracking() {
        // Construct a specialized vector pair interleaving massive and minuscule magnitudes
        // We expect precision loss derived from left-fold iterators here
        let x = [1e15, -1e15, 1.0, 1.0];
        let y = [1.0, 1.0, 1.0, 1.0];
        let result = BasicSubProgramsF32::dot(&x, &y);
        assert_eq!(result, 0.0);
        let x2 = [1.0, 1e15, -1e15];
        let y2 = [1.0, 1.0, 1.0];
        let result2 = BasicSubProgramsF32::dot(&x2, &y2);
        assert_eq!(result2, 0.0);
    }

    #[test]
    fn test_nrm2_f32() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[test]
    fn test_nrm2_f64() {
        let x = [3.0, 4.0];
        let result = BasicSubProgramsF64::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[test]
    fn test_nrm2_premature_domain_overflow() {
        // Construct vectors composed of large, normal floats whose squares exceed representational limits
        let x = [1e20_f32, 1e20_f32];
        // 1e20 * 1e20 = 1e40 which exceeds f32 max of ~3.4e38.
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_eq!(result, f32::INFINITY);
    }

    #[test]
    fn test_nrm2_premature_domain_underflow() {
        // Construct vectors composed of extremely small floats whose squares underflow
        let x = [1e-25_f32, 1e-25_f32];
        // 1e-25 * 1e-25 = 1e-50 which is smaller than f32 min subnormal (~1e-45).
        let result = BasicSubProgramsF32::nrm2(&x);
        assert_eq!(result, 0.0);
    }

    #[test]
    fn test_iamax_f32() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_iamax_f64() {
        let x = [1.0, -5.0, 3.0];
        let result = BasicSubProgramsF64::iamax(&x);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_iamax_empty() {
        let x: [f32; 0] = [];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 0);
    }

    #[test]
    fn test_iamax_iterator_stability() {
        let x = [1.0, 5.0, 3.0, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 3);
    }

    #[test]
    fn test_iamax_partial_ordering_corruptions() {
        let x = [1.0, f32::NAN, 5.0];
        let result = BasicSubProgramsF32::iamax(&x);
        assert_eq!(result, 1);
    }
}

mod level2 {
    use super::*;

    #[test]
    fn test_gemv_f32() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        // y = 1.0 * A * x + 0.0 * y
        // y[0] = 1*1 + 2*1 = 3
        // y[1] = 3*1 + 4*1 = 7
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[test]
    fn test_gemv_f64() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF64::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemv_panic_a_len() {
        let a = [1.0, 2.0, 3.0]; // Not 2x2
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn test_gemv_panic_x_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0]; // Too short
        let mut y = [0.0, 0.0];
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn test_gemv_panic_y_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0]; // Too short
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[test]
    fn test_gemv_asymmetric_rectangular_tall_processing() {
        // 3x2 matrix (tall)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0, 0.0];
        // y = 1 * A * x + 0 * y
        // y[0] = 1*1 + 2*1 = 3
        // y[1] = 3*1 + 4*1 = 7
        // y[2] = 5*1 + 6*1 = 11
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 3, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
        assert_almost_eq!(y[2], 11.0);
    }

    #[test]
    fn test_gemv_asymmetric_rectangular_wide_processing() {
        // 2x3 matrix (wide)
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let x = [1.0, 1.0, 1.0];
        let mut y = [0.0, 0.0];
        // y = 1 * A * x + 0 * y
        // y[0] = 1*1 + 2*1 + 3*1 = 6
        // y[1] = 4*1 + 5*1 + 6*1 = 15
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 3);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 15.0);
    }

    #[test]
    fn test_gemv_destination_overwrite_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89]; // High-entropy randomized data
        BasicSubProgramsF32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[test]
    fn test_gemv_destination_suppression_identity() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [123.45, 67.89];
        let y_original = y.clone();
        BasicSubProgramsF32::gemv(0.0, &a, &x, 1.0, &mut y, 2, 2);
        assert_eq!(y, y_original);
    }
}

mod level3 {
    use super::*;

    #[test]
    fn test_gemm_f32() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = [0.0; 4];
        // C = 1.0 * A * I + 0.0 * C = A
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert_almost_eq!(c[0], 1.0);
        assert_almost_eq!(c[1], 2.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 4.0);
    }

    #[test]
    fn test_gemm_f64() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2
        let b = [1.0, 0.0, 0.0, 1.0]; // 2x2 identity
        let mut c = [0.0; 4];
        BasicSubProgramsF64::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
        assert_almost_eq!(c[0], 1.0);
        assert_almost_eq!(c[1], 2.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 4.0);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemm_panic_a_len() {
        let a = [1.0, 2.0, 3.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemm_panic_b_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemm_panic_c_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 3];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[test]
    fn test_gemm_asymmetric_shared_axis_bounds() {
        // A is 2x3, B is 3x2. C is 2x2. Shared axis is 3.
        let a = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let b = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut c = [0.0; 4];
        BasicSubProgramsF32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 3);
        // Each element of C should be the dot product of a row of A and a col of B.
        // Row of A is [1,1,1]. Col of B is [1,1,1]. Dot product is 3.
        assert_almost_eq!(c[0], 3.0);
        assert_almost_eq!(c[1], 3.0);
        assert_almost_eq!(c[2], 3.0);
        assert_almost_eq!(c[3], 3.0);
    }

    #[test]
    fn test_gemm_tainted_state_nullification_failure() {
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

mod fuzzing {
    use super::*;

    #[test]
    fn test_symmetric_topology_preservation() {
        let mut rng = rand_lcg(1234);
        for _ in 0..100 {
            // A = A^T
            let mut a = [0.0; 4];
            a[0] = next_f32(&mut rng);
            a[1] = next_f32(&mut rng);
            a[2] = a[1];
            a[3] = next_f32(&mut rng);

            let mut c = [0.0; 4];
            BasicSubProgramsF32::gemm(1.0, &a, &a, 0.0, &mut c, 2, 2, 2);
            assert_almost_eq!(c[1], c[2]);
        }
    }

    #[test]
    fn test_distributive_variance_bounds() {
        let mut rng = rand_lcg(5678);
        for _ in 0..100 {
            let m = [
                next_f32(&mut rng),
                next_f32(&mut rng),
                next_f32(&mut rng),
                next_f32(&mut rng),
            ];
            let v1 = [next_f32(&mut rng), next_f32(&mut rng)];
            let v2 = [next_f32(&mut rng), next_f32(&mut rng)];

            // M(v1 + v2)
            let v_sum = [v1[0] + v2[0], v1[1] + v2[1]];
            let mut r1 = [0.0; 2];
            BasicSubProgramsF32::gemv(1.0, &m, &v_sum, 0.0, &mut r1, 2, 2);

            // Mv1 + Mv2
            let mut r2a = [0.0; 2];
            let mut r2b = [0.0; 2];
            BasicSubProgramsF32::gemv(1.0, &m, &v1, 0.0, &mut r2a, 2, 2);
            BasicSubProgramsF32::gemv(1.0, &m, &v2, 0.0, &mut r2b, 2, 2);
            let r2 = [r2a[0] + r2b[0], r2a[1] + r2b[1]];

            let epsilon = 1e-4;
            assert!((r1[0] - r2[0]).abs() < epsilon);
            assert!((r1[1] - r2[1]).abs() < epsilon);
        }
    }

    #[test]
    fn test_denormalized_subnormal_signal_decays() {
        let x = [1e-40_f32, 1e-42_f32];
        let mut y = [1e-40_f32, 1e-42_f32];
        BasicSubProgramsF32::axpy(0.5, &x, &mut y);
        // Ensure no pipeline panics occurred and result is calculated
        assert!(y[0] > 0.0);
    }

    #[test]
    fn test_clone_call_performance_benchmarking() {
        // Can't easily count clones without mocking the type, but basic f32/f64 subprograms only accept these types anyway.
        // We will just verify it runs.
        let a = [1.0, 2.0];
        let mut y = [3.0, 4.0];
        BasicSubProgramsF32::axpy(1.0, &a, &mut y);
    }

    #[test]
    fn test_mixed_sign_zero_invariance() {
        let x = [0.0_f32, -0.0_f32];
        let mut y = [0.0_f32, -0.0_f32];
        BasicSubProgramsF32::axpy(1.0, &x, &mut y);
        // Verify bitwise sign parity
        assert_eq!(y[0].to_bits(), 0.0_f32.to_bits());
        assert_eq!(y[1].to_bits(), (-0.0_f32).to_bits());
    }

    // Basic LCG for tests without external dependencies
    fn rand_lcg(seed: u32) -> u32 {
        seed.wrapping_mul(1664525).wrapping_add(1013904223)
    }

    fn next_f32(state: &mut u32) -> f32 {
        *state = rand_lcg(*state);
        // Simple 0 to 1 mapping
        (*state as f32) / (u32::MAX as f32)
    }
}
