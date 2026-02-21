//! # Subprogram Tests
//!
//! These tests cover `[subprograms]`.
#![allow(unused_imports)]

use crate::math::subprograms::{
    NaiveBlasf32, NaiveBlasf64,
    level1::{AXPY, DOT, IAMAX, NRM2},
    level2::GEMV,
    level3::GEMM,
};
use crate::{assert_almost_eq, assert_not_almost_eq};

mod level1 {
    use super::*;

    #[test]
    fn test_axpy_f32() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        NaiveBlasf32::axpy(2.0, &x, &mut y);
        assert_almost_eq!(y[0], 6.0);
        assert_almost_eq!(y[1], 9.0);
        assert_almost_eq!(y[2], 12.0);
    }

    #[test]
    fn test_axpy_f64() {
        let x = [1.0, 2.0, 3.0];
        let mut y = [4.0, 5.0, 6.0];
        NaiveBlasf64::axpy(2.0, &x, &mut y);
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
        NaiveBlasf32::axpy(2.0, &x, &mut y);
    }

    #[test]
    fn test_dot_f32() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = NaiveBlasf32::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[test]
    fn test_dot_f64() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let result = NaiveBlasf64::dot(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[test]
    #[should_panic(expected = "")]
    fn test_dot_panic_length_mismatch() {
        let x = [1.0, 2.0];
        let y = [4.0, 5.0, 6.0];
        NaiveBlasf32::dot(&x, &y);
    }

    #[test]
    fn test_nrm2_f32() {
        let x = [3.0, 4.0];
        let result = NaiveBlasf32::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[test]
    fn test_nrm2_f64() {
        let x = [3.0, 4.0];
        let result = NaiveBlasf64::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[test]
    fn test_iamax_f32() {
        let x = [1.0, -5.0, 3.0];
        let result = NaiveBlasf32::iamax(&x);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_iamax_f64() {
        let x = [1.0, -5.0, 3.0];
        let result = NaiveBlasf64::iamax(&x);
        assert_eq!(result, 1);
    }

    #[test]
    fn test_iamax_empty() {
        let x: [f32; 0] = [];
        let result = NaiveBlasf32::iamax(&x);
        assert_eq!(result, 0);
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
        NaiveBlasf32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
        assert_almost_eq!(y[0], 3.0);
        assert_almost_eq!(y[1], 7.0);
    }

    #[test]
    fn test_gemv_f64() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2x2 matrix
        let x = [1.0, 1.0];
        let mut y = [0.0, 0.0];
        NaiveBlasf64::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
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
        NaiveBlasf32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn test_gemv_panic_x_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0]; // Too short
        let mut y = [0.0, 0.0];
        NaiveBlasf32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 1\n right: 2"
    )]
    fn test_gemv_panic_y_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let x = [1.0, 1.0];
        let mut y = [0.0]; // Too short
        NaiveBlasf32::gemv(1.0, &a, &x, 0.0, &mut y, 2, 2);
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
        NaiveBlasf32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
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
        NaiveBlasf64::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
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
        NaiveBlasf32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemm_panic_b_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0];
        let mut c = [0.0; 4];
        NaiveBlasf32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }

    #[test]
    #[should_panic(
        expected = "assertion `left == right` failed\n  left: 3\n right: 4"
    )]
    fn test_gemm_panic_c_len() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0; 3];
        NaiveBlasf32::gemm(1.0, &a, &b, 0.0, &mut c, 2, 2, 2);
    }
}
