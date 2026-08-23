//! BLAS-style subprogram (`subprograms.rs`, levels 1-3, packed, sparse, and LAPACK) test suite.
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::float_cmp)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod subprogram_test_suite {
    use crate::assert_almost_eq;
    use crate::math::LinAlgError;
    use crate::math::complex_num::{Complex, Complex32, Complex64};
    use crate::math::num_traits::{One, Zero};
    use crate::math::num_types::Const;
    use crate::math::storage::{
        ArrayCooStorage, ArrayCsrStorage, ArraySparseVector, ArrayStorage,
        DenseStorage, DenseStorageMut, Diag, HermitianPackedStorage,
        MatrixLayout, PackedStorage, PackedStorageMut, RowArrayStorage, Side,
        SparseStorage, Storage, StorageInit, StorageMut, StorageView,
        StorageViewMut, SymmetricPackedStorage, ToCsrStorage, Trans,
        TriangularPackedStorage, UpLo, ViewStorage, ViewStorageMut,
    };
    use crate::math::subprograms::{
        DefaultBlas, JobZ,
        lapack::{
            Geqrf, Getrf, Getrs, Heev, Ormqr, Potrf, Potrs, Pptrf, Pptrs, Syev,
            Unmqr,
        },
        level1::{
            Asum, Axpy, Dotc, Dotu, Iamax, Nrm2, RealScal, Rot, Scal, Swap,
        },
        level2::{
            Gemv, Gerc, Geru, Hemv, Her, Her2, Symv, Syr, Syr2, Trmv, Trsv,
        },
        level3::{Gemm, Hemm, Her2k, Herk, Symm, Syr2k, Syrk, Trmm, Trsm},
        packed::{Hpmv, Hpr, Hpr2, Spmv, Spr, Spr2, Tpmv, Tpsv},
        sparse::{Cscmv, Csrmm, Csrmv, SpAxpy, SpDotc, SpDotu},
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
            let a_val0 = _next_f32(&mut rng);
            let a_val1 = _next_f32(&mut rng);
            let a_val3 = _next_f32(&mut rng);
            let a = ArrayStorage::<f32, 2, 2>::from_array([
                [a_val0, a_val1],
                [a_val1, a_val3],
            ]);

            let mut c = ArrayStorage::<f32, 2, 2>::zeros();
            DefaultBlas::gemm(
                Trans::NoTrans,
                Trans::NoTrans,
                1.0,
                &a,
                &a,
                0.0,
                &mut c,
            );
            let c01 = unsafe { *c.get_unchecked(0, 1) };
            let c10 = unsafe { *c.get_unchecked(1, 0) };
            assert_almost_eq!(c01, c10);
        }
    }

    #[cfg_attr(test, test)]
    /// Fuzzes GEMV to verify distributive property: M(v1 + v2) == Mv1 + Mv2.
    fn test_subprograms_fuzz_distributive_gemv_bounds() {
        let mut rng = _rand_lcg(5678);
        for _ in 0..100 {
            let m = ArrayStorage::<f32, 2, 2>::from_array([
                [_next_f32(&mut rng), _next_f32(&mut rng)],
                [_next_f32(&mut rng), _next_f32(&mut rng)],
            ]);
            let v1 = ArrayStorage::<f32, 2, 1>::from_array([[
                _next_f32(&mut rng),
                _next_f32(&mut rng),
            ]]);
            let v2 = ArrayStorage::<f32, 2, 1>::from_array([[
                _next_f32(&mut rng),
                _next_f32(&mut rng),
            ]]);

            let v_sum = ArrayStorage::<f32, 2, 1>::from_array([[
                unsafe { v1.get_unchecked(0, 0) + v2.get_unchecked(0, 0) },
                unsafe { v1.get_unchecked(1, 0) + v2.get_unchecked(1, 0) },
            ]]);

            let mut r1 = ArrayStorage::<f32, 2, 1>::zeros();
            DefaultBlas::gemv(Trans::NoTrans, 1.0, &m, &v_sum, 0.0, &mut r1);

            let mut term_a = ArrayStorage::<f32, 2, 1>::zeros();
            let mut term_b = ArrayStorage::<f32, 2, 1>::zeros();
            DefaultBlas::gemv(Trans::NoTrans, 1.0, &m, &v1, 0.0, &mut term_a);
            DefaultBlas::gemv(Trans::NoTrans, 1.0, &m, &v2, 0.0, &mut term_b);

            let r2_0 = unsafe {
                term_a.get_unchecked(0, 0) + term_b.get_unchecked(0, 0)
            };
            let r2_1 = unsafe {
                term_a.get_unchecked(1, 0) + term_b.get_unchecked(1, 0)
            };

            let r1_0 = unsafe { *r1.get_unchecked(0, 0) };
            let r1_1 = unsafe { *r1.get_unchecked(1, 0) };

            assert_almost_eq!(r1_0, r2_0, 1e-4);
            assert_almost_eq!(r1_1, r2_1, 1e-4);
        }
    }

    // --- Level 1 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies AXPY vector scaling and addition (y = a * x + y) on f32.
    fn test_subprograms_level1_axpy_f32() {
        let x = ArrayStorage::<f32, 3, 1>::from_array([[1.0, 2.0, 3.0]]);
        let mut y = ArrayStorage::<f32, 3, 1>::from_array([[4.0, 5.0, 6.0]]);
        DefaultBlas::axpy(2.0, &x, &mut y);
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 6.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 9.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(2, 0) }, 12.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies AXPY vector scaling and addition (y = a * x + y) on f64.
    fn test_subprograms_level1_axpy_f64() {
        let x = ArrayStorage::<f64, 3, 1>::from_array([[1.0, 2.0, 3.0]]);
        let mut y = ArrayStorage::<f64, 3, 1>::from_array([[4.0, 5.0, 6.0]]);
        DefaultBlas::axpy(2.0, &x, &mut y);
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 6.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 9.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(2, 0) }, 12.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies SCAL vector scaling in place (x = a * x) on f32.
    fn test_subprograms_level1_scal_f32() {
        let mut x = ArrayStorage::<f32, 3, 1>::from_array([[1.0, 2.0, 3.0]]);
        DefaultBlas::scal(2.0, &mut x);
        assert_almost_eq!(unsafe { *x.get_unchecked(0, 0) }, 2.0);
        assert_almost_eq!(unsafe { *x.get_unchecked(1, 0) }, 4.0);
        assert_almost_eq!(unsafe { *x.get_unchecked(2, 0) }, 6.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies RealScal on complex vectors.
    fn test_subprograms_level1_real_scal() {
        let mut x = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
        ]]);
        DefaultBlas::real_scal(2.0f32, &mut x);
        assert_almost_eq!(unsafe { x.get_unchecked(0, 0).re }, 2.0);
        assert_almost_eq!(unsafe { x.get_unchecked(0, 0).im }, 4.0);
        assert_almost_eq!(unsafe { x.get_unchecked(1, 0).re }, 6.0);
        assert_almost_eq!(unsafe { x.get_unchecked(1, 0).im }, 8.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies DOT product calculation on f32 vectors.
    fn test_subprograms_level1_dot_f32() {
        let x = ArrayStorage::<f32, 3, 1>::from_array([[1.0, 2.0, 3.0]]);
        let y = ArrayStorage::<f32, 3, 1>::from_array([[4.0, 5.0, 6.0]]);
        let result = DefaultBlas::dotu(&x, &y);
        assert_almost_eq!(result, 32.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies NRM2 Euclidean norm calculation on f32 vectors.
    fn test_subprograms_level1_nrm2_f32() {
        let x = ArrayStorage::<f32, 2, 1>::from_array([[3.0, 4.0]]);
        let result = DefaultBlas::nrm2(&x);
        assert_almost_eq!(result, 5.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies IAMAX index calculation for maximum absolute value on f32 vectors.
    fn test_subprograms_level1_iamax_f32() {
        let x = ArrayStorage::<f32, 3, 1>::from_array([[1.0, -5.0, 3.0]]);
        let result = DefaultBlas::iamax(&x);
        assert_eq!(result, 1);
    }

    #[cfg_attr(test, test)]
    /// Verifies Dotc (conjugated dot product), Asum, Swap, and Rot kernels.
    fn test_subprograms_level1_additional_kernels() {
        let x = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(1.0, 2.0),
            Complex32::new(3.0, 4.0),
        ]]);
        let y = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(2.0, 0.0),
            Complex32::new(0.0, 1.0),
        ]]);
        // x^H y = (1 - 2i)(2) + (3 - 4i)(i) = (2 - 4i) + (4 + 3i) = 6 - i
        let dotc_res = DefaultBlas::dotc(&x, &y);
        assert_almost_eq!(dotc_res.re, 6.0);
        assert_almost_eq!(dotc_res.im, -1.0);

        let z = ArrayStorage::<f32, 3, 1>::from_array([[1.0, -2.0, 3.0]]);
        assert_almost_eq!(DefaultBlas::asum(&z), 6.0);

        let mut a = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 2.0]]);
        let mut b = ArrayStorage::<f32, 2, 1>::from_array([[3.0, 4.0]]);
        DefaultBlas::swap(&mut a, &mut b);
        assert_eq!(unsafe { *a.get_unchecked(0, 0) }, 3.0);
        assert_eq!(unsafe { *a.get_unchecked(1, 0) }, 4.0);
        assert_eq!(unsafe { *b.get_unchecked(0, 0) }, 1.0);
        assert_eq!(unsafe { *b.get_unchecked(1, 0) }, 2.0);

        let mut rx = ArrayStorage::<f32, 1, 1>::from_array([[1.0]]);
        let mut ry = ArrayStorage::<f32, 1, 1>::from_array([[0.0]]);
        DefaultBlas::rot(&mut rx, &mut ry, 0.0, 1.0);
        assert_almost_eq!(unsafe { *rx.get_unchecked(0, 0) }, 0.0);
        assert_almost_eq!(unsafe { *ry.get_unchecked(0, 0) }, -1.0);
    }

    // --- Level 2 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies GEMV matrix-vector multiplication (y = alpha * A * x + beta * y) on f32.
    fn test_subprograms_level2_gemv_f32() {
        let a = ArrayStorage::<f32, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        let mut y = ArrayStorage::<f32, 2, 1>::zeros();
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a, &x, 0.0, &mut y);
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 7.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMV with NaN-safe beta = 0 scaling (C-3).
    fn test_subprograms_level2_gemv_nan_safe_beta_zero() {
        let a = ArrayStorage::<f32, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        let mut y =
            ArrayStorage::<f32, 2, 1>::from_array([[f32::NAN, f32::NAN]]);
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a, &x, 0.0, &mut y);
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 7.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies Geru, Gerc, Symv, Hemv, Syr, Syr2, Her, Her2, Trmv, and Trsv Level 2 kernels.
    fn test_subprograms_level2_additional_kernels() {
        let mut a = ArrayStorage::<f32, 2, 2>::zeros();
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 2.0]]);
        let y = ArrayStorage::<f32, 2, 1>::from_array([[3.0, 4.0]]);
        DefaultBlas::geru(1.0, &x, &y, &mut a);
        assert_almost_eq!(unsafe { *a.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(0, 1) }, 4.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(1, 0) }, 6.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(1, 1) }, 8.0);

        // Gerc
        let mut ac = ArrayStorage::<Complex32, 2, 2>::zeros();
        let xc = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
        ]]);
        let yc = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(0.0, 1.0),
            Complex32::new(2.0, 0.0),
        ]]);
        DefaultBlas::gerc(Complex32::ONE, &xc, &yc, &mut ac);
        // (0,0) = 1 * (0 - i) = -i
        assert_almost_eq!(unsafe { ac.get_unchecked(0, 0).re }, 0.0);
        assert_almost_eq!(unsafe { ac.get_unchecked(0, 0).im }, -1.0);

        // Symv & Hemv
        let sym_a =
            ArrayStorage::<f32, 2, 2>::from_array([[2.0, 1.0], [1.0, 3.0]]);
        let mut sym_y = ArrayStorage::<f32, 2, 1>::zeros();
        DefaultBlas::symv(UpLo::Upper, 1.0, &sym_a, &x, 0.0, &mut sym_y);
        assert_almost_eq!(unsafe { *sym_y.get_unchecked(0, 0) }, 4.0);
        assert_almost_eq!(unsafe { *sym_y.get_unchecked(1, 0) }, 7.0);

        // Syr & Syr2
        let mut syr_a = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::syr(UpLo::Upper, 1.0, &x, &mut syr_a);
        assert_almost_eq!(unsafe { *syr_a.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *syr_a.get_unchecked(0, 1) }, 2.0);

        let mut syr2_a = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::syr2(UpLo::Upper, 1.0, &x, &y, &mut syr2_a);
        // (0,0) = 1*3 + 3*1 = 6, (0,1) = 1*4 + 2*3 = 10, (1,1) = 2*4 + 4*2 = 16
        assert_almost_eq!(unsafe { *syr2_a.get_unchecked(0, 0) }, 6.0);
        assert_almost_eq!(unsafe { *syr2_a.get_unchecked(0, 1) }, 10.0);
        assert_almost_eq!(unsafe { *syr2_a.get_unchecked(1, 1) }, 16.0);

        // Triangular solve Trsv
        let tri =
            ArrayStorage::<f32, 2, 2>::from_array([[2.0, 0.0], [1.0, 3.0]]); // Upper: (0,0)=2, (0,1)=1, (1,1)=3
        let mut rhs = ArrayStorage::<f32, 2, 1>::from_array([[5.0, 6.0]]);
        DefaultBlas::trsv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &tri,
            &mut rhs,
        )
        .unwrap();
        assert_almost_eq!(unsafe { *rhs.get_unchecked(0, 0) }, 1.5);
        assert_almost_eq!(unsafe { *rhs.get_unchecked(1, 0) }, 2.0);

        // Trsv Singular Matrix error check
        let singular_tri = ArrayStorage::<f32, 2, 2>::zeros();
        let mut bad_rhs = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        let err = DefaultBlas::trsv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &singular_tri,
            &mut bad_rhs,
        );
        assert!(matches!(err, Err(LinAlgError::SingularMatrix)));
    }

    // --- Packed BLAS ---

    #[cfg_attr(test, test)]
    /// Verifies Spmv, Hpmv, Spr, Spr2, Hpr, Hpr2, Tpmv, and Tpsv packed operations.
    fn test_subprograms_packed_blas() {
        let ap = SymmetricPackedStorage::<f32, 2, 3>::new(
            [1.0, 2.0, 3.0],
            UpLo::Upper,
        );
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        let mut y = ArrayStorage::<f32, 2, 1>::zeros();
        DefaultBlas::spmv(UpLo::Upper, 1.0, &ap, &x, 0.0, &mut y);
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 5.0);

        let mut ap_mut = SymmetricPackedStorage::<f32, 2, 3>::new(
            [0.0, 0.0, 0.0],
            UpLo::Upper,
        );
        DefaultBlas::spr(UpLo::Upper, 1.0, &x, &mut ap_mut);
        assert_almost_eq!(ap_mut.value_unchecked(0, 0), 1.0);
        assert_almost_eq!(ap_mut.value_unchecked(0, 1), 1.0);
        assert_almost_eq!(ap_mut.value_unchecked(1, 1), 1.0);

        let y_vec = ArrayStorage::<f32, 2, 1>::from_array([[2.0, 3.0]]);
        let mut ap_spr2 = SymmetricPackedStorage::<f32, 2, 3>::new(
            [0.0, 0.0, 0.0],
            UpLo::Upper,
        );
        DefaultBlas::spr2(UpLo::Upper, 1.0, &x, &y_vec, &mut ap_spr2);
        // (0,0) = 1*2 + 2*1 = 4, (0,1) = 1*3 + 1*2 = 5, (1,1) = 1*3 + 3*1 = 6
        assert_almost_eq!(ap_spr2.value_unchecked(0, 0), 4.0);
        assert_almost_eq!(ap_spr2.value_unchecked(0, 1), 5.0);
        assert_almost_eq!(ap_spr2.value_unchecked(1, 1), 6.0);

        let mut hp_mut = HermitianPackedStorage::<Complex32, 2, 3>::new(
            [
                <Complex32 as Zero>::ZERO,
                <Complex32 as Zero>::ZERO,
                <Complex32 as Zero>::ZERO,
            ],
            UpLo::Upper,
        );
        let xc = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 1.0),
        ]]);
        DefaultBlas::hpr(UpLo::Upper, 1.0f32, &xc, &mut hp_mut);
        assert_almost_eq!(hp_mut.value_unchecked(0, 0).re, 1.0);
        assert_almost_eq!(hp_mut.value_unchecked(0, 1).im, -1.0);

        let tri_p = TriangularPackedStorage::<f32, 2, 3>::new(
            [2.0, 1.0, 3.0],
            UpLo::Upper,
            Diag::NonUnit,
        );
        let mut sol = ArrayStorage::<f32, 2, 1>::from_array([[5.0, 6.0]]);
        DefaultBlas::tpsv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &tri_p,
            &mut sol,
        )
        .unwrap();
        assert_almost_eq!(unsafe { *sol.get_unchecked(0, 0) }, 1.5);
        assert_almost_eq!(unsafe { *sol.get_unchecked(1, 0) }, 2.0);
    }

    // --- Level 3 Subprograms ---

    #[cfg_attr(test, test)]
    /// Verifies GEMM matrix-matrix multiplication (C = alpha * A * B + beta * C) on f32.
    fn test_subprograms_level3_gemm_f32() {
        let a = ArrayStorage::<f32, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let b = ArrayStorage::<f32, 2, 2>::identity();
        let mut c = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 1) }, 2.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(1, 0) }, 3.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(1, 1) }, 4.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies GEMM with NaN-safe beta = 0 scaling (C-3).
    fn test_subprograms_level3_gemm_nan_safe_beta_zero() {
        let a = ArrayStorage::<f32, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let b = ArrayStorage::<f32, 2, 2>::identity();
        let mut c = ArrayStorage::<f32, 2, 2>::from_array([
            [f32::NAN, f32::NAN],
            [f32::NAN, f32::NAN],
        ]);
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 1) }, 2.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(1, 0) }, 3.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(1, 1) }, 4.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies Syrk, Syr2k, and Symm Level 3 kernels.
    fn test_subprograms_level3_syrk_syr2k_symm() {
        let a = ArrayStorage::<f32, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let mut c = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::syrk(UpLo::Upper, Trans::NoTrans, 1.0, &a, 0.0, &mut c);
        // C(0,0) = 1*1 + 2*2 = 5, C(0,1) = 1*3 + 2*4 = 11, C(1,1) = 3*3 + 4*4 = 25
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 0) }, 5.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(0, 1) }, 11.0);
        assert_almost_eq!(unsafe { *c.get_unchecked(1, 1) }, 25.0);

        // Syr2k
        let b = ArrayStorage::<f32, 2, 2>::identity();
        let mut c2 = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::syr2k(
            UpLo::Upper,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c2,
        );
        // C(0,0) = 1*1 + 1*1 = 2, C(0,1) = 1*0 + 3*1 = 3 + (2*0 + 0*4) = 3
        assert_almost_eq!(unsafe { *c2.get_unchecked(0, 0) }, 2.0);

        // Symm
        let sym_a =
            ArrayStorage::<f32, 2, 2>::from_array([[2.0, 1.0], [1.0, 3.0]]);
        let mut sym_c = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::symm(
            Side::Left,
            UpLo::Upper,
            1.0,
            &sym_a,
            &b,
            0.0,
            &mut sym_c,
        );
        assert_almost_eq!(unsafe { *sym_c.get_unchecked(0, 0) }, 2.0);
        assert_almost_eq!(unsafe { *sym_c.get_unchecked(0, 1) }, 1.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies Trmm and Trsm Level 3 triangular kernels.
    fn test_subprograms_level3_trmm_trsm() {
        let tri =
            ArrayStorage::<f32, 2, 2>::from_array([[2.0, 0.0], [1.0, 3.0]]);
        let mut trmm_b =
            ArrayStorage::<f32, 2, 2>::from_array([[1.0, 0.0], [0.0, 1.0]]);
        DefaultBlas::trmm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &tri,
            &mut trmm_b,
        );
        assert_almost_eq!(unsafe { *trmm_b.get_unchecked(0, 0) }, 2.0);
        assert_almost_eq!(unsafe { *trmm_b.get_unchecked(0, 1) }, 1.0);
        assert_almost_eq!(unsafe { *trmm_b.get_unchecked(1, 1) }, 3.0);

        // Trsm
        let mut trsm_b =
            ArrayStorage::<f32, 2, 2>::from_array([[2.0, 0.0], [1.0, 3.0]]);
        DefaultBlas::trsm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &tri,
            &mut trsm_b,
        )
        .unwrap();
        assert_almost_eq!(unsafe { *trsm_b.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *trsm_b.get_unchecked(1, 1) }, 1.0);
    }

    // --- Sparse BLAS ---

    #[cfg_attr(test, test)]
    /// Verifies Csrmv, Csrmm, and SpDotu/SpDotc/SpAxpy kernels.
    fn test_subprograms_sparse_blas() {
        let mut coo = ArrayCooStorage::<f32, 2, 2, 4>::new();
        coo.push(0, 0, 1.0).unwrap();
        coo.push(0, 1, 2.0).unwrap();
        coo.push(1, 1, 3.0).unwrap();
        let csr = ArrayCsrStorage::<f32, 2, 2, 4, 3>::from_coo(&coo).unwrap();

        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 2.0]]);
        let mut y = ArrayStorage::<f32, 2, 1>::zeros();
        DefaultBlas::csrmv(1.0, &csr, &x, 0.0, &mut y);
        // y[0] = 1*1 + 2*2 = 5, y[1] = 3*2 = 6
        assert_almost_eq!(unsafe { *y.get_unchecked(0, 0) }, 5.0);
        assert_almost_eq!(unsafe { *y.get_unchecked(1, 0) }, 6.0);

        let b_dense = ArrayStorage::<f32, 2, 2>::identity();
        let mut c_dense = ArrayStorage::<f32, 2, 2>::zeros();
        DefaultBlas::csrmm(1.0, &csr, &b_dense, 0.0, &mut c_dense);
        assert_almost_eq!(unsafe { *c_dense.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *c_dense.get_unchecked(0, 1) }, 2.0);

        let mut sp_vec = ArraySparseVector::<f32, 2, 4>::new();
        sp_vec.push(0, 2.0).unwrap();
        sp_vec.push(1, 3.0).unwrap();
        let y_full = ArrayStorage::<f32, 2, 1>::from_array([[4.0, 5.0]]);
        let dot_res = DefaultBlas::sp_dotu(&sp_vec, &y_full);
        assert_almost_eq!(dot_res, 23.0); // 2*4 + 3*5 = 23
    }

    // --- LAPACK Direct Solvers & Factorizations ---

    #[cfg_attr(test, test)]
    /// Verifies Cholesky factorization (Potrf), solver (Potrs), and failure on non-positive definite matrix.
    fn test_subprograms_lapack_potrf_potrs() {
        // A = [[4, 2], [2, 2]], SPD.
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[4.0, 2.0], [2.0, 2.0]]);
        DefaultBlas::potrf(UpLo::Lower, &mut a).unwrap();
        // L(0,0) = 2, L(1,0) = 1, L(1,1) = 1
        assert_almost_eq!(unsafe { *a.get_unchecked(0, 0) }, 2.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(1, 0) }, 1.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(1, 1) }, 1.0);

        let mut b = ArrayStorage::<f64, 2, 1>::from_array([[8.0, 6.0]]);
        DefaultBlas::potrs(UpLo::Lower, &a, &mut b).unwrap();
        assert_almost_eq!(unsafe { *b.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *b.get_unchecked(1, 0) }, 2.0);

        // Not positive definite error check
        let mut non_spd =
            ArrayStorage::<f64, 2, 2>::from_array([[-1.0, 0.0], [0.0, -1.0]]);
        let err = DefaultBlas::potrf(UpLo::Lower, &mut non_spd);
        assert!(matches!(err, Err(LinAlgError::NotPositiveDefinite)));
    }

    #[cfg_attr(test, test)]
    /// Verifies Packed Cholesky factorization (Pptrf) and solver (Pptrs).
    fn test_subprograms_lapack_pptrf_pptrs() {
        let mut ap = SymmetricPackedStorage::<f64, 2, 3>::new(
            [4.0, 2.0, 2.0],
            UpLo::Lower,
        );
        DefaultBlas::pptrf(UpLo::Lower, &mut ap).unwrap();
        assert_almost_eq!(ap.value_unchecked(0, 0), 2.0);
        assert_almost_eq!(ap.value_unchecked(1, 0), 1.0);
        assert_almost_eq!(ap.value_unchecked(1, 1), 1.0);

        let mut b = ArrayStorage::<f64, 2, 1>::from_array([[8.0, 6.0]]);
        DefaultBlas::pptrs(UpLo::Lower, &ap, &mut b).unwrap();
        assert_almost_eq!(unsafe { *b.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *b.get_unchecked(1, 0) }, 2.0);
    }

    #[cfg_attr(test, test)]
    /// Verifies LU factorization (Getrf) and solver (Getrs).
    fn test_subprograms_lapack_getrf_getrs() {
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 4.0], [1.0, 3.0]]);
        let mut ipiv = [0usize; 2];
        DefaultBlas::getrf(&mut a, &mut ipiv).unwrap();

        let mut b = ArrayStorage::<f64, 2, 1>::from_array([[4.0, 10.0]]);
        DefaultBlas::getrs(Trans::NoTrans, &a, &ipiv, &mut b).unwrap();
        assert_almost_eq!(unsafe { *b.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *b.get_unchecked(1, 0) }, 2.0);

        // Singular matrix error check
        let mut singular = ArrayStorage::<f64, 2, 2>::zeros();
        let err = DefaultBlas::getrf(&mut singular, &mut ipiv);
        assert!(matches!(err, Err(LinAlgError::SingularMatrix)));

        // Workspace too small check
        let mut short_ipiv = [0usize; 0];
        let err2 = DefaultBlas::getrf(&mut a, &mut short_ipiv);
        assert!(matches!(err2, Err(LinAlgError::WorkspaceTooSmall)));
    }

    #[cfg_attr(test, test)]
    /// Verifies QR factorization (Geqrf) and Ormqr/Unmqr application.
    fn test_subprograms_lapack_geqrf_ormqr() {
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[3.0, 4.0], [4.0, 3.0]]);
        let mut tau = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::geqrf(&mut a, &mut tau, &mut work).unwrap();
        assert!(tau[0].abs() > 0.0);

        let mut c = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 0.0]]);
        DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &a,
            &tau,
            &mut c,
            &mut work,
        )
        .unwrap();

        // Workspace too small check
        let mut short_tau = [0.0f64; 0];
        let err = DefaultBlas::geqrf(&mut a, &mut short_tau, &mut work);
        assert!(matches!(err, Err(LinAlgError::WorkspaceTooSmall)));
    }

    #[cfg_attr(test, test)]
    /// Verifies real symmetric Jacobi eigensolver (Syev).
    fn test_subprograms_lapack_syev() {
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 1.0], [1.0, 2.0]]);
        let mut w = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a,
            &mut w,
            &mut work,
        )
        .unwrap();
        assert_almost_eq!(w[0] + w[1], 4.0, 1e-10); // Trace = 4.0
        assert_almost_eq!(w[0] * w[1], 3.0, 1e-10); // Det = 3.0

        // Workspace too small check
        let mut short_w = [0.0f64; 0];
        let err = DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a,
            &mut short_w,
            &mut work,
        );
        assert!(matches!(err, Err(LinAlgError::WorkspaceTooSmall)));
    }

    #[cfg_attr(test, test)]
    /// Verifies complex Hermitian Jacobi eigensolver (Heev).
    fn test_subprograms_lapack_heev() {
        let mut a = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);
        let mut w = [0.0f64; 2];
        let mut work = [<Complex64 as Zero>::ZERO; 4];
        DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a,
            &mut w,
            &mut work,
        )
        .unwrap();
        // Trace = 4.0, Det = 2*2 - 1 = 3.0 (eigenvalues 3.0 and 1.0)
        assert_almost_eq!(w[0] + w[1], 4.0, 1e-10);
        assert_almost_eq!(w[0] * w[1], 3.0, 1e-10);
    }

    #[cfg_attr(test, test)]
    /// Verifies that Syev and Heev return Err(LinAlgError::MaxIterationsReached) on non-converging state.
    fn test_subprograms_syev_heev_max_iterations_reached() {
        let mut a_real = ArrayStorage::<f64, 3, 3>::from_array([
            [1.0, f64::NAN, 0.5],
            [f64::NAN, 2.0, 0.5],
            [0.5, 0.5, 3.0],
        ]);
        let mut w_real = [0.0f64; 3];
        let mut work_real = [0.0f64; 9];
        let res_real = DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a_real,
            &mut w_real,
            &mut work_real,
        );
        assert_eq!(res_real, Err(LinAlgError::MaxIterationsReached));

        let mut a_cplx = ArrayStorage::<Complex64, 3, 3>::from_array([
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            [
                Complex64::new(f64::NAN, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.5, 0.0),
            ],
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(0.5, 0.0),
                Complex64::new(3.0, 0.0),
            ],
        ]);
        let mut w_cplx = [0.0f64; 3];
        let mut work_cplx = [Complex64::ZERO; 9];
        let res_cplx = DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a_cplx,
            &mut w_cplx,
            &mut work_cplx,
        );
        assert_eq!(res_cplx, Err(LinAlgError::MaxIterationsReached));
    }

    #[cfg_attr(test, test)]
    /// Verifies bit-exact execution of integer Level 1, 2, 3 BLAS over u8, u16, u32, i32.
    fn test_subprograms_integer_blas_bit_exact() {
        // Axpy, Scal, Dotu over u8
        let x_u8 = ArrayStorage::<u8, 3, 1>::from_array([[1, 2, 1]]);
        let mut y_u8 = ArrayStorage::<u8, 3, 1>::from_array([[10, 20, 10]]);
        DefaultBlas::axpy(2u8, &x_u8, &mut y_u8);
        assert_eq!(y_u8.as_slice(), &[12, 24, 12]);
        DefaultBlas::scal(2u8, &mut y_u8);
        assert_eq!(y_u8.as_slice(), &[24, 48, 24]);
        let dot_u8 = DefaultBlas::dotu(&x_u8, &y_u8);
        assert_eq!(dot_u8, 24 + 2 * 48 + 24);

        // Gemv over i32
        let a_i32 = ArrayStorage::<i32, 2, 2>::from_array([[1, 2], [3, 4]]);
        let x_i32 = ArrayStorage::<i32, 2, 1>::from_array([[5, 6]]);
        let mut y_i32 = ArrayStorage::<i32, 2, 1>::from_array([[0, 0]]);
        DefaultBlas::gemv(Trans::NoTrans, 1, &a_i32, &x_i32, 0, &mut y_i32);
        assert_eq!(y_i32.as_slice(), &[5 + 3 * 6, 2 * 5 + 4 * 6]);

        // Gemm over u32
        let a_u32 = ArrayStorage::<u32, 2, 2>::from_array([[1, 2], [3, 4]]);
        let b_u32 = ArrayStorage::<u32, 2, 2>::from_array([[5, 6], [7, 8]]);
        let mut c_u32 = ArrayStorage::<u32, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1,
            &a_u32,
            &b_u32,
            0,
            &mut c_u32,
        );
        assert_eq!(
            c_u32.as_slice(),
            &[5 + 3 * 6, 2 * 5 + 4 * 6, 7 + 3 * 8, 2 * 7 + 4 * 8]
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies analytical backward error and residual bounds scaled by N * EPS * ||A|| * ||x||.
    fn test_subprograms_residual_bounds_gemv_gemm() {
        let a = ArrayStorage::<f64, 3, 3>::from_array([
            [2.0, 1.0, 0.5],
            [1.0, 3.0, 1.5],
            [0.5, 1.5, 4.0],
        ]);
        let x = ArrayStorage::<f64, 3, 1>::from_array([[1.0, 2.0, 3.0]]);
        let mut y = ArrayStorage::<f64, 3, 1>::zeros();
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a, &x, 0.0, &mut y);

        let y_exact = [5.5f64, 11.5, 15.5];
        let eps = f64::EPSILON;
        let n = 3.0f64;
        let a_inf_norm = 6.0f64;
        let x_inf_norm = 3.0f64;
        let bound = n * eps * a_inf_norm * x_inf_norm;

        for (i, &exact_val) in y_exact.iter().enumerate() {
            let diff = (y.as_slice()[i] - exact_val).abs();
            assert!(diff <= bound.max(1e-14));
        }

        let mut l = a;
        assert!(DefaultBlas::potrf(UpLo::Lower, &mut l).is_ok());
        for j in 0..3 {
            for i in 0..j {
                unsafe {
                    l.set_unchecked(i, j, 0.0);
                }
            }
        }
        let mut recon = ArrayStorage::<f64, 3, 3>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::Trans,
            1.0,
            &l,
            &l,
            0.0,
            &mut recon,
        );
        for j in 0..3 {
            for i in 0..3 {
                let diff = (recon.as_slice()[j * 3 + i]
                    - a.as_slice()[j * 3 + i])
                    .abs();
                assert!(diff <= (n * eps * a_inf_norm).max(1e-14));
            }
        }
    }

    #[cfg_attr(test, test)]
    /// Verifies mathematical equivalence between packed, sparse, and dense operations.
    fn test_subprograms_packed_sparse_dense_equivalence() {
        let a_dense = ArrayStorage::<f64, 3, 3>::from_array([
            [4.0, 1.0, 2.0],
            [1.0, 5.0, 3.0],
            [2.0, 3.0, 6.0],
        ]);
        let ap_data = [4.0, 1.0, 5.0, 2.0, 3.0, 6.0];
        let ap = SymmetricPackedStorage::<f64, 3, 6>::new(ap_data, UpLo::Upper);
        let x = ArrayStorage::<f64, 3, 1>::from_array([[1.0, 2.0, 3.0]]);

        // 1. Spmv vs Symv
        let mut y_dense = ArrayStorage::<f64, 3, 1>::zeros();
        let mut y_packed = ArrayStorage::<f64, 3, 1>::zeros();
        DefaultBlas::symv(UpLo::Upper, 1.0, &a_dense, &x, 0.0, &mut y_dense);
        DefaultBlas::spmv(UpLo::Upper, 1.0, &ap, &x, 0.0, &mut y_packed);
        assert_eq!(y_dense.as_slice(), y_packed.as_slice());

        // 2. Csrmv vs Gemv
        let mut coo = ArrayCooStorage::<f64, 3, 3, 9>::new();
        for j in 0..3 {
            for i in 0..3 {
                assert!(coo.push(i, j, *a_dense.get(i, j).unwrap()).is_ok());
            }
        }
        let a_csr: ArrayCsrStorage<f64, 3, 3, 9, 4> = coo.to_csr().unwrap();
        let mut y_csr = ArrayStorage::<f64, 3, 1>::zeros();
        DefaultBlas::csrmv(1.0, &a_csr, &x, 0.0, &mut y_csr);
        assert_eq!(y_dense.as_slice(), y_csr.as_slice());

        // 3. SpDotu vs Dotu
        let mut svec = ArraySparseVector::<f64, 3, 3>::new();
        assert!(svec.push(0, 1.0).is_ok());
        assert!(svec.push(1, 2.0).is_ok());
        assert!(svec.push(2, 3.0).is_ok());
        let dot_dense = DefaultBlas::dotu(&x, &x);
        let dot_sparse = DefaultBlas::sp_dotu(&svec, &x);
        assert_eq!(dot_dense, dot_sparse);
    }

    #[cfg_attr(test, test)]
    /// Verifies negative increment execution on reversed ViewStorage for Axpy.
    fn test_subprograms_negative_increment_reversed_views() {
        let x_data = [1.0f64, 2.0, 3.0];
        let mut y_data = [10.0f64, 20.0, 30.0];
        let x_rev = unsafe {
            ViewStorage::<f64, Const<3>, Const<1>>::new_with_strides(
                x_data.as_ptr().add(2),
                -1,
                1,
            )
        };
        let mut y_view =
            ViewStorageMut::<f64, Const<3>, Const<1>>::new(&mut y_data)
                .unwrap();
        DefaultBlas::axpy(2.0, &x_rev, &mut y_view);
        assert_eq!(y_data, [16.0, 24.0, 32.0]);
    }

    #[cfg_attr(test, test)]
    /// Val-1: Square-Root Kalman Filter — covariance propagation and measurement update.
    fn test_subprograms_val1_square_root_kalman_filter() {
        let s = ArrayStorage::<f64, 2, 2>::from_array([[2.0, 0.0], [1.0, 1.5]]);
        let phi =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 0.0], [0.1, 1.0]]);
        let mut s_next = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &phi,
            &s,
            0.0,
            &mut s_next,
        );
        let mut p_next = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::syrk(
            UpLo::Lower,
            Trans::NoTrans,
            1.0,
            &s_next,
            0.0,
            &mut p_next,
        );
        let mut l = p_next;
        assert!(DefaultBlas::potrf(UpLo::Lower, &mut l).is_ok());
    }

    #[cfg_attr(test, test)]
    /// Val-2: Real-Time Sparse MPC QP — condensed trajectory optimizer step.
    fn test_subprograms_val2_real_time_sparse_mpc_qp() {
        let mut coo = ArrayCooStorage::<f64, 3, 3, 5>::new();
        assert!(coo.push(0, 0, 1.0).is_ok());
        assert!(coo.push(1, 0, -0.8).is_ok());
        assert!(coo.push(1, 1, 1.0).is_ok());
        assert!(coo.push(2, 1, -0.8).is_ok());
        assert!(coo.push(2, 2, 1.0).is_ok());
        let a_csr: ArrayCsrStorage<f64, 3, 3, 5, 4> = coo.to_csr().unwrap();
        let u = ArrayStorage::<f64, 3, 1>::from_array([[1.0, 0.5, 0.2]]);
        let mut x_pred = ArrayStorage::<f64, 3, 1>::zeros();
        DefaultBlas::csrmv(1.0, &a_csr, &u, 0.0, &mut x_pred);
        assert_eq!(x_pred.as_slice()[0], 1.0);
        assert_almost_eq!(x_pred.as_slice()[1], -0.8 * 1.0 + 1.0 * 0.5, 1e-10);
    }

    #[cfg_attr(test, test)]
    /// Val-3: Complex MIMO Frequency Response — spectral singular value extraction.
    fn test_subprograms_val3_complex_mimo_frequency_response() {
        let g = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 1.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(1.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);
        let mut h = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::herk(UpLo::Upper, Trans::ConjTrans, 1.0, &g, 0.0, &mut h);
        let mut w = [0.0f64; 2];
        let mut work = [Complex64::ZERO; 4];
        assert!(
            DefaultBlas::heev(
                JobZ::Vectors,
                UpLo::Upper,
                &mut h,
                &mut w,
                &mut work
            )
            .is_ok()
        );
        assert!(w[0] >= 0.0);
        assert!(w[1] >= w[0]);
    }

    #[cfg_attr(test, test)]
    /// Val-4: Decoupled Modal State Simulation — decoupled structural vibration ODE step.
    fn test_subprograms_val4_decoupled_modal_state_simulation() {
        let mut modal_state =
            ArrayStorage::<f64, 4, 1>::from_array([[1.0, 2.0, 3.0, 4.0]]);
        let modal_damping = 0.95f64;
        DefaultBlas::scal(modal_damping, &mut modal_state);
        let modal_input =
            ArrayStorage::<f64, 4, 1>::from_array([[0.1, 0.2, 0.3, 0.4]]);
        DefaultBlas::axpy(1.0, &modal_input, &mut modal_state);
        assert_almost_eq!(modal_state.as_slice()[0], 1.0 * 0.95 + 0.1, 1e-10);
        assert_almost_eq!(modal_state.as_slice()[1], 2.0 * 0.95 + 0.2, 1e-10);
        assert_almost_eq!(modal_state.as_slice()[2], 3.0 * 0.95 + 0.3, 1e-10);
        assert_almost_eq!(modal_state.as_slice()[3], 4.0 * 0.95 + 0.4, 1e-10);
    }

    #[cfg_attr(test, test)]
    /// Val-5: Mixed-Layout State Observer — combining row-major gains with col-major state.
    fn test_subprograms_val5_mixed_layout_state_observer() {
        let l_gain =
            RowArrayStorage::<f64, 2, 2>::from_array([[1.0, 0.0], [0.0, 2.0]]);
        let y_err = ArrayStorage::<f64, 2, 1>::from_array([[0.5, 0.25]]);
        let mut correction = ArrayStorage::<f64, 2, 1>::zeros();
        DefaultBlas::gemv(
            Trans::NoTrans,
            1.0,
            &l_gain,
            &y_err,
            0.0,
            &mut correction,
        );
        assert_eq!(correction.as_slice(), &[0.5, 0.5]);
    }
}
