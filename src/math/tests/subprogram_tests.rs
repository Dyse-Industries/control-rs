//! BLAS-style subprogram (`subprograms.rs`, levels 1-3, packed, sparse, and LAPACK) test suite.
#![allow(unused_imports)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::arithmetic_side_effects)]
#![allow(clippy::indexing_slicing)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::similar_names)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod subprogram_test_suite {
    use crate::assert_almost_eq;
    use crate::math::LinAlgError;
    use crate::math::complex_num::{Complex, Complex32, Complex64};
    use crate::math::num_traits::{One, Scalar, Zero};
    use crate::math::num_types::{Const, Dim};
    use crate::math::storage::{
        ArrayCooStorage, ArrayCscStorage, ArrayCsrStorage, ArraySparseVector,
        ArrayStorage, DenseStorage, DenseStorageMut, Diag,
        HermitianPackedStorage, MatrixLayout, PackedStorage, PackedStorageMut,
        RowArrayStorage, Side, SparseStorage, Storage, StorageInit, StorageMut,
        StorageView, StorageViewMut, SymmetricPackedStorage, ToCscStorage,
        ToCsrStorage, Trans, TriangularPackedStorage, UpLo,
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

    fn _inf_norm_f64<const R: usize, const C: usize>(
        a: &ArrayStorage<f64, R, C>,
    ) -> f64
    where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        let mut max = 0.0f64;
        for i in 0..R {
            let mut sum = 0.0;
            for j in 0..C {
                sum += unsafe { a.get_unchecked(i, j) }.abs();
            }
            max = max.max(sum);
        }
        max
    }

    fn _residual_inf_f64<const R: usize, const C: usize>(
        computed: &ArrayStorage<f64, R, C>,
        exact: &ArrayStorage<f64, R, C>,
    ) -> f64
    where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        let mut max = 0.0f64;
        for i in 0..R {
            let mut sum = 0.0;
            for j in 0..C {
                sum += (unsafe { *computed.get_unchecked(i, j) }
                    - unsafe { *exact.get_unchecked(i, j) })
                .abs();
            }
            max = max.max(sum);
        }
        max
    }

    fn _inf_norm_c64<const R: usize, const C: usize>(
        a: &ArrayStorage<Complex64, R, C>,
    ) -> f64
    where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        let mut max = 0.0f64;
        for i in 0..R {
            let mut sum = 0.0;
            for j in 0..C {
                sum += unsafe { a.get_unchecked(i, j) }.abs();
            }
            max = max.max(sum);
        }
        max
    }

    fn _residual_inf_c64<const R: usize, const C: usize>(
        computed: &ArrayStorage<Complex64, R, C>,
        exact: &ArrayStorage<Complex64, R, C>,
    ) -> f64
    where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        let mut max = 0.0f64;
        for i in 0..R {
            let mut sum = 0.0;
            for j in 0..C {
                let d = *unsafe { computed.get_unchecked(i, j) }
                    - *unsafe { exact.get_unchecked(i, j) };
                sum += d.abs();
            }
            max = max.max(sum);
        }
        max
    }

    fn _higham_bound(n: f64, a_inf: f64, b_inf: f64) -> f64 {
        n * f64::EPSILON * a_inf * b_inf
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
        assert_almost_eq!(unsafe { *a.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *a.get_unchecked(1, 0) }, 4.0);
        assert_almost_eq!(unsafe { *b.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *b.get_unchecked(1, 0) }, 2.0);

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
    /// Verifies Hemv with NaN-safe beta = 0 scaling (C-3, §6.1.2).
    fn test_subprograms_level2_hemv_nan_safe_beta_zero() {
        let a = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            [Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let x = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ZERO,
        ]]);
        let mut y = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::new(f64::NAN, f64::NAN),
            Complex64::new(f64::NAN, f64::NAN),
        ]]);
        DefaultBlas::hemv(
            UpLo::Upper,
            Complex64::ONE,
            &a,
            &x,
            Complex64::ZERO,
            &mut y,
        );
        assert_almost_eq!(y.get(0, 0).unwrap().re, 2.0);
        assert_almost_eq!(y.get(0, 0).unwrap().im, 0.0);
        assert_almost_eq!(y.get(1, 0).unwrap().re, 1.0);
        assert_almost_eq!(y.get(1, 0).unwrap().im, 1.0);
        assert!(y.get(0, 0).unwrap().re.is_finite());
        assert!(y.get(1, 0).unwrap().im.is_finite());
    }

    #[cfg_attr(test, test)]
    /// Verifies Spmv with NaN-safe beta = 0 scaling (C-3, §6.1.2).
    fn test_subprograms_packed_spmv_nan_safe_beta_zero() {
        let ap = SymmetricPackedStorage::<f32, 2, 3>::new(
            [2.0, 1.0, 3.0],
            UpLo::Upper,
        );
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        let mut y =
            ArrayStorage::<f32, 2, 1>::from_array([[f32::NAN, f32::NAN]]);
        DefaultBlas::spmv(UpLo::Upper, 1.0, &ap, &x, 0.0, &mut y);
        assert_almost_eq!(*y.get(0, 0).unwrap(), 3.0);
        assert_almost_eq!(*y.get(1, 0).unwrap(), 4.0);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines, clippy::cognitive_complexity)]
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

        let ha = ArrayStorage::<Complex32, 2, 2>::from_array([
            [Complex32::new(2.0, 0.0), Complex32::new(1.0, 1.0)],
            [Complex32::new(1.0, -1.0), Complex32::new(3.0, 0.0)],
        ]);
        let hx = ArrayStorage::<Complex32, 2, 1>::from_array([[
            Complex32::new(1.0, 0.0),
            Complex32::new(0.0, 0.0),
        ]]);
        let mut hy = ArrayStorage::<Complex32, 2, 1>::zeros();
        DefaultBlas::hemv(
            UpLo::Upper,
            Complex32::ONE,
            &ha,
            &hx,
            Complex32::ZERO,
            &mut hy,
        );
        assert_almost_eq!(hy.get(0, 0).unwrap().re, 2.0);
        assert_almost_eq!(hy.get(0, 0).unwrap().im, 0.0);
        assert_almost_eq!(hy.get(1, 0).unwrap().re, 1.0);
        assert_almost_eq!(hy.get(1, 0).unwrap().im, 1.0);

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

        let mut trmv_x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        DefaultBlas::trmv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &tri,
            &mut trmv_x,
        );
        assert_almost_eq!(unsafe { *trmv_x.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *trmv_x.get_unchecked(1, 0) }, 3.0);

        let tp = TriangularPackedStorage::<f32, 2, 3>::new(
            [2.0, 1.0, 3.0],
            UpLo::Upper,
            Diag::NonUnit,
        );
        let mut tpmv_x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 1.0]]);
        DefaultBlas::tpmv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &tp,
            &mut tpmv_x,
        );
        assert_almost_eq!(unsafe { *tpmv_x.get_unchecked(0, 0) }, 3.0);
        assert_almost_eq!(unsafe { *tpmv_x.get_unchecked(1, 0) }, 3.0);

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

        let mut trsm_right =
            ArrayStorage::<f32, 2, 2>::from_array([[1.0, 0.0], [0.0, 1.0]]);
        DefaultBlas::trsm(
            Side::Right,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &tri,
            &mut trsm_right,
        )
        .unwrap();
        assert_almost_eq!(unsafe { *trsm_right.get_unchecked(0, 0) }, 0.5);
        assert_almost_eq!(
            unsafe { *trsm_right.get_unchecked(0, 1) },
            -1.0 / 6.0
        );
        assert_almost_eq!(unsafe { *trsm_right.get_unchecked(1, 0) }, 0.0);
        assert_almost_eq!(
            unsafe { *trsm_right.get_unchecked(1, 1) },
            1.0 / 3.0
        );
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

        // Complex non-HPD error check
        let mut complex_non_hpd =
            ArrayStorage::<Complex64, 2, 2>::from_array([
                [Complex64::new(-1.0, 0.0), Complex64::ZERO],
                [Complex64::ZERO, Complex64::new(1.0, 0.0)],
            ]);
        let err_c = DefaultBlas::potrf(UpLo::Lower, &mut complex_non_hpd);
        assert_eq!(err_c, Err(LinAlgError::NotPositiveDefinite));
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

        // Not positive definite error check for Pptrf
        let mut non_spd_ap = SymmetricPackedStorage::<f64, 2, 3>::new(
            [-1.0, 0.0, -1.0],
            UpLo::Lower,
        );
        let err_packed = DefaultBlas::pptrf(UpLo::Lower, &mut non_spd_ap);
        assert_eq!(err_packed, Err(LinAlgError::NotPositiveDefinite));

        let mut ap_u = SymmetricPackedStorage::<f64, 2, 3>::new(
            [4.0, 2.0, 2.0],
            UpLo::Upper,
        );
        DefaultBlas::pptrf(UpLo::Upper, &mut ap_u).unwrap();
        let mut b_u = ArrayStorage::<f64, 2, 1>::from_array([[8.0, 6.0]]);
        DefaultBlas::pptrs(UpLo::Upper, &ap_u, &mut b_u).unwrap();
        assert_almost_eq!(unsafe { *b_u.get_unchecked(0, 0) }, 1.0);
        assert_almost_eq!(unsafe { *b_u.get_unchecked(1, 0) }, 2.0);
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
        let mut factored =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 4.0], [1.0, 3.0]]);
        let mut ipiv_ok = [0usize; 2];
        DefaultBlas::getrf(&mut factored, &mut ipiv_ok).unwrap();
        let mut rhs = ArrayStorage::<f64, 2, 1>::from_array([[4.0, 10.0]]);
        let err_getrs = DefaultBlas::getrs(
            Trans::NoTrans,
            &factored,
            &[] as &[usize],
            &mut rhs,
        );
        assert_eq!(err_getrs, Err(LinAlgError::WorkspaceTooSmall));
    }

    #[cfg_attr(test, test)]
    /// Verifies QR factorization (Geqrf) and Ormqr/Unmqr application.
    fn test_subprograms_lapack_geqrf_ormqr() {
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[3.0, 4.0], [4.0, 3.0]]);
        let mut tau = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::geqrf(&mut a, &mut tau, &mut work).unwrap();

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
    #[allow(clippy::too_many_lines)]
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

        // §6.1.2 oracle: a Jacobi budget of zero on a well-conditioned operand
        // returns MaxIterationsReached. The budget is passed explicitly to the
        // crate-private seam; NaN-poisoned matrices above are a separate case,
        // not this oracle.
        let mut a_real_budget =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 2.0], [2.0, 3.0]]);
        let mut w_real_budget = [0.0f64; 2];
        let mut work_real_budget = [0.0f64; 4];
        let res_real_budget = crate::math::subprograms::syev_impl(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a_real_budget,
            &mut w_real_budget,
            &mut work_real_budget,
            0,
        );
        assert_eq!(res_real_budget, Err(LinAlgError::MaxIterationsReached));

        let mut a_cplx_budget = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 1.0)],
            [Complex64::new(2.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let mut w_cplx_budget = [0.0f64; 2];
        let mut work_cplx_budget = [Complex64::ZERO; 4];
        let res_cplx_budget = crate::math::subprograms::heev_impl(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a_cplx_budget,
            &mut w_cplx_budget,
            &mut work_cplx_budget,
            0,
        );
        assert_eq!(res_cplx_budget, Err(LinAlgError::MaxIterationsReached));

        // The default budget converges on the same real operand, so the
        // assertions above isolate the budget rather than the matrix.
        let mut a_real_default =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 2.0], [2.0, 3.0]]);
        let mut w_real_default = [0.0f64; 2];
        let mut work_real_default = [0.0f64; 4];
        assert_eq!(
            DefaultBlas::syev(
                JobZ::Vectors,
                UpLo::Upper,
                &mut a_real_default,
                &mut w_real_default,
                &mut work_real_default,
            ),
            Ok(())
        );
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// Verifies WorkspaceTooSmall errors for LAPACK routines Geqrf, Ormqr, Unmqr, Syev, and Heev.
    fn test_subprograms_lapack_workspace_too_small() {
        let mut a =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 2.0], [3.0, 4.0]]);
        let mut tau = [0.0f64; 2];
        let mut work_short = [0.0f64; 1]; // Geqrf needs work.len() >= n = 2
        let mut work_ok = [0.0f64; 2];

        // 1. Geqrf
        let err_geqrf = DefaultBlas::geqrf(&mut a, &mut tau, &mut work_short);
        assert_eq!(err_geqrf, Err(LinAlgError::WorkspaceTooSmall));

        // 2. Ormqr
        let mut c = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 0.0]]);
        let tau_short = [0.0f64; 0];
        let err_ormqr_tau = DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &a,
            &tau_short,
            &mut c,
            &mut work_ok,
        );
        assert_eq!(err_ormqr_tau, Err(LinAlgError::WorkspaceTooSmall));

        let mut work_ormqr_short = [0.0f64; 0]; // Left needs work.len() >= cols of C = 1
        let err_ormqr_work = DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &a,
            &tau,
            &mut c,
            &mut work_ormqr_short,
        );
        assert_eq!(err_ormqr_work, Err(LinAlgError::WorkspaceTooSmall));

        // 3. Unmqr
        let a_c = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);
        let mut c_c = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ZERO,
        ]]);
        let tau_c = [Complex64::ZERO; 2];
        let mut work_unmqr_short = [Complex64::ZERO; 0];
        let err_unmqr = DefaultBlas::unmqr(
            Side::Left,
            Trans::NoTrans,
            &a_c,
            &tau_c,
            &mut c_c,
            &mut work_unmqr_short,
        );
        assert_eq!(err_unmqr, Err(LinAlgError::WorkspaceTooSmall));

        // 4. Syev
        let mut w = [0.0f64; 2];
        let mut work_syev_short = [0.0f64; 3]; // Vectors needs work.len() >= n * n = 4
        let err_syev = DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a,
            &mut w,
            &mut work_syev_short,
        );
        assert_eq!(err_syev, Err(LinAlgError::WorkspaceTooSmall));

        // 5. Heev
        let mut a_c_mut = a_c;
        let mut w_c = [0.0f64; 2];
        let mut work_heev_short = [Complex64::ZERO; 3]; // Vectors needs work.len() >= n * n = 4
        let err_heev = DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut a_c_mut,
            &mut w_c,
            &mut work_heev_short,
        );
        assert_eq!(err_heev, Err(LinAlgError::WorkspaceTooSmall));
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// Verifies Level 2 complex Hermitian updates (Hemv, Her, Her2) and algebraic invariants.
    fn test_subprograms_level2_and_complex_invariants() {
        use crate::math::complex_num::Complex64;
        use crate::math::storage::{
            ArrayStorage, DenseStorageMut, Trans, UpLo,
        };
        use crate::math::subprograms::level2::{Hemv, Her, Her2};

        // 1. Dotc(x, y) == Dotu(conj(x), y)
        let x = ArrayStorage::<Complex64, 3, 1>::from_array([[
            Complex64::new(1.0, 2.0),
            Complex64::new(3.0, 4.0),
            Complex64::new(5.0, 6.0),
        ]]);
        let y = ArrayStorage::<Complex64, 3, 1>::from_array([[
            Complex64::new(7.0, 8.0),
            Complex64::new(9.0, 10.0),
            Complex64::new(11.0, 12.0),
        ]]);
        let dotc_val = DefaultBlas::dotc(&x, &y);

        let mut x_conj = x;
        for i in 0..3 {
            unsafe {
                x_conj.set_unchecked(i, 0, x.get_unchecked(i, 0).conj());
            }
        }
        let dotu_conj_val = DefaultBlas::dotu(&x_conj, &y);
        assert_almost_eq!(dotc_val.re, dotu_conj_val.re);
        assert_almost_eq!(dotc_val.im, dotu_conj_val.im);

        // 2. Level 2 complex Hermitian updates: Hemv, Her, Her2
        let mut a = ArrayStorage::<Complex64, 3, 3>::from_array([
            [
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(0.5, 2.0),
            ],
            [
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(1.5, -1.0),
            ],
            [
                Complex64::new(0.5, -2.0),
                Complex64::new(1.5, 1.0),
                Complex64::new(4.0, 0.0),
            ],
        ]);
        let mut y_out = ArrayStorage::<Complex64, 3, 1>::zeros();
        DefaultBlas::hemv(
            UpLo::Upper,
            Complex64::ONE,
            &a,
            &x,
            Complex64::ZERO,
            &mut y_out,
        );
        let mut y_ref = ArrayStorage::<Complex64, 3, 1>::zeros();
        DefaultBlas::gemv(
            Trans::NoTrans,
            Complex64::ONE,
            &a,
            &x,
            Complex64::ZERO,
            &mut y_ref,
        );
        for i in 0..3 {
            assert_almost_eq!(
                y_out.get(i, 0).unwrap().re,
                y_ref.get(i, 0).unwrap().re
            );
            assert_almost_eq!(
                y_out.get(i, 0).unwrap().im,
                y_ref.get(i, 0).unwrap().im
            );
        }

        // Her: A = A + alpha * x * x^H
        DefaultBlas::her(UpLo::Upper, 2.0, &x, &mut a);

        // Her2: A = A + alpha * x * y^H + conj(alpha) * y * x^H
        DefaultBlas::her2(
            UpLo::Upper,
            Complex64::new(1.0, 1.0),
            &x,
            &y,
            &mut a,
        );

        // 3. (AB)^H == B^H A^H across Gemm(ConjTrans)
        let a_gemm = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 2.0), Complex64::new(3.0, 4.0)],
            [Complex64::new(5.0, 6.0), Complex64::new(7.0, 8.0)],
        ]);
        let b_gemm = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(9.0, 10.0), Complex64::new(11.0, 12.0)],
            [Complex64::new(13.0, 14.0), Complex64::new(15.0, 16.0)],
        ]);
        let mut ab = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &a_gemm,
            &b_gemm,
            Complex64::ZERO,
            &mut ab,
        );
        let mut ab_h = ArrayStorage::<Complex64, 2, 2>::zeros();
        for i in 0..2 {
            for j in 0..2 {
                unsafe {
                    ab_h.set_unchecked(i, j, ab.get_unchecked(j, i).conj());
                }
            }
        }

        // Compute B^H A^H
        let mut bha_h = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::ConjTrans,
            Trans::ConjTrans,
            Complex64::ONE,
            &b_gemm,
            &a_gemm,
            Complex64::ZERO,
            &mut bha_h,
        );
        for i in 0..2 {
            for j in 0..2 {
                assert_almost_eq!(
                    ab_h.get(i, j).unwrap().re,
                    bha_h.get(i, j).unwrap().re
                );
                assert_almost_eq!(
                    ab_h.get(i, j).unwrap().im,
                    bha_h.get(i, j).unwrap().im
                );
            }
        }

        // 4. 1x1 Edge cases
        let a_1x1 =
            ArrayStorage::<Complex64, 1, 1>::from_array([[Complex64::new(
                2.0, 3.0,
            )]]);
        let x_1x1 =
            ArrayStorage::<Complex64, 1, 1>::from_array([[Complex64::new(
                4.0, 5.0,
            )]]);
        let mut y_1x1 = ArrayStorage::<Complex64, 1, 1>::zeros();
        DefaultBlas::gemv(
            Trans::NoTrans,
            Complex64::ONE,
            &a_1x1,
            &x_1x1,
            Complex64::ZERO,
            &mut y_1x1,
        );
        let expected_1x1 = Complex64::new(2.0, 3.0) * Complex64::new(4.0, 5.0);
        assert_almost_eq!(y_1x1.get(0, 0).unwrap().re, expected_1x1.re);
        assert_almost_eq!(y_1x1.get(0, 0).unwrap().im, expected_1x1.im);
    }

    #[cfg_attr(test, test)]
    /// Verifies bit-exact execution of integer Level 1, 2, 3 BLAS over u8, u16, u32, i32.
    fn test_subprograms_integer_blas_bit_exact() {
        type Q8 = crate::math::fixed_num::Fixed<i16, 8>;

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

        // Gemv over u16
        let a_u16 = ArrayStorage::<u16, 2, 2>::from_array([[1, 2], [3, 4]]);
        let x_u16 = ArrayStorage::<u16, 2, 1>::from_array([[5, 6]]);
        let mut y_u16 = ArrayStorage::<u16, 2, 1>::from_array([[0, 0]]);
        DefaultBlas::gemv(Trans::NoTrans, 1, &a_u16, &x_u16, 0, &mut y_u16);
        assert_eq!(y_u16.as_slice(), &[5 + 3 * 6, 2 * 5 + 4 * 6]);

        // Gemv over Fixed at a Scalar-capable scale
        let a_q = ArrayStorage::<Q8, 2, 2>::from_array([
            [Q8::from_num(1.0), Q8::from_num(0.0)],
            [Q8::from_num(0.0), Q8::from_num(1.0)],
        ]);
        let x_q = ArrayStorage::<Q8, 2, 1>::from_array([[
            Q8::from_num(3.0),
            Q8::from_num(4.0),
        ]]);
        let mut y_q = ArrayStorage::<Q8, 2, 1>::zeros();
        DefaultBlas::gemv(
            Trans::NoTrans,
            Q8::from_num(1.0),
            &a_q,
            &x_q,
            Q8::from_num(0.0),
            &mut y_q,
        );
        assert_eq!(*y_q.get(0, 0).unwrap(), Q8::from_num(3.0));
        assert_eq!(*y_q.get(1, 0).unwrap(), Q8::from_num(4.0));
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

        for (i, &exact_val) in y_exact.iter().enumerate() {
            let diff = (y.as_slice()[i] - exact_val).abs();
            // Exact dyadic case must have 0.0 residual
            assert_almost_eq!(diff, 0.0);
        }

        // Inexact gemv case using non-dyadic values
        let a_inexact = ArrayStorage::<f64, 2, 2>::from_array([
            [1.0 / 3.0, 1.0 / 7.0],
            [1.0 / 11.0, 1.0 / 13.0],
        ]);
        let x_inexact = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 2.0]]);
        let mut y_inexact = ArrayStorage::<f64, 2, 1>::zeros();
        DefaultBlas::gemv(
            Trans::NoTrans,
            1.0,
            &a_inexact,
            &x_inexact,
            0.0,
            &mut y_inexact,
        );
        let y_exact_inexact = [17.0 / 33.0, 27.0 / 91.0];
        let bound_inexact = 2.0 * eps * 1.0 * 2.0; // n=2, ||A||_inf < 1.0, ||x||_inf = 2.0
        for (i, &exact_val) in y_exact_inexact.iter().enumerate() {
            let diff = (y_inexact.as_slice()[i] - exact_val).abs();
            assert!(diff <= bound_inexact); // No floor!
        }
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// GEMM residual `‖C_comp − C_exact‖_∞ ≤ N·EPS·‖A‖_∞‖B‖_∞` with no floor.
    fn test_subprograms_residual_bounds_gemm() {
        let a = ArrayStorage::<f64, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let b = ArrayStorage::<f64, 2, 2>::from_array([[5.0, 7.0], [6.0, 8.0]]);
        let mut c = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &b,
            0.0,
            &mut c,
        );
        let c_exact =
            ArrayStorage::<f64, 2, 2>::from_array([[19.0, 43.0], [22.0, 50.0]]);
        assert!(_residual_inf_f64(&c, &c_exact) <= 0.0);

        let a_inexact = ArrayStorage::<f64, 2, 2>::from_array([
            [1.0 / 3.0, 1.0 / 11.0],
            [1.0 / 7.0, 1.0 / 13.0],
        ]);
        let b_inexact =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 3.0], [2.0, 4.0]]);
        let mut c_inexact = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a_inexact,
            &b_inexact,
            0.0,
            &mut c_inexact,
        );
        let c_ref = ArrayStorage::<f64, 2, 2>::from_array([
            [1.0 / 3.0 + 3.0 / 7.0, 1.0 / 11.0 + 3.0 / 13.0],
            [2.0 / 3.0 + 4.0 / 7.0, 2.0 / 11.0 + 4.0 / 13.0],
        ]);
        let bound = _higham_bound(
            2.0,
            _inf_norm_f64(&a_inexact),
            _inf_norm_f64(&b_inexact),
        );
        assert!(_residual_inf_f64(&c_inexact, &c_ref) <= bound);
    }

    #[cfg_attr(test, test)]
    /// Verifies POTRF reconstruction residual against a Higham-style bound.
    fn test_subprograms_residual_bounds_potrf() {
        let a = ArrayStorage::<f64, 3, 3>::from_array([
            [2.0, 1.0, 0.5],
            [1.0, 3.0, 1.5],
            [0.5, 1.5, 4.0],
        ]);
        let eps = f64::EPSILON;
        let n = 3.0f64;
        let a_inf_norm = 6.0f64;

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
                // Inexact potrf reconstruction - assert Higham bound without floor
                assert!(diff <= n * eps * a_inf_norm);
            }
        }
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// QR residual `‖A − QR‖_∞` and orthogonality `‖QᵀQ − I‖_∞` without a floor.
    fn test_subprograms_residual_bounds_geqrf() {
        let a = ArrayStorage::<f64, 2, 2>::from_array([[3.0, 4.0], [4.0, 3.0]]);
        let mut qr = a;
        let mut tau = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::geqrf(&mut qr, &mut tau, &mut work).unwrap();

        let mut r = ArrayStorage::<f64, 2, 2>::zeros();
        for j in 0..2 {
            for i in 0..=j {
                unsafe {
                    r.set_unchecked(i, j, *qr.get_unchecked(i, j));
                }
            }
        }
        let mut q =
            ArrayStorage::<f64, 2, 2>::from_fn(
                |i, j| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        DefaultBlas::ormqr(
            Side::Left,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut q,
            &mut work,
        )
        .unwrap();
        let mut recon = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &q,
            &r,
            0.0,
            &mut recon,
        );
        let n = 2.0;
        assert!(
            _residual_inf_f64(&recon, &a)
                <= _higham_bound(n, _inf_norm_f64(&a), 1.0)
        );

        let mut qtq = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &q,
            &q,
            0.0,
            &mut qtq,
        );
        let ident =
            ArrayStorage::<f64, 2, 2>::from_fn(
                |i, j| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        assert!(_residual_inf_f64(&qtq, &ident) <= n * f64::EPSILON);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// LU residual `‖PA − LU‖_∞` without a floor.
    fn test_subprograms_residual_bounds_getrf() {
        let a = ArrayStorage::<f64, 2, 2>::from_array([[2.0, 4.0], [1.0, 3.0]]);
        let mut lu = a;
        let mut ipiv = [0usize; 2];
        DefaultBlas::getrf(&mut lu, &mut ipiv).unwrap();

        let mut pa = a;
        for (k, &p) in ipiv.iter().enumerate() {
            if p != k {
                for j in 0..2 {
                    unsafe {
                        let vk = *pa.get_unchecked(k, j);
                        let vp = *pa.get_unchecked(p, j);
                        pa.set_unchecked(k, j, vp);
                        pa.set_unchecked(p, j, vk);
                    }
                }
            }
        }
        let l = ArrayStorage::<f64, 2, 2>::from_fn(|i, j| match i.cmp(&j) {
            core::cmp::Ordering::Equal => 1.0,
            core::cmp::Ordering::Greater => unsafe { *lu.get_unchecked(i, j) },
            core::cmp::Ordering::Less => 0.0,
        });
        let u = ArrayStorage::<f64, 2, 2>::from_fn(|i, j| {
            if i <= j {
                unsafe { *lu.get_unchecked(i, j) }
            } else {
                0.0
            }
        });
        let mut recon = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &l,
            &u,
            0.0,
            &mut recon,
        );
        assert!(
            _residual_inf_f64(&recon, &pa)
                <= _higham_bound(2.0, _inf_norm_f64(&a), 1.0)
        );
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// Symmetric eigen residual `‖AV − VΛ‖_∞` and `‖VᵀV − I‖_∞`.
    fn test_subprograms_residual_bounds_syev() {
        let a = ArrayStorage::<f64, 2, 2>::from_array([[2.0, 1.0], [1.0, 2.0]]);
        let mut v = a;
        let mut w = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut v,
            &mut w,
            &mut work,
        )
        .unwrap();
        let mut av = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            1.0,
            &a,
            &v,
            0.0,
            &mut av,
        );
        let vl = ArrayStorage::<f64, 2, 2>::from_fn(|i, j| unsafe {
            *v.get_unchecked(i, j) * w[j]
        });
        let n = 2.0;
        assert!(
            _residual_inf_f64(&av, &vl)
                <= _higham_bound(n, _inf_norm_f64(&a), 1.0)
        );
        let mut vtv = ArrayStorage::<f64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::NoTrans,
            1.0,
            &v,
            &v,
            0.0,
            &mut vtv,
        );
        let ident =
            ArrayStorage::<f64, 2, 2>::from_fn(
                |i, j| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        assert!(_residual_inf_f64(&vtv, &ident) <= n * f64::EPSILON);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// Hermitian eigen residual `‖AU − UΛ‖_∞` and `‖UᴴU − I‖_∞`.
    fn test_subprograms_residual_bounds_heev() {
        let a = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(0.0, 1.0)],
            [Complex64::new(0.0, -1.0), Complex64::new(2.0, 0.0)],
        ]);
        let mut u = a;
        let mut w = [0.0f64; 2];
        let mut work = [Complex64::ZERO; 4];
        DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut u,
            &mut w,
            &mut work,
        )
        .unwrap();
        let mut au = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &a,
            &u,
            Complex64::ZERO,
            &mut au,
        );
        let ul = ArrayStorage::<Complex64, 2, 2>::from_fn(|i, j| unsafe {
            *u.get_unchecked(i, j) * Complex64::from_real(w[j])
        });
        let n = 2.0;
        assert!(
            _residual_inf_c64(&au, &ul)
                <= _higham_bound(n, _inf_norm_c64(&a), 1.0)
        );
        let mut uhu = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::ConjTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &u,
            &u,
            Complex64::ZERO,
            &mut uhu,
        );
        let ident = ArrayStorage::<Complex64, 2, 2>::from_fn(|i, j| {
            if i == j {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        assert!(_residual_inf_c64(&uhu, &ident) <= n * f64::EPSILON);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// 3×3 Hermitian Heev: eigenvalues and `‖AU − UΛ‖_∞` (catches `s`/`s̄` swap).
    ///
    /// The prior residual oracle used a 2×2 with purely imaginary off-diagonal,
    /// where the off-diag Jacobi update is a no-op and a mistaken conjugate
    /// transpose of the vector workspace can still pass. A non-trivial 3×3
    /// with mixed complex couplings exposes both the eigenvalue drift and the
    /// broken `A U = U Λ` residual.
    fn test_subprograms_heev_3x3_complex_coupling() {
        let a = ArrayStorage::<Complex64, 3, 3>::from_array([
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, -1.0),
                Complex64::new(0.5, 0.0),
            ],
            [
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(0.2, 0.3),
            ],
            [
                Complex64::new(0.5, 0.0),
                Complex64::new(0.2, -0.3),
                Complex64::new(2.0, 0.0),
            ],
        ]);
        let mut u = a;
        let mut w = [0.0f64; 3];
        let mut work = [Complex64::ZERO; 9];
        DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut u,
            &mut w,
            &mut work,
        )
        .unwrap();

        let mut sorted = w;
        sorted.sort_by(|x, y| x.partial_cmp(y).unwrap());
        // NumPy `linalg.eigh` reference for this operand.
        assert_almost_eq!(sorted[0], 1.856_872_55, 1e-7);
        assert_almost_eq!(sorted[1], 2.022_388_22, 1e-7);
        assert_almost_eq!(sorted[2], 5.120_739_22, 1e-7);

        let mut au = ArrayStorage::<Complex64, 3, 3>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &a,
            &u,
            Complex64::ZERO,
            &mut au,
        );
        let ul = ArrayStorage::<Complex64, 3, 3>::from_fn(|i, j| unsafe {
            *u.get_unchecked(i, j) * Complex64::from_real(w[j])
        });
        let n = 3.0;
        let resid = _residual_inf_c64(&au, &ul);
        // Pre-fix residual was O(1); Jacobi FP accumulation for this 3×3 is ~1e-12.
        assert!(
            resid <= 1e-9,
            "‖AU − UΛ‖_∞ too large: resid={resid} w={w:?}"
        );

        let mut uhu = ArrayStorage::<Complex64, 3, 3>::zeros();
        DefaultBlas::gemm(
            Trans::ConjTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &u,
            &u,
            Complex64::ZERO,
            &mut uhu,
        );
        let ident = ArrayStorage::<Complex64, 3, 3>::from_fn(|i, j| {
            if i == j {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        assert!(_residual_inf_c64(&uhu, &ident) <= n * f64::EPSILON * 10.0);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    /// `Unmqr` `Side::Right` residual `C ← C Q` against formed `Q`.
    fn test_subprograms_residual_bounds_unmqr_right() {
        let a0 = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 1.0)],
            [Complex64::new(4.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let mut qr = a0;
        let mut tau = [Complex64::ZERO; 2];
        let mut work = [Complex64::ZERO; 4];
        DefaultBlas::geqrf(&mut qr, &mut tau, &mut work).unwrap();
        let mut q = ArrayStorage::<Complex64, 2, 2>::from_fn(|i, j| {
            if i == j {
                Complex64::ONE
            } else {
                Complex64::ZERO
            }
        });
        DefaultBlas::unmqr(
            Side::Left,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut q,
            &mut work,
        )
        .unwrap();
        let c0 = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
            [Complex64::new(2.0, 0.0), Complex64::new(4.0, 0.0)],
        ]);
        let mut c_right = c0;
        DefaultBlas::unmqr(
            Side::Right,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut c_right,
            &mut work,
        )
        .unwrap();
        let mut c_ref = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::gemm(
            Trans::NoTrans,
            Trans::NoTrans,
            Complex64::ONE,
            &c0,
            &q,
            Complex64::ZERO,
            &mut c_ref,
        );
        assert!(
            _residual_inf_c64(&c_right, &c_ref)
                <= _higham_bound(2.0, _inf_norm_c64(&c0), _inf_norm_c64(&q))
        );
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
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
        assert_almost_eq!(dot_dense, dot_sparse);

        // 4. Hpmv vs Hemv (complex hermitian packed vs dense)
        let a_cplx_dense = ArrayStorage::<Complex64, 3, 3>::from_array([
            [
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, -2.0),
                Complex64::new(2.0, 3.0),
            ],
            [
                Complex64::new(1.0, 2.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(3.0, -1.0),
            ],
            [
                Complex64::new(2.0, -3.0),
                Complex64::new(3.0, 1.0),
                Complex64::new(6.0, 0.0),
            ],
        ]);
        let ap_cplx_data = [
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 2.0),
            Complex64::new(5.0, 0.0),
            Complex64::new(2.0, -3.0),
            Complex64::new(3.0, 1.0),
            Complex64::new(6.0, 0.0),
        ];
        let ap_cplx = HermitianPackedStorage::<Complex64, 3, 6>::new(
            ap_cplx_data,
            UpLo::Upper,
        );
        let x_cplx = ArrayStorage::<Complex64, 3, 1>::from_array([[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
            Complex64::new(3.0, 2.0),
        ]]);
        let mut y_cplx_dense = ArrayStorage::<Complex64, 3, 1>::zeros();
        let mut y_cplx_packed = ArrayStorage::<Complex64, 3, 1>::zeros();
        DefaultBlas::hemv(
            UpLo::Upper,
            Complex64::ONE,
            &a_cplx_dense,
            &x_cplx,
            Complex64::ZERO,
            &mut y_cplx_dense,
        );
        DefaultBlas::hpmv(
            UpLo::Upper,
            Complex64::ONE,
            &ap_cplx,
            &x_cplx,
            Complex64::ZERO,
            &mut y_cplx_packed,
        );
        for i in 0..3 {
            assert_almost_eq!(
                y_cplx_dense.get(i, 0).unwrap().re,
                y_cplx_packed.get(i, 0).unwrap().re
            );
            assert_almost_eq!(
                y_cplx_dense.get(i, 0).unwrap().im,
                y_cplx_packed.get(i, 0).unwrap().im
            );
        }

        // 5. Tpsv vs Trsv (packed triangular solver vs dense)
        let a_tri_dense = ArrayStorage::<f64, 3, 3>::from_array([
            [4.0, 0.0, 0.0],
            [1.0, 5.0, 0.0],
            [2.0, 3.0, 6.0],
        ]);
        let ap_tri_data = [4.0, 1.0, 5.0, 2.0, 3.0, 6.0];
        let ap_tri = TriangularPackedStorage::<f64, 3, 6>::new(
            ap_tri_data,
            UpLo::Upper,
            Diag::NonUnit,
        );
        let mut b_trsv =
            ArrayStorage::<f64, 3, 1>::from_array([[8.0, 6.0, 12.0]]);
        let mut b_tpsv =
            ArrayStorage::<f64, 3, 1>::from_array([[8.0, 6.0, 12.0]]);
        DefaultBlas::trsv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &a_tri_dense,
            &mut b_trsv,
        )
        .unwrap();
        DefaultBlas::tpsv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &ap_tri,
            &mut b_tpsv,
        )
        .unwrap();
        assert_almost_eq!(
            *b_trsv.get(0, 0).unwrap(),
            *b_tpsv.get(0, 0).unwrap()
        );
        assert_almost_eq!(
            *b_trsv.get(1, 0).unwrap(),
            *b_tpsv.get(1, 0).unwrap()
        );
        assert_almost_eq!(
            *b_trsv.get(2, 0).unwrap(),
            *b_tpsv.get(2, 0).unwrap()
        );

        // 6. Cscmv vs Gemv
        let mut coo_csc = ArrayCooStorage::<f64, 3, 3, 9>::new();
        for j in 0..3 {
            for i in 0..3 {
                assert!(
                    coo_csc.push(i, j, *a_dense.get(i, j).unwrap()).is_ok()
                );
            }
        }
        let a_csc: ArrayCscStorage<f64, 3, 3, 9, 4> = coo_csc.to_csc().unwrap();
        let mut y_csc = ArrayStorage::<f64, 3, 1>::zeros();
        DefaultBlas::cscmv(1.0, &a_csc, &x, 0.0, &mut y_csc);
        assert_eq!(y_dense.as_slice(), y_csc.as_slice());

        // 7. SpDotc vs Dotc
        let mut svec_c = ArraySparseVector::<Complex64, 3, 3>::new();
        assert!(svec_c.push(0, Complex64::new(1.0, 1.0)).is_ok());
        assert!(svec_c.push(1, Complex64::new(2.0, -1.0)).is_ok());
        assert!(svec_c.push(2, Complex64::new(3.0, 2.0)).is_ok());
        let dot_c_dense = DefaultBlas::dotc(&x_cplx, &x_cplx);
        let dot_c_sparse = DefaultBlas::sp_dotc(&svec_c, &x_cplx);
        assert_almost_eq!(dot_c_dense.re, dot_c_sparse.re);
        assert_almost_eq!(dot_c_dense.im, dot_c_sparse.im);
    }

    #[cfg_attr(test, test)]
    /// Verifies negative increment execution on reversed StorageView for Axpy.
    fn test_subprograms_negative_increment_reversed_views() {
        let x_data = [1.0f64, 2.0, 3.0];
        let mut y_data = [10.0f64, 20.0, 30.0];
        let x_rev = unsafe {
            StorageView::<f64, Const<3>, Const<1>>::new_with_strides_unchecked(
                x_data.as_ptr().add(2),
                -1,
                1,
            )
        };
        let mut y_view =
            StorageViewMut::<f64, Const<3>, Const<1>>::new_with_strides(
                &mut y_data,
                1,
                1,
            )
            .unwrap();
        DefaultBlas::axpy(2.0, &x_rev, &mut y_view);
        assert_eq!(
            y_data.map(f64::to_bits),
            [16.0f64.to_bits(), 24.0f64.to_bits(), 32.0f64.to_bits()]
        );

        let a_rev_data = [1.0f64, 0.0, 0.0, 1.0];
        let a_rev = unsafe {
            StorageView::<f64, Const<2>, Const<2>>::new_with_strides_unchecked(
                a_rev_data.as_ptr(),
                1,
                2,
            )
        };
        let x_g = [3.0f64, 4.0];
        let xg = unsafe {
            StorageView::<f64, Const<2>, Const<1>>::new_with_strides_unchecked(
                x_g.as_ptr(),
                1,
                1,
            )
        };
        let mut yg_data = [0.0f64, 0.0];
        let mut yg =
            StorageViewMut::<f64, Const<2>, Const<1>>::new_with_strides(
                &mut yg_data,
                1,
                1,
            )
            .unwrap();
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a_rev, &xg, 0.0, &mut yg);
        assert_almost_eq!(yg_data[0], 3.0);
        assert_almost_eq!(yg_data[1], 4.0);
        let mut yg_rev_data = [0.0f64, 0.0];
        let mut yg_rev = unsafe {
            StorageViewMut::<f64, Const<2>, Const<1>>::new_with_strides_unchecked(
                yg_rev_data.as_mut_ptr().add(1),
                -1,
                1,
            )
        };
        DefaultBlas::gemv(Trans::NoTrans, 1.0, &a_rev, &xg, 0.0, &mut yg_rev);
        assert_almost_eq!(yg_rev_data[0], 4.0);
        assert_almost_eq!(yg_rev_data[1], 3.0);
    }

    #[cfg_attr(test, test)]
    /// Kernel smoke: `Syrk` then `Potrf` on a 2×2 SPD Gram matrix.
    /// Not a §6.2 Val-1 discharge (10k-step covariance is deferred).
    fn test_subprograms_syrk_potrf_kernel_smoke() {
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
    /// Kernel smoke: `Csrmv` on a small lower-bidiagonal CSR operand.
    /// Not a §6.2 Val-2 discharge (Cortex-M7 timing is deferred).
    fn test_subprograms_csrmv_kernel_smoke() {
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
        assert_almost_eq!(x_pred.as_slice()[0], 1.0);
        assert_almost_eq!(x_pred.as_slice()[1], -0.8 * 1.0 + 1.0 * 0.5, 1e-10);
    }

    #[cfg_attr(test, test)]
    /// Kernel smoke: `Herk` then `Heev` on a 2×2 Gram matrix.
    /// Not a §6.2 Val-3 discharge (MATLAB/SciPy spectra are deferred).
    fn test_subprograms_herk_heev_kernel_smoke() {
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
    /// Kernel smoke: `Scal` then `Axpy` on a length-4 state.
    /// Not a §6.2 Val-4 discharge (O(N) scaling is deferred).
    fn test_subprograms_scal_axpy_kernel_smoke() {
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
    /// Kernel smoke: `Gemv` with a row-major gain and column-major residual.
    /// Not a §6.2 Val-5 discharge (observer workflow is deferred).
    fn test_subprograms_gemv_mixed_layout_kernel_smoke() {
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

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    fn test_subprograms_verification_oracles() {
        let ha = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            [Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let hx = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ZERO,
        ]]);
        let mut hy = ArrayStorage::<Complex64, 2, 1>::zeros();
        DefaultBlas::hemv(
            UpLo::Upper,
            Complex64::ONE,
            &ha,
            &hx,
            Complex64::ZERO,
            &mut hy,
        );
        assert_almost_eq!(hy.get(0, 0).unwrap().re, 2.0);
        assert_almost_eq!(hy.get(0, 0).unwrap().im, 0.0);
        assert_almost_eq!(hy.get(1, 0).unwrap().re, 1.0);
        assert_almost_eq!(hy.get(1, 0).unwrap().im, 1.0);

        let mut her_a = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::her(UpLo::Upper, 1.0, &hx, &mut her_a);
        assert_almost_eq!(her_a.get(0, 0).unwrap().re, 1.0);
        let mut her2_a = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::her2(UpLo::Upper, Complex64::ONE, &hx, &hx, &mut her2_a);
        assert_almost_eq!(her2_a.get(0, 0).unwrap().re, 2.0);

        let a0 = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, 1.0)],
            [Complex64::new(4.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let mut qr = a0;
        let mut tau = [Complex64::ZERO; 2];
        let mut work = [Complex64::ZERO; 4];
        DefaultBlas::geqrf(&mut qr, &mut tau, &mut work).unwrap();
        let mut qh_a = a0;
        DefaultBlas::unmqr(
            Side::Left,
            Trans::ConjTrans,
            &qr,
            &tau,
            &mut qh_a,
            &mut work,
        )
        .unwrap();
        assert_almost_eq!(qh_a.get(0, 0).unwrap().re, qr.get(0, 0).unwrap().re);
        assert_almost_eq!(qh_a.get(0, 1).unwrap().re, qr.get(0, 1).unwrap().re);
        assert!(qh_a.get(1, 0).unwrap().abs2() < 1e-12);

        let mut c_right = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE],
        ]);
        DefaultBlas::unmqr(
            Side::Right,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut c_right,
            &mut work,
        )
        .unwrap();
        assert!(c_right.get(0, 0).unwrap().re.is_finite());

        let mut coo = ArrayCooStorage::<f32, 2, 2, 4>::new();
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 1.0).unwrap();
        let a_csc: ArrayCscStorage<f32, 2, 2, 4, 3> = coo.to_csc().unwrap();
        let x = ArrayStorage::<f32, 2, 1>::from_array([[1.0, 2.0]]);
        let mut y_row =
            ArrayStorage::<f32, 1, 2>::from_array([[f32::NAN], [f32::NAN]]);
        DefaultBlas::cscmv(1.0, &a_csc, &x, 0.0, &mut y_row);
        assert_almost_eq!(*y_row.get(0, 0).unwrap(), 1.0);
        assert_almost_eq!(*y_row.get(0, 1).unwrap(), 2.0);

        let mut ident =
            ArrayStorage::<f64, 9, 9>::from_fn(
                |i, j| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        let mut w9 = [0.0f64; 9];
        let mut work9 = [0.0f64; 81];
        DefaultBlas::syev(
            JobZ::Vectors,
            UpLo::Upper,
            &mut ident,
            &mut w9,
            &mut work9,
        )
        .unwrap();
        for (i, w) in w9.iter().enumerate() {
            assert_almost_eq!(*w, 1.0);
            assert_almost_eq!(*ident.get(i, i).unwrap(), 1.0);
        }

        let mut garbage =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 99.0], [1.0, 3.0]]);
        let mut wg = [0.0f64; 2];
        let mut workg = [0.0f64; 4];
        DefaultBlas::syev(
            JobZ::NoVectors,
            UpLo::Upper,
            &mut garbage,
            &mut wg,
            &mut workg,
        )
        .unwrap();
        let mut clean =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 0.0], [1.0, 3.0]]);
        let mut wc = [0.0f64; 2];
        let mut workc = [0.0f64; 4];
        DefaultBlas::syev(
            JobZ::NoVectors,
            UpLo::Upper,
            &mut clean,
            &mut wc,
            &mut workc,
        )
        .unwrap();
        assert_almost_eq!(wg[0], wc[0]);
        assert_almost_eq!(wg[1], wc[1]);

        let mut t9 =
            ArrayStorage::<f64, 9, 9>::from_fn(
                |i, j| {
                    if i == j { 1.0 } else { 0.0 }
                },
            );
        let mut x9 = ArrayStorage::<f64, 9, 1>::from_array([[
            0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ]]);
        DefaultBlas::trmv(
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            &t9,
            &mut x9,
        );
        assert_almost_eq!(*x9.get(8, 0).unwrap(), 8.0);
        let a9 = t9;
        DefaultBlas::trmm(
            Side::Left,
            UpLo::Upper,
            Trans::NoTrans,
            Diag::NonUnit,
            1.0,
            &a9,
            &mut t9,
        );
        assert_almost_eq!(*t9.get(0, 0).unwrap(), 1.0);
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines, clippy::float_cmp)]
    fn test_subprograms_untaken_trans_uplo_side_diag_arms() {
        let a_c = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 1.0), Complex64::new(2.0, 0.0)],
            [Complex64::new(3.0, 0.0), Complex64::new(4.0, -1.0)],
        ]);
        let x_c = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 1.0),
        ]]);
        let mut y_row = ArrayStorage::<Complex64, 1, 2>::from_row([
            Complex64::ONE,
            Complex64::ONE,
        ]);
        DefaultBlas::gemv(
            Trans::ConjTrans,
            Complex64::new(2.0, 0.0),
            &a_c,
            &x_c,
            Complex64::new(0.5, 0.0),
            &mut y_row,
        );
        assert!(y_row.get(0, 0).unwrap().abs2() > 0.0);

        let mut y_col = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ONE,
        ]]);
        DefaultBlas::gemv(
            Trans::Trans,
            Complex64::ONE,
            &a_c,
            &x_c,
            Complex64::new(2.0, 0.0),
            &mut y_col,
        );

        let mut ha = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 1.0)],
            [Complex64::new(1.0, -1.0), Complex64::new(3.0, 0.0)],
        ]);
        let hx = ArrayStorage::<Complex64, 1, 2>::from_row([
            Complex64::ONE,
            Complex64::new(0.0, 1.0),
        ]);
        let mut hy = ArrayStorage::<Complex64, 1, 2>::from_row([
            Complex64::ONE,
            Complex64::ONE,
        ]);
        DefaultBlas::hemv(
            UpLo::Lower,
            Complex64::new(0.5, 0.0),
            &ha,
            &hx,
            Complex64::new(2.0, 0.0),
            &mut hy,
        );
        let sa =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 1.0], [1.0, 3.0]]);
        let sx = ArrayStorage::<f64, 1, 2>::from_row([1.0, 1.0]);
        let mut sy_row = ArrayStorage::<f64, 1, 2>::from_row([1.0, 1.0]);
        DefaultBlas::symv(UpLo::Lower, 2.0, &sa, &sx, 0.5, &mut sy_row);

        let mut syr_a = ArrayStorage::<f64, 2, 2>::zeros();
        let x_row = ArrayStorage::<f64, 1, 2>::from_row([1.0, 2.0]);
        DefaultBlas::syr(UpLo::Lower, 1.0, &x_row, &mut syr_a);
        DefaultBlas::syr2(UpLo::Lower, 1.0, &x_row, &x_row, &mut syr_a);

        let mut her_a = ha;
        DefaultBlas::her(UpLo::Lower, 1.0, &hx, &mut her_a);
        DefaultBlas::her2(UpLo::Lower, Complex64::ONE, &hx, &hx, &mut her_a);

        let tri_l =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 0.0], [1.0, 3.0]]);
        let mut x_tr = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 1.0]]);
        DefaultBlas::trmv(
            UpLo::Lower,
            Trans::Trans,
            Diag::Unit,
            &tri_l,
            &mut x_tr,
        );
        let mut x_cj = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ONE,
        ]]);
        DefaultBlas::trmv(
            UpLo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            &ha,
            &mut x_cj,
        );
        let mut rhs = ArrayStorage::<f64, 2, 1>::from_array([[4.0, 5.0]]);
        DefaultBlas::trsv(
            UpLo::Lower,
            Trans::Trans,
            Diag::Unit,
            &tri_l,
            &mut rhs,
        )
        .unwrap();

        let hp = HermitianPackedStorage::<Complex64, 2, 3>::new(
            [
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
            ],
            UpLo::Lower,
        );
        let mut y_hp = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::ONE,
            Complex64::ONE,
        ]]);
        DefaultBlas::hpmv(
            UpLo::Lower,
            Complex64::new(0.5, 0.0),
            &hp,
            &x_c,
            Complex64::new(2.0, 0.0),
            &mut y_hp,
        );
        let mut hp_mut = hp;
        DefaultBlas::hpr(UpLo::Lower, 1.0, &x_c, &mut hp_mut);
        DefaultBlas::hpr2(UpLo::Upper, Complex64::ONE, &x_c, &x_c, &mut hp_mut);
        DefaultBlas::hpr2(UpLo::Lower, Complex64::ONE, &hx, &hx, &mut hp_mut);

        let mut ap_lo = SymmetricPackedStorage::<f64, 2, 3>::new(
            [1.0, 2.0, 3.0],
            UpLo::Lower,
        );
        DefaultBlas::spr(UpLo::Lower, 1.0, &x_row, &mut ap_lo);
        DefaultBlas::spr2(UpLo::Lower, 1.0, &x_row, &x_row, &mut ap_lo);
        let mut y_sp = ArrayStorage::<f64, 1, 2>::from_row([1.0, 1.0]);
        DefaultBlas::spmv(UpLo::Lower, 0.5, &ap_lo, &x_row, 2.0, &mut y_sp);

        let tp = TriangularPackedStorage::<f64, 2, 3>::new(
            [2.0, 1.0, 3.0],
            UpLo::Lower,
            Diag::Unit,
        );
        let mut tpx = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 1.0]]);
        DefaultBlas::tpmv(UpLo::Lower, Trans::Trans, Diag::Unit, &tp, &mut tpx);
        let tpc = TriangularPackedStorage::<Complex64, 2, 3>::new(
            [
                Complex64::new(2.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(3.0, 0.0),
            ],
            UpLo::Lower,
            Diag::NonUnit,
        );
        let mut tpxc = x_c;
        DefaultBlas::tpmv(
            UpLo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            &tpc,
            &mut tpxc,
        );
        let mut tps = ArrayStorage::<f64, 2, 1>::from_array([[4.0, 5.0]]);
        DefaultBlas::tpsv(UpLo::Lower, Trans::Trans, Diag::Unit, &tp, &mut tps)
            .unwrap();
        let mut tpsc = x_c;
        DefaultBlas::tpsv(
            UpLo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            &tpc,
            &mut tpsc,
        )
        .unwrap();
        let tp_sing = TriangularPackedStorage::<f64, 2, 3>::new(
            [0.0, 1.0, 0.0],
            UpLo::Lower,
            Diag::NonUnit,
        );
        let mut bad = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 1.0]]);
        assert!(
            DefaultBlas::tpsv(
                UpLo::Lower,
                Trans::NoTrans,
                Diag::NonUnit,
                &tp_sing,
                &mut bad,
            )
            .is_err()
        );

        let mut c_beta = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::ONE, Complex64::ONE],
            [Complex64::ONE, Complex64::ONE],
        ]);
        DefaultBlas::gemm(
            Trans::Trans,
            Trans::Trans,
            Complex64::ONE,
            &a_c,
            &a_c,
            Complex64::new(0.5, 0.0),
            &mut c_beta,
        );

        let mut sc = ArrayStorage::<f64, 2, 2>::zeros();
        let b_id = ArrayStorage::<f64, 2, 2>::identity();
        let sym_a =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 1.0], [1.0, 3.0]]);
        DefaultBlas::symm(
            Side::Right,
            UpLo::Lower,
            1.0,
            &sym_a,
            &b_id,
            0.5,
            &mut sc,
        );
        DefaultBlas::hemm(
            Side::Left,
            UpLo::Lower,
            Complex64::ONE,
            &ha,
            &a_c,
            Complex64::new(0.5, 0.0),
            &mut c_beta,
        );
        DefaultBlas::hemm(
            Side::Right,
            UpLo::Upper,
            Complex64::ONE,
            &ha,
            &a_c,
            Complex64::ZERO,
            &mut c_beta,
        );
        DefaultBlas::hemm(
            Side::Right,
            UpLo::Lower,
            Complex64::ONE,
            &ha,
            &a_c,
            Complex64::ONE,
            &mut c_beta,
        );

        let mut rk =
            ArrayStorage::<f64, 2, 2>::from_array([[1.0, 0.0], [0.0, 1.0]]);
        DefaultBlas::syrk(UpLo::Lower, Trans::Trans, 1.0, &sym_a, 0.5, &mut rk);
        DefaultBlas::syr2k(
            UpLo::Lower,
            Trans::Trans,
            1.0,
            &sym_a,
            &b_id,
            0.5,
            &mut rk,
        );
        let mut hrk = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::herk(
            UpLo::Lower,
            Trans::NoTrans,
            1.0,
            &a_c,
            0.0,
            &mut hrk,
        );
        DefaultBlas::herk(
            UpLo::Lower,
            Trans::ConjTrans,
            0.5,
            &a_c,
            2.0,
            &mut hrk,
        );
        DefaultBlas::her2k(
            UpLo::Lower,
            Trans::NoTrans,
            Complex64::ONE,
            &a_c,
            &a_c,
            0.0,
            &mut hrk,
        );
        DefaultBlas::her2k(
            UpLo::Upper,
            Trans::ConjTrans,
            Complex64::new(1.0, 1.0),
            &a_c,
            &a_c,
            0.5,
            &mut hrk,
        );
        DefaultBlas::her2k(
            UpLo::Lower,
            Trans::Trans,
            Complex64::ONE,
            &a_c,
            &a_c,
            1.0,
            &mut hrk,
        );

        let mut tb = b_id;
        DefaultBlas::trmm(
            Side::Right,
            UpLo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            1.0,
            &tri_l,
            &mut tb,
        );
        DefaultBlas::trmm(
            Side::Left,
            UpLo::Lower,
            Trans::Trans,
            Diag::NonUnit,
            1.0,
            &tri_l,
            &mut tb,
        );
        DefaultBlas::trmm(
            Side::Right,
            UpLo::Upper,
            Trans::Trans,
            Diag::NonUnit,
            1.0,
            &tri_l,
            &mut tb,
        );
        let mut tbc = a_c;
        DefaultBlas::trmm(
            Side::Left,
            UpLo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::ONE,
            &ha,
            &mut tbc,
        );
        let mut ts = ArrayStorage::<f64, 2, 2>::identity();
        DefaultBlas::trsm(
            Side::Right,
            UpLo::Lower,
            Trans::NoTrans,
            Diag::Unit,
            1.0,
            &tri_l,
            &mut ts,
        )
        .unwrap();
        DefaultBlas::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::Trans,
            Diag::NonUnit,
            1.0,
            &tri_l,
            &mut ts,
        )
        .unwrap();
        let mut tsc = a_c;
        DefaultBlas::trsm(
            Side::Left,
            UpLo::Lower,
            Trans::ConjTrans,
            Diag::NonUnit,
            Complex64::ONE,
            &ha,
            &mut tsc,
        )
        .unwrap();

        let mut coo = ArrayCooStorage::<f64, 2, 2, 4>::new();
        coo.push(0, 0, 1.0).unwrap();
        coo.push(1, 1, 2.0).unwrap();
        let csr = ArrayCsrStorage::<f64, 2, 2, 4, 3>::from_coo(&coo).unwrap();
        let csc = ArrayCscStorage::<f64, 2, 2, 4, 3>::from_coo(&coo).unwrap();
        let x_sp = ArrayStorage::<f64, 1, 2>::from_row([1.0, 1.0]);
        let mut y_sp2 = ArrayStorage::<f64, 1, 2>::from_row([1.0, 1.0]);
        DefaultBlas::csrmv(1.0, &csr, &x_sp, 0.5, &mut y_sp2);
        DefaultBlas::cscmv(1.0, &csc, &x_sp, 2.0, &mut y_sp2);
        let mut cm = ArrayStorage::<f64, 2, 2>::identity();
        DefaultBlas::csrmm(1.0, &csr, &b_id, 0.5, &mut cm);
        let mut svec = ArraySparseVector::<f64, 2, 2>::new();
        svec.push(0, 1.0).unwrap();
        let y_row_f = ArrayStorage::<f64, 1, 2>::from_row([3.0, 4.0]);
        let _ = DefaultBlas::sp_dotu(&svec, &y_row_f);
        let _ = DefaultBlas::sp_dotc(&svec, &y_row_f);
        let mut y_ax = y_row_f;
        DefaultBlas::sp_axpy(1.0, &svec, &mut y_ax);

        let mut chol =
            ArrayStorage::<f64, 2, 2>::from_array([[4.0, 1.0], [1.0, 3.0]]);
        DefaultBlas::potrf(UpLo::Upper, &mut chol).unwrap();
        let mut rhs_p = ArrayStorage::<f64, 2, 1>::from_array([[1.0, 1.0]]);
        DefaultBlas::potrs(UpLo::Upper, &chol, &mut rhs_p).unwrap();
        let mut pp = SymmetricPackedStorage::<f64, 2, 3>::new(
            [4.0, 1.0, 3.0],
            UpLo::Upper,
        );
        DefaultBlas::pptrf(UpLo::Upper, &mut pp).unwrap();
        DefaultBlas::pptrs(UpLo::Upper, &pp, &mut rhs_p).unwrap();

        let mut lu = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(2.0, 0.0), Complex64::new(1.0, 0.0)],
            [Complex64::new(1.0, 0.0), Complex64::new(3.0, 0.0)],
        ]);
        let mut ipiv = [0usize; 2];
        DefaultBlas::getrf(&mut lu, &mut ipiv).unwrap();
        let mut b_lu = a_c;
        DefaultBlas::getrs(Trans::Trans, &lu, &ipiv, &mut b_lu).unwrap();
        DefaultBlas::getrs(Trans::ConjTrans, &lu, &ipiv, &mut b_lu).unwrap();

        let mut qr =
            ArrayStorage::<f64, 2, 2>::from_array([[3.0, 4.0], [4.0, 3.0]]);
        let mut tau = [0.0f64; 2];
        let mut work = [0.0f64; 4];
        DefaultBlas::geqrf(&mut qr, &mut tau, &mut work).unwrap();
        let mut c_left = ArrayStorage::<f64, 2, 2>::identity();
        DefaultBlas::ormqr(
            Side::Left,
            Trans::Trans,
            &qr,
            &tau,
            &mut c_left,
            &mut work,
        )
        .unwrap();
        DefaultBlas::ormqr(
            Side::Right,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut c_left,
            &mut work,
        )
        .unwrap();
        DefaultBlas::ormqr(
            Side::Right,
            Trans::Trans,
            &qr,
            &tau,
            &mut c_left,
            &mut work,
        )
        .unwrap();
        DefaultBlas::ormqr(
            Side::Left,
            Trans::ConjTrans,
            &qr,
            &tau,
            &mut c_left,
            &mut work,
        )
        .unwrap();

        let mut qrc = a_c;
        let mut tauc = [Complex64::ZERO; 2];
        let mut workc = [Complex64::ZERO; 4];
        DefaultBlas::geqrf(&mut qrc, &mut tauc, &mut workc).unwrap();
        let mut cc = a_c;
        DefaultBlas::unmqr(
            Side::Left,
            Trans::NoTrans,
            &qrc,
            &tauc,
            &mut cc,
            &mut workc,
        )
        .unwrap();
        DefaultBlas::unmqr(
            Side::Right,
            Trans::ConjTrans,
            &qrc,
            &tauc,
            &mut cc,
            &mut workc,
        )
        .unwrap();
        DefaultBlas::unmqr(
            Side::Right,
            Trans::Trans,
            &qrc,
            &tauc,
            &mut cc,
            &mut workc,
        )
        .unwrap();

        let mut sy =
            ArrayStorage::<f64, 2, 2>::from_array([[2.0, 1.0], [1.0, 2.0]]);
        let mut w = [0.0f64; 2];
        DefaultBlas::syev(
            JobZ::NoVectors,
            UpLo::Lower,
            &mut sy,
            &mut w,
            &mut work,
        )
        .unwrap();
        let mut he = ha;
        DefaultBlas::heev(
            JobZ::NoVectors,
            UpLo::Lower,
            &mut he,
            &mut w,
            &mut workc,
        )
        .unwrap();
        DefaultBlas::heev(
            JobZ::Vectors,
            UpLo::Lower,
            &mut ha,
            &mut w,
            &mut workc,
        )
        .unwrap();
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::too_many_lines)]
    fn test_herk_unmqr_trans_distinct_from_conjtrans() {
        // HERK: Trans must preserve a real Hermitian diagonal (coerced to AᴴA).
        let a = ArrayStorage::<Complex64, 2, 1>::from_array([[
            Complex64::new(1.0, 1.0),
            Complex64::new(2.0, -1.0),
        ]]);
        let mut c_t = ArrayStorage::<Complex64, 2, 2>::zeros();
        let mut c_c = ArrayStorage::<Complex64, 2, 2>::zeros();
        DefaultBlas::herk(UpLo::Lower, Trans::Trans, 1.0, &a, 0.0, &mut c_t);
        DefaultBlas::herk(
            UpLo::Lower,
            Trans::ConjTrans,
            1.0,
            &a,
            0.0,
            &mut c_c,
        );
        let t00 = *c_t.get(0, 0).unwrap();
        let c00 = *c_c.get(0, 0).unwrap();
        // |1+i|² + |2−i|² = 7
        assert_almost_eq!(c00.re, 7.0, 1e-12);
        assert_almost_eq!(c00.im, 0.0, 1e-12);
        assert_almost_eq!(t00.re, 7.0, 1e-12);
        assert_almost_eq!(t00.im, 0.0, 1e-12);

        // UNMQR: Trans applies Qᵀ, ConjTrans applies Qᴴ — distinct for complex Q.
        let mut qr = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::new(1.0, 2.0), Complex64::new(0.5, 0.25)],
            [Complex64::new(3.0, -1.0), Complex64::new(2.0, 1.0)],
        ]);
        let mut tau = [Complex64::ZERO; 2];
        let mut work = [Complex64::ZERO; 4];
        DefaultBlas::geqrf(&mut qr, &mut tau, &mut work).unwrap();

        let mut q = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE],
        ]);
        DefaultBlas::unmqr(
            Side::Left,
            Trans::NoTrans,
            &qr,
            &tau,
            &mut q,
            &mut work,
        )
        .unwrap();

        let mut i_t = ArrayStorage::<Complex64, 2, 2>::from_array([
            [Complex64::ONE, Complex64::ZERO],
            [Complex64::ZERO, Complex64::ONE],
        ]);
        let mut i_c = i_t;
        DefaultBlas::unmqr(
            Side::Left,
            Trans::Trans,
            &qr,
            &tau,
            &mut i_t,
            &mut work,
        )
        .unwrap();
        DefaultBlas::unmqr(
            Side::Left,
            Trans::ConjTrans,
            &qr,
            &tau,
            &mut i_c,
            &mut work,
        )
        .unwrap();

        for i in 0..2 {
            for j in 0..2 {
                let q_ji = *q.get(j, i).unwrap();
                let qh_ij = q_ji.conj();
                let qt_ij = q_ji; // Qᵀ[i,j] = Q[j,i]
                let got_h = *i_c.get(i, j).unwrap();
                let got_t = *i_t.get(i, j).unwrap();
                assert_almost_eq!(got_h.re, qh_ij.re, 1e-10);
                assert_almost_eq!(got_h.im, qh_ij.im, 1e-10);
                assert_almost_eq!(got_t.re, qt_ij.re, 1e-10);
                assert_almost_eq!(got_t.im, qt_ij.im, 1e-10);
            }
        }
        // Sanity: Qᵀ ≠ Qᴴ for this complex factor.
        let diff = (*i_t.get(0, 1).unwrap() - *i_c.get(0, 1).unwrap()).abs();
        assert!(diff > 1e-6);
    }
}
