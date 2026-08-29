//! ARM NEON subprogram backend implementor for aarch64.
//!
//! Implements Level 1, 2, and 3 BLAS subprogram traits over `f32` and `f64` using
//! `core::arch::aarch64` SIMD vector intrinsics on contiguous layouts, delegating
//! to `DefaultBlas` for non-contiguous strides or remainder tails.

#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

use control_rs::math::storage::{DenseStorage, DenseStorageMut, Trans};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::math::subprograms::level1::{Axpy, Dotu, Nrm2, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::Gemm;

/// Zero-sized marker type for the ARM NEON accelerated subprogram backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NeonBlas;

// =============================================================================
// Level 1: Axpy
// =============================================================================

impl<X: DenseStorage<f32>, Y: DenseStorageMut<f32>> Axpy<f32, X, Y>
    for NeonBlas
{
    #[inline(always)]
    fn axpy(alpha: f32, x: &X, y: &mut Y) {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            debug_assert_eq!(n, y.rows() * y.cols());
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            if x_stride == 1 && y_stride == 1 {
                let chunks = n / 4;
                let rem = n % 4;
                let alpha_vec = unsafe { vdupq_n_f32(alpha) };
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 4;
                        let xv = vld1q_f32(x_ptr.add(offset));
                        let yv = vld1q_f32(y_ptr.add(offset));
                        let res = vfmaq_f32(yv, alpha_vec, xv);
                        vst1q_f32(y_ptr.add(offset), res);
                    }
                }
                for i in (n - rem)..n {
                    unsafe {
                        let xi = *x_ptr.add(i);
                        let yi = *y_ptr.add(i);
                        *y_ptr.add(i) = alpha * xi + yi;
                    }
                }
                return;
            }
        }
        DefaultBlas::axpy(alpha, x, y);
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorageMut<f64>> Axpy<f64, X, Y>
    for NeonBlas
{
    #[inline(always)]
    fn axpy(alpha: f64, x: &X, y: &mut Y) {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            debug_assert_eq!(n, y.rows() * y.cols());
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            if x_stride == 1 && y_stride == 1 {
                let chunks = n / 2;
                let rem = n % 2;
                let alpha_vec = unsafe { vdupq_n_f64(alpha) };
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 2;
                        let xv = vld1q_f64(x_ptr.add(offset));
                        let yv = vld1q_f64(y_ptr.add(offset));
                        let res = vfmaq_f64(yv, alpha_vec, xv);
                        vst1q_f64(y_ptr.add(offset), res);
                    }
                }
                for i in (n - rem)..n {
                    unsafe {
                        let xi = *x_ptr.add(i);
                        let yi = *y_ptr.add(i);
                        *y_ptr.add(i) = alpha * xi + yi;
                    }
                }
                return;
            }
        }
        DefaultBlas::axpy(alpha, x, y);
    }
}

// =============================================================================
// Level 1: Scal
// =============================================================================

impl<X: DenseStorageMut<f32>> Scal<f32, X> for NeonBlas {
    #[inline(always)]
    fn scal(alpha: f32, x: &mut X) {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            if x_stride == 1 {
                let chunks = n / 4;
                let rem = n % 4;
                let alpha_vec = unsafe { vdupq_n_f32(alpha) };
                let x_ptr = x.as_mut_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 4;
                        let xv = vld1q_f32(x_ptr.add(offset));
                        let res = vmulq_f32(xv, alpha_vec);
                        vst1q_f32(x_ptr.add(offset), res);
                    }
                }
                for i in (n - rem)..n {
                    unsafe {
                        *x_ptr.add(i) *= alpha;
                    }
                }
                return;
            }
        }
        DefaultBlas::scal(alpha, x);
    }
}

impl<X: DenseStorageMut<f64>> Scal<f64, X> for NeonBlas {
    #[inline(always)]
    fn scal(alpha: f64, x: &mut X) {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            if x_stride == 1 {
                let chunks = n / 2;
                let rem = n % 2;
                let alpha_vec = unsafe { vdupq_n_f64(alpha) };
                let x_ptr = x.as_mut_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 2;
                        let xv = vld1q_f64(x_ptr.add(offset));
                        let res = vmulq_f64(xv, alpha_vec);
                        vst1q_f64(x_ptr.add(offset), res);
                    }
                }
                for i in (n - rem)..n {
                    unsafe {
                        *x_ptr.add(i) *= alpha;
                    }
                }
                return;
            }
        }
        DefaultBlas::scal(alpha, x);
    }
}

// =============================================================================
// Level 1: Dotu
// =============================================================================

impl<X: DenseStorage<f32>, Y: DenseStorage<f32>> Dotu<f32, X, Y> for NeonBlas {
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            debug_assert_eq!(n, y.rows() * y.cols());
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            if x_stride == 1 && y_stride == 1 {
                let chunks = n / 4;
                let rem = n % 4;
                let mut acc_vec = unsafe { vdupq_n_f32(0.0) };
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 4;
                        let xv = vld1q_f32(x_ptr.add(offset));
                        let yv = vld1q_f32(y_ptr.add(offset));
                        acc_vec = vfmaq_f32(acc_vec, xv, yv);
                    }
                }
                let mut sum = unsafe { vaddvq_f32(acc_vec) };
                for i in (n - rem)..n {
                    unsafe {
                        sum += *x_ptr.add(i) * *y_ptr.add(i);
                    }
                }
                return sum;
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorage<f64>> Dotu<f64, X, Y> for NeonBlas {
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f64 {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            debug_assert_eq!(n, y.rows() * y.cols());
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            if x_stride == 1 && y_stride == 1 {
                let chunks = n / 2;
                let rem = n % 2;
                let mut acc_vec = unsafe { vdupq_n_f64(0.0) };
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 2;
                        let xv = vld1q_f64(x_ptr.add(offset));
                        let yv = vld1q_f64(y_ptr.add(offset));
                        acc_vec = vfmaq_f64(acc_vec, xv, yv);
                    }
                }
                let mut sum = unsafe { vaddvq_f64(acc_vec) };
                for i in (n - rem)..n {
                    unsafe {
                        sum += *x_ptr.add(i) * *y_ptr.add(i);
                    }
                }
                return sum;
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

// =============================================================================
// Level 1: Nrm2
// =============================================================================

impl<X: DenseStorage<f32>> Nrm2<f32, X> for NeonBlas {
    #[inline(always)]
    fn nrm2(x: &X) -> f32 {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride == 1 {
                let chunks = n / 4;
                let rem = n % 4;
                let mut acc_vec = unsafe { vdupq_n_f32(0.0) };
                let x_ptr = x.as_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 4;
                        let xv = vld1q_f32(x_ptr.add(offset));
                        acc_vec = vfmaq_f32(acc_vec, xv, xv);
                    }
                }
                let mut sum = unsafe { vaddvq_f32(acc_vec) };
                for i in (n - rem)..n {
                    unsafe {
                        let xi = *x_ptr.add(i);
                        sum += xi * xi;
                    }
                }
                return libm::sqrtf(sum);
            }
        }
        DefaultBlas::nrm2(x)
    }
}

impl<X: DenseStorage<f64>> Nrm2<f64, X> for NeonBlas {
    #[inline(always)]
    fn nrm2(x: &X) -> f64 {
        #[cfg(target_arch = "aarch64")]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride == 1 {
                let chunks = n / 2;
                let rem = n % 2;
                let mut acc_vec = unsafe { vdupq_n_f64(0.0) };
                let x_ptr = x.as_ptr();

                for i in 0..chunks {
                    unsafe {
                        let offset = i * 2;
                        let xv = vld1q_f64(x_ptr.add(offset));
                        acc_vec = vfmaq_f64(acc_vec, xv, xv);
                    }
                }
                let mut sum = unsafe { vaddvq_f64(acc_vec) };
                for i in (n - rem)..n {
                    unsafe {
                        let xi = *x_ptr.add(i);
                        sum += xi * xi;
                    }
                }
                return libm::sqrt(sum);
            }
        }
        DefaultBlas::nrm2(x)
    }
}

// =============================================================================
// Level 2: Gemv
// =============================================================================

impl<A: DenseStorage<f32>, X: DenseStorage<f32>, Y: DenseStorageMut<f32>>
    Gemv<f32, A, X, Y> for NeonBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y) {
        #[cfg(target_arch = "aarch64")]
        {
            let (m, n) = match trans {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            // Accelerated fast path: NoTrans with row-major A and contiguous vectors
            if trans == Trans::NoTrans
                && a.c_stride() == 1
                && x_stride == 1
                && y_stride == 1
            {
                let chunks = n / 4;
                let rem = n % 4;
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let lda = a.r_stride();

                for i in 0..m {
                    let mut acc_vec = unsafe { vdupq_n_f32(0.0) };
                    let row_ptr =
                        unsafe { a_ptr.offset(i.cast_signed() * lda) };

                    for k in 0..chunks {
                        unsafe {
                            let offset = k * 4;
                            let av = vld1q_f32(row_ptr.add(offset));
                            let xv = vld1q_f32(x_ptr.add(offset));
                            acc_vec = vfmaq_f32(acc_vec, av, xv);
                        }
                    }
                    let mut dot = unsafe { vaddvq_f32(acc_vec) };
                    for k in (n - rem)..n {
                        unsafe {
                            dot += *row_ptr.add(k) * *x_ptr.add(k);
                        }
                    }

                    unsafe {
                        let y_val = if beta == 0.0 {
                            0.0
                        } else {
                            *y_ptr.add(i) * beta
                        };
                        *y_ptr.add(i) = alpha * dot + y_val;
                    }
                }
                return;
            }
        }
        DefaultBlas::gemv(trans, alpha, a, x, beta, y);
    }
}

impl<A: DenseStorage<f64>, X: DenseStorage<f64>, Y: DenseStorageMut<f64>>
    Gemv<f64, A, X, Y> for NeonBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f64, a: &A, x: &X, beta: f64, y: &mut Y) {
        #[cfg(target_arch = "aarch64")]
        {
            let (m, n) = match trans {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };
            let y_stride = if y.rows() >= y.cols() {
                y.r_stride()
            } else {
                y.c_stride()
            };

            if trans == Trans::NoTrans
                && a.c_stride() == 1
                && x_stride == 1
                && y_stride == 1
            {
                let chunks = n / 2;
                let rem = n % 2;
                let x_ptr = x.as_ptr();
                let y_ptr = y.as_mut_ptr();
                let a_ptr = a.as_ptr();
                let lda = a.r_stride();

                for i in 0..m {
                    let mut acc_vec = unsafe { vdupq_n_f64(0.0) };
                    let row_ptr =
                        unsafe { a_ptr.offset(i.cast_signed() * lda) };

                    for k in 0..chunks {
                        unsafe {
                            let offset = k * 2;
                            let av = vld1q_f64(row_ptr.add(offset));
                            let xv = vld1q_f64(x_ptr.add(offset));
                            acc_vec = vfmaq_f64(acc_vec, av, xv);
                        }
                    }
                    let mut dot = unsafe { vaddvq_f64(acc_vec) };
                    for k in (n - rem)..n {
                        unsafe {
                            dot += *row_ptr.add(k) * *x_ptr.add(k);
                        }
                    }

                    unsafe {
                        let y_val = if beta == 0.0 {
                            0.0
                        } else {
                            *y_ptr.add(i) * beta
                        };
                        *y_ptr.add(i) = alpha * dot + y_val;
                    }
                }
                return;
            }
        }
        DefaultBlas::gemv(trans, alpha, a, x, beta, y);
    }
}

// =============================================================================
// Level 3: Gemm
// =============================================================================

impl<A: DenseStorage<f32>, B: DenseStorage<f32>, C: DenseStorageMut<f32>>
    Gemm<f32, A, B, C> for NeonBlas
{
    #[inline(always)]
    fn gemm(
        ta: Trans,
        tb: Trans,
        alpha: f32,
        a: &A,
        b: &B,
        beta: f32,
        c: &mut C,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            let (m, k_a) = match ta {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let (k_b, n) = match tb {
                Trans::NoTrans => (b.rows(), b.cols()),
                Trans::Trans | Trans::ConjTrans => (b.cols(), b.rows()),
            };
            debug_assert_eq!(k_a, k_b);
            let k = k_a;

            // Fast path: NoTrans + row-major contiguous for A, B, and C
            if ta == Trans::NoTrans
                && tb == Trans::NoTrans
                && a.c_stride() == 1
                && b.c_stride() == 1
                && c.c_stride() == 1
            {
                let a_lda = a.r_stride();
                let b_ldb = b.r_stride();
                let c_ldc = c.r_stride();
                let a_ptr = a.as_ptr();
                let b_ptr = b.as_ptr();
                let c_ptr = c.as_mut_ptr();

                let n_chunks = n / 4;
                let n_rem = n % 4;

                // Scale C by beta (NaN-safe)
                for i in 0..m {
                    let c_row =
                        unsafe { c_ptr.offset(i.cast_signed() * c_ldc) };
                    if beta == 0.0 {
                        for j in 0..n {
                            unsafe {
                                *c_row.add(j) = 0.0;
                            }
                        }
                    } else if beta != 1.0 {
                        for j in 0..n {
                            unsafe {
                                *c_row.add(j) *= beta;
                            }
                        }
                    }
                }

                // Vectorized row update: C[i, :] += alpha * A[i, p] * B[p, :]
                for i in 0..m {
                    let a_row =
                        unsafe { a_ptr.offset(i.cast_signed() * a_lda) };
                    let c_row =
                        unsafe { c_ptr.offset(i.cast_signed() * c_ldc) };

                    for p in 0..k {
                        let a_ip = unsafe { *a_row.add(p) };
                        let scale = alpha * a_ip;
                        let scale_vec = unsafe { vdupq_n_f32(scale) };
                        let b_row =
                            unsafe { b_ptr.offset(p.cast_signed() * b_ldb) };

                        for j in 0..n_chunks {
                            unsafe {
                                let offset = j * 4;
                                let bv = vld1q_f32(b_row.add(offset));
                                let cv = vld1q_f32(c_row.add(offset));
                                let updated = vfmaq_f32(cv, scale_vec, bv);
                                vst1q_f32(c_row.add(offset), updated);
                            }
                        }
                        for j in (n - n_rem)..n {
                            unsafe {
                                *c_row.add(j) += scale * *b_row.add(j);
                            }
                        }
                    }
                }
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}

impl<A: DenseStorage<f64>, B: DenseStorage<f64>, C: DenseStorageMut<f64>>
    Gemm<f64, A, B, C> for NeonBlas
{
    #[inline(always)]
    fn gemm(
        ta: Trans,
        tb: Trans,
        alpha: f64,
        a: &A,
        b: &B,
        beta: f64,
        c: &mut C,
    ) {
        #[cfg(target_arch = "aarch64")]
        {
            let (m, k_a) = match ta {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let (k_b, n) = match tb {
                Trans::NoTrans => (b.rows(), b.cols()),
                Trans::Trans | Trans::ConjTrans => (b.cols(), b.rows()),
            };
            debug_assert_eq!(k_a, k_b);
            let k = k_a;

            if ta == Trans::NoTrans
                && tb == Trans::NoTrans
                && a.c_stride() == 1
                && b.c_stride() == 1
                && c.c_stride() == 1
            {
                let a_lda = a.r_stride();
                let b_ldb = b.r_stride();
                let c_ldc = c.r_stride();
                let a_ptr = a.as_ptr();
                let b_ptr = b.as_ptr();
                let c_ptr = c.as_mut_ptr();

                let n_chunks = n / 2;
                let n_rem = n % 2;

                for i in 0..m {
                    let c_row =
                        unsafe { c_ptr.offset(i.cast_signed() * c_ldc) };
                    if beta == 0.0 {
                        for j in 0..n {
                            unsafe {
                                *c_row.add(j) = 0.0;
                            }
                        }
                    } else if beta != 1.0 {
                        for j in 0..n {
                            unsafe {
                                *c_row.add(j) *= beta;
                            }
                        }
                    }
                }

                for i in 0..m {
                    let a_row =
                        unsafe { a_ptr.offset(i.cast_signed() * a_lda) };
                    let c_row =
                        unsafe { c_ptr.offset(i.cast_signed() * c_ldc) };

                    for p in 0..k {
                        let a_ip = unsafe { *a_row.add(p) };
                        let scale = alpha * a_ip;
                        let scale_vec = unsafe { vdupq_n_f64(scale) };
                        let b_row =
                            unsafe { b_ptr.offset(p.cast_signed() * b_ldb) };

                        for j in 0..n_chunks {
                            unsafe {
                                let offset = j * 2;
                                let bv = vld1q_f64(b_row.add(offset));
                                let cv = vld1q_f64(c_row.add(offset));
                                let updated = vfmaq_f64(cv, scale_vec, bv);
                                vst1q_f64(c_row.add(offset), updated);
                            }
                        }
                        for j in (n - n_rem)..n {
                            unsafe {
                                *c_row.add(j) += scale * *b_row.add(j);
                            }
                        }
                    }
                }
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}
