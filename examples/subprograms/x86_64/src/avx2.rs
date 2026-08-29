//! x86_64 AVX2 + FMA hardware-accelerated subprogram backend.
//!
//! Implements Level 1, 2, and 3 BLAS subprograms using 8-lane (`f32`) and 4-lane (`f64`)
//! `core::arch::x86_64` intrinsics, guarded by runtime CPU feature detection.

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

use control_rs::math::storage::{DenseStorage, DenseStorageMut, Trans};
use control_rs::math::subprograms::level1::{Axpy, Dotu, Nrm2, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::Gemm;
use control_rs::math::subprograms::DefaultBlas;

/// Zero-sized marker type for the AVX2+FMA accelerated subprogram backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Avx2Blas;

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_ps(v: __m256) -> f32 {
    let vlow = _mm256_castps256_ps128(v);
    let vhigh = _mm256_extractf128_ps(v, 1);
    let sum128 = _mm_add_ps(vlow, vhigh);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    _mm_cvtss_f32(final_sum)
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256_pd(v: __m256d) -> f64 {
    let vlow = _mm256_extractf128_pd(v, 0);
    let vhigh = _mm256_extractf128_pd(v, 1);
    let sum128 = _mm_add_pd(vlow, vhigh);
    let shuf = _mm_unpackhi_pd(sum128, sum128);
    let final_sum = _mm_add_sd(sum128, shuf);
    _mm_cvtsd_f64(final_sum)
}

// =============================================================================
// Level 1: Axpy
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f32_avx2(n: usize, alpha: f32, x_ptr: *const f32, y_ptr: *mut f32) {
    let chunks = n / 8;
    let rem = n % 8;
    let alpha_vec = _mm256_set1_ps(alpha);

    for i in 0..chunks {
        let offset = i * 8;
        let xv = _mm256_loadu_ps(x_ptr.add(offset));
        let yv = _mm256_loadu_ps(y_ptr.add(offset));
        let res = _mm256_fmadd_ps(alpha_vec, xv, yv);
        _mm256_storeu_ps(y_ptr.add(offset), res);
    }
    for i in (n - rem)..n {
        let xi = *x_ptr.add(i);
        let yi = *y_ptr.add(i);
        *y_ptr.add(i) = alpha * xi + yi;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn axpy_f64_avx2(n: usize, alpha: f64, x_ptr: *const f64, y_ptr: *mut f64) {
    let chunks = n / 4;
    let rem = n % 4;
    let alpha_vec = _mm256_set1_pd(alpha);

    for i in 0..chunks {
        let offset = i * 4;
        let xv = _mm256_loadu_pd(x_ptr.add(offset));
        let yv = _mm256_loadu_pd(y_ptr.add(offset));
        let res = _mm256_fmadd_pd(alpha_vec, xv, yv);
        _mm256_storeu_pd(y_ptr.add(offset), res);
    }
    for i in (n - rem)..n {
        let xi = *x_ptr.add(i);
        let yi = *y_ptr.add(i);
        *y_ptr.add(i) = alpha * xi + yi;
    }
}

impl<X: DenseStorage<f32>, Y: DenseStorageMut<f32>> Axpy<f32, X, Y> for Avx2Blas {
    #[inline(always)]
    fn axpy(alpha: f32, x: &X, y: &mut Y) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if x_stride == 1 && y_stride == 1 {
                    unsafe {
                        axpy_f32_avx2(n, alpha, x.as_ptr(), y.as_mut_ptr());
                    }
                    return;
                }
            }
        }
        DefaultBlas::axpy(alpha, x, y);
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorageMut<f64>> Axpy<f64, X, Y> for Avx2Blas {
    #[inline(always)]
    fn axpy(alpha: f64, x: &X, y: &mut Y) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if x_stride == 1 && y_stride == 1 {
                    unsafe {
                        axpy_f64_avx2(n, alpha, x.as_ptr(), y.as_mut_ptr());
                    }
                    return;
                }
            }
        }
        DefaultBlas::axpy(alpha, x, y);
    }
}

// =============================================================================
// Level 1: Scal
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scal_f32_avx2(n: usize, alpha: f32, x_ptr: *mut f32) {
    let chunks = n / 8;
    let rem = n % 8;
    let alpha_vec = _mm256_set1_ps(alpha);

    for i in 0..chunks {
        let offset = i * 8;
        let xv = _mm256_loadu_ps(x_ptr.add(offset));
        let res = _mm256_mul_ps(alpha_vec, xv);
        _mm256_storeu_ps(x_ptr.add(offset), res);
    }
    for i in (n - rem)..n {
        *x_ptr.add(i) *= alpha;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scal_f64_avx2(n: usize, alpha: f64, x_ptr: *mut f64) {
    let chunks = n / 4;
    let rem = n % 4;
    let alpha_vec = _mm256_set1_pd(alpha);

    for i in 0..chunks {
        let offset = i * 4;
        let xv = _mm256_loadu_pd(x_ptr.add(offset));
        let res = _mm256_mul_pd(alpha_vec, xv);
        _mm256_storeu_pd(x_ptr.add(offset), res);
    }
    for i in (n - rem)..n {
        *x_ptr.add(i) *= alpha;
    }
}

impl<X: DenseStorageMut<f32>> Scal<f32, X> for Avx2Blas {
    #[inline(always)]
    fn scal(alpha: f32, x: &mut X) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let n = x.rows() * x.cols();
                let stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                if stride == 1 {
                    unsafe {
                        scal_f32_avx2(n, alpha, x.as_mut_ptr());
                    }
                    return;
                }
            }
        }
        DefaultBlas::scal(alpha, x);
    }
}

impl<X: DenseStorageMut<f64>> Scal<f64, X> for Avx2Blas {
    #[inline(always)]
    fn scal(alpha: f64, x: &mut X) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") {
                let n = x.rows() * x.cols();
                let stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                if stride == 1 {
                    unsafe {
                        scal_f64_avx2(n, alpha, x.as_mut_ptr());
                    }
                    return;
                }
            }
        }
        DefaultBlas::scal(alpha, x);
    }
}

// =============================================================================
// Level 1: Dotu & Nrm2
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dotu_f32_avx2(n: usize, x_ptr: *const f32, y_ptr: *const f32) -> f32 {
    let chunks = n / 8;
    let rem = n % 8;
    let mut acc = _mm256_setzero_ps();

    for i in 0..chunks {
        let offset = i * 8;
        let xv = _mm256_loadu_ps(x_ptr.add(offset));
        let yv = _mm256_loadu_ps(y_ptr.add(offset));
        acc = _mm256_fmadd_ps(xv, yv, acc);
    }

    let mut sum = hsum256_ps(acc);
    for i in (n - rem)..n {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dotu_f64_avx2(n: usize, x_ptr: *const f64, y_ptr: *const f64) -> f64 {
    let chunks = n / 4;
    let rem = n % 4;
    let mut acc = _mm256_setzero_pd();

    for i in 0..chunks {
        let offset = i * 4;
        let xv = _mm256_loadu_pd(x_ptr.add(offset));
        let yv = _mm256_loadu_pd(y_ptr.add(offset));
        acc = _mm256_fmadd_pd(xv, yv, acc);
    }

    let mut sum = hsum256_pd(acc);
    for i in (n - rem)..n {
        sum += *x_ptr.add(i) * *y_ptr.add(i);
    }
    sum
}

impl<X: DenseStorage<f32>, Y: DenseStorage<f32>> Dotu<f32, X, Y> for Avx2Blas {
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if x_stride == 1 && y_stride == 1 {
                    return unsafe { dotu_f32_avx2(n, x.as_ptr(), y.as_ptr()) };
                }
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorage<f64>> Dotu<f64, X, Y> for Avx2Blas {
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if x_stride == 1 && y_stride == 1 {
                    return unsafe { dotu_f64_avx2(n, x.as_ptr(), y.as_ptr()) };
                }
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

impl<X: DenseStorage<f32>> Nrm2<f32, X> for Avx2Blas {
    #[inline(always)]
    fn nrm2(x: &X) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                if x_stride == 1 {
                    let sum = unsafe { dotu_f32_avx2(n, x.as_ptr(), x.as_ptr()) };
                    return libm::sqrtf(sum);
                }
            }
        }
        DefaultBlas::nrm2(x)
    }
}

impl<X: DenseStorage<f64>> Nrm2<f64, X> for Avx2Blas {
    #[inline(always)]
    fn nrm2(x: &X) -> f64 {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let n = x.rows() * x.cols();
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                if x_stride == 1 {
                    let sum = unsafe { dotu_f64_avx2(n, x.as_ptr(), x.as_ptr()) };
                    return libm::sqrt(sum);
                }
            }
        }
        DefaultBlas::nrm2(x)
    }
}

// =============================================================================
// Level 2: Gemv
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_notrans_rowmajor_f32_avx2(
    m: usize,
    n: usize,
    alpha: f32,
    a_ptr: *const f32,
    lda: isize,
    x_ptr: *const f32,
    beta: f32,
    y_ptr: *mut f32,
) {
    let chunks = n / 8;
    let rem = n % 8;

    for i in 0..m {
        let mut acc_vec = _mm256_setzero_ps();
        let row_ptr = a_ptr.offset(i.cast_signed() * lda);

        for k in 0..chunks {
            let offset = k * 8;
            let av = _mm256_loadu_ps(row_ptr.add(offset));
            let xv = _mm256_loadu_ps(x_ptr.add(offset));
            acc_vec = _mm256_fmadd_ps(av, xv, acc_vec);
        }
        let mut dot = hsum256_ps(acc_vec);
        for k in (n - rem)..n {
            dot += *row_ptr.add(k) * *x_ptr.add(k);
        }

        let y_val = if beta == 0.0 { 0.0 } else { *y_ptr.add(i) * beta };
        *y_ptr.add(i) = alpha * dot + y_val;
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemv_notrans_rowmajor_f64_avx2(
    m: usize,
    n: usize,
    alpha: f64,
    a_ptr: *const f64,
    lda: isize,
    x_ptr: *const f64,
    beta: f64,
    y_ptr: *mut f64,
) {
    let chunks = n / 4;
    let rem = n % 4;

    for i in 0..m {
        let mut acc_vec = _mm256_setzero_pd();
        let row_ptr = a_ptr.offset(i.cast_signed() * lda);

        for k in 0..chunks {
            let offset = k * 4;
            let av = _mm256_loadu_pd(row_ptr.add(offset));
            let xv = _mm256_loadu_pd(x_ptr.add(offset));
            acc_vec = _mm256_fmadd_pd(av, xv, acc_vec);
        }
        let mut dot = hsum256_pd(acc_vec);
        for k in (n - rem)..n {
            dot += *row_ptr.add(k) * *x_ptr.add(k);
        }

        let y_val = if beta == 0.0 { 0.0 } else { *y_ptr.add(i) * beta };
        *y_ptr.add(i) = alpha * dot + y_val;
    }
}

impl<A: DenseStorage<f32>, X: DenseStorage<f32>, Y: DenseStorageMut<f32>>
    Gemv<f32, A, X, Y> for Avx2Blas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if trans == Trans::NoTrans && a.c_stride() == 1 && x_stride == 1 && y_stride == 1 {
                    unsafe {
                        gemv_notrans_rowmajor_f32_avx2(
                            a.rows(),
                            a.cols(),
                            alpha,
                            a.as_ptr(),
                            a.r_stride(),
                            x.as_ptr(),
                            beta,
                            y.as_mut_ptr(),
                        );
                    }
                    return;
                }
            }
        }
        DefaultBlas::gemv(trans, alpha, a, x, beta, y);
    }
}

impl<A: DenseStorage<f64>, X: DenseStorage<f64>, Y: DenseStorageMut<f64>>
    Gemv<f64, A, X, Y> for Avx2Blas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f64, a: &A, x: &X, beta: f64, y: &mut Y) {
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
                let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

                if trans == Trans::NoTrans && a.c_stride() == 1 && x_stride == 1 && y_stride == 1 {
                    unsafe {
                        gemv_notrans_rowmajor_f64_avx2(
                            a.rows(),
                            a.cols(),
                            alpha,
                            a.as_ptr(),
                            a.r_stride(),
                            x.as_ptr(),
                            beta,
                            y.as_mut_ptr(),
                        );
                    }
                    return;
                }
            }
        }
        DefaultBlas::gemv(trans, alpha, a, x, beta, y);
    }
}

// =============================================================================
// Level 3: Gemm
// =============================================================================

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_notrans_rowmajor_f32_avx2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    a_ptr: *const f32,
    a_lda: isize,
    b_ptr: *const f32,
    b_ldb: isize,
    beta: f32,
    c_ptr: *mut f32,
    c_ldc: isize,
) {
    let n_chunks = n / 8;
    let n_rem = n % 8;

    for i in 0..m {
        let c_row = c_ptr.offset(i.cast_signed() * c_ldc);
        if beta == 0.0 {
            for j in 0..n {
                *c_row.add(j) = 0.0;
            }
        } else if beta != 1.0 {
            for j in 0..n {
                *c_row.add(j) *= beta;
            }
        }
    }

    for i in 0..m {
        let a_row = a_ptr.offset(i.cast_signed() * a_lda);
        let c_row = c_ptr.offset(i.cast_signed() * c_ldc);

        for p in 0..k {
            let a_ip = *a_row.add(p);
            let scale = alpha * a_ip;
            let scale_vec = _mm256_set1_ps(scale);
            let b_row = b_ptr.offset(p.cast_signed() * b_ldb);

            for j in 0..n_chunks {
                let offset = j * 8;
                let bv = _mm256_loadu_ps(b_row.add(offset));
                let cv = _mm256_loadu_ps(c_row.add(offset));
                let updated = _mm256_fmadd_ps(scale_vec, bv, cv);
                _mm256_storeu_ps(c_row.add(offset), updated);
            }
            for j in (n - n_rem)..n {
                *c_row.add(j) += scale * *b_row.add(j);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn gemm_notrans_rowmajor_f64_avx2(
    m: usize,
    n: usize,
    k: usize,
    alpha: f64,
    a_ptr: *const f64,
    a_lda: isize,
    b_ptr: *const f64,
    b_ldb: isize,
    beta: f64,
    c_ptr: *mut f64,
    c_ldc: isize,
) {
    let n_chunks = n / 4;
    let n_rem = n % 4;

    for i in 0..m {
        let c_row = c_ptr.offset(i.cast_signed() * c_ldc);
        if beta == 0.0 {
            for j in 0..n {
                *c_row.add(j) = 0.0;
            }
        } else if beta != 1.0 {
            for j in 0..n {
                *c_row.add(j) *= beta;
            }
        }
    }

    for i in 0..m {
        let a_row = a_ptr.offset(i.cast_signed() * a_lda);
        let c_row = c_ptr.offset(i.cast_signed() * c_ldc);

        for p in 0..k {
            let a_ip = *a_row.add(p);
            let scale = alpha * a_ip;
            let scale_vec = _mm256_set1_pd(scale);
            let b_row = b_ptr.offset(p.cast_signed() * b_ldb);

            for j in 0..n_chunks {
                let offset = j * 4;
                let bv = _mm256_loadu_pd(b_row.add(offset));
                let cv = _mm256_loadu_pd(c_row.add(offset));
                let updated = _mm256_fmadd_pd(scale_vec, bv, cv);
                _mm256_storeu_pd(c_row.add(offset), updated);
            }
            for j in (n - n_rem)..n {
                *c_row.add(j) += scale * *b_row.add(j);
            }
        }
    }
}

impl<
    A: DenseStorage<f32>,
    B: DenseStorage<f32>,
    C: DenseStorageMut<f32>,
> Gemm<f32, A, B, C> for Avx2Blas {
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
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                if ta == Trans::NoTrans
                    && tb == Trans::NoTrans
                    && a.c_stride() == 1
                    && b.c_stride() == 1
                    && c.c_stride() == 1
                {
                    unsafe {
                        gemm_notrans_rowmajor_f32_avx2(
                            a.rows(),
                            b.cols(),
                            a.cols(),
                            alpha,
                            a.as_ptr(),
                            a.r_stride(),
                            b.as_ptr(),
                            b.r_stride(),
                            beta,
                            c.as_mut_ptr(),
                            c.r_stride(),
                        );
                    }
                    return;
                }
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}

impl<
    A: DenseStorage<f64>,
    B: DenseStorage<f64>,
    C: DenseStorageMut<f64>,
> Gemm<f64, A, B, C> for Avx2Blas {
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
        #[cfg(target_arch = "x86_64")]
        {
            if std::arch::is_x86_feature_detected!("avx2") && std::arch::is_x86_feature_detected!("fma") {
                if ta == Trans::NoTrans
                    && tb == Trans::NoTrans
                    && a.c_stride() == 1
                    && b.c_stride() == 1
                    && c.c_stride() == 1
                {
                    unsafe {
                        gemm_notrans_rowmajor_f64_avx2(
                            a.rows(),
                            b.cols(),
                            a.cols(),
                            alpha,
                            a.as_ptr(),
                            a.r_stride(),
                            b.as_ptr(),
                            b.r_stride(),
                            beta,
                            c.as_mut_ptr(),
                            c.r_stride(),
                        );
                    }
                    return;
                }
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}
