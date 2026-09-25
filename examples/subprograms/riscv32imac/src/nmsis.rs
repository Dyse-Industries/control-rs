//! Nuclei RISC-V NMSIS-DSP subprogram backend implementor.
//!
//! Bridges `control-rs` subprogram traits to NMSIS-DSP C static library functions
//! (`riscv_mat_mult_f32`, `riscv_mat_vec_mult_f32`, `riscv_dot_prod_f32`, etc.).
//!
//! # ABI and C Source Reference
//! The C sources in `c_src/` represent a portable standalone reference stand-in for the
//! vendor NMSIS-DSP library. When compiling with an external RISC-V C toolchain (`feature = "c_nmsis"`),
//! `build.rs` compiles `c_src/` into `.o` objects. When compiling without an external C toolchain,
//! the fallback Rust implementations below satisfy the exact NMSIS C ABI symbols without symbol collisions.

use control_rs::math::complex_num::Complex;
use control_rs::math::storage::{
    DenseStorage, DenseStorageMut, Diag, Side, Trans, UpLo,
};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::math::subprograms::lapack::Potrf;
use control_rs::math::subprograms::level1::{Dotc, Dotu, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::{Gemm, Trsm};
use control_rs::math::{LinAlgError, LinAlgResult};

/// Zero-sized marker type for RISC-V NMSIS-DSP backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct NmsisDspBlas;

#[repr(C)]
pub struct RiscvMatrixInstanceF32 {
    pub num_rows: u16,
    pub num_cols: u16,
    pub p_data: *mut f32,
}

pub mod ffi {
    use super::RiscvMatrixInstanceF32;

    #[allow(dead_code)]
    unsafe extern "C" {
        pub fn riscv_mat_init_f32(
            s: *mut RiscvMatrixInstanceF32,
            n_rows: u16,
            n_cols: u16,
            p_data: *mut f32,
        );
        pub fn riscv_mat_mult_f32(
            src_a: *const RiscvMatrixInstanceF32,
            src_b: *const RiscvMatrixInstanceF32,
            dst: *mut RiscvMatrixInstanceF32,
        ) -> i32;
        pub fn riscv_mat_vec_mult_f32(
            src_mat: *const RiscvMatrixInstanceF32,
            p_vec: *const f32,
            p_dst: *mut f32,
        );
        pub fn riscv_dot_prod_f32(
            src_a: *const f32,
            src_b: *const f32,
            block_size: u32,
            result: *mut f32,
        );
        pub fn riscv_cmplx_dot_prod_f32(
            src_a: *const f32,
            src_b: *const f32,
            num_samples: u32,
            real_result: *mut f32,
            imag_result: *mut f32,
        );
        pub fn riscv_scale_f32(
            src: *const f32,
            scale: f32,
            dst: *mut f32,
            block_size: u32,
        );
        pub fn riscv_mat_cholesky_f32(
            src: *const RiscvMatrixInstanceF32,
            dst: *mut RiscvMatrixInstanceF32,
        ) -> i32;
        pub fn riscv_mat_solve_upper_triangular_f32(
            src_a: *const RiscvMatrixInstanceF32,
            src_b: *const RiscvMatrixInstanceF32,
            dst: *mut RiscvMatrixInstanceF32,
        ) -> i32;
    }
}

// Fallback Rust implementations of NMSIS C symbols, active when C static library is not compiled

/// Rust stand-in for `riscv_scale_f32`: writes `src[i] * scale` to `dst[i]`
/// for each `i < block_size`.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src` is valid for reads and `dst` is valid for writes of `block_size`
///   properly aligned `f32` values.
/// - `dst` either equals `src` (in-place scaling) or does not overlap it.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_scale_f32(
    src: *const f32,
    scale: f32,
    dst: *mut f32,
    block_size: u32,
) {
    for i in 0..block_size as usize {
        *dst.add(i) = *src.add(i) * scale;
    }
}

/// Rust stand-in for `riscv_dot_prod_f32`: writes the dot product of the
/// first `block_size` elements of `src_a` and `src_b` to `result`.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src_a` and `src_b` are each valid for reads of `block_size` properly
///   aligned `f32` values.
/// - `result` is valid for a write of one properly aligned `f32`.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_dot_prod_f32(
    src_a: *const f32,
    src_b: *const f32,
    block_size: u32,
    result: *mut f32,
) {
    let mut sum = 0.0f32;
    for i in 0..block_size as usize {
        sum += *src_a.add(i) * *src_b.add(i);
    }
    *result = sum;
}

/// Rust stand-in for `riscv_cmplx_dot_prod_f32`: writes the conjugated dot
/// product `sum(conj(a[i]) * b[i])` over `num_samples` interleaved
/// `(re, im)` pairs to `real_result` and `imag_result`.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src_a` and `src_b` are each valid for reads of `2 * num_samples`
///   properly aligned `f32` values.
/// - `real_result` and `imag_result` are each valid for a write of one
///   properly aligned `f32`.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_cmplx_dot_prod_f32(
    src_a: *const f32,
    src_b: *const f32,
    num_samples: u32,
    real_result: *mut f32,
    imag_result: *mut f32,
) {
    let mut sum_r = 0.0f32;
    let mut sum_i = 0.0f32;
    for i in 0..num_samples as usize {
        let ar = *src_a.add(2 * i);
        let ai = *src_a.add(2 * i + 1);
        let br = *src_b.add(2 * i);
        let bi = *src_b.add(2 * i + 1);
        sum_r += ar * br + ai * bi;
        sum_i += ar * bi - ai * br;
    }
    *real_result = sum_r;
    *imag_result = sum_i;
}

/// Rust stand-in for `riscv_mat_vec_mult_f32`: writes the product of the
/// row-major matrix `src_mat` and the vector `p_vec` to `p_dst`.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src_mat` points to a valid `RiscvMatrixInstanceF32` whose `p_data` is
///   valid for reads of `num_rows * num_cols` properly aligned `f32` values.
/// - `p_vec` is valid for reads of `num_cols` properly aligned `f32` values.
/// - `p_dst` is valid for writes of `num_rows` properly aligned `f32` values
///   and overlaps neither the matrix data nor `p_vec`.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_mat_vec_mult_f32(
    src_mat: *const RiscvMatrixInstanceF32,
    p_vec: *const f32,
    p_dst: *mut f32,
) {
    let rows = (*src_mat).num_rows as usize;
    let cols = (*src_mat).num_cols as usize;
    let mat = (*src_mat).p_data;
    for r in 0..rows {
        let mut sum = 0.0f32;
        for c in 0..cols {
            sum += *mat.add(r * cols + c) * *p_vec.add(c);
        }
        *p_dst.add(r) = sum;
    }
}

/// Rust stand-in for `riscv_mat_mult_f32`: writes the row-major product
/// `src_a * src_b` to `dst`.
///
/// Returns `0` on success, or `-3` (`RISCV_MATH_SIZE_MISMATCH`) before any
/// `p_data` access when the three shapes do not chain.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src_a`, `src_b` and `dst` each point to a valid
///   `RiscvMatrixInstanceF32`.
/// - The `p_data` of each instance is valid for `num_rows * num_cols`
///   properly aligned `f32` values: reads for `src_a` and `src_b`, writes for
///   `dst`.
/// - The data of `dst` overlaps neither the data of `src_a` nor of `src_b`.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_mat_mult_f32(
    src_a: *const RiscvMatrixInstanceF32,
    src_b: *const RiscvMatrixInstanceF32,
    dst: *mut RiscvMatrixInstanceF32,
) -> i32 {
    let rows_a = (*src_a).num_rows as usize;
    let cols_a = (*src_a).num_cols as usize;
    let rows_b = (*src_b).num_rows as usize;
    let cols_b = (*src_b).num_cols as usize;
    let rows_c = (*dst).num_rows as usize;
    let cols_c = (*dst).num_cols as usize;

    if cols_a != rows_b || rows_a != rows_c || cols_b != cols_c {
        return -3;
    }

    let a_ptr = (*src_a).p_data;
    let b_ptr = (*src_b).p_data;
    let c_ptr = (*dst).p_data;

    for r in 0..rows_a {
        for c in 0..cols_b {
            let mut sum = 0.0f32;
            for k in 0..cols_a {
                sum += *a_ptr.add(r * cols_a + k) * *b_ptr.add(k * cols_b + c);
            }
            *c_ptr.add(r * cols_b + c) = sum;
        }
    }
    0
}

/// Rust stand-in for `riscv_mat_cholesky_f32`: writes the lower-triangular
/// Cholesky factor of the `n x n` matrix `src` to `dst`, where `n` is the
/// `num_rows` of `src`. Entries above the diagonal keep the values of `src`.
///
/// Returns `0` on success, or `-7` (`RISCV_MATH_DECOMPOSITION_FAILURE`) when
/// the matrix is not positive definite.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src` and `dst` each point to a valid `RiscvMatrixInstanceF32`.
/// - The `p_data` of `src` is valid for reads, and the `p_data` of `dst` is
///   valid for reads and writes, of `n * n` properly aligned `f32` values.
///   Unlike the C reference, this stand-in does not check the shapes.
/// - The two `p_data` pointers are either equal (in-place factorization) or
///   do not overlap.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_mat_cholesky_f32(
    src: *const RiscvMatrixInstanceF32,
    dst: *mut RiscvMatrixInstanceF32,
) -> i32 {
    let n = (*src).num_rows as usize;
    let p_a = (*src).p_data;
    let p_l = (*dst).p_data;

    if p_a != p_l {
        for k in 0..n * n {
            *p_l.add(k) = *p_a.add(k);
        }
    }

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0f32;
            for k in 0..j {
                sum += *p_l.add(i * n + k) * *p_l.add(j * n + k);
            }
            if i == j {
                let val = *p_l.add(i * n + i) - sum;
                if val <= 0.0 {
                    return -7;
                }
                *p_l.add(i * n + i) = libm::sqrtf(val);
            } else {
                let diag = *p_l.add(j * n + j);
                if diag == 0.0 {
                    return -7;
                }
                *p_l.add(i * n + j) = (*p_l.add(i * n + j) - sum) / diag;
            }
        }
    }
    0
}

/// Rust stand-in for `riscv_mat_solve_upper_triangular_f32`: solves
/// `src_a * X = src_b` by back substitution and writes `X` to `dst`, where
/// `src_a` is `n x n` upper triangular and `src_b` is `n x m`, with `n` the
/// `num_rows` of `src_a` and `m` the `num_cols` of `src_b`.
///
/// Returns `0` on success, or `-5` (`RISCV_MATH_SINGULAR`) on a zero
/// diagonal entry.
///
/// # Safety
///
/// The caller must ensure that:
/// - `src_a`, `src_b` and `dst` each point to a valid
///   `RiscvMatrixInstanceF32`.
/// - The `p_data` of `src_a` is valid for reads of `n * n` properly aligned
///   `f32` values, the `p_data` of `src_b` for reads of `n * m`, and the
///   `p_data` of `dst` for reads and writes of `n * m`. Unlike the C
///   reference, this stand-in does not check the shapes.
/// - The `p_data` of `dst` either equals that of `src_b` (in-place solve) or
///   overlaps neither input.
#[cfg(not(feature = "c_nmsis"))]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn riscv_mat_solve_upper_triangular_f32(
    src_a: *const RiscvMatrixInstanceF32,
    src_b: *const RiscvMatrixInstanceF32,
    dst: *mut RiscvMatrixInstanceF32,
) -> i32 {
    let n = (*src_a).num_rows as usize;
    let m = (*src_b).num_cols as usize;
    let p_u = (*src_a).p_data;
    let p_b = (*src_b).p_data;
    let p_x = (*dst).p_data;

    for c in 0..m {
        for i in (0..n).rev() {
            let mut sum = 0.0f32;
            for k in (i + 1)..n {
                sum += *p_u.add(i * n + k) * *p_x.add(k * m + c);
            }
            let diag = *p_u.add(i * n + i);
            if diag == 0.0 {
                return -5;
            }
            *p_x.add(i * m + c) = (*p_b.add(i * m + c) - sum) / diag;
        }
    }
    0
}

// =============================================================================
// Level 1: Scal & Dot
// =============================================================================

impl<X: DenseStorageMut<f32>> Scal<f32, X> for NmsisDspBlas {
    #[inline(always)]
    fn scal(alpha: f32, x: &mut X) {
        let n = x.rows() * x.cols();
        let stride = if x.rows() >= x.cols() {
            x.r_stride()
        } else {
            x.c_stride()
        };
        if stride == 1 {
            unsafe {
                ffi::riscv_scale_f32(
                    x.as_mut_ptr(),
                    alpha,
                    x.as_mut_ptr(),
                    u32::try_from(n).unwrap_or(0),
                );
            }
        } else {
            DefaultBlas::scal(alpha, x);
        }
    }
}

impl<X: DenseStorage<f32>, Y: DenseStorage<f32>> Dotu<f32, X, Y>
    for NmsisDspBlas
{
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f32 {
        let n = x.rows() * x.cols();
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
            let mut res = 0.0f32;
            unsafe {
                ffi::riscv_dot_prod_f32(
                    x.as_ptr(),
                    y.as_ptr(),
                    u32::try_from(n).unwrap_or(0),
                    &mut res,
                );
            }
            res
        } else {
            DefaultBlas::dotu(x, y)
        }
    }
}

impl<X: DenseStorage<Complex<f32>>, Y: DenseStorage<Complex<f32>>>
    Dotc<Complex<f32>, X, Y> for NmsisDspBlas
{
    #[inline(always)]
    fn dotc(x: &X, y: &Y) -> Complex<f32> {
        let n = x.rows() * x.cols();
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
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            unsafe {
                ffi::riscv_cmplx_dot_prod_f32(
                    x.as_ptr().cast::<f32>(),
                    y.as_ptr().cast::<f32>(),
                    u32::try_from(n).unwrap_or(0),
                    &mut re,
                    &mut im,
                );
            }
            Complex::new(re, im)
        } else {
            DefaultBlas::dotc(x, y)
        }
    }
}

// =============================================================================
// Level 2: Gemv
// =============================================================================

impl<A: DenseStorage<f32>, X: DenseStorage<f32>, Y: DenseStorageMut<f32>>
    Gemv<f32, A, X, Y> for NmsisDspBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y) {
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
            && alpha == 1.0
            && beta == 0.0
            && a.c_stride() == 1
            && a.r_stride() == a.cols().cast_signed()
            && x_stride == 1
            && y_stride == 1
        {
            let a_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            unsafe {
                ffi::riscv_mat_vec_mult_f32(&a_mat, x.as_ptr(), y.as_mut_ptr());
            }
        } else {
            DefaultBlas::gemv(trans, alpha, a, x, beta, y);
        }
    }
}

// =============================================================================
// Level 3: Gemm
// =============================================================================

impl<A: DenseStorage<f32>, B: DenseStorage<f32>, C: DenseStorageMut<f32>>
    Gemm<f32, A, B, C> for NmsisDspBlas
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
        if ta == Trans::NoTrans
            && tb == Trans::NoTrans
            && alpha == 1.0
            && beta == 0.0
            && a.c_stride() == 1
            && a.r_stride() == a.cols().cast_signed()
            && b.c_stride() == 1
            && b.r_stride() == b.cols().cast_signed()
            && c.c_stride() == 1
            && c.r_stride() == c.cols().cast_signed()
        {
            let a_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            let b_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(b.rows()).unwrap_or(0),
                num_cols: u16::try_from(b.cols()).unwrap_or(0),
                p_data: b.as_ptr() as *mut f32,
            };
            let mut c_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(c.rows()).unwrap_or(0),
                num_cols: u16::try_from(c.cols()).unwrap_or(0),
                p_data: c.as_mut_ptr(),
            };

            let status =
                unsafe { ffi::riscv_mat_mult_f32(&a_mat, &b_mat, &mut c_mat) };
            if status == 0 {
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}

// =============================================================================
// LAPACK: Potrf & Trsm
// =============================================================================

impl<A: DenseStorageMut<f32>> Potrf<f32, A> for NmsisDspBlas {
    #[inline(always)]
    fn potrf(uplo: UpLo, a: &mut A) -> LinAlgResult<()> {
        let n = a.rows();
        if uplo == UpLo::Lower
            && a.c_stride() == 1
            && a.r_stride() == n.cast_signed()
        {
            let mut l_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(n).unwrap_or(0),
                num_cols: u16::try_from(n).unwrap_or(0),
                p_data: a.as_mut_ptr(),
            };
            let status =
                unsafe { ffi::riscv_mat_cholesky_f32(&l_mat, &mut l_mat) };
            if status == 0 {
                return Ok(());
            } else {
                return Err(LinAlgError::NotPositiveDefinite);
            }
        }
        DefaultBlas::potrf(uplo, a)
    }
}

impl<A: DenseStorage<f32>, B: DenseStorageMut<f32>> Trsm<f32, A, B>
    for NmsisDspBlas
{
    #[inline(always)]
    fn trsm(
        side: Side,
        uplo: UpLo,
        trans: Trans,
        diag: Diag,
        alpha: f32,
        a: &A,
        b: &mut B,
    ) -> LinAlgResult<()> {
        if side == Side::Left
            && uplo == UpLo::Upper
            && trans == Trans::NoTrans
            && diag == Diag::NonUnit
            && alpha == 1.0
            && a.c_stride() == 1
            && a.r_stride() == a.cols().cast_signed()
            && b.c_stride() == 1
            && b.r_stride() == b.cols().cast_signed()
        {
            let a_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            let mut b_mat = RiscvMatrixInstanceF32 {
                num_rows: u16::try_from(b.rows()).unwrap_or(0),
                num_cols: u16::try_from(b.cols()).unwrap_or(0),
                p_data: b.as_mut_ptr(),
            };

            let status = unsafe {
                ffi::riscv_mat_solve_upper_triangular_f32(
                    &a_mat, &b_mat, &mut b_mat,
                )
            };
            if status == 0 {
                return Ok(());
            } else {
                return Err(LinAlgError::SingularMatrix);
            }
        }
        DefaultBlas::trsm(side, uplo, trans, diag, alpha, a, b)
    }
}
