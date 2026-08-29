//! ARM CMSIS-DSP subprogram backend implementor for Cortex-M.
//!
//! Bridges `control-rs` subprogram traits to CMSIS-DSP C static library functions
//! (`arm_mat_mult_f32`, `arm_mat_vec_mult_f32`, `arm_dot_prod_f32`, etc.).

use control_rs::math::complex_num::Complex;
use control_rs::math::storage::{DenseStorage, DenseStorageMut, Diag, Side, Trans, UpLo};
use control_rs::math::subprograms::lapack::Potrf;
use control_rs::math::subprograms::level1::{Dotc, Dotu, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::{Gemm, Trsm};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::math::{LinAlgError, LinAlgResult};

/// Zero-sized marker type for ARM CMSIS-DSP backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CmsisDspBlas;

#[repr(C)]
struct ArmMatrixInstanceF32 {
    num_rows: u16,
    num_cols: u16,
    p_data: *mut f32,
}

#[allow(dead_code)]
unsafe extern "C" {
    fn arm_mat_init_f32(s: *mut ArmMatrixInstanceF32, n_rows: u16, n_cols: u16, p_data: *mut f32);
    fn arm_mat_mult_f32(
        src_a: *const ArmMatrixInstanceF32,
        src_b: *const ArmMatrixInstanceF32,
        dst: *mut ArmMatrixInstanceF32,
    ) -> i32;
    fn arm_mat_vec_mult_f32(src_mat: *const ArmMatrixInstanceF32, p_vec: *const f32, p_dst: *mut f32);
    fn arm_dot_prod_f32(src_a: *const f32, src_b: *const f32, block_size: u32, result: *mut f32);
    fn arm_cmplx_dot_prod_f32(
        src_a: *const f32,
        src_b: *const f32,
        num_samples: u32,
        real_result: *mut f32,
        imag_result: *mut f32,
    );
    fn arm_scale_f32(src: *const f32, scale: f32, dst: *mut f32, block_size: u32);
    fn arm_mat_cholesky_f32(
        src: *const ArmMatrixInstanceF32,
        dst: *mut ArmMatrixInstanceF32,
    ) -> i32;
    fn arm_mat_solve_upper_triangular_f32(
        src_a: *const ArmMatrixInstanceF32,
        src_b: *const ArmMatrixInstanceF32,
        dst: *mut ArmMatrixInstanceF32,
    ) -> i32;
}

// =============================================================================
// Level 1: Scal & Dot
// =============================================================================

impl<X: DenseStorageMut<f32>> Scal<f32, X> for CmsisDspBlas {
    #[inline(always)]
    fn scal(alpha: f32, x: &mut X) {
        let n = x.rows() * x.cols();
        let stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
        if stride == 1 {
            unsafe {
                arm_scale_f32(
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

impl<X: DenseStorage<f32>, Y: DenseStorage<f32>> Dotu<f32, X, Y> for CmsisDspBlas {
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f32 {
        let n = x.rows() * x.cols();
        let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
        let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

        if x_stride == 1 && y_stride == 1 {
            let mut res = 0.0f32;
            unsafe {
                arm_dot_prod_f32(
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
    Dotc<Complex<f32>, X, Y> for CmsisDspBlas
{
    #[inline(always)]
    fn dotc(x: &X, y: &Y) -> Complex<f32> {
        let n = x.rows() * x.cols();
        let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
        let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

        if x_stride == 1 && y_stride == 1 {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            unsafe {
                arm_cmplx_dot_prod_f32(
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
    Gemv<f32, A, X, Y> for CmsisDspBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y) {
        let x_stride = if x.rows() >= x.cols() { x.r_stride() } else { x.c_stride() };
        let y_stride = if y.rows() >= y.cols() { y.r_stride() } else { y.c_stride() };

        // CMSIS-DSP fast path: row-major contiguous, NoTrans, alpha=1.0, beta=0.0
        if trans == Trans::NoTrans
            && alpha == 1.0
            && beta == 0.0
            && a.c_stride() == 1
            && a.r_stride() == a.cols().cast_signed()
            && x_stride == 1
            && y_stride == 1
        {
            let a_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            unsafe {
                arm_mat_vec_mult_f32(&a_mat, x.as_ptr(), y.as_mut_ptr());
            }
        } else {
            DefaultBlas::gemv(trans, alpha, a, x, beta, y);
        }
    }
}

// =============================================================================
// Level 3: Gemm
// =============================================================================

impl<
    A: DenseStorage<f32>,
    B: DenseStorage<f32>,
    C: DenseStorageMut<f32>,
> Gemm<f32, A, B, C> for CmsisDspBlas {
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
        // CMSIS-DSP fast path: NoTrans on row-major contiguous operands with alpha=1.0, beta=0.0
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
            let a_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            let b_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(b.rows()).unwrap_or(0),
                num_cols: u16::try_from(b.cols()).unwrap_or(0),
                p_data: b.as_ptr() as *mut f32,
            };
            let mut c_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(c.rows()).unwrap_or(0),
                num_cols: u16::try_from(c.cols()).unwrap_or(0),
                p_data: c.as_mut_ptr(),
            };

            let status = unsafe { arm_mat_mult_f32(&a_mat, &b_mat, &mut c_mat) };
            if status == 0 {
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}

// =============================================================================
// LAPACK: Potrf (Cholesky) & Trsm (Upper Triangular Solve)
// =============================================================================

impl<A: DenseStorageMut<f32>> Potrf<f32, A> for CmsisDspBlas {
    #[inline(always)]
    fn potrf(uplo: UpLo, a: &mut A) -> LinAlgResult<()> {
        let n = a.rows();
        if uplo == UpLo::Lower && a.c_stride() == 1 && a.r_stride() == n.cast_signed() {
            // Buffer to compute Cholesky in place
            let mut l_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(n).unwrap_or(0),
                num_cols: u16::try_from(n).unwrap_or(0),
                p_data: a.as_mut_ptr(),
            };
            let status = unsafe { arm_mat_cholesky_f32(&l_mat, &mut l_mat) };
            if status == 0 {
                return Ok(());
            } else {
                return Err(LinAlgError::NotPositiveDefinite);
            }
        }
        DefaultBlas::potrf(uplo, a)
    }
}

impl<A: DenseStorage<f32>, B: DenseStorageMut<f32>> Trsm<f32, A, B> for CmsisDspBlas {
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
            let a_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(a.rows()).unwrap_or(0),
                num_cols: u16::try_from(a.cols()).unwrap_or(0),
                p_data: a.as_ptr() as *mut f32,
            };
            let mut b_mat = ArmMatrixInstanceF32 {
                num_rows: u16::try_from(b.rows()).unwrap_or(0),
                num_cols: u16::try_from(b.cols()).unwrap_or(0),
                p_data: b.as_mut_ptr(),
            };

            let status = unsafe {
                arm_mat_solve_upper_triangular_f32(&a_mat, &b_mat, &mut b_mat)
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
