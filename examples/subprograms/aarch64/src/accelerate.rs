//! Apple Accelerate framework (vecLib CBLAS) backend implementor for macOS on Apple Silicon.
//!
//! Attaches to subprogram traits via `-framework Accelerate`.

use control_rs::math::storage::{DenseStorage, DenseStorageMut, Trans};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::math::subprograms::level1::{Axpy, Dotu, Nrm2, Scal};
use control_rs::math::subprograms::level2::Gemv;
use control_rs::math::subprograms::level3::Gemm;

/// Zero-sized marker type for Apple Accelerate framework backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AccelerateBlas;

#[cfg(all(target_vendor = "apple", feature = "accelerate"))]
#[link(name = "Accelerate", kind = "framework")]
#[allow(dead_code)]
unsafe extern "C" {
    fn cblas_saxpy(
        n: i32,
        alpha: f32,
        x: *const f32,
        incx: i32,
        y: *mut f32,
        incy: i32,
    );
    fn cblas_daxpy(
        n: i32,
        alpha: f64,
        x: *const f64,
        incx: i32,
        y: *mut f64,
        incy: i32,
    );
    fn cblas_sscal(n: i32, alpha: f32, x: *mut f32, incx: i32);
    fn cblas_dscal(n: i32, alpha: f64, x: *mut f64, incx: i32);
    fn cblas_sdot(
        n: i32,
        x: *const f32,
        incx: i32,
        y: *const f32,
        incy: i32,
    ) -> f32;
    fn cblas_ddot(
        n: i32,
        x: *const f64,
        incx: i32,
        y: *const f64,
        incy: i32,
    ) -> f64;
    fn cblas_snrm2(n: i32, x: *const f32, incx: i32) -> f32;
    fn cblas_dnrm2(n: i32, x: *const f64, incx: i32) -> f64;
    fn cblas_sgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        x: *const f32,
        incx: i32,
        beta: f32,
        y: *mut f32,
        incy: i32,
    );
    fn cblas_dgemv(
        order: i32,
        trans: i32,
        m: i32,
        n: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        x: *const f64,
        incx: i32,
        beta: f64,
        y: *mut f64,
        incy: i32,
    );
    fn cblas_sgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f32,
        a: *const f32,
        lda: i32,
        b: *const f32,
        ldb: i32,
        beta: f32,
        c: *mut f32,
        ldc: i32,
    );
    fn cblas_dgemm(
        order: i32,
        transa: i32,
        transb: i32,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: *const f64,
        lda: i32,
        b: *const f64,
        ldb: i32,
        beta: f64,
        c: *mut f64,
        ldc: i32,
    );
}

#[allow(dead_code)]
const CBLAS_ROW_MAJOR: i32 = 101;
#[allow(dead_code)]
const CBLAS_COL_MAJOR: i32 = 102;
#[allow(dead_code)]
const CBLAS_NO_TRANS: i32 = 111;
#[allow(dead_code)]
const CBLAS_TRANS: i32 = 112;

// =============================================================================
// Level 1: Axpy
// =============================================================================

impl<X: DenseStorage<f32>, Y: DenseStorageMut<f32>> Axpy<f32, X, Y>
    for AccelerateBlas
{
    #[inline(always)]
    fn axpy(alpha: f32, x: &X, y: &mut Y) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
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

            if x_stride > 0 && y_stride > 0 {
                unsafe {
                    cblas_saxpy(
                        i32::try_from(n).unwrap_or(0),
                        alpha,
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        y.as_mut_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    );
                }
                return;
            }
        }
        DefaultBlas::axpy(alpha, x, y);
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorageMut<f64>> Axpy<f64, X, Y>
    for AccelerateBlas
{
    #[inline(always)]
    fn axpy(alpha: f64, x: &X, y: &mut Y) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
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

            if x_stride > 0 && y_stride > 0 {
                unsafe {
                    cblas_daxpy(
                        i32::try_from(n).unwrap_or(0),
                        alpha,
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        y.as_mut_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    );
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

impl<X: DenseStorageMut<f32>> Scal<f32, X> for AccelerateBlas {
    #[inline(always)]
    fn scal(alpha: f32, x: &mut X) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride > 0 {
                unsafe {
                    cblas_sscal(
                        i32::try_from(n).unwrap_or(0),
                        alpha,
                        x.as_mut_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                    );
                }
                return;
            }
        }
        DefaultBlas::scal(alpha, x);
    }
}

impl<X: DenseStorageMut<f64>> Scal<f64, X> for AccelerateBlas {
    #[inline(always)]
    fn scal(alpha: f64, x: &mut X) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride > 0 {
                unsafe {
                    cblas_dscal(
                        i32::try_from(n).unwrap_or(0),
                        alpha,
                        x.as_mut_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                    );
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

impl<X: DenseStorage<f32>, Y: DenseStorage<f32>> Dotu<f32, X, Y>
    for AccelerateBlas
{
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f32 {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
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

            if x_stride > 0 && y_stride > 0 {
                return unsafe {
                    cblas_sdot(
                        i32::try_from(n).unwrap_or(0),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        y.as_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    )
                };
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

impl<X: DenseStorage<f64>, Y: DenseStorage<f64>> Dotu<f64, X, Y>
    for AccelerateBlas
{
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> f64 {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
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

            if x_stride > 0 && y_stride > 0 {
                return unsafe {
                    cblas_ddot(
                        i32::try_from(n).unwrap_or(0),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        y.as_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    )
                };
            }
        }
        DefaultBlas::dotu(x, y)
    }
}

// =============================================================================
// Level 1: Nrm2
// =============================================================================

impl<X: DenseStorage<f32>> Nrm2<f32, X> for AccelerateBlas {
    #[inline(always)]
    fn nrm2(x: &X) -> f32 {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride > 0 {
                return unsafe {
                    cblas_snrm2(
                        i32::try_from(n).unwrap_or(0),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                    )
                };
            }
        }
        DefaultBlas::nrm2(x)
    }
}

impl<X: DenseStorage<f64>> Nrm2<f64, X> for AccelerateBlas {
    #[inline(always)]
    fn nrm2(x: &X) -> f64 {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let n = x.rows() * x.cols();
            let x_stride = if x.rows() >= x.cols() {
                x.r_stride()
            } else {
                x.c_stride()
            };

            if x_stride > 0 {
                return unsafe {
                    cblas_dnrm2(
                        i32::try_from(n).unwrap_or(0),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                    )
                };
            }
        }
        DefaultBlas::nrm2(x)
    }
}

// =============================================================================
// Level 2: Gemv
// =============================================================================

impl<A: DenseStorage<f32>, X: DenseStorage<f32>, Y: DenseStorageMut<f32>>
    Gemv<f32, A, X, Y> for AccelerateBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let is_row_major = a.c_stride() == 1;
            let is_col_major = a.r_stride() == 1;
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

            if (is_row_major || is_col_major) && x_stride > 0 && y_stride > 0 {
                let order = if is_row_major {
                    CBLAS_ROW_MAJOR
                } else {
                    CBLAS_COL_MAJOR
                };
                let lda = if is_row_major {
                    a.r_stride()
                } else {
                    a.c_stride()
                };
                let trans_code = match trans {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };

                unsafe {
                    cblas_sgemv(
                        order,
                        trans_code,
                        i32::try_from(a.rows()).unwrap_or(0),
                        i32::try_from(a.cols()).unwrap_or(0),
                        alpha,
                        a.as_ptr(),
                        i32::try_from(lda).unwrap_or(1),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        beta,
                        y.as_mut_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    );
                }
                return;
            }
        }
        DefaultBlas::gemv(trans, alpha, a, x, beta, y);
    }
}

impl<A: DenseStorage<f64>, X: DenseStorage<f64>, Y: DenseStorageMut<f64>>
    Gemv<f64, A, X, Y> for AccelerateBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: f64, a: &A, x: &X, beta: f64, y: &mut Y) {
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let is_row_major = a.c_stride() == 1;
            let is_col_major = a.r_stride() == 1;
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

            if (is_row_major || is_col_major) && x_stride > 0 && y_stride > 0 {
                let order = if is_row_major {
                    CBLAS_ROW_MAJOR
                } else {
                    CBLAS_COL_MAJOR
                };
                let lda = if is_row_major {
                    a.r_stride()
                } else {
                    a.c_stride()
                };
                let trans_code = match trans {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };

                unsafe {
                    cblas_dgemv(
                        order,
                        trans_code,
                        i32::try_from(a.rows()).unwrap_or(0),
                        i32::try_from(a.cols()).unwrap_or(0),
                        alpha,
                        a.as_ptr(),
                        i32::try_from(lda).unwrap_or(1),
                        x.as_ptr(),
                        i32::try_from(x_stride).unwrap_or(1),
                        beta,
                        y.as_mut_ptr(),
                        i32::try_from(y_stride).unwrap_or(1),
                    );
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
    Gemm<f32, A, B, C> for AccelerateBlas
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
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let (m, k) = match ta {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let n = match tb {
                Trans::NoTrans => b.cols(),
                Trans::Trans | Trans::ConjTrans => b.rows(),
            };

            let a_row = a.c_stride() == 1;
            let b_row = b.c_stride() == 1;
            let c_row = c.c_stride() == 1;
            let a_col = a.r_stride() == 1;
            let b_col = b.r_stride() == 1;
            let c_col = c.r_stride() == 1;

            let layout = if a_row && b_row && c_row {
                Some((
                    CBLAS_ROW_MAJOR,
                    a.r_stride(),
                    b.r_stride(),
                    c.r_stride(),
                ))
            } else if a_col && b_col && c_col {
                Some((
                    CBLAS_COL_MAJOR,
                    a.c_stride(),
                    b.c_stride(),
                    c.c_stride(),
                ))
            } else {
                None
            };

            if let Some((order, lda, ldb, ldc)) = layout
                && lda > 0
                && ldb > 0
                && ldc > 0
            {
                let transa_code = match ta {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };
                let transb_code = match tb {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };

                unsafe {
                    cblas_sgemm(
                        order,
                        transa_code,
                        transb_code,
                        i32::try_from(m).unwrap_or(0),
                        i32::try_from(n).unwrap_or(0),
                        i32::try_from(k).unwrap_or(0),
                        alpha,
                        a.as_ptr(),
                        i32::try_from(lda).unwrap_or(1),
                        b.as_ptr(),
                        i32::try_from(ldb).unwrap_or(1),
                        beta,
                        c.as_mut_ptr(),
                        i32::try_from(ldc).unwrap_or(1),
                    );
                }
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}

impl<A: DenseStorage<f64>, B: DenseStorage<f64>, C: DenseStorageMut<f64>>
    Gemm<f64, A, B, C> for AccelerateBlas
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
        #[cfg(all(target_vendor = "apple", feature = "accelerate"))]
        {
            let (m, k) = match ta {
                Trans::NoTrans => (a.rows(), a.cols()),
                Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
            };
            let n = match tb {
                Trans::NoTrans => b.cols(),
                Trans::Trans | Trans::ConjTrans => b.rows(),
            };

            let a_row = a.c_stride() == 1;
            let b_row = b.c_stride() == 1;
            let c_row = c.c_stride() == 1;
            let a_col = a.r_stride() == 1;
            let b_col = b.r_stride() == 1;
            let c_col = c.r_stride() == 1;

            let layout = if a_row && b_row && c_row {
                Some((
                    CBLAS_ROW_MAJOR,
                    a.r_stride(),
                    b.r_stride(),
                    c.r_stride(),
                ))
            } else if a_col && b_col && c_col {
                Some((
                    CBLAS_COL_MAJOR,
                    a.c_stride(),
                    b.c_stride(),
                    c.c_stride(),
                ))
            } else {
                None
            };

            if let Some((order, lda, ldb, ldc)) = layout
                && lda > 0
                && ldb > 0
                && ldc > 0
            {
                let transa_code = match ta {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };
                let transb_code = match tb {
                    Trans::NoTrans => CBLAS_NO_TRANS,
                    Trans::Trans | Trans::ConjTrans => CBLAS_TRANS,
                };

                unsafe {
                    cblas_dgemm(
                        order,
                        transa_code,
                        transb_code,
                        i32::try_from(m).unwrap_or(0),
                        i32::try_from(n).unwrap_or(0),
                        i32::try_from(k).unwrap_or(0),
                        alpha,
                        a.as_ptr(),
                        i32::try_from(lda).unwrap_or(1),
                        b.as_ptr(),
                        i32::try_from(ldb).unwrap_or(1),
                        beta,
                        c.as_mut_ptr(),
                        i32::try_from(ldc).unwrap_or(1),
                    );
                }
                return;
            }
        }
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}
