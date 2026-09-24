//! Panic-free numeric helpers shared by the verification kernels.
//!
//! The kernels index small fixed-size matrices and sweep integer grids. These
//! helpers keep every element access fallible and every index-to-float
//! conversion exact, so the emitted containers carry no hidden panic path.

use control_rs::math::num_types::{Const, Dim};
use control_rs::matrix::Owned;

/// Result of a kernel step; the error is a human-readable message.
pub type KernelResult<T> = Result<T, String>;

/// A sampled signal or flattened matrix produced by a kernel.
pub type SeriesResult = KernelResult<Vec<f64>>;

/// Converts an index to `f64`.
///
/// Exact for every index below 2^32; larger indices saturate at
/// `u32::MAX`. Kernel indices are bounded by the matrix and grid sizes.
#[must_use]
pub fn index_f64(i: usize) -> f64 {
    f64::from(u32::try_from(i).unwrap_or(u32::MAX))
}

/// Converts an index to `f32`.
///
/// Exact for every index below 2^16; larger indices saturate at
/// `u16::MAX`. Kernel indices are bounded by the tensor and grid sizes.
#[must_use]
pub fn index_f32(i: usize) -> f32 {
    f32::from(u16::try_from(i).unwrap_or(u16::MAX))
}

/// Element `(i, j)` of `m`, or an error naming the out-of-range index.
///
/// # Errors
///
/// Returns an error when `(i, j)` lies outside the `R x C` matrix.
pub fn entry<const R: usize, const C: usize>(
    m: &Owned<f64, R, C>,
    i: usize,
    j: usize,
) -> KernelResult<f64>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    m.get(i, j)
        .copied()
        .ok_or_else(|| format!("index ({i}, {j}) outside {R}x{C} matrix"))
}

/// Row-major copy of every element of `m`.
///
/// # Errors
///
/// Returns an error if an element cannot be read, which the loop bounds
/// rule out.
pub fn row_major<const R: usize, const C: usize>(
    m: &Owned<f64, R, C>,
) -> SeriesResult
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut out = Vec::with_capacity(R.saturating_mul(C));
    for i in 0..R {
        for j in 0..C {
            out.push(entry(m, i, j)?);
        }
    }
    Ok(out)
}

/// Column `j` of `m` as a vector.
///
/// # Errors
///
/// Returns an error when `j` lies outside the matrix.
pub fn column<const R: usize, const C: usize>(
    m: &Owned<f64, R, C>,
    j: usize,
) -> SeriesResult
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    (0..R).map(|i| entry(m, i, j)).collect()
}

/// Matrix product `a * b` through the library's default BLAS backend.
#[must_use]
pub fn matmul<const M: usize, const N: usize, const P: usize>(
    a: &Owned<f64, M, N>,
    b: &Owned<f64, N, P>,
) -> Owned<f64, M, P>
where
    Const<M>: Dim,
    Const<N>: Dim,
    Const<P>: Dim,
{
    let mut out = Owned::<f64, M, P>::zero();
    a.mul_into(b, &mut out);
    out
}

/// Element-wise `a + b`.
#[must_use]
pub fn add<const R: usize, const C: usize>(
    a: &Owned<f64, R, C>,
    b: &Owned<f64, R, C>,
) -> Owned<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    zip_with(a, b, |x, y| x + y)
}

/// Element-wise `a - b`.
#[must_use]
pub fn sub<const R: usize, const C: usize>(
    a: &Owned<f64, R, C>,
    b: &Owned<f64, R, C>,
) -> Owned<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    zip_with(a, b, |x, y| x - y)
}

/// Applies `f` to corresponding elements of two equally shaped matrices.
fn zip_with<const R: usize, const C: usize>(
    a: &Owned<f64, R, C>,
    b: &Owned<f64, R, C>,
    f: impl Fn(f64, f64) -> f64,
) -> Owned<f64, R, C>
where
    Const<R>: Dim,
    Const<C>: Dim,
{
    let mut out = Owned::<f64, R, C>::zero();
    for ((o, &x), &y) in out
        .as_mut_slice()
        .iter_mut()
        .zip(a.as_slice())
        .zip(b.as_slice())
    {
        *o = f(x, y);
    }
    out
}
