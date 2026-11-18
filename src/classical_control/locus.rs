//! Root locus over a caller-supplied gain set (FR-1).
//!
//! For each gain `k_i` the closed-loop characteristic polynomial
//! `D(s) + k_i N(s)` is formed with coefficient arithmetic and solved with
//! `Polynomial::roots`. Each output row is an unordered root set; root order
//! across consecutive gains is not matched.

use super::{ClassicalError, TypeNum};
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim, DimMax};
use crate::polynomial::ArrayPolynomial;
use crate::transfer_function::ArrayTransferFunction;

/// Closed-loop roots per gain: one row of `D` roots for each of `K` gains, or
/// the reason root finding failed.
pub type LocusResult<T, const D: usize, const K: usize> =
    Result<[[Complex<T>; D]; K], ClassicalError>;

/// Computes the closed-loop roots of `D(s) + k_i N(s)` for each gain
/// (FR-1).
///
/// Each row holds the `D - 1` roots in its first entries and a zero in the
/// last, the layout of
/// [`TransferFunction::poles`](crate::transfer_function::TransferFunction::poles).
/// The open loop must be proper (`N <= D`). The cost is `K` root solves.
///
/// # Errors
/// [`ClassicalError::Root`] when a characteristic polynomial has a zero
/// leading coefficient or root finding fails.
///
/// # Example
/// ```
/// use control_rs::classical_control::root_locus;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// let l = ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 2.0, 1.0]);
/// let rows = root_locus(&l, &[1.0])?;
/// // s^2 + 2s + 1: double root at -1.
/// assert!(rows[0][..2].iter().all(|r| (r.re + 1.0).abs() < 1e-6));
/// # Ok::<(), control_rs::classical_control::ClassicalError>(())
/// ```
///
/// An improper open loop (`N > D`) is rejected at compile time:
///
/// ```compile_fail,E0271
/// use control_rs::classical_control::root_locus;
/// use control_rs::transfer_function::ArrayTransferFunction;
///
/// let l = ArrayTransferFunction::<f64, 3, 2>::continuous([1.0, 0.0, 1.0], [1.0, 1.0]);
/// let _ = root_locus(&l, &[2.0]);
/// ```
pub fn root_locus<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const K: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    gains: &[T; K],
) -> LocusResult<T, D, K>
where
    Const<N>: Dim,
    Const<D>: Dim,
    TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    let mut out = [[Complex::new(T::ZERO, T::ZERO); D]; K];
    for (row, &k) in out.iter_mut().zip(gains) {
        let mut c = [T::ZERO; D];
        c.iter_mut().zip(sys.den_slice()).for_each(|(o, &d)| *o = d);
        c.iter_mut()
            .zip(sys.num_slice())
            .for_each(|(o, &n)| *o = o.saturating_add(&k.saturating_mul(&n)));
        *row = ArrayPolynomial::<T, D>::from_coefficients(c).roots()?;
    }
    Ok(out)
}
