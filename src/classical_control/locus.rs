//! Root locus over a caller-supplied gain set (FR-1).
//!
//! For each gain `k_i` the closed-loop characteristic polynomial
//! `D(s) + k_i N(s)` is formed with coefficient arithmetic and solved with
//! `Polynomial::roots`. Each output row is an unordered root set; root order
//! across consecutive gains is not matched.

use super::{ClassicalError, Succ, TypeNum};
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim, DimAdd, DimMax};
use crate::polynomial::ArrayPolynomial;
use crate::transfer_function::ArrayTransferFunction;

/// Closed-loop roots per gain, `ORDER` roots in each of `K` rows, or the
/// reason root finding failed.
pub type LocusResult<T, const ORDER: usize, const K: usize> =
    Result<[[Complex<T>; ORDER]; K], ClassicalError>;

/// Computes the closed-loop roots of `D(s) + k_i N(s)` for each gain
/// (FR-1).
///
/// `ORDER` is the denominator degree `D - 1` and `N <= D` (proper
/// open loop). The cost is `K` root solves.
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
/// let rows = root_locus::<_, 1, 3, 2, 1>(&l, &[1.0])?;
/// // s^2 + 2s + 1: double root at -1.
/// assert!(rows[0].iter().all(|r| (r.re + 1.0).abs() < 1e-6));
/// # Ok::<(), control_rs::classical_control::ClassicalError>(())
/// ```
pub fn root_locus<
    T: Float + Copy,
    const N: usize,
    const D: usize,
    const ORDER: usize,
    const K: usize,
>(
    sys: &ArrayTransferFunction<T, N, D>,
    gains: &[T; K],
) -> LocusResult<T, ORDER, K>
where
    Const<N>: Dim,
    Const<ORDER>: Dim,
    TypeNum<ORDER>: DimAdd<Const<1>>,
    Const<D>: Dim<TypeNum = Succ<ORDER>>,
    TypeNum<N>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    let mut out = [[Complex::new(T::ZERO, T::ZERO); ORDER]; K];
    for (row, &k) in out.iter_mut().zip(gains) {
        let mut c = [T::ZERO; D];
        c.iter_mut().zip(sys.den_slice()).for_each(|(o, &d)| *o = d);
        c.iter_mut()
            .zip(sys.num_slice())
            .for_each(|(o, &n)| *o = o.saturating_add(&k.saturating_mul(&n)));
        let roots = ArrayPolynomial::<T, D>::from_coefficients(c).roots()?;
        row.iter_mut().zip(roots).for_each(|(o, r)| *o = r);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const U: f64 = f64::EPSILON / 2.0;

    /// `L(s) = (s + 2) / ((s + 1)(s + 3)(s + 5))`.
    fn plant() -> ArrayTransferFunction<f64, 2, 4> {
        ArrayTransferFunction::continuous([2.0, 1.0], [15.0, 23.0, 9.0, 1.0])
    }

    #[test]
    fn locus_roots_satisfy_characteristic() {
        let gains: [f64; 20] =
            core::array::from_fn(|k| 2.5 * f64::from(u8::try_from(k).unwrap()));
        let rows = root_locus::<_, 2, 4, 3, 20>(&plant(), &gains).unwrap();
        for (row, &k) in rows.iter().zip(&gains) {
            let c = [15.0 + 2.0 * k, 23.0 + k, 9.0, 1.0];
            let poly = ArrayPolynomial::<f64, 4>::from_coefficients(c);
            for &p in row {
                let residual = poly.evaluate_complex(p).magnitude();
                let scale: f64 = c
                    .iter()
                    .zip(0..)
                    .map(|(cj, j)| cj.abs() * p.magnitude().powi(j))
                    .sum();
                assert!(
                    residual / scale <= 10.0 * 3.0 * U,
                    "k={k} r={residual}"
                );
            }
        }
    }

    #[test]
    fn locus_zero_gain() {
        let rows = root_locus::<_, 2, 4, 3, 1>(&plant(), &[0.0]).unwrap();
        let poles = plant().poles().unwrap();
        for p in &rows[0] {
            let nearest = poles
                .iter()
                .take(3)
                .map(|q| (*p - *q).magnitude())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= 10.0 * 3.0 * U * 15.0);
        }
    }
}
