//! Root locus over a caller-supplied gain set (FR-1).
//!
//! For each gain `k_i` the closed-loop characteristic polynomial
//! `D(s) + k_i N(s)` is formed with coefficient arithmetic and solved with
//! `Polynomial::roots`. Each output row is an unordered root set; root order
//! across consecutive gains is not matched.

use super::ClassicalError;
use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
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
/// The open loop must be proper: a numerator of higher degree than the
/// denominator cannot fit in the length-`D` characteristic polynomial, and
/// truncating it would drop closed-loop poles. The cost is `K` root solves.
///
/// # Errors
/// - [`ClassicalError::Improper`]: `deg(num) > deg(den)`, or the
///   denominator is identically zero.
/// - [`ClassicalError::Root`]: a characteristic polynomial has a zero
///   leading coefficient or root finding fails.
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
{
    // The stack characteristic polynomial has length `D`. A higher-degree
    // numerator would be truncated by the coefficient zip and invent the
    // wrong closed-loop roots.
    let num_deg = sys.num_slice().iter().rposition(|v| *v != T::ZERO);
    let den_deg = sys.den_slice().iter().rposition(|v| *v != T::ZERO);
    match (num_deg, den_deg) {
        (_, None) => return Err(ClassicalError::Improper),
        (Some(n), Some(d)) if n > d => return Err(ClassicalError::Improper),
        _ => {}
    }
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
        let rows = root_locus(&plant(), &gains).unwrap();
        for (row, &k) in rows.iter().zip(&gains) {
            let c = [15.0 + 2.0 * k, 23.0 + k, 9.0, 1.0];
            let poly = ArrayPolynomial::<f64, 4>::from_coefficients(c);
            for &p in &row[..3] {
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
        let rows = root_locus(&plant(), &[0.0]).unwrap();
        let poles = plant().poles().unwrap();
        // Every open-loop pole is a locus point at k = 0.
        for q in &poles[..3] {
            let nearest = rows[0][..3]
                .iter()
                .map(|p| (*p - *q).magnitude())
                .fold(f64::INFINITY, f64::min);
            assert!(nearest <= 10.0 * 3.0 * U);
        }
    }

    #[test]
    fn improper_open_loop_is_rejected() {
        // L(s) = (1 + s^2) / (1 + s). Without the degree gate the length-2
        // characteristic zip would drop the s^2 term and report the root of
        // 3 + s at k = 2 instead of the quadratic 2 s^2 + s + 3.
        let improper = ArrayTransferFunction::<f64, 3, 2>::continuous(
            [1.0, 0.0, 1.0],
            [1.0, 1.0],
        );
        assert_eq!(
            root_locus(&improper, &[2.0]),
            Err(ClassicalError::Improper)
        );
        let zero_den =
            ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [0.0, 0.0]);
        assert_eq!(
            root_locus(&zero_den, &[1.0]),
            Err(ClassicalError::Improper)
        );
        // Trailing zero padding that keeps deg(N) <= deg(D) stays accepted.
        let padded = ArrayTransferFunction::<f64, 3, 2>::continuous(
            [1.0, 0.0, 0.0],
            [1.0, 1.0],
        );
        assert!(root_locus(&padded, &[1.0]).is_ok());
    }
}
