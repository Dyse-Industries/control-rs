//! PID, lead and lag compensators as `TransferFunction` values (FR-6, FR-7).
//!
//! Constructed compensators are ordinary `TransferFunction` data and compose
//! through `series`, `parallel` and `feedback`. A discrete compensator is the
//! Tustin image of the continuous one, obtained through
//! `TransferFunction::to_discrete_tustin` with prewarping at the design
//! crossover.

use super::{ClassicalError, TfResult};
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::transfer_function::ArrayTransferFunction;

/// PID parameterization (FR-6).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PidForm<T> {
    /// `K_p (1 + (1) / (T_i s) + T_d s)`.
    Standard {
        /// Proportional gain `K_p`.
        kp: T,
        /// Integral time `T_i > 0`.
        ti: T,
        /// Derivative time `T_d >= 0`.
        td: T,
    },
    /// `K_c (1 + (1) / (tau_i s))(tau_d s + 1)`.
    Series {
        /// Gain `K_c`.
        kc: T,
        /// Integral time `tau_i > 0`.
        tau_i: T,
        /// Derivative time `tau_d >= 0`.
        tau_d: T,
    },
    /// `K_p + (K_i) / (s) + K_d s`.
    Parallel {
        /// Proportional gain `K_p`.
        kp: T,
        /// Integral gain `K_i`.
        ki: T,
        /// Derivative gain `K_d`.
        kd: T,
    },
}

/// Builds a PID compensator (FR-6).
///
/// With a filter time constant `tf = Some(T_f)`, `s` in the derivative term
/// is replaced by `s / (1 + T_f s)` and the result is proper with
/// numerator and denominator of degree 2:
///
/// | Form | Numerator (ascending) | Denominator (ascending) |
/// |:--|:--|:--|
/// | Standard | `K_p [1, T_i + T_f, T_i (T_f + T_d)]` | `[0, T_i, T_i T_f]` |
/// | Series | `K_c [1, tau_i + tau_d + T_f, tau_i (tau_d + T_f)]` | `[0, tau_i, tau_i T_f]` |
/// | Parallel | `[K_i, K_p + K_i T_f, K_p T_f + K_d]` | `[0, 1, T_f]` |
///
/// Without a filter the same expansion with `T_f = 0` is improper unless
/// the derivative term vanishes. The result is written into capacities
/// `(N, D)`, which must equal the numerator and denominator degrees plus
/// one and satisfy `N <= D`.
///
/// # Errors
/// - [`ClassicalError::InvalidParameter`]: a time constant `T_i` or
///   `tau_i` is not positive, `T_d` or `tau_d` is negative, or `tf` is
///   `Some` and not positive.
/// - [`ClassicalError::Improper`]: the result is improper or does not fit
///   `(N, D)`.
pub fn pid<T: Float + Copy, const N: usize, const D: usize>(
    form: PidForm<T>,
    tf: Option<T>,
) -> TfResult<T, N, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let t_f = match tf {
        Some(v) if v > T::ZERO => v,
        Some(_) => return Err(ClassicalError::InvalidParameter),
        None => T::ZERO,
    };
    let (num, den) = match form {
        PidForm::Standard { kp, ti, td } => {
            check_times(ti, td)?;
            let num = [
                T::ONE,
                ti.saturating_add(&t_f),
                ti.saturating_mul(&t_f.saturating_add(&td)),
            ];
            (
                num.map(|c| kp.saturating_mul(&c)),
                [T::ZERO, ti, ti.saturating_mul(&t_f)],
            )
        }
        PidForm::Series { kc, tau_i, tau_d } => {
            check_times(tau_i, tau_d)?;
            let lead = tau_d.saturating_add(&t_f);
            let num = [
                T::ONE,
                tau_i.saturating_add(&lead),
                tau_i.saturating_mul(&lead),
            ];
            (
                num.map(|c| kc.saturating_mul(&c)),
                [T::ZERO, tau_i, tau_i.saturating_mul(&t_f)],
            )
        }
        PidForm::Parallel { kp, ki, kd } => {
            let num = [
                ki,
                kp.saturating_add(&ki.saturating_mul(&t_f)),
                kp.saturating_mul(&t_f).saturating_add(&kd),
            ];
            (num, [T::ZERO, T::ONE, t_f])
        }
    };
    fit(num, den)
}

/// Builds the lead compensator `K (1 + s/b) / (1 + s/(bN))` (FR-7).
///
/// # Errors
/// [`ClassicalError::InvalidParameter`] when `b` or `n` is not positive.
pub fn lead<T: Float + Copy>(k: T, b: T, n: T) -> TfResult<T, 2, 2> {
    if !(b > T::ZERO && n > T::ZERO) {
        return Err(ClassicalError::InvalidParameter);
    }
    Ok(ArrayTransferFunction::continuous(
        [k, k.saturating_div(&b)],
        [T::ONE, T::ONE.saturating_div(&b.saturating_mul(&n))],
    ))
}

/// Builds the lag compensator `M (1 + s/a) / (1 + sM/a)` (FR-7).
///
/// # Errors
/// [`ClassicalError::InvalidParameter`] when `m` or `a` is not positive.
pub fn lag<T: Float + Copy>(m: T, a: T) -> TfResult<T, 2, 2> {
    if !(m > T::ZERO && a > T::ZERO) {
        return Err(ClassicalError::InvalidParameter);
    }
    let ratio = m.saturating_div(&a);
    Ok(ArrayTransferFunction::continuous(
        [m, ratio],
        [T::ONE, ratio],
    ))
}

/// Rejects a non-positive integral time or a negative derivative time.
fn check_times<T: Float + Copy>(
    integral: T,
    derivative: T,
) -> Result<(), ClassicalError> {
    if integral > T::ZERO && derivative >= T::ZERO {
        Ok(())
    } else {
        Err(ClassicalError::InvalidParameter)
    }
}

/// Index of the highest nonzero coefficient, `None` for the zero polynomial.
fn degree<T: Float + Copy>(c: &[T]) -> Option<usize> {
    c.iter().rposition(|v| *v != T::ZERO)
}

/// Writes degree-2 numerator and denominator into capacities `(N, D)`.
fn fit<T: Float + Copy, const N: usize, const D: usize>(
    num: [T; 3],
    den: [T; 3],
) -> TfResult<T, N, D>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let dd = degree(&den).ok_or(ClassicalError::Improper)?;
    let nd = degree(&num).unwrap_or(0);
    if nd > dd || dd.saturating_add(1) != D || nd >= N || N > D {
        return Err(ClassicalError::Improper);
    }
    let mut n_out = [T::ZERO; N];
    let mut d_out = [T::ZERO; D];
    n_out.iter_mut().zip(num).for_each(|(o, c)| *o = c);
    d_out.iter_mut().zip(den).for_each(|(o, c)| *o = c);
    Ok(ArrayTransferFunction::continuous(n_out, d_out))
}
