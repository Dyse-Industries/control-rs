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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pid_forms() {
        let (kp, ti, td, tf) = (2.0, 0.5, 0.25, 0.01);
        let std = pid::<f64, 3, 3>(PidForm::Standard { kp, ti, td }, Some(tf))
            .unwrap();
        assert_eq!(
            std.num_slice(),
            [kp, kp * (ti + tf), kp * (ti * (tf + td))]
        );
        assert_eq!(std.den_slice(), [0.0, ti, ti * tf]);

        let (kc, tau_i, tau_d) = (3.0, 0.4, 0.1);
        let ser =
            pid::<f64, 3, 3>(PidForm::Series { kc, tau_i, tau_d }, Some(tf))
                .unwrap();
        assert_eq!(
            ser.num_slice(),
            [kc, kc * (tau_i + (tau_d + tf)), kc * (tau_i * (tau_d + tf))]
        );
        assert_eq!(ser.den_slice(), [0.0, tau_i, tau_i * tf]);

        let (ki, kd) = (4.0, 0.3);
        let par = pid::<f64, 3, 3>(PidForm::Parallel { kp, ki, kd }, Some(tf))
            .unwrap();
        assert_eq!(par.num_slice(), [ki, kp + ki * tf, kp * tf + kd]);
        assert_eq!(par.den_slice(), [0.0, 1.0, tf]);
        assert!(par.is_continuous());
    }

    #[test]
    fn pid_filtered_proper() {
        let form = PidForm::Parallel {
            kp: 1.0,
            ki: 1.0,
            kd: 1.0,
        };
        let filtered = pid::<f64, 3, 3>(form, Some(0.1)).unwrap();
        assert!(filtered.num_slice().len() <= filtered.den_slice().len());
        assert_eq!(pid::<f64, 3, 3>(form, None), Err(ClassicalError::Improper));
        let pi = PidForm::Parallel {
            kp: 2.0,
            ki: 3.0,
            kd: 0.0,
        };
        let tf = pid::<f64, 2, 2>(pi, None).unwrap();
        assert_eq!(tf.num_slice(), [3.0, 2.0]);
        assert_eq!(tf.den_slice(), [0.0, 1.0]);
        assert_eq!(
            pid::<f64, 3, 3>(form, Some(0.0)),
            Err(ClassicalError::InvalidParameter)
        );
    }

    #[test]
    fn lead_lag() {
        let (gain, corner, ratio) = (2.0, 3.0, 10.0);
        let ld = lead(gain, corner, ratio).unwrap();
        assert_eq!(ld.num_slice(), [gain, gain / corner]);
        assert_eq!(ld.den_slice(), [1.0, 1.0 / (corner * ratio)]);
        let (lag_gain, lag_corner) = (5.0, 0.2);
        let lg = lag(lag_gain, lag_corner).unwrap();
        assert_eq!(lg.num_slice(), [lag_gain, lag_gain / lag_corner]);
        assert_eq!(lg.den_slice(), [1.0, lag_gain / lag_corner]);
        let bad = Err(ClassicalError::InvalidParameter);
        assert_eq!(lead(gain, 0.0, ratio), bad);
        assert_eq!(lead(gain, corner, -1.0), bad);
        assert_eq!(lag(0.0, lag_corner), bad);
        assert_eq!(lag(lag_gain, -lag_corner), bad);
        assert_eq!(lead(gain, corner, 0.0), bad);
        assert_eq!(lag(lag_gain, 0.0), bad);
    }

    #[test]
    fn pid_time_bounds() {
        let bad = Err(ClassicalError::InvalidParameter);
        let std = |ti, td| PidForm::Standard { kp: 1.0, ti, td };
        assert_eq!(pid::<f64, 3, 3>(std(0.0, 0.25), Some(0.1)), bad);
        assert_eq!(pid::<f64, 3, 3>(std(0.5, -0.25), Some(0.1)), bad);
        let ser = PidForm::Series {
            kc: 1.0,
            tau_i: 0.0,
            tau_d: 0.1,
        };
        assert_eq!(pid::<f64, 3, 3>(ser, Some(0.1)), bad);
    }

    #[test]
    fn pid_fit_conditions() {
        let form = PidForm::Parallel {
            kp: 1.0,
            ki: 1.0,
            kd: 1.0,
        };
        let improper = ClassicalError::Improper;
        assert_eq!(pid::<f64, 3, 4>(form, Some(0.1)).unwrap_err(), improper);
        assert_eq!(pid::<f64, 2, 3>(form, Some(0.1)).unwrap_err(), improper);
        assert_eq!(pid::<f64, 4, 3>(form, Some(0.1)).unwrap_err(), improper);
        let low = PidForm::Parallel {
            kp: 0.0,
            ki: 1.0,
            kd: 0.0,
        };
        assert!(pid::<f64, 3, 3>(low, Some(0.1)).is_ok());
    }
}
