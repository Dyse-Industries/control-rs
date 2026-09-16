//! # Lead, Lag, and Lead-Lag Compensator Synthesis
//!
//! Rational compensator factories (Iqbal, 2023) emitting native
//! [`ArrayTransferFunction`] models:
//! $C(s) = K \dfrac{s + 1/T}{s + 1/(\alpha T)}$, with $\alpha < 1$ producing
//! a lead network (phase lead near the zero/pole pair's geometric mean
//! frequency) and $\alpha > 1$ producing a lag network (low-frequency gain
//! boost with high-frequency attenuation). [`lead_lag`] cascades one of each
//! via [`ArrayTransferFunction::series`].
//!
//! Discrete-time realizations are obtained by discretizing the continuous
//! result with
//! [`crate::transfer_function::TransferFunction::to_discrete_tustin`] rather
//! than by a dedicated discrete constructor, satisfying FR-8's discrete
//! compensator requirement through composition instead of duplicated
//! discretization logic. The standard PID network
//! ($C(s) = K_p(1 + 1/(sT_i) + sT_d/(1+sT_f))$) is provided by
//! [`crate::classical_tools::pid::Pid::to_transfer_function`].
#![allow(
    clippy::doc_markdown,
    clippy::too_long_first_doc_paragraph,
    clippy::type_complexity,
    clippy::too_many_arguments,
    clippy::arithmetic_side_effects
)]

use crate::math::num_traits::Float;
use crate::transfer_function::ArrayTransferFunction;
use core::fmt;

/// Errors from [`lead`], [`lag`], and [`lead_lag`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompensatorError {
    /// `alpha` did not satisfy the range required for the requested network
    /// ($\alpha < 1$ for [`lead`], $\alpha > 1$ for [`lag`]).
    InvalidAlpha,
}

impl fmt::Display for CompensatorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidAlpha => {
                write!(f, "alpha out of range for the requested compensator")
            }
        }
    }
}

/// Lead compensator $C(s) = K \dfrac{s + 1/T}{s + 1/(\alpha T)}$ with
/// $\alpha < 1$: places the zero closer to the origin than the pole,
/// contributing phase lead near their geometric-mean frequency
/// $1/(T\sqrt{\alpha})$.
///
/// # Errors
/// [`CompensatorError::InvalidAlpha`] if `alpha >= 1`.
pub fn lead<T: Float + Copy>(
    k: T,
    t: T,
    alpha: T,
) -> Result<ArrayTransferFunction<T, 2, 2>, CompensatorError> {
    if alpha >= T::ONE {
        return Err(CompensatorError::InvalidAlpha);
    }
    Ok(lead_lag_stage(k, t, alpha))
}

/// Lag compensator $C(s) = K \dfrac{s + 1/T}{s + 1/(\alpha T)}$ with
/// $\alpha > 1$: places the pole closer to the origin than the zero,
/// boosting low-frequency gain while attenuating high frequencies.
///
/// # Errors
/// [`CompensatorError::InvalidAlpha`] if `alpha <= 1`.
pub fn lag<T: Float + Copy>(
    k: T,
    t: T,
    alpha: T,
) -> Result<ArrayTransferFunction<T, 2, 2>, CompensatorError> {
    if alpha <= T::ONE {
        return Err(CompensatorError::InvalidAlpha);
    }
    Ok(lead_lag_stage(k, t, alpha))
}

/// Cascades a lead stage ($\alpha_{\text{lead}} < 1$) with a lag stage
/// ($\alpha_{\text{lag}} > 1$) into a single lead-lag compensator network
/// via [`ArrayTransferFunction::series`]: low-frequency gain and
/// high-frequency attenuation from the lag stage, phase lead near
/// crossover from the lead stage.
///
/// # Errors
/// [`CompensatorError::InvalidAlpha`] if `alpha_lead >= 1` or
/// `alpha_lag <= 1`.
pub fn lead_lag<T: Float + Copy>(
    k_lead: T,
    t_lead: T,
    alpha_lead: T,
    k_lag: T,
    t_lag: T,
    alpha_lag: T,
) -> Result<ArrayTransferFunction<T, 3, 3>, CompensatorError> {
    let lead_stage = lead(k_lead, t_lead, alpha_lead)?;
    let lag_stage = lag(k_lag, t_lag, alpha_lag)?;
    Ok(lead_stage.series::<2, 2, 3, 3>(&lag_stage))
}

/// Builds $C(s) = K \dfrac{s + 1/T}{s + 1/(\alpha T)}$ (ascending
/// coefficients `num = [K/T, K]`, `den = [1/(\alpha T), 1]`) without
/// validating `alpha`'s sign convention; shared by [`lead`] and [`lag`]
/// after each checks its own required range.
fn lead_lag_stage<T: Float + Copy>(
    k: T,
    t: T,
    alpha: T,
) -> ArrayTransferFunction<T, 2, 2> {
    let zero = T::ONE / t;
    let pole = T::ONE / (alpha * t);
    ArrayTransferFunction::continuous([k * zero, k], [pole, T::ONE])
}
