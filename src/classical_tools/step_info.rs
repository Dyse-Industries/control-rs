//! # Step Response Transient Metrics Extraction
//!
//! Standardized extraction of transient step response metrics ($t_r$, $M_p$,
//! $t_s$, $e_{ss}$) from time-series trajectories for classical SISO control systems.
#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::doc_markdown
)]

use crate::math::num_traits::Float;

/// Transient performance metrics extracted from a step response trajectory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInfo<T> {
    /// 10% to 90% rise time in seconds.
    pub rise_time: T,
    /// Peak overshoot percentage relative to target step magnitude:
    /// $$M_p = \frac{y_{\mathrm{peak}} - y_{\mathrm{target}}}{y_{\mathrm{target}} - y_{\mathrm{initial}}} \times 100\%$$
    pub peak_overshoot_pct: T,
    /// Peak response value.
    pub peak_value: T,
    /// Timestamp corresponding to peak value.
    pub peak_time: T,
    /// Settling time within the specified tolerance band of the target setpoint.
    /// Returns `None` if the response does not settle within the target band
    /// (e.g. Type-0 steady-state offset larger than the settling tolerance).
    pub settling_time: Option<T>,
    /// Settling time within the specified tolerance band of the achieved steady-state value.
    pub settling_time_achieved: T,
    /// Absolute steady-state error $|y_{\mathrm{final}} - y_{\mathrm{target}}|$.
    pub steady_state_error: T,
    /// Achieved steady-state value $y_{\mathrm{final}}$.
    pub steady_state_value: T,
}

#[inline]
fn max_t<T: PartialOrd>(a: T, b: T) -> T {
    if a > b { a } else { b }
}

/// Computes transient step response metrics from sampled time-series data.
///
/// # Arguments
/// - `time`: Slice of strictly increasing sample timestamps.
/// - `response`: Slice of output values corresponding to `time`.
/// - `step_time`: Timestamp at which the setpoint step is initiated.
/// - `initial_val`: Value of the output prior to the step.
/// - `target_val`: Target setpoint value.
/// - `settling_pct`: Settling band fraction (e.g. `Some(0.02)` for a 2% band).
///   Defaults to 2% if `None`.
#[must_use]
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn step_info<T: Float + Copy + PartialOrd>(
    time: &[T],
    response: &[T],
    step_time: T,
    initial_val: T,
    target_val: T,
    settling_pct: Option<T>,
) -> StepInfo<T> {
    let zero = T::ZERO;
    let one = T::ONE;
    let ten = T::from_usize(10);
    let hundred = T::from_usize(100);
    let default_band_pct = T::from_usize(2) / hundred;
    let band_pct = settling_pct.unwrap_or(default_band_pct);

    let step_mag = target_val - initial_val;
    let n = time.len().min(response.len());

    let epsilon = T::from_usize(1) / T::from_usize(1_000_000_000);
    if n == 0 || step_mag.abs() < epsilon {
        return StepInfo {
            rise_time: zero,
            peak_overshoot_pct: zero,
            peak_value: initial_val,
            peak_time: step_time,
            settling_time: None,
            settling_time_achieved: zero,
            steady_state_error: zero,
            steady_state_value: initial_val,
        };
    }

    let final_val = response[n - 1];
    let sse = (final_val - target_val).abs();
    let actual_step = final_val - initial_val;

    let micro = T::from_usize(1) / T::from_usize(1_000_000);
    let (val_10, val_90) = if actual_step.abs() > micro {
        (
            initial_val + (one / ten) * actual_step,
            initial_val + (T::from_usize(9) / ten) * actual_step,
        )
    } else {
        (
            initial_val + (one / ten) * step_mag,
            initial_val + (T::from_usize(9) / ten) * step_mag,
        )
    };

    let mut t_10 = None;
    let mut t_90 = None;
    let mut peak_val = initial_val;
    let mut peak_t = step_time;

    for i in 0..n {
        let t = time[i];
        if t < step_time {
            continue;
        }
        let y = response[i];

        if step_mag > zero {
            if y > peak_val {
                peak_val = y;
                peak_t = t;
            }
            if t_10.is_none() && y >= val_10 {
                t_10 = Some(t);
            }
            if t_90.is_none() && y >= val_90 {
                t_90 = Some(t);
            }
        } else {
            if y < peak_val {
                peak_val = y;
                peak_t = t;
            }
            if t_10.is_none() && y <= val_10 {
                t_10 = Some(t);
            }
            if t_90.is_none() && y <= val_90 {
                t_90 = Some(t);
            }
        }
    }

    let rise_time = match (t_10, t_90) {
        (Some(t1), Some(t2)) if t2 >= t1 => t2 - t1,
        _ => zero,
    };

    let overshoot_pct = if step_mag > zero && peak_val > target_val {
        ((peak_val - target_val) / step_mag) * hundred
    } else if step_mag < zero && peak_val < target_val {
        ((target_val - peak_val) / step_mag.abs()) * hundred
    } else {
        zero
    };

    // Settling band tolerances
    let target_band = band_pct * step_mag.abs();
    let achieved_band = band_pct * max_t(actual_step.abs(), target_band);

    // Target settling time: only defined if the steady-state value is within target band
    let settling_time = if sse <= target_band {
        let mut t_settle = zero;
        for i in (0..n).rev() {
            let t = time[i];
            if t < step_time {
                break;
            }
            let y = response[i];
            if (y - target_val).abs() > target_band {
                t_settle = max_t(t - step_time, zero);
                break;
            }
        }
        Some(t_settle)
    } else {
        None
    };

    // Achieved settling time: settling relative to the final achieved value
    let mut settling_time_achieved = zero;
    for i in (0..n).rev() {
        let t = time[i];
        if t < step_time {
            break;
        }
        let y = response[i];
        if (y - final_val).abs() > achieved_band {
            settling_time_achieved = max_t(t - step_time, zero);
            break;
        }
    }

    StepInfo {
        rise_time,
        peak_overshoot_pct: overshoot_pct,
        peak_value: peak_val,
        peak_time: peak_t,
        settling_time,
        settling_time_achieved,
        steady_state_error: sse,
        steady_state_value: final_val,
    }
}
