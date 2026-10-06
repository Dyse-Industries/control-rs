//! Step-response metrics from sampled data (FR-15).
//!
//! With `y_norm = (y - y_init) / (y_final - y_init)`: rise time
//! runs from 10% to 90% of the way to `y_final`, with crossings
//! interpolated linearly; settling time is the first sample time after
//! which `|y - y_final| <= theta |y_final - y_init|` holds for every
//! later sample; overshoot is `max(0, 100 max (y_norm,k - 1))`; peak
//! is `max |y_k - y_init|` at the peak time. The routine takes response
//! data only, so it applies equally to simulated and on-target logs.

use super::ClassicalError;
use crate::math::num_traits::Float;

/// Reference values and threshold of a step-response evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepOptions<T> {
    /// Initial value `y_init` (default 0).
    pub y_init: T,
    /// Final value `y_final`; `None` takes the last sample.
    pub y_final: Option<T>,
    /// Settling threshold `theta` as a fraction (default 0.02).
    pub threshold: T,
}

/// Step-response metrics (FR-15).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepInfo<T> {
    /// Time from 10% to 90% of the transition; `None` if a level is never
    /// reached.
    pub rise_time: Option<T>,
    /// First sample time after which the response stays within the
    /// threshold; `None` if the last sample is outside it.
    pub settling_time: Option<T>,
    /// Percent overshoot.
    pub overshoot: T,
    /// Peak deviation `max |y_k - y_init|`.
    pub peak: T,
    /// Time of the peak.
    pub peak_time: T,
}

/// Step-response metrics or the reason they cannot be evaluated.
pub type StepResult<T> = Result<StepInfo<T>, ClassicalError>;

/// Peak deviation, its time and the maximum normalized response.
type Peak<T> = (T, T, T);

impl<T: Float + Copy> Default for StepOptions<T> {
    fn default() -> Self {
        Self {
            y_init: T::ZERO,
            y_final: None,
            threshold: T::ONE.saturating_div(&T::from_usize(50)),
        }
    }
}

/// Evaluates step-response metrics on samples `(t_k, y_k)` (FR-15).
///
/// # Errors
/// [`ClassicalError::InvalidParameter`] when `K == 0`, `y_final = y_init`
/// or the threshold is negative.
pub fn step_info<T: Float + Copy, const K: usize>(
    t: &[T; K],
    y: &[T; K],
    opts: &StepOptions<T>,
) -> StepResult<T> {
    let last = y.last().copied().ok_or(ClassicalError::InvalidParameter)?;
    let y_final = opts.y_final.unwrap_or(last);
    let span = y_final.saturating_sub(&opts.y_init);
    if span == T::ZERO || opts.threshold < T::ZERO {
        return Err(ClassicalError::InvalidParameter);
    }
    let norm = |v: T| v.saturating_sub(&opts.y_init).saturating_div(&span);
    let ten = T::from_usize(10);
    let lo = crossing_time(t, y, norm, T::ONE.saturating_div(&ten));
    let hi = crossing_time(t, y, norm, T::from_usize(9).saturating_div(&ten));
    let band = opts.threshold.saturating_mul(&span.abs());
    let (peak, peak_time, max_norm) = peak_of(t, y, opts.y_init, norm);
    let excess = max_norm.saturating_sub(&T::ONE);
    Ok(StepInfo {
        rise_time: lo.zip(hi).map(|(a, b)| b.saturating_sub(&a)),
        settling_time: settling(t, y, y_final, band),
        overshoot: if excess > T::ZERO {
            T::from_usize(100).saturating_mul(&excess)
        } else {
            T::ZERO
        },
        peak,
        peak_time,
    })
}

/// First time `norm(y)` reaches `level`, interpolated linearly.
fn crossing_time<T: Float + Copy>(
    t: &[T],
    y: &[T],
    norm: impl Fn(T) -> T,
    level: T,
) -> Option<T> {
    let pts = t.iter().zip(y).map(|(&tk, &yk)| (tk, norm(yk)));
    let mut prev = None;
    for (tk, nk) in pts {
        if nk >= level {
            return Some(prev.map_or(tk, |(tp, np): (T, T)| {
                let frac = level
                    .saturating_sub(&np)
                    .saturating_div(&nk.saturating_sub(&np));
                tp.saturating_add(&tk.saturating_sub(&tp).saturating_mul(&frac))
            }));
        }
        prev = Some((tk, nk));
    }
    None
}

/// First sample time after the last sample outside `band` of `y_final`.
fn settling<T: Float + Copy>(
    t: &[T],
    y: &[T],
    y_final: T,
    band: T,
) -> Option<T> {
    let outside = |v: &T| v.saturating_sub(&y_final).abs() > band;
    y.iter().rposition(outside).map_or_else(
        || t.first().copied(),
        |k| t.get(k.saturating_add(1)).copied(),
    )
}

/// Peak deviation, its time and the maximum normalized response.
fn peak_of<T: Float + Copy>(
    t: &[T],
    y: &[T],
    y_init: T,
    norm: impl Fn(T) -> T,
) -> Peak<T> {
    let mut best = (T::ZERO, t.first().copied().unwrap_or(T::ZERO), T::ZERO);
    let mut first = true;
    for (&tk, &yk) in t.iter().zip(y) {
        let dev = yk.saturating_sub(&y_init).abs();
        let n = norm(yk);
        if first || dev > best.0 {
            best.0 = dev;
            best.1 = tk;
        }
        if first || n > best.2 {
            best.2 = n;
        }
        first = false;
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    const H: f64 = 0.001;
    const WN: f64 = 2.0;
    const ZETA: f64 = 0.3;

    /// Sample times and values.
    type Trace = ([f64; 8001], [f64; 8001]);

    fn response() -> Trace {
        let wd = WN * (1.0 - ZETA * ZETA).sqrt();
        let phi = (1.0 - ZETA * ZETA).sqrt().atan2(ZETA);
        let t: [f64; 8001] =
            core::array::from_fn(|k| H * f64::from(u16::try_from(k).unwrap()));
        let y = t.map(|tk| {
            1.0 - (-ZETA * WN * tk).exp() * (wd * tk + phi).sin()
                / (1.0 - ZETA * ZETA).sqrt()
        });
        (t, y)
    }

    fn settle_time(theta: f64, t: &[f64], y: &[f64]) -> f64 {
        let last_out = y.iter().rposition(|v| (v - 1.0).abs() > theta).unwrap();
        *t.get(last_out.saturating_add(1)).unwrap()
    }

    #[test]
    fn second_order_closed_form() {
        let (t, y) = response();
        let opts = StepOptions {
            y_final: Some(1.0),
            ..StepOptions::default()
        };
        let info = step_info(&t, &y, &opts).unwrap();
        let wd = WN * (1.0 - ZETA * ZETA).sqrt();
        let tp = core::f64::consts::PI / wd;
        let os = 100.0
            * (-ZETA * core::f64::consts::PI / (1.0 - ZETA * ZETA).sqrt())
                .exp();
        assert!((info.peak_time - tp).abs() <= H);
        let ddy = WN * WN * (os / 100.0);
        assert!(
            (info.overshoot - os).abs() <= 100.0 * ddy * H * H / 8.0 + 1e-9
        );
        assert!((info.peak - (1.0 + os / 100.0)).abs() <= ddy * H * H);
        // Rise and settling times against a fine-grid evaluation of the
        // closed form.
        let wd_t = |tk: f64| {
            let phi = (1.0 - ZETA * ZETA).sqrt().atan2(ZETA);
            1.0 - (-ZETA * WN * tk).exp() * (wd * tk + phi).sin()
                / (1.0 - ZETA * ZETA).sqrt()
        };
        let cross = |level: f64| {
            let mut lo = 0.0;
            let mut hi = tp;
            for _ in 0..60 {
                let mid = f64::midpoint(lo, hi);
                if wd_t(mid) < level {
                    lo = mid;
                } else {
                    hi = mid;
                }
            }
            lo
        };
        let rise = cross(0.9) - cross(0.1);
        assert!((info.rise_time.unwrap() - rise).abs() <= H);
        let st = info.settling_time.unwrap();
        assert!((st - settle_time(0.02, &t, &y)).abs() <= H);
    }

    #[test]
    fn defaults_and_threshold() {
        let (t, y) = response();
        let default = step_info(&t, &y, &StepOptions::default()).unwrap();
        let explicit = StepOptions {
            y_final: y.last().copied(),
            ..StepOptions::default()
        };
        assert_eq!(default, step_info(&t, &y, &explicit).unwrap());
        let five = StepOptions {
            y_final: Some(1.0),
            threshold: 0.05,
            ..StepOptions::default()
        };
        let info = step_info(&t, &y, &five).unwrap();
        // Envelope bound of the 5% settling time.
        let ts_env = -(0.05 * (1.0 - ZETA * ZETA).sqrt()).ln() / (ZETA * WN);
        let st = info.settling_time.unwrap();
        assert!(st <= ts_env + H);
        assert!((st - settle_time(0.05, &t, &y)).abs() <= H);
        let flat = [1.0; 4];
        assert_eq!(
            step_info(
                &[0.0, 1.0, 2.0, 3.0],
                &flat,
                &StepOptions {
                    y_init: 1.0,
                    ..StepOptions::default()
                }
            ),
            Err(ClassicalError::InvalidParameter)
        );
    }
}
