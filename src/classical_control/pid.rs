//! Discrete PID controller with anti-windup and bumpless gain change
//! (FR-11 to FR-14).
//!
//! The derivative acts on the measurement through the filter
//! `T_f D' + D = -K_d y'` discretized by backward difference, so
//! `a_d = T_f / (T_f + h) in [0, 1]`; the filtered derivative `delta` is
//! stored without `K_d`. The integral uses forward Euler. Coefficients are
//! computed once at construction, so `step` runs a fixed sequence (NFR-4).

use super::compensator::{PidForm, pid};
use super::{ClassicalError, TfResult};
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};

#[cfg(any(test, feature = "ets"))]
#[cfg_attr(not(test), control_rs_macros::ets_suite)]
/// Unit and ETS test suite for the discrete PID controller.
pub mod tests {
    use super::*;
    use crate::math::num_traits::Trig;

    /// Parallel gains with a filter, wide limits and the given mode.
    fn params(anti_windup: AntiWindup<f64>, limit: f64) -> PidParams<f64> {
        PidParams {
            kp: 2.0,
            ki: 1.5,
            kd: 0.2,
            tf: 0.05,
            h: 0.01,
            u_min: -limit,
            u_max: limit,
            anti_windup,
        }
    }

    /// Unsaturated output `v` recomputed from the last sample.
    fn unsaturated(c: &Pid<f64>) -> f64 {
        c.params.kp * c.e_last + c.integral + c.params.kd * c.delta
    }

    #[cfg_attr(test, test)]
    /// Within limits, `f32` outputs follow the §4.10 recurrence evaluated
    /// in `f64` with relative error at most `10 k u` at sample `k` (FR-11).
    fn unsaturated_matches_recurrence() {
        let p32 = PidParams {
            kp: 2.0f32,
            ki: 1.5,
            kd: 0.2,
            tf: 0.05,
            h: 0.01,
            u_min: -1e6,
            u_max: 1e6,
            anti_windup: AntiWindup::Clamping,
        };
        let mut ctl = Pid::new(p32).unwrap();
        let (kp, ki, kd, tf, period) = (
            2.0f64,
            f64::from(1.5f32),
            f64::from(0.2f32),
            f64::from(0.05f32),
            f64::from(0.01f32),
        );
        let (ad, bd) = (tf / (tf + period), 1.0 / (tf + period));
        let (mut integ, mut filt, mut y_prev) = (0.0f64, 0.0f64, None::<f64>);
        let u32_round = f64::from(f32::EPSILON) / 2.0;
        for step in 1..=200u16 {
            let time = f32::from(step) * 0.01;
            let (sp, meas) = (10.0f32, 0.5 * Trig::sin(3.0 * time));
            let (sp64, meas64) = (f64::from(sp), f64::from(meas));
            let err = sp64 - meas64;
            filt = ad * filt - y_prev.map_or(0.0, |yp| meas64 - yp) * bd;
            let raw = kp * err + integ + kd * filt;
            integ += ki * period * err;
            y_prev = Some(meas64);
            let out = f64::from(ctl.step(sp, meas));
            let bound = 10.0 * f64::from(step) * u32_round * raw.abs();
            assert!((out - raw).abs() <= bound);
        }
    }

    #[cfg_attr(test, test)]
    /// A setpoint step at constant measurement leaves the derivative state
    /// unchanged exactly (FR-11).
    fn setpoint_step_no_derivative_kick() {
        let mut a = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
        let mut b = a;
        for k in 0..50u8 {
            let r = if k < 10 { 0.0 } else { 5.0 };
            a.step(r, 1.0);
            b.step(0.0, 1.0);
            assert_eq!(a.delta.to_bits(), b.delta.to_bits());
        }
    }

    #[cfg_attr(test, test)]
    /// Outputs stay within the limits and invalid parameters are rejected
    /// (FR-11).
    fn output_within_limits() {
        let mut c = Pid::new(params(AntiWindup::Clamping, 0.5)).unwrap();
        for k in 0..100u8 {
            let r = if k % 20 < 10 { 10.0 } else { -10.0 };
            let u = c.step(r, 0.0);
            assert!((-0.5..=0.5).contains(&u));
        }
        let base = params(AntiWindup::Clamping, 1.0);
        for bad in [
            PidParams { u_min: 2.0, ..base },
            PidParams { h: 0.0, ..base },
            PidParams { tf: -1.0, ..base },
            PidParams {
                anti_windup: AntiWindup::BackCalculation { tt: 0.0 },
                ..base
            },
        ] {
            assert_eq!(Pid::new(bad), Err(ClassicalError::InvalidParameter));
        }
    }

    #[cfg_attr(test, test)]
    /// Under sustained saturation in `Clamping` the integrator is constant
    /// exactly and integrates again once the error changes sign (FR-12).
    fn clamping_holds_integrator() {
        let mut c = Pid::new(params(AntiWindup::Clamping, 1.0)).unwrap();
        c.step(0.4, 0.0);
        let held = c.integral;
        for _ in 0..100 {
            assert_eq!(c.step(10.0, 0.0).to_bits(), 1.0f64.to_bits());
            assert_eq!(c.integral.to_bits(), held.to_bits());
        }
        c.step(-0.1, 0.0);
        assert!(c.integral < held);
    }

    #[cfg_attr(test, test)]
    /// Under sustained saturation in `BackCalculation` the integrator
    /// converges to `u_lim - K_p e - K_d delta + T_t K_i e` (FR-12).
    fn back_calculation_fixed_point() {
        let tt = 0.2;
        let p = params(AntiWindup::BackCalculation { tt }, 1.0);
        let mut c = Pid::new(p).unwrap();
        let e = 3.0;
        for _ in 0..400 {
            c.step(e, 0.0);
        }
        let fixed = p.u_max - p.kp * e - p.kd * c.delta + tt * p.ki * e;
        assert!(((c.integral - fixed) / fixed).abs() <= 1e-6);
    }

    #[cfg_attr(test, test)]
    /// `set_gains` keeps the unsaturated output of the last sample within
    /// `4u (|K_p e| + |I| + |K_d delta|)` (FR-13).
    fn bumpless_gain_change() {
        let mut c = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
        for k in 0..30u8 {
            c.step(1.0, 0.1 * Trig::sin(f64::from(k)));
        }
        let v = unsaturated(&c);
        let scale = (c.params.kp * c.e_last).abs()
            + c.integral.abs()
            + (c.params.kd * c.delta).abs();
        c.set_gains(5.0, 0.7, 0.9);
        let v2 = unsaturated(&c);
        assert!((v2 - v).abs() <= 4.0 * (f64::EPSILON / 2.0) * scale);
    }

    #[cfg_attr(test, test)]
    /// `to_transfer_function` equals the parallel form with filter exactly
    /// (FR-14).
    fn analysis_model_matches_compensator() {
        let p = params(AntiWindup::Clamping, 1.0);
        let c = Pid::new(p).unwrap();
        let got = c.to_transfer_function::<3, 3>().unwrap();
        let form = PidForm::Parallel {
            kp: p.kp,
            ki: p.ki,
            kd: p.kd,
        };
        assert_eq!(got, pid::<f64, 3, 3>(form, Some(p.tf)).unwrap());
    }
}

/// Integrator update under output saturation (FR-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiWindup<T> {
    /// Conditional integration: the integrator holds while the output is
    /// saturated and the error has the sign of `v - u`.
    Clamping,
    /// Back-calculation `I <- I + h (K_i e + (u - v) / T_t)`.
    ///
    /// A rule of thumb is `T_d < T_t < T_i` and `T_t = sqrt(T_i T_d)` with
    /// `T_i = K_p / K_i` and `T_d = K_d / K_p`.
    BackCalculation {
        /// Tracking time constant `T_t > 0`.
        tt: T,
    },
}

/// Construction parameters of a [`Pid`] (parallel gains).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PidParams<T> {
    /// Proportional gain `K_p`.
    pub kp: T,
    /// Integral gain `K_i`.
    pub ki: T,
    /// Derivative gain `K_d`.
    pub kd: T,
    /// Derivative filter time constant `T_f >= 0`.
    pub tf: T,
    /// Sample period `h > 0`.
    pub h: T,
    /// Lower output limit `u_min`.
    pub u_min: T,
    /// Upper output limit `u_max >= u_min`.
    pub u_max: T,
    /// Anti-windup mode.
    pub anti_windup: AntiWindup<T>,
}

/// Discrete PID controller (FR-11 to FR-14). Requires `T: Float` (C-5).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pid<T> {
    params: PidParams<T>,
    /// `a_d = T_f / (T_f + h)`.
    ad: T,
    /// `1 / (T_f + h)`.
    bd: T,
    /// `K_i h`.
    kih: T,
    /// `h / T_t` for back-calculation, else zero.
    hbt: T,
    integral: T,
    delta: T,
    y_prev: Option<T>,
    e_last: T,
}

impl<T: Float + Copy> Pid<T> {
    /// Builds a controller with zero state.
    ///
    /// # Errors
    /// [`ClassicalError::InvalidParameter`] when `u_min > u_max`,
    /// `h <= 0`, `T_f < 0` or a back-calculation `T_t <= 0`.
    pub fn new(params: PidParams<T>) -> Result<Self, ClassicalError> {
        let tt_ok = match params.anti_windup {
            AntiWindup::Clamping => true,
            AntiWindup::BackCalculation { tt } => tt > T::ZERO,
        };
        let valid = params.u_min <= params.u_max
            && params.h > T::ZERO
            && params.tf >= T::ZERO
            && tt_ok;
        if !valid {
            return Err(ClassicalError::InvalidParameter);
        }
        let sum = params.tf.saturating_add(&params.h);
        let hbt = match params.anti_windup {
            AntiWindup::Clamping => T::ZERO,
            AntiWindup::BackCalculation { tt } => params.h.saturating_div(&tt),
        };
        Ok(Self {
            params,
            ad: params.tf.saturating_div(&sum),
            bd: T::ONE.saturating_div(&sum),
            kih: params.ki.saturating_mul(&params.h),
            hbt,
            integral: T::ZERO,
            delta: T::ZERO,
            y_prev: None,
            e_last: T::ZERO,
        })
    }

    /// Computes the output for setpoint `r` and measurement `y`, then
    /// updates the integrator (FR-11, FR-12).
    ///
    /// `u = min(max(K_p e + I + K_d delta, u_min), u_max)` with
    /// `e = r - y`; the output is computed before the integrator update.
    #[inline]
    pub fn step(&mut self, r: T, y: T) -> T {
        let params = &self.params;
        let err = r.saturating_sub(&y);
        let dy = self.y_prev.map_or(T::ZERO, |prev| y.saturating_sub(&prev));
        self.delta = self
            .ad
            .saturating_mul(&self.delta)
            .saturating_sub(&dy.saturating_mul(&self.bd));
        let raw = params
            .kp
            .saturating_mul(&err)
            .saturating_add(&self.integral)
            .saturating_add(&params.kd.saturating_mul(&self.delta));
        let out = if raw < params.u_min {
            params.u_min
        } else if raw > params.u_max {
            params.u_max
        } else {
            raw
        };
        self.integral = self
            .integral
            .saturating_add(&self.integrator_step(err, raw, out));
        self.y_prev = Some(y);
        self.e_last = err;
        out
    }

    /// Clears the integrator, the derivative state and the stored
    /// measurement.
    pub const fn reset(&mut self) {
        self.integral = T::ZERO;
        self.delta = T::ZERO;
        self.y_prev = None;
        self.e_last = T::ZERO;
    }

    /// Changes the gains between samples, correcting the integrator so the
    /// unsaturated output recomputed from the last sample is unchanged
    /// (FR-13): `I <- I + (K_p - K_p') e_k + (K_d - K_d') delta_k`.
    pub fn set_gains(&mut self, kp: T, ki: T, kd: T) {
        let dp = self
            .params
            .kp
            .saturating_sub(&kp)
            .saturating_mul(&self.e_last);
        let dd = self
            .params
            .kd
            .saturating_sub(&kd)
            .saturating_mul(&self.delta);
        self.integral = self.integral.saturating_add(&dp).saturating_add(&dd);
        self.params.kp = kp;
        self.params.ki = ki;
        self.params.kd = kd;
        self.kih = ki.saturating_mul(&self.params.h);
    }

    /// Continuous-time model of the unsaturated controller: the parallel
    /// form of [`pid`] with derivative filter `T_f` (FR-14).
    ///
    /// # Errors
    /// [`ClassicalError::Improper`] when `T_f = 0` with a nonzero
    /// derivative, or the result does not fit `(N, D)`.
    pub fn to_transfer_function<const N: usize, const D: usize>(
        &self,
    ) -> TfResult<T, N, D>
    where
        Const<N>: Dim,
        Const<D>: Dim,
    {
        let p = &self.params;
        let form = PidForm::Parallel {
            kp: p.kp,
            ki: p.ki,
            kd: p.kd,
        };
        pid(form, (p.tf > T::ZERO).then_some(p.tf))
    }

    /// Integrator increment for error `e`, unsaturated output `v` and
    /// output `u`.
    fn integrator_step(&self, e: T, v: T, u: T) -> T {
        let euler = self.kih.saturating_mul(&e);
        match self.params.anti_windup {
            AntiWindup::Clamping => {
                let pushing = (e > T::ZERO && v > u) || (e < T::ZERO && v < u);
                if pushing { T::ZERO } else { euler }
            }
            AntiWindup::BackCalculation { .. } => euler.saturating_add(
                &self.hbt.saturating_mul(&u.saturating_sub(&v)),
            ),
        }
    }
}
