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

/// Integrator update under output saturation (FR-12).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntiWindup<T> {
    /// Conditional integration: the integrator holds while the output is
    /// saturated and the integrator increment `K_i h e` has the sign of
    /// `v - u`.
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
    pub(crate) params: PidParams<T>,
    /// `a_d = T_f / (T_f + h)`.
    ad: T,
    /// `1 / (T_f + h)`.
    bd: T,
    /// `K_i h`.
    kih: T,
    /// `h / T_t` for back-calculation, else zero.
    hbt: T,
    pub(crate) integral: T,
    pub(crate) delta: T,
    pub(crate) y_prev: Option<T>,
    pub(crate) e_last: T,
}

impl<T: Float + Copy> Pid<T> {
    /// Builds a controller with zero state.
    ///
    /// # Errors
    /// [`ClassicalError::InvalidParameter`] when `u_min > u_max`,
    /// `h <= 0`, `T_f < 0`, a back-calculation `T_t <= 0`, or any parameter
    /// is NaN or a gain or time parameter (`h`, `T_f`, `T_t`) is infinite.
    /// An infinite limit means no limit.
    pub fn new(params: PidParams<T>) -> Result<Self, ClassicalError> {
        // `x - x` is zero only for a finite `x`; NaN and infinity give NaN.
        let finite = |x: T| x.saturating_sub(&x) == T::ZERO;
        let tt_ok = match params.anti_windup {
            AntiWindup::Clamping => true,
            AntiWindup::BackCalculation { tt } => tt > T::ZERO && finite(tt),
        };
        let valid = params.u_min <= params.u_max
            && params.h > T::ZERO
            && params.tf >= T::ZERO
            && finite(params.kp)
            && finite(params.ki)
            && finite(params.kd)
            && finite(params.h)
            && finite(params.tf)
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
    ///
    /// A non-finite `r` or `y` enters the integrator and filter state and
    /// the output, and stays there. The caller screens sensor faults and
    /// calls [`reset`](Self::reset) to recover.
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
                let pushing =
                    (euler > T::ZERO && v > u) || (euler < T::ZERO && v < u);
                if pushing { T::ZERO } else { euler }
            }
            AntiWindup::BackCalculation { .. } => euler.saturating_add(
                &self.hbt.saturating_mul(&u.saturating_sub(&v)),
            ),
        }
    }
}
