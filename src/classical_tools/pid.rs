//! # Discrete PID Controller
//!
//! A scalar PID controller with derivative-on-measurement (avoiding
//! setpoint-driven derivative kick), a single-pole low-pass filter on the
//! derivative term, and conditional (clamping) integrator anti-windup
//! (Beauregard, 2011; Åström & Hägglund, 2006).
//!
//! Continuous-time transfer function, parallel form:
//! $C(s) = K_p + \dfrac{K_i}{s} + \dfrac{K_d s}{1 + s T_f}$, exposed for
//! frequency-domain analysis via [`Pid::to_transfer_function`].
//!
//! The filtered derivative term is discretized by backward Euler applied to
//! $T_f \dot{d}(t) + d(t) = -K_d \dot{y}(t)$, giving the one-step recursion
//! $d_k = \dfrac{T_f}{T_f + \Delta t} d_{k-1} - \dfrac{K_d}{T_f + \Delta t}
//! (y_k - y_{k-1})$.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::doc_markdown,
    clippy::too_many_arguments
)]

use crate::math::num_traits::Float;
use crate::transfer_function::ArrayTransferFunction;
use core::fmt;

/// Errors from [`Pid::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PidError {
    /// Sample period `dt` was not strictly positive, or $T_f + dt$ cancelled
    /// the derivative-filter denominator.
    InvalidSamplePeriod,
}

/// A scalar discrete-time PID controller.
///
/// Construct with [`Pid::new`], then call [`Pid::step`] once per control
/// period. [`Pid::to_transfer_function`] exposes the continuous-time
/// parallel-form transfer function for frequency-domain analysis (Bode,
/// margins, root locus); discretize it with
/// [`crate::transfer_function::TransferFunction::to_discrete_tustin`] if a
/// discrete-domain rational model is needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pid<T> {
    /// Proportional gain $K_p$.
    pub kp: T,
    /// Integral gain $K_i$.
    pub ki: T,
    /// Derivative gain $K_d$.
    pub kd: T,
    /// Derivative low-pass filter time constant $T_f > 0$.
    pub tf: T,
    /// Minimum saturated control output.
    pub u_min: T,
    /// Maximum saturated control output.
    pub u_max: T,
    /// Integrator leak factor applied while saturated, $\gamma \in [0, 1]$.
    /// $\gamma = 1$ freezes the integrator during saturation; $\gamma = 0$
    /// resets it immediately.
    pub gamma: T,
    /// Accumulated integral term.
    integrator: T,
    /// Measurement from the previous [`Pid::step`] call.
    prev_measurement: T,
    /// Filtered derivative term from the previous [`Pid::step`] call.
    filtered_derivative: T,
    /// `true` once [`Pid::step`] has been called at least once, so the first
    /// call does not derivative-kick from an arbitrary zeroed measurement.
    has_run: bool,
}

impl fmt::Display for PidError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSamplePeriod => {
                write!(f, "PID sample period must be strictly positive")
            }
        }
    }
}

impl<T: Float + Copy> Pid<T> {
    /// Builds a PID controller with zeroed internal state.
    ///
    /// `tf` should be strictly positive; a non-positive filter time constant
    /// leaves the derivative term unfiltered (undamped) and, via
    /// [`Pid::to_transfer_function`], produces a non-proper rational
    /// transfer function representation.
    #[must_use]
    pub const fn new(
        kp: T,
        ki: T,
        kd: T,
        tf: T,
        u_min: T,
        u_max: T,
        gamma: T,
    ) -> Self {
        Self {
            kp,
            ki,
            kd,
            tf,
            u_min,
            u_max,
            gamma,
            integrator: T::ZERO,
            prev_measurement: T::ZERO,
            filtered_derivative: T::ZERO,
            has_run: false,
        }
    }

    /// Resets all internal state (integrator, filtered derivative, and the
    /// derivative-kick guard) to zero.
    pub const fn reset(&mut self) {
        self.integrator = T::ZERO;
        self.prev_measurement = T::ZERO;
        self.filtered_derivative = T::ZERO;
        self.has_run = false;
    }

    /// Advances the controller by one step of duration `dt` and returns the
    /// saturated control output $u \in [u_{min}, u_{max}]$.
    ///
    /// The derivative acts on `measurement` rather than the error, so a
    /// setpoint step does not spike the derivative term. The integrator uses
    /// conditional (clamping) anti-windup: while the unsaturated candidate
    /// output would exceed the output limits, the integrator leaks toward
    /// zero at rate `gamma` instead of continuing to accumulate.
    ///
    /// # Errors
    /// [`PidError::InvalidSamplePeriod`] if `dt` is not strictly positive or
    /// $T_f + dt$ is not strictly positive.
    pub fn step(
        &mut self,
        setpoint: T,
        measurement: T,
        dt: T,
    ) -> Result<T, PidError> {
        if dt <= T::ZERO || self.tf + dt <= T::ZERO {
            return Err(PidError::InvalidSamplePeriod);
        }
        let error = setpoint - measurement;
        let delta_measurement = if self.has_run {
            measurement - self.prev_measurement
        } else {
            T::ZERO
        };

        let denom = self.tf + dt;
        self.filtered_derivative = (self.tf / denom) * self.filtered_derivative
            - (self.kd / denom) * delta_measurement;

        let candidate_integrator = self.integrator + self.ki * error * dt;
        let unclamped =
            self.kp * error + candidate_integrator + self.filtered_derivative;
        let output = unclamped.clamp(self.u_min, self.u_max);

        self.integrator = if output == unclamped {
            candidate_integrator
        } else {
            self.gamma * self.integrator
        };

        self.prev_measurement = measurement;
        self.has_run = true;
        Ok(output)
    }

    /// Continuous-time parallel-form transfer function
    /// $C(s) = K_p + \dfrac{K_i}{s} + \dfrac{K_d s}{1 + s T_f}$, combined
    /// over the common denominator $s(1 + s T_f)$:
    ///
    /// $C(s) = \dfrac{K_i + (K_p + K_i T_f) s + (K_p T_f + K_d) s^2}
    /// {s + T_f s^2}$.
    #[must_use]
    pub fn to_transfer_function(&self) -> ArrayTransferFunction<T, 3, 3> {
        let num = [
            self.ki,
            self.kp + self.ki * self.tf,
            self.kp * self.tf + self.kd,
        ];
        let den = [T::ZERO, T::ONE, self.tf];
        ArrayTransferFunction::continuous(num, den)
    }
}
