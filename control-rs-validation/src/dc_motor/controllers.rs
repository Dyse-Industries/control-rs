//! src/controllers.rs
//!
//! Real-time discrete firmware execution topologies:
//! - **Controller A: Analytical Lead Compensator** realized via [`DirectForm2T<f64, 1>`].
//! - **Controller B: Discrete PID with Filtered Derivative** via [`control_rs::classical_tools::pid::Pid`].

use control_rs::classical_tools::pid::Pid;
#[cfg(test)]
use control_rs::classical_tools::realization::Biquad;
use control_rs::classical_tools::realization::DirectForm2T;
use control_rs::transfer_function::ArrayTransferFunction;

/// Unified interface for discrete motor position controllers.
pub trait MotorController {
    /// Resets all internal filter states and accumulators.
    fn reset(&mut self);

    /// Advances the controller by one sample period $T_s$, returning the command voltage $u[k] \in [-V_{\max}, +V_{\max}]$.
    fn update(
        &mut self,
        setpoint_rad: f64,
        measurement_rad: f64,
        dt_s: f64,
    ) -> f64;

    /// Human-readable name of the controller strategy.
    fn name(&self) -> &'static str;
}

/// Converts a 1st-order discrete rational transfer function $C(z) = \frac{n_0 + n_1 z}{d_0 + d_1 z}$
/// into a minimal-delay [`DirectForm2T<f64, 1>`].
#[must_use]
pub fn lead_to_df2t(
    c_discrete: &ArrayTransferFunction<f64, 2, 2>,
) -> DirectForm2T<f64, 1> {
    let num = c_discrete.num_slice();
    let den = c_discrete.den_slice();

    let d1 = den[1];
    let b0 = num[1] / d1;
    let b1 = num[0] / d1;
    let a1 = den[0] / d1;

    DirectForm2T::new(b0, [b1], [a1])
}

/// Discrete PID proportional, integral, derivative, and filter parameters.
#[derive(Clone, Copy, Debug)]
pub struct PidGains {
    /// Proportional gain $K_p$.
    pub kp: f64,
    /// Integral gain $K_i$.
    pub ki: f64,
    /// Derivative gain $K_d$.
    pub kd: f64,
    /// Derivative filter time constant $T_f$ (seconds).
    pub tf: f64,
}

/// Converts a 2nd-order discrete rational transfer function $C(z) = \frac{n_0 + n_1 z + n_2 z^2}{d_0 + d_1 z + d_2 z^2}$
/// into a [`DirectForm2T<f64, 2>`].
#[cfg(test)]
#[must_use]
fn lead_lag_to_df2t(
    c_discrete: &ArrayTransferFunction<f64, 3, 3>,
) -> DirectForm2T<f64, 2> {
    let num = c_discrete.num_slice();
    let den = c_discrete.den_slice();

    let d2 = den[2];
    let b0 = num[2] / d2;
    let b1 = num[1] / d2;
    let b2 = num[0] / d2;
    let a1 = den[1] / d2;
    let a2 = den[0] / d2;

    DirectForm2T::new(b0, [b1, b2], [a1, a2])
}

/// Converts a 2nd-order discrete rational transfer function into a single [`Biquad<f64>`] section.
#[cfg(test)]
#[must_use]
fn lead_lag_to_biquad(
    c_discrete: &ArrayTransferFunction<f64, 3, 3>,
) -> Biquad<f64> {
    let num = c_discrete.num_slice();
    let den = c_discrete.den_slice();

    let d2 = den[2];
    let b0 = num[2] / d2;
    let b1 = num[1] / d2;
    let b2 = num[0] / d2;
    let a1 = den[1] / d2;
    let a2 = den[0] / d2;

    Biquad::new(b0, b1, b2, a1, a2)
}

/// Controller A: Discrete Lead Compensator realized in Transposed Direct Form II.
#[derive(Debug, Clone)]
pub struct LeadDf2tController {
    df2t: DirectForm2T<f64, 1>,
    v_max: f64,
}

impl LeadDf2tController {
    /// Creates a new Lead controller from a discretized 1st-order transfer function.
    #[must_use]
    pub fn new(
        c_discrete: &ArrayTransferFunction<f64, 2, 2>,
        v_max: f64,
    ) -> Self {
        Self {
            df2t: lead_to_df2t(c_discrete),
            v_max,
        }
    }
}

impl MotorController for LeadDf2tController {
    fn reset(&mut self) {
        self.df2t.reset();
    }

    fn update(
        &mut self,
        setpoint_rad: f64,
        measurement_rad: f64,
        _dt_s: f64,
    ) -> f64 {
        let err = setpoint_rad - measurement_rad;
        let u = self.df2t.update(err);
        u.clamp(-self.v_max, self.v_max)
    }

    fn name(&self) -> &'static str {
        "Lead (DirectForm2T)"
    }
}

/// Controller B: Discrete PID controller with derivative filter and integrator anti-windup.
#[derive(Debug, Clone)]
pub struct PidMotorController {
    pid: Pid<f64>,
}

impl PidMotorController {
    /// Creates a new PID controller for the motor position servo.
    #[must_use]
    pub fn new(gains: PidGains, v_max: f64) -> Self {
        Self {
            pid: Pid::new(
                gains.kp, gains.ki, gains.kd, gains.tf, -v_max, v_max, 1.0,
            ),
        }
    }
}

impl MotorController for PidMotorController {
    fn reset(&mut self) {
        self.pid.reset();
    }

    fn update(
        &mut self,
        setpoint_rad: f64,
        measurement_rad: f64,
        dt_s: f64,
    ) -> f64 {
        self.pid.step(setpoint_rad, measurement_rad, dt_s)
    }

    fn name(&self) -> &'static str {
        "PID (Filtered Derivative)"
    }
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dc_motor::motor::V_MAX;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_lead_to_df2t_conversion_equivalence() {
        // (1 + 2z) / (3 + 4z) -> b0 = 2/4 = 0.5, b1 = 1/4 = 0.25, a1 = 3/4 = 0.75
        let c_disc = ArrayTransferFunction::<f64, 2, 2>::continuous(
            [1.0, 2.0],
            [3.0, 4.0],
        );
        let mut df2t = lead_to_df2t(&c_disc);

        assert_eq!(df2t.b0, 0.5);
        assert_eq!(df2t.b[0], 0.25);
        assert_eq!(df2t.a[0], 0.75);

        // Step 0: x = 1.0 -> y = 0.5 * 1.0 + 0 = 0.5
        let y0 = df2t.update(1.0);
        assert_eq!(y0, 0.5);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_lead_lag_to_df2t_and_biquad_equivalence() {
        // (1 + 2z + 4z^2) / (2 + 3z + 5z^2)
        let c_disc = ArrayTransferFunction::<f64, 3, 3>::continuous(
            [1.0, 2.0, 4.0],
            [2.0, 3.0, 5.0],
        );
        let mut df2t = lead_lag_to_df2t(&c_disc);
        let mut biquad = lead_lag_to_biquad(&c_disc);

        assert_eq!(df2t.b0, 0.8);
        assert_eq!(df2t.b[0], 0.4);
        assert_eq!(df2t.b[1], 0.2);
        assert_eq!(df2t.a[0], 0.6);
        assert_eq!(df2t.a[1], 0.4);

        // Run sequential inputs through both and assert exact equivalence
        for u in [1.0, 0.5, -0.2, 0.0, 0.8] {
            let y_df2t = df2t.update(u);
            let y_bq = biquad.update(u);
            assert!((y_df2t - y_bq).abs() < 1e-12);
        }
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2
    /// Method: Requirements-based test
    fn test_pid_controller_step_and_clamping() {
        let mut ctrl = PidMotorController::new(
            PidGains {
                kp: 10.0,
                ki: 2.0,
                kd: 0.1,
                tf: 0.01,
            },
            V_MAX,
        );

        // Large error -> should saturate at +V_MAX (12.0 V)
        let u1 = ctrl.update(5.0, 0.0, 0.001);
        assert_eq!(u1, V_MAX);

        // Large negative error -> should saturate at -V_MAX (-12.0 V)
        let u2 = ctrl.update(-5.0, 0.0, 0.001);
        assert_eq!(u2, -V_MAX);
    }
}
