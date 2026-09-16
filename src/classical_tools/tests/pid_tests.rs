//! # Discrete PID Controller Unit and Verification Tests
#![allow(clippy::arithmetic_side_effects, clippy::suboptimal_flops)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod pid_test_suite {
    use crate::classical_tools::pid::Pid;
    use crate::math::complex_num::Complex;

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_proportional_only_matches_exact_formula() {
        // kp = 2, ki = kd = 0, wide-open limits: u = Kp * e exactly, no
        // internal state involved.
        let mut pid = Pid::<f64>::new(2.0, 0.0, 0.0, 1.0, -1.0e9, 1.0e9, 0.0);
        let output = pid.step(5.0, 1.0, 0.1);
        assert!((output - 8.0).abs() < 1e-12, "output = {output}");
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_integral_only_matches_closed_form_running_sum() {
        // kp = kd = 0, wide-open limits: u[k] = Ki * dt * sum_{j<=k} e[j], an
        // independent closed-form oracle that does not share code with the
        // controller's own recursive accumulation.
        let ki = 0.5_f64;
        let dt = 0.1_f64;
        let mut pid = Pid::<f64>::new(0.0, ki, 0.0, 1.0, -1.0e9, 1.0e9, 0.0);
        let measurements = [1.0, 2.0, -1.0, 0.5, 3.0];
        let setpoint = 0.0;

        let mut running_sum = 0.0_f64;
        for &measurement in &measurements {
            let output = pid.step(setpoint, measurement, dt);
            let error = setpoint - measurement;
            running_sum += ki * error * dt;
            assert!(
                (output - running_sum).abs() < 1e-9,
                "output = {output}, expected {running_sum}"
            );
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_derivative_only_settles_to_steady_state_ramp_slope() {
        // kp = ki = 0, so u[k] is exactly the filtered derivative term. For a
        // measurement ramping at a constant slope, the continuous-time
        // steady-state solution of `Tf*d' + d = -Kd*y'` is the ODE fixed
        // point `d_ss = -Kd * slope` (independent of the discrete recursion
        // used by `step`): with `d' = 0`, `d = -Kd * y' = -Kd * slope`.
        let kd = 2.0_f64;
        let tf = 0.05_f64;
        let dt = 0.001_f64;
        let slope = 3.0_f64;
        let mut pid = Pid::<f64>::new(0.0, 0.0, kd, tf, -1.0e9, 1.0e9, 0.0);

        let mut measurement = 0.0_f64;
        let mut output = 0.0_f64;
        for _ in 0..5000 {
            measurement += slope * dt;
            output = pid.step(0.0, measurement, dt);
        }

        let expected = -kd * slope;
        assert!(
            (output - expected).abs() < 1e-3,
            "output = {output}, expected ~{expected}"
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_saturation_anti_windup_recovers_with_zero_lag() {
        // kp = 1, ki = 1, kd = 0, gamma = 0 (immediate integrator reset while
        // saturated). A huge constant error drives the unclamped output far
        // past `u_max`, so the controller must hold exactly at `u_max` every
        // step and carry no accumulated integrator once the error vanishes,
        // per the design doc's "recovers with zero lag" anti-windup
        // requirement: the very next zero-error step must return exactly
        // zero, not decay from a wound-up integrator.
        let mut pid = Pid::<f64>::new(1.0, 1.0, 0.0, 1.0, -1.0, 1.0, 0.0);
        for _ in 0..20 {
            let output = pid.step(100.0, 0.0, 0.1);
            assert!((output - 1.0).abs() < 1e-12, "output = {output}");
        }
        let recovered = pid.step(0.0, 0.0, 0.1);
        assert!(recovered.abs() < 1e-12, "recovered output = {recovered}");
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_reset_clears_internal_state() {
        let mut pid = Pid::<f64>::new(0.0, 1.0, 0.0, 1.0, -1.0e9, 1.0e9, 0.0);
        let _ = pid.step(1.0, 0.0, 0.1);
        let _ = pid.step(1.0, 0.0, 0.1);
        pid.reset();
        // Immediately after a reset the integrator is zero, so a fresh
        // integral-only step reproduces the very first step's output.
        let after_reset = pid.step(1.0, 0.0, 0.1);
        assert!((after_reset - 0.1).abs() < 1e-12, "output = {after_reset}");
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-7, classical-tools#FR-8
    /// Method: Requirements-based test
    fn test_transfer_function_matches_independent_frequency_formula() {
        // Cross-checks `Pid::to_transfer_function` against the parallel-form
        // definition C(s) = Kp + Ki/s + Kd*s/(1+s*Tf) evaluated directly at
        // s = j*omega via `Complex` arithmetic, independent of the
        // ascending-coefficient combination used by `to_transfer_function`.
        let kp = 1.5_f64;
        let ki = 0.7_f64;
        let kd = 0.3_f64;
        let tf = 0.05_f64;
        let omega = 2.0_f64;

        let pid = Pid::<f64>::new(kp, ki, kd, tf, -1.0, 1.0, 0.0);
        let actual = pid.to_transfer_function().eval_frequency(omega);

        let proportional = Complex::new(kp, 0.0);
        let integral = Complex::new(0.0, -ki / omega);
        let derivative =
            Complex::new(0.0, kd * omega) / Complex::new(1.0, omega * tf);
        let expected = proportional + integral + derivative;

        assert!(
            (actual - expected).magnitude() < 1e-9,
            "actual = {actual:?}, expected = {expected:?}"
        );
    }
}
