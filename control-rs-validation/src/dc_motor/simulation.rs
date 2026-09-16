//! src/simulation.rs
//!
//! Continuous-discrete closed-loop simulation engine for the DC motor position servo:
//! - Multi-rate numerical solver: RK4 continuous integration for plant electromechanics
//!   coupled with discrete processor updates at sample period $T_s = 500\,\mu\text{s}$.
//! - Realistic hardware peripherals: actuator delay & noise, 16-bit absolute encoder
//!   quantization, sensor delay & noise.
//! - Comparative benchmarking: Controller A (Lead DirectForm2T) vs. Controller B (PID) vs. Ideal Baseline.

use crate::dc_motor::controllers::MotorController;
use crate::dc_motor::motor::{V_MAX, rk4_step};
use crate::dc_motor::peripherals::{Actuator, EncoderSensor};

/// Performance metrics extracted from closed-loop position step response.
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct StepMetrics {
    /// 10% to 90% rise time (seconds).
    pub rise_time_s: f64,
    /// Peak overshoot percentage $M_p$ (%).
    pub peak_overshoot_pct: f64,
    /// Peak shaft angle $\theta_{\text{peak}}$ (radians).
    pub peak_angle_rad: f64,
    /// 2% settling time (seconds).
    pub settling_time_s: f64,
    /// Steady-state position error before disturbance (radians).
    pub steady_state_error_rad: f64,
    /// High-frequency steady-state tracking jitter (RMS standard deviation in radians).
    pub tracking_jitter_rms_rad: f64,
    /// Maximum position dip caused by load torque disturbance (radians).
    pub disturbance_dip_rad: f64,
    /// Restored steady-state angle after load torque disturbance (radians).
    pub restored_angle_rad: f64,
}

/// Sampled simulation trajectory time-series.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MotorTrajectory {
    /// Time timestamps (seconds).
    pub time: Vec<f64>,
    /// Actual physical shaft angle $\theta(t)$ (radians).
    pub theta_true: Vec<f64>,
    /// Quantized and noisy sensor angle measurement $\theta_{\text{meas}}[k]$ (radians).
    pub theta_meas: Vec<f64>,
    /// Shaft angular velocity $\omega(t)$ (rad/s).
    pub omega: Vec<f64>,
    /// Armature current $i_a(t)$ (Amperes).
    pub i_a: Vec<f64>,
    /// Controller raw command voltage $u[k]$ (Volts).
    pub u_command: Vec<f64>,
    /// Actual noisy & delayed terminal voltage $v_a(t)$ applied to motor (Volts).
    pub v_applied: Vec<f64>,
    /// Target setpoint trajectory $\theta_{\text{ref}}(t)$ (radians).
    pub theta_ref: Vec<f64>,
    /// Applied disturbance torque $\tau_L(t)$ ($\text{N}\cdot\text{m}$).
    pub tau_load: Vec<f64>,
    /// Extracted step response performance metrics.
    pub metrics: StepMetrics,
}

/// Integration horizon and sample period for a closed-loop run.
#[derive(Clone, Copy, Debug)]
pub struct SampledHorizon {
    /// Total simulated time (s).
    pub total_time_s: f64,
    /// Controller sample period $T_s$ (s).
    pub sample_time_s: f64,
}

/// Position reference step applied at `time_s`.
#[derive(Clone, Copy, Debug)]
pub struct PositionStep {
    /// Step instant (s).
    pub time_s: f64,
    /// Step amplitude (rad).
    pub angle_rad: f64,
}

/// Load-torque disturbance applied at `time_s`.
#[derive(Clone, Copy, Debug)]
pub struct LoadDisturbance {
    /// Disturbance instant (s).
    pub time_s: f64,
    /// Torque amplitude (N·m).
    pub torque_nm: f64,
}

/// Closed-loop servo experiment: horizon, position step, and load torque.
#[derive(Clone, Copy, Debug)]
pub struct ServoScenario {
    /// Integration horizon and sample period.
    pub horizon: SampledHorizon,
    /// Position reference step.
    pub step: PositionStep,
    /// Load-torque disturbance.
    pub disturbance: LoadDisturbance,
}

/// Sampled traces used to extract step-response metrics.
pub struct StepMetricRef<'a> {
    /// Sample timestamps (s).
    pub time: &'a [f64],
    /// True shaft angle (rad).
    pub theta_true: &'a [f64],
    /// Position reference step.
    pub step: PositionStep,
    /// Load-torque disturbance instant (s).
    pub dist_time_s: f64,
}

/// Computes step response performance metrics from the simulated position trajectory.
#[must_use]
pub fn compute_step_metrics(signals: StepMetricRef<'_>) -> StepMetrics {
    let time = signals.time;
    let theta_true = signals.theta_true;
    let step_time_s = signals.step.time_s;
    let step_val_rad = signals.step.angle_rad;
    let dist_time_s = signals.dist_time_s;
    let n = time.len();
    if n == 0 || step_val_rad.abs() < 1e-9 {
        return StepMetrics {
            rise_time_s: 0.0,
            peak_overshoot_pct: 0.0,
            peak_angle_rad: 0.0,
            settling_time_s: 0.0,
            steady_state_error_rad: 0.0,
            tracking_jitter_rms_rad: 0.0,
            disturbance_dip_rad: 0.0,
            restored_angle_rad: 0.0,
        };
    }

    let step_idx = time.iter().position(|&t| t >= step_time_s).unwrap_or(0);
    let dist_idx = time.iter().position(|&t| t >= dist_time_s).unwrap_or(n - 1);

    // Analyze pre-disturbance step response (between step_idx and dist_idx)
    let pre_dist_theta = &theta_true[step_idx..dist_idx];
    let pre_dist_time = &time[step_idx..dist_idx];

    let val_10 = 0.1 * step_val_rad;
    let val_90 = 0.9 * step_val_rad;

    let t_10 = pre_dist_time
        .iter()
        .zip(pre_dist_theta.iter())
        .find(|(_, th)| **th >= val_10)
        .map_or(step_time_s, |(&t, _)| t);

    let t_90 = pre_dist_time
        .iter()
        .zip(pre_dist_theta.iter())
        .find(|(_, th)| **th >= val_90)
        .map_or(
            pre_dist_time.last().copied().unwrap_or(step_time_s),
            |(&t, _)| t,
        );

    let rise_time_s = (t_90 - t_10).max(0.0);

    let mut peak_val = 0.0_f64;
    for &th in pre_dist_theta {
        if th > peak_val {
            peak_val = th;
        }
    }
    let overshoot_rad = (peak_val - step_val_rad).max(0.0);
    let peak_overshoot_pct = (overshoot_rad / step_val_rad) * 100.0;

    // 2% Settling time: within [0.98, 1.02] * step_val_rad
    let band_lo = 0.98 * step_val_rad;
    let band_hi = 1.02 * step_val_rad;

    let mut settling_time_s = 0.0;
    for (i, (&_t, &th)) in pre_dist_time
        .iter()
        .zip(pre_dist_theta.iter())
        .enumerate()
        .rev()
    {
        if th < band_lo || th > band_hi {
            let next_idx = (i + 1).min(pre_dist_time.len() - 1);
            settling_time_s = pre_dist_time[next_idx] - step_time_s;
            break;
        }
    }

    // Steady state before disturbance (average of last 50 points before disturbance)
    let avg_window = 50.min(pre_dist_theta.len());
    let ss_slice = &pre_dist_theta[pre_dist_theta.len() - avg_window..];
    let mean_ss = ss_slice.iter().sum::<f64>() / (avg_window as f64);
    let steady_state_error_rad = (step_val_rad - mean_ss).abs();

    // High frequency jitter RMS
    let var_ss = ss_slice
        .iter()
        .map(|&th| (th - mean_ss).powi(2))
        .sum::<f64>()
        / (avg_window as f64);
    let tracking_jitter_rms_rad = var_ss.sqrt();

    // Disturbance response (after dist_idx)
    let post_dist_theta = &theta_true[dist_idx..];
    let mut min_during_dist = mean_ss;
    for &th in post_dist_theta {
        if th < min_during_dist {
            min_during_dist = th;
        }
    }
    let disturbance_dip_rad = (mean_ss - min_during_dist).max(0.0);

    let post_avg_window = 50.min(post_dist_theta.len());
    let post_slice =
        &post_dist_theta[post_dist_theta.len() - post_avg_window..];
    let restored_angle_rad =
        post_slice.iter().sum::<f64>() / (post_avg_window as f64);

    StepMetrics {
        rise_time_s,
        peak_overshoot_pct,
        peak_angle_rad: peak_val,
        settling_time_s,
        steady_state_error_rad,
        tracking_jitter_rms_rad,
        disturbance_dip_rad,
        restored_angle_rad,
    }
}

/// Simulates the closed-loop motor servo system under realistic delayed and noisy peripherals.
#[must_use]
pub fn simulate_motor_system<C: MotorController>(
    controller: &mut C,
    actuator: &mut Actuator,
    sensor: &mut EncoderSensor,
    scenario: ServoScenario,
) -> MotorTrajectory {
    controller.reset();
    actuator.reset();
    sensor.reset();

    let ts_s = scenario.horizon.sample_time_s;
    let step_time_s = scenario.step.time_s;
    let step_val_rad = scenario.step.angle_rad;
    let dist_time_s = scenario.disturbance.time_s;
    let dist_torque_nm = scenario.disturbance.torque_nm;
    let num_control_steps =
        (scenario.horizon.total_time_s / ts_s).round() as usize;
    let sub_steps = 10;
    let dt_plant = ts_s / (sub_steps as f64);

    let mut time = Vec::with_capacity(num_control_steps);
    let mut theta_true = Vec::with_capacity(num_control_steps);
    let mut theta_meas = Vec::with_capacity(num_control_steps);
    let mut omega = Vec::with_capacity(num_control_steps);
    let mut i_a = Vec::with_capacity(num_control_steps);
    let mut u_command = Vec::with_capacity(num_control_steps);
    let mut v_applied = Vec::with_capacity(num_control_steps);
    let mut theta_ref = Vec::with_capacity(num_control_steps);
    let mut tau_load = Vec::with_capacity(num_control_steps);

    let mut x = [0.0, 0.0, 0.0]; // [i_a, omega, theta]

    for k in 0..num_control_steps {
        let t = (k as f64) * ts_s;
        let r = if t >= step_time_s { step_val_rad } else { 0.0 };
        let tau_l = if t >= dist_time_s {
            dist_torque_nm
        } else {
            0.0
        };

        // 1. Sensor reads physical shaft angle x[2], applies 16-bit quantization, noise, and sensor delay
        let meas = sensor.step(x[2]);

        // 2. Controller computes desired terminal voltage u[k]
        let u_cmd = controller.update(r, meas, ts_s);

        // 3. Actuator applies actuator delay, voltage noise, and clamping
        let v_act = actuator.step(u_cmd);

        // Record sampled variables at start of control step
        time.push(t);
        theta_true.push(x[2]);
        theta_meas.push(meas);
        omega.push(x[1]);
        i_a.push(x[0]);
        u_command.push(u_cmd);
        v_applied.push(v_act);
        theta_ref.push(r);
        tau_load.push(tau_l);

        // 4. Integrate physical plant over sub-steps using RK4
        for _ in 0..sub_steps {
            x = rk4_step(x, v_act, tau_l, dt_plant);
        }
    }

    let metrics = compute_step_metrics(StepMetricRef {
        time: &time,
        theta_true: &theta_true,
        step: scenario.step,
        dist_time_s,
    });

    MotorTrajectory {
        time,
        theta_true,
        theta_meas,
        omega,
        i_a,
        u_command,
        v_applied,
        theta_ref,
        tau_load,
        metrics,
    }
}

/// Simulates ideal baseline (zero delay, zero noise, perfect infinite-resolution sensor).
#[must_use]
pub fn simulate_ideal_baseline<C: MotorController>(
    controller: &mut C,
    scenario: ServoScenario,
) -> MotorTrajectory {
    let mut ideal_actuator = Actuator::new(0, 0.0, V_MAX, 1);
    let mut ideal_sensor = EncoderSensor::new(0, 0.0, 1);

    simulate_motor_system(
        controller,
        &mut ideal_actuator,
        &mut ideal_sensor,
        scenario,
    )
}

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::dc_motor::analysis::synthesize_delay_compensated_lead;
    use crate::dc_motor::controllers::{
        LeadDf2tController, PidGains, PidMotorController,
    };
    use crate::dc_motor::motor::plant_position;

    fn tracking_scenario(ts_s: f64) -> ServoScenario {
        ServoScenario {
            horizon: SampledHorizon {
                total_time_s: 1.2,
                sample_time_s: ts_s,
            },
            step: PositionStep {
                time_s: 0.05,
                angle_rad: 1.0,
            },
            disturbance: LoadDisturbance {
                time_s: 0.45,
                torque_nm: 0.02,
            },
        }
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_lead_df2t_closed_loop_tracking() {
        let plant = plant_position();
        let target_wc = 50.0;
        let target_pm = 50.0;
        let delay_s = 0.001; // 1 ms total loop delay (2 steps at Ts=500us)
        let ts_s = 0.0005; // 500 us

        let lead_design = synthesize_delay_compensated_lead(
            &plant, target_wc, target_pm, delay_s,
        )
        .expect("Compensator synthesis failed");
        let c_disc = lead_design
            .compensator_tf
            .to_discrete_tustin(ts_s, Some(target_wc));

        let mut controller = LeadDf2tController::new(&c_disc, V_MAX);
        let mut actuator = Actuator::new(1, 0.02, V_MAX, 101);
        let mut sensor = EncoderSensor::new(1, 5e-5, 202);

        let traj = simulate_motor_system(
            &mut controller,
            &mut actuator,
            &mut sensor,
            tracking_scenario(ts_s),
        );

        // Assert step tracking success
        assert!(
            traj.metrics.rise_time_s > 0.0 && traj.metrics.rise_time_s < 0.1
        );
        assert!(
            traj.metrics.steady_state_error_rad < 0.05,
            "Lead ss error should be < 0.05 rad: {}",
            traj.metrics.steady_state_error_rad
        );
        assert!(
            traj.metrics.peak_angle_rad > 0.9
                && traj.metrics.peak_angle_rad < 1.6
        );
        assert!(traj.metrics.disturbance_dip_rad > 0.0);
        // Lead suffers steady-state droop under constant torque load
        assert!(
            traj.metrics.restored_angle_rad < 0.95,
            "Lead restored angle should reflect droop (< 0.95 rad): {}",
            traj.metrics.restored_angle_rad
        );
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_pid_closed_loop_tracking() {
        let ts_s = 0.0005;
        // Damped PID tuning: Kp=10.0, Ki=30.0, Kd=0.3, Tf=0.002
        let mut controller = PidMotorController::new(
            PidGains {
                kp: 10.0,
                ki: 30.0,
                kd: 0.3,
                tf: 0.002,
            },
            V_MAX,
        );
        let mut actuator = Actuator::new(1, 0.02, V_MAX, 303);
        let mut sensor = EncoderSensor::new(1, 5e-5, 404);

        let traj = simulate_motor_system(
            &mut controller,
            &mut actuator,
            &mut sensor,
            tracking_scenario(ts_s),
        );

        // PID should have small pre-disturbance steady state error (< 0.05 rad)
        assert!(
            traj.metrics.rise_time_s > 0.0 && traj.metrics.rise_time_s < 0.15
        );
        assert!(
            traj.metrics.steady_state_error_rad < 0.05,
            "PID ss error should be < 0.05 rad: {}",
            traj.metrics.steady_state_error_rad
        );
        // PID integral action rejects constant torque disturbance: theta -> 1.0 rad
        assert!(
            (traj.metrics.restored_angle_rad - 1.0).abs() < 0.05,
            "PID restored angle should return to ~1.0 rad: {}",
            traj.metrics.restored_angle_rad
        );
    }
}
