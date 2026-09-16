//! src/simulation.rs
//!
//! Closed-loop simulation and verification module for the buck converter example:
//! - Discretization of the lead compensator via Tustin bilinear transform.
//! - Firmware execution topologies: [`DirectForm2T`] and [`Biquad`].
//! - Linear sampled-data state-space simulation (exact ZOH plant).
//! - Continuous-conduction mode (CCM) averaged circuit nonlinear simulation (RK4).
//! - Transient performance metrics extraction ($t_r$, $M_p$, $t_s$, $e_{ss}$).

use crate::buck_converter::analysis::LeadTf;
use crate::buck_converter::circuit::{
    AveragedCircuit, AveragedDrive, LOAD_RESISTANCE, V_IN, V_OUT,
    nonlinear_dynamics, operating_duty_cycle, state_space_plant,
};
use control_rs::classical_tools::realization::{Biquad, DirectForm2T};
use control_rs::matrix::Owned;

/// Transient performance metrics extracted from a step response simulation.
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct TransientMetrics {
    /// 10% to 90% rise time in seconds.
    pub rise_time_s: f64,
    /// 10% to 90% rise time in microseconds.
    pub rise_time_us: f64,
    /// Peak overshoot percentage $M_p$ (%) relative to target step.
    pub peak_overshoot_pct: f64,
    /// 2% settling time to target in seconds (`None` if Type-0 offset > 2% band).
    pub settling_time_target_s: Option<f64>,
    /// 2% settling time to final achieved value in seconds.
    pub settling_time_final_s: f64,
    /// 2% settling time to final achieved value in microseconds.
    pub settling_time_us: f64,
    /// Steady-state voltage error $|V_{\mathrm{final}} - V_{\mathrm{target}}|$ in Volts.
    pub steady_state_error_v: f64,
    /// Peak output voltage in Volts.
    pub peak_voltage_v: f64,
}

/// Load transient recovery metrics.
#[derive(
    Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize,
)]
pub struct LoadTransientMetrics {
    /// Nominal voltage before load step (V).
    pub nominal_v: f64,
    /// Peak voltage spike during load step (V).
    pub peak_v: f64,
    /// Time in seconds from load step until voltage returns within 1% of nominal.
    pub recovery_time_s: f64,
    /// Recovery time in microseconds.
    pub recovery_time_us: f64,
    /// Final restored voltage (V).
    pub restored_v: f64,
}

/// Simulation trajectory recording sampled time-series data.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SimulationTrajectory {
    /// Sample timestamps (seconds).
    pub time_s: Vec<f64>,
    /// Output voltage trajectory $v_o(t)$ (Volts).
    pub v_out_v: Vec<f64>,
    /// Inductor current trajectory $i_L(t)$ (Amperes).
    pub i_l_a: Vec<f64>,
    /// Duty cycle command trajectory $d(t) \in [0, 1]$.
    pub duty_cycle: Vec<f64>,
    /// Extracted transient performance metrics.
    pub metrics: TransientMetrics,
}

/// Load step response trajectory and recovery metrics.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoadTrajectory {
    /// Sample timestamps (seconds).
    pub time_s: Vec<f64>,
    /// Output voltage trajectory $v_o(t)$ (Volts).
    pub v_out_v: Vec<f64>,
    /// Inductor current trajectory $i_L(t)$ (Amperes).
    pub i_l_a: Vec<f64>,
    /// Duty cycle command trajectory $d(t) \in [0, 1]$.
    pub duty_cycle: Vec<f64>,
    /// Nominal voltage before load step (V).
    pub nominal_v: f64,
    /// Peak voltage spike during load step (V).
    pub peak_v: f64,
    /// Recovery time (seconds) to return within 1% of nominal.
    pub recovery_time_s: f64,
    /// Recovery time (microseconds).
    pub recovery_time_us: f64,
    /// Final restored voltage (V).
    pub restored_v: f64,
    /// Recovery metrics subobject.
    pub metrics: LoadTransientMetrics,
}

/// Integration horizon and sample period for a closed-loop run.
#[derive(Clone, Copy, Debug)]
pub struct SampledHorizon {
    /// Total simulated time (s).
    pub total_time_s: f64,
    /// Controller sample period $T_s$ (s).
    pub sample_time_s: f64,
}

/// Output-voltage step applied at `time_s`.
#[derive(Clone, Copy, Debug)]
pub struct VoltageStep {
    /// Step instant (s).
    pub time_s: f64,
    /// Step amplitude (V).
    pub delta_v: f64,
}

/// Load-resistance step applied at `time_s`.
#[derive(Clone, Copy, Debug)]
pub struct LoadStep {
    /// Step instant (s).
    pub time_s: f64,
    /// New load resistance ($\Omega$).
    pub resistance_ohm: f64,
}

/// Discretizes a continuous rational transfer function $C(s)$ using Tustin bilinear transform.
#[must_use]
pub fn discretize_compensator(
    c: &LeadTf,
    sample_time_s: f64,
    prewarp_wc: Option<f64>,
) -> LeadTf {
    c.to_discrete_tustin(sample_time_s, prewarp_wc)
}

/// Converts a discrete 1st-order rational transfer function $C(z) = \frac{n_0 + n_1 z}{d_0 + d_1 z}$
/// into a minimal-delay [`DirectForm2T<f64, 1>`] structure:
///
/// $$H(z) = \frac{b_0 + b_1 z^{-1}}{1 + a_1 z^{-1}} = \frac{(n_1/d_1) + (n_0/d_1) z^{-1}}{1 + (d_0/d_1) z^{-1}}$$
#[must_use]
pub fn controller_to_df2t(c_discrete: &LeadTf) -> DirectForm2T<f64, 1> {
    let num = c_discrete.num_slice();
    let den = c_discrete.den_slice();

    let d1 = den[1];
    let b0 = num[1] / d1;
    let b1 = num[0] / d1;
    let a1 = den[0] / d1;

    DirectForm2T::new(b0, [b1], [a1])
}

/// Converts a discrete 1st-order rational transfer function into a single [`Biquad<f64>`] section.
#[must_use]
pub fn controller_to_biquad(c_discrete: &LeadTf) -> Biquad<f64> {
    let num = c_discrete.num_slice();
    let den = c_discrete.den_slice();

    let d1 = den[1];
    let b0 = num[1] / d1;
    let b1 = num[0] / d1;
    let a1 = den[0] / d1;

    Biquad::new(b0, b1, 0.0, a1, 0.0)
}

/// Reference values used to extract step-response metrics.
#[derive(Clone, Copy, Debug)]
pub struct StepMetricRef {
    /// Instant of the commanded step (s).
    pub t_step: f64,
    /// Output before the step (V).
    pub v_initial: f64,
    /// Commanded output after the step (V).
    pub v_target: f64,
}

/// Extracts transient step response metrics from output voltage trajectory using [`control_rs::classical_tools::step_info`].
#[must_use]
pub fn compute_step_metrics(
    time: &[f64],
    v_out: &[f64],
    spec: StepMetricRef,
) -> TransientMetrics {
    let info = control_rs::classical_tools::step_info::step_info(
        time,
        v_out,
        spec.t_step,
        spec.v_initial,
        spec.v_target,
        Some(0.02),
    );
    TransientMetrics {
        rise_time_s: info.rise_time,
        rise_time_us: info.rise_time * 1e6,
        peak_overshoot_pct: info.peak_overshoot_pct,
        settling_time_target_s: info.settling_time,
        settling_time_final_s: info.settling_time_achieved,
        settling_time_us: info.settling_time_achieved * 1e6,
        steady_state_error_v: info.steady_state_error,
        peak_voltage_v: info.peak_value,
    }
}

/// Simulates the linear sampled-data closed-loop system using exact ZOH plant discretization
/// and a discrete [`DirectForm2T<f64, 1>`] compensator.
#[must_use]
pub fn simulate_linear_closed_loop(
    compensator: &LeadTf,
    horizon: SampledHorizon,
    step: VoltageStep,
) -> SimulationTrajectory {
    let total_time_s = horizon.total_time_s;
    let sample_time_s = horizon.sample_time_s;
    let t_step_s = step.time_s;
    let v_step_s = step.delta_v;
    let num_steps = (total_time_s / sample_time_s).round() as usize;
    let ss_c = state_space_plant();
    let ss_d = ss_c.to_discrete_zoh(sample_time_s);

    let c_d = discretize_compensator(compensator, sample_time_s, Some(3.0e4));
    let mut controller = controller_to_df2t(&c_d);

    let d0 = operating_duty_cycle();
    let i0 = V_OUT / LOAD_RESISTANCE;

    let mut x = Owned::<f64, 2, 1>::zero(); // small-signal states [i_hat, v_hat]
    let mut time = Vec::with_capacity(num_steps);
    let mut v_out = Vec::with_capacity(num_steps);
    let mut i_l = Vec::with_capacity(num_steps);
    let mut duty_cycle = Vec::with_capacity(num_steps);

    for k in 0..num_steps {
        let t = (k as f64) * sample_time_s;
        let v_ref = if t >= t_step_s {
            V_OUT + v_step_s
        } else {
            V_OUT
        };

        let v_hat = x.get(1, 0).copied().unwrap_or(0.0);
        let i_hat = x.get(0, 0).copied().unwrap_or(0.0);
        let vo_actual = V_OUT + v_hat;
        let il_actual = i0 + i_hat;

        // Feedback error:
        let error = v_ref - vo_actual;
        let d_hat = controller.update(error);
        let d_command = (d0 + d_hat).clamp(0.0, 1.0);
        let u_effective = d_command - d0;

        time.push(t);
        v_out.push(vo_actual);
        i_l.push(il_actual);
        duty_cycle.push(d_command);

        let u_mat = Owned::<f64, 1, 1>::scalar(u_effective);
        let (x_next, _) = ss_d.step(&x, &u_mat);
        x = x_next;
    }

    let target = if total_time_s >= t_step_s {
        V_OUT + v_step_s
    } else {
        V_OUT
    };
    let metrics = compute_step_metrics(
        &time,
        &v_out,
        StepMetricRef {
            t_step: t_step_s,
            v_initial: V_OUT,
            v_target: target,
        },
    );

    SimulationTrajectory {
        time_s: time,
        v_out_v: v_out,
        i_l_a: i_l,
        duty_cycle,
        metrics,
    }
}

/// Simulates the nonlinear continuous-conduction mode averaged circuit differential equations:
///
/// $$\frac{di_L}{dt} = \frac{d \cdot V_{in} - v_o}{L}$$
/// $$\frac{dv_o}{dt} = \frac{i_L - v_o / R_L}{C}$$
///
/// integrated via 4th-order Runge-Kutta across each PWM sampling interval $T_s$.
#[must_use]
pub fn simulate_nonlinear_circuit(
    compensator: &LeadTf,
    horizon: SampledHorizon,
    step: VoltageStep,
    load_step: Option<LoadStep>,
) -> SimulationTrajectory {
    let total_time_s = horizon.total_time_s;
    let sample_time_s = horizon.sample_time_s;
    let t_step_s = step.time_s;
    let v_step_s = step.delta_v;
    let num_steps = (total_time_s / sample_time_s).round() as usize;
    let c_d = discretize_compensator(compensator, sample_time_s, Some(3.0e4));
    let mut controller = controller_to_df2t(&c_d);

    let d0 = operating_duty_cycle();
    let mut il = V_OUT / LOAD_RESISTANCE;
    let mut vo = V_OUT;

    let sub_steps = 10;
    let dt_sub = sample_time_s / (sub_steps as f64);

    let mut time = Vec::with_capacity(num_steps);
    let mut v_out = Vec::with_capacity(num_steps);
    let mut i_l = Vec::with_capacity(num_steps);
    let mut duty_cycle = Vec::with_capacity(num_steps);

    for k in 0..num_steps {
        let t = (k as f64) * sample_time_s;
        let v_ref = if t >= t_step_s {
            V_OUT + v_step_s
        } else {
            V_OUT
        };

        let r_load = match load_step {
            Some(load) if t >= load.time_s => load.resistance_ohm,
            _ => LOAD_RESISTANCE,
        };

        // ADC sampling:
        let error = v_ref - vo;
        let d_hat = controller.update(error);
        let d_command = (d0 + d_hat).clamp(0.0, 1.0);

        time.push(t);
        v_out.push(vo);
        i_l.push(il);
        duty_cycle.push(d_command);

        // RK4 integration over sampling period:
        for _ in 0..sub_steps {
            // k1
            let (k1_il, k1_vo) = nonlinear_dynamics(
                AveragedCircuit {
                    inductor_current_a: il,
                    output_voltage_v: vo,
                },
                AveragedDrive {
                    duty: d_command,
                    v_in: V_IN,
                    r_load,
                },
            );
            // k2
            let il_k2 = il + 1.0 * dt_sub * k1_il;
            let vo_k2 = vo + 0.5 * dt_sub * k1_vo;
            let (k2_il, k2_vo) = nonlinear_dynamics(
                AveragedCircuit {
                    inductor_current_a: il_k2,
                    output_voltage_v: vo_k2,
                },
                AveragedDrive {
                    duty: d_command,
                    v_in: V_IN,
                    r_load,
                },
            );
            // k3
            let il_k3 = il + 0.5 * dt_sub * k2_il;
            let vo_k3 = vo + 0.5 * dt_sub * k2_vo;
            let (k3_il, k3_vo) = nonlinear_dynamics(
                AveragedCircuit {
                    inductor_current_a: il_k3,
                    output_voltage_v: vo_k3,
                },
                AveragedDrive {
                    duty: d_command,
                    v_in: V_IN,
                    r_load,
                },
            );
            // k4
            let il_k4 = il + dt_sub * k3_il;
            let vo_k4 = vo + dt_sub * k3_vo;
            let (k4_il, k4_vo) = nonlinear_dynamics(
                AveragedCircuit {
                    inductor_current_a: il_k4,
                    output_voltage_v: vo_k4,
                },
                AveragedDrive {
                    duty: d_command,
                    v_in: V_IN,
                    r_load,
                },
            );

            il += (dt_sub / 6.0) * (k1_il + 2.0 * k2_il + 2.0 * k3_il + k4_il);
            vo += (dt_sub / 6.0) * (k1_vo + 2.0 * k2_vo + 2.0 * k3_vo + k4_vo);
        }
    }

    let target = if total_time_s >= t_step_s {
        V_OUT + v_step_s
    } else {
        V_OUT
    };
    let metrics = compute_step_metrics(
        &time,
        &v_out,
        StepMetricRef {
            t_step: t_step_s,
            v_initial: V_OUT,
            v_target: target,
        },
    );

    SimulationTrajectory {
        time_s: time,
        v_out_v: v_out,
        i_l_a: i_l,
        duty_cycle,
        metrics,
    }
}

/// Simulates a 50% load drop ($R_L: 1.0\,\Omega \to 2.0\,\Omega$) and extracts recovery metrics.
#[must_use]
pub fn simulate_load_transient(
    compensator: &LeadTf,
    horizon: SampledHorizon,
    load: LoadStep,
) -> LoadTrajectory {
    let traj = simulate_nonlinear_circuit(
        compensator,
        horizon,
        VoltageStep {
            time_s: 0.0,
            delta_v: 0.0,
        },
        Some(load),
    );

    let step_idx = traj
        .time_s
        .iter()
        .position(|&t| t >= load.time_s)
        .unwrap_or(0);
    let nominal_v = traj.v_out_v[step_idx];

    let mut peak_v = nominal_v;
    for &v in &traj.v_out_v[step_idx..] {
        if v > peak_v {
            peak_v = v;
        }
    }

    // 1% recovery threshold: time until voltage returns and remains within 1% (50 mV) of nominal
    let threshold_1pct = 0.01 * nominal_v;
    let mut recovery_time_s = 0.0;
    let post_step_t = &traj.time_s[step_idx..];
    let post_step_v = &traj.v_out_v[step_idx..];

    for (i, (&_t, &v)) in
        post_step_t.iter().zip(post_step_v.iter()).enumerate().rev()
    {
        if (v - nominal_v).abs() > threshold_1pct {
            let next_idx = (i + 1).min(post_step_t.len() - 1);
            recovery_time_s = (post_step_t[next_idx] - load.time_s).max(0.0);
            break;
        }
    }

    let restored_v = *traj.v_out_v.last().unwrap_or(&nominal_v);

    let metrics = LoadTransientMetrics {
        nominal_v,
        peak_v,
        recovery_time_s,
        recovery_time_us: recovery_time_s * 1e6,
        restored_v,
    };

    LoadTrajectory {
        time_s: traj.time_s,
        v_out_v: traj.v_out_v,
        i_l_a: traj.i_l_a,
        duty_cycle: traj.duty_cycle,
        nominal_v,
        peak_v,
        recovery_time_s,
        recovery_time_us: recovery_time_s * 1e6,
        restored_v,
        metrics,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::buck_converter::analysis::synthesize_lead_compensator;
    use crate::buck_converter::circuit::plant;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_compensator_discretization_and_realization_equivalence() {
        let p = plant();
        let design =
            synthesize_lead_compensator(&p, 3.0e4, 50.0).expect("Lead failed");
        let ts = 10.0e-6; // 100 kHz

        let c_d =
            discretize_compensator(&design.compensator_tf, ts, Some(3.0e4));
        let mut df2t = controller_to_df2t(&c_d);
        let mut biquad = controller_to_biquad(&c_d);

        let test_inputs = [0.1, -0.05, 0.2, 0.0, -0.1, 0.05];
        for &u in &test_inputs {
            let y_df2t = df2t.update(u);
            let y_bq = biquad.update(u);
            assert!(
                (y_df2t - y_bq).abs() < 1e-12,
                "DF2T and Biquad must produce identical outputs"
            );
        }
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_linear_simulation_step_response() {
        let p = plant();
        let design =
            synthesize_lead_compensator(&p, 3.0e4, 50.0).expect("Lead failed");
        let ts = 10.0e-6;
        let horizon = SampledHorizon {
            total_time_s: 0.003,
            sample_time_s: ts,
        };
        let sim = simulate_linear_closed_loop(
            &design.compensator_tf,
            horizon,
            VoltageStep {
                time_s: 0.0005,
                delta_v: 0.5,
            },
        );

        // Achieved settling time should be well under 1 ms
        assert!(
            sim.metrics.settling_time_final_s < 1.0e-3,
            "Achieved settling time: {} s",
            sim.metrics.settling_time_final_s
        );
        // Target settling time is None because steady-state error (90.9 mV) exceeds 2% band (10 mV)
        assert!(
            sim.metrics.settling_time_target_s.is_none(),
            "Target settling time must be None for Type-0 offset > 2% band"
        );
        assert!(
            sim.metrics.peak_overshoot_pct < 20.0,
            "Overshoot: {}%",
            sim.metrics.peak_overshoot_pct
        );
        assert!(
            sim.metrics.rise_time_s > 10.0e-6,
            "Rise time: {} s",
            sim.metrics.rise_time_s
        );
        assert!(
            (sim.metrics.steady_state_error_v - 0.0909).abs() < 0.005,
            "SSE: {} V",
            sim.metrics.steady_state_error_v
        );
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_load_transient_recovery() {
        let p = plant();
        let design =
            synthesize_lead_compensator(&p, 3.0e4, 50.0).expect("Lead failed");
        let ts = 10.0e-6;
        let load_sim = simulate_load_transient(
            &design.compensator_tf,
            SampledHorizon {
                total_time_s: 0.003,
                sample_time_s: ts,
            },
            LoadStep {
                time_s: 0.001,
                resistance_ohm: 2.0,
            },
        );
        assert!((load_sim.nominal_v - 5.0).abs() < 0.01);
        assert!(load_sim.peak_v > 5.4 && load_sim.peak_v < 5.8);
        assert!(
            load_sim.recovery_time_s > 0.0 && load_sim.recovery_time_s < 1.0e-3
        );
        assert!((load_sim.restored_v - 5.0).abs() < 0.01);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#NFR-1
    /// Method: Requirements-based test
    fn test_nonlinear_simulation_matches_linear_closely() {
        let p = plant();
        let design =
            synthesize_lead_compensator(&p, 3.0e4, 50.0).expect("Lead failed");
        let ts = 10.0e-6;
        let horizon = SampledHorizon {
            total_time_s: 0.002,
            sample_time_s: ts,
        };
        let step = VoltageStep {
            time_s: 0.0005,
            delta_v: 0.5,
        };
        let sim_lin =
            simulate_linear_closed_loop(&design.compensator_tf, horizon, step);
        let sim_nonlin = simulate_nonlinear_circuit(
            &design.compensator_tf,
            horizon,
            step,
            None,
        );

        assert!(
            (sim_lin.metrics.peak_voltage_v
                - sim_nonlin.metrics.peak_voltage_v)
                .abs()
                < 0.1
        );
        assert!(
            (sim_lin.metrics.peak_overshoot_pct
                - sim_nonlin.metrics.peak_overshoot_pct)
                .abs()
                < 8.0
        );
    }
}
