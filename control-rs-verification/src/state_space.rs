//! State-space numerical model control-rs-verification suite.
//!
//! Evaluates continuous and discrete state-space kernels:
//! 1. Inverted pendulum continuous model linearized and simulated via ZOH
//! 2. Phase portrait trajectory ($\theta, \dot{\theta}$)
//! 3. Closed-loop discrete-time step response trajectory
//! 4. Matrix exponential ZOH discretization numerical consistency (`A_d`, `B_d`)

#![allow(missing_docs)]

use std::path::Path;

use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;

use crate::h5_writer::H5Writer;
use crate::numeric::{KernelResult, column, row_major};

/// Sampled `(theta, theta_dot)` trajectory of the pendulum.
pub struct PhasePortrait {
    /// Angle samples.
    pub theta: Vec<f64>,
    /// Angular-rate samples.
    pub theta_dot: Vec<f64>,
}

pub struct PendulumSim {
    pub sys_d: ArrayStateSpace<f64, 2, 1, 1>,
}

impl PendulumSim {
    #[must_use]
    pub fn new(omega0: f64, b: f64, dt: f64) -> Self {
        let omega0_sq = omega0 * omega0;
        let a_c =
            Owned::<f64, 2, 2>::from_row_arrays([[0.0, 1.0], [-omega0_sq, -b]]);
        let b_c = Owned::<f64, 2, 1>::from_column([0.0, 1.0]);
        let c_c = Owned::<f64, 1, 2>::from_row([1.0, 0.0]);
        let d_c = Owned::<f64, 1, 1>::scalar(0.0);

        let sys_c = ArrayStateSpace::continuous(a_c, b_c, c_c, d_c);
        let sys_d = sys_c.to_discrete_zoh(dt);
        Self { sys_d }
    }

    #[must_use]
    pub fn simulate(
        &self,
        x0: [f64; 2],
        n_steps: usize,
        u_val: f64,
    ) -> PhasePortrait {
        let mut x_k = Owned::<f64, 2, 1>::from_column(x0);
        let u_k = Owned::<f64, 1, 1>::scalar(u_val);

        let mut theta = Vec::with_capacity(n_steps);
        let mut theta_dot = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let th = x_k.get(0, 0).copied().unwrap_or(0.0);
            let th_dot = x_k.get(1, 0).copied().unwrap_or(0.0);
            theta.push(th);
            theta_dot.push(th_dot);
            let (x_next, _) = self.sys_d.step(&x_k, &u_k);
            x_k = x_next;
        }

        PhasePortrait { theta, theta_dot }
    }

    #[must_use]
    pub fn step_response(&self, n_steps: usize) -> Vec<f64> {
        let mut x_k = Owned::<f64, 2, 1>::zero();
        let u_k = Owned::<f64, 1, 1>::scalar(1.0);
        let mut step_data = Vec::with_capacity(n_steps);

        for _ in 0..n_steps {
            let (_, y_k) = self.sys_d.step(&x_k, &u_k);
            step_data.push(y_k.get(0, 0).copied().unwrap_or(0.0));
            let (x_next, _) = self.sys_d.step(&x_k, &u_k);
            x_k = x_next;
        }

        step_data
    }
}

/// Executes the state-space control-rs-verification kernel and writes `target/verification/state_space.rust.h5`.
///
/// # Errors
///
/// Returns an error if a discretized matrix entry cannot be read or the
/// container cannot be written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
    let mut writer = H5Writer::new();
    let sim = PendulumSim::new(2.0, 0.8, 0.05);

    let PhasePortrait { theta, theta_dot } =
        sim.simulate([std::f64::consts::PI - 0.15, 0.5], 200, 0.0);
    writer.add_dataset("phase_portrait/theta", &theta);
    writer.set_tolerance("phase_portrait/theta", "abs", 1e-4);

    writer.add_dataset("phase_portrait/theta_dot", &theta_dot);
    writer.set_tolerance("phase_portrait/theta_dot", "abs", 1e-4);

    let step_data = sim.step_response(100);
    writer.add_dataset("transient/step_data", &step_data);
    writer.set_tolerance("transient/step_data", "abs", 1e-4);

    let a_d_mat = row_major(&sim.sys_d.a())?;
    writer.add_dataset("discretization/a_d", &a_d_mat);
    writer.set_tolerance("discretization/a_d", "abs", 1e-4);

    let b_d_vec = column(&sim.sys_d.b(), 0)?;
    writer.add_dataset("discretization/b_d", &b_d_vec);
    writer.set_tolerance("discretization/b_d", "abs", 1e-4);

    writer.write_to_file(output_path)
}
