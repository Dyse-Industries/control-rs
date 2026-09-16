//! src/buck_converter.rs
//!
//! Circuit definition for the `classical_tools` validation example
//! (`documentation/control-toolboxes/classical-tools-design.md`, §6.6): a
//! synchronous buck converter's averaged small-signal duty-cycle-to-
//! output-voltage model in continuous conduction mode (CCM).
//!
//! ```text
//!        S1        L
//!  12V ──/ ──●───UUUU───┬──────┬── V_out
//!            │          │      │
//!           D1(=)       C(=)  R_L
//!            │          │      │
//!            └──────────┴──────┘
//!                       GND
//! ```
//!
//! State-space averaging replaces the switch/diode pair with their duty-cycle
//! weighted average over a switching period, then linearizes the resulting
//! averaged model about a steady-state operating point (Erickson and
//! Maksimovic, 2001). Perturbing the duty cycle `D` by a small signal
//! `d_hat` around its steady-state value produces the control-to-output
//! transfer function:
//!
//! $$G_{vd}(s) = \frac{\hat v_o(s)}{\hat d(s)}
//!    = \frac{V_{in}}{LCs^2 + (L/R_L)s + 1}
//!    = \frac{V_{in}/(LC)}{s^2 + (1/(R_L C))s + 1/(LC)}$$
//!
//! i.e. a standard second-order low-pass with natural frequency
//! $\omega_n = 1/\sqrt{LC}$ and damping ratio
//! $\zeta = (1/(R_L C)) / (2\omega_n) = (1/(2R_L))\sqrt{L/C}$ (Erickson and
//! Maksimovic, 2001).
//!
//! With the component values below ($L=100\,\mu\text{H}$,
//! $C=100\,\mu\text{F}$, $R_L=1\,\Omega$), $\omega_n=10^4\,\text{rad/s}$ and
//! $\zeta=0.5$, matching the design doc's closed form
//! $G_{vd}(s) = 1.2\times10^9 / (s^2+10^4s+10^8)$.

use control_rs::state_space::ArrayStateSpace;
use control_rs::transfer_function::ArrayTransferFunction;

/// Input rail voltage $V_{in}$ (V).
pub const V_IN: f64 = 12.0;

/// Output filter inductance $L$ (H).
pub const INDUCTANCE: f64 = 100e-6;

/// Output filter capacitance $C$ (F).
pub const CAPACITANCE: f64 = 100e-6;

/// Load resistance $R_L$ (Ohm).
pub const LOAD_RESISTANCE: f64 = 1.0;

/// Steady-state target output voltage $V_{out}$ (V), fixing the operating
/// duty cycle via $D = V_{out}/V_{in}$.
pub const V_OUT: f64 = 5.0;

/// Steady-state operating duty cycle $D = V_{out}/V_{in}$ the small-signal
/// model in [`plant`] is linearized about.
#[must_use]
pub fn operating_duty_cycle() -> f64 {
    V_OUT / V_IN
}

/// LC output filter natural frequency $\omega_n = 1/\sqrt{LC}$ (rad/s).
#[must_use]
pub fn natural_frequency() -> f64 {
    1.0 / (INDUCTANCE * CAPACITANCE).sqrt()
}

/// LC output filter damping ratio
/// $\zeta = (1/(2R_L))\sqrt{L/C}$.
#[must_use]
pub fn damping_ratio() -> f64 {
    (INDUCTANCE / CAPACITANCE).sqrt() / (2.0 * LOAD_RESISTANCE)
}

/// The buck converter's control-to-output transfer function
/// $G_{vd}(s) = \hat v_o(s)/\hat d(s)$, built from the physical component
/// values above rather than pre-collapsed constants, so changing `L`, `C`,
/// or `R_L` updates the plant automatically.
///
/// Ascending-power coefficients: numerator $[V_{in}/(LC)]$, denominator
/// $[1/(LC),\ 1/(R_L C),\ 1]$ (i.e. $\omega_n^2$, $2\zeta\omega_n$, $1$).
#[must_use]
pub fn plant() -> ArrayTransferFunction<f64, 1, 3> {
    let wn_sq = natural_frequency().powi(2);
    let two_zeta_wn = 1.0 / (LOAD_RESISTANCE * CAPACITANCE);
    let dc_numerator = V_IN * wn_sq;
    ArrayTransferFunction::continuous([dc_numerator], [wn_sq, two_zeta_wn, 1.0])
}

/// The buck converter's small-signal state-space plant
/// $\dot{x} = A x + B \hat{d}$, $y = C x + D \hat{d}$ where
/// $x = [\hat{i}_L, \hat{v}_o]^T$ and $y = \hat{v}_o$.
#[must_use]
pub fn state_space_plant() -> ArrayStateSpace<f64, 2, 1, 1> {
    ArrayStateSpace::continuous(
        [
            [0.0, -1.0 / INDUCTANCE],
            [1.0 / CAPACITANCE, -1.0 / (LOAD_RESISTANCE * CAPACITANCE)],
        ],
        [[V_IN / INDUCTANCE], [0.0]],
        [[0.0, 1.0]],
        [[0.0]],
    )
}

/// Instantaneous inductor current and capacitor voltage of the averaged CCM model.
#[derive(Clone, Copy, Debug)]
pub struct AveragedCircuit {
    /// Inductor current $i_L$ (A).
    pub inductor_current_a: f64,
    /// Output voltage $v_o$ (V).
    pub output_voltage_v: f64,
}

/// Duty cycle, input rail, and load resistance driving the averaged CCM model.
#[derive(Clone, Copy, Debug)]
pub struct AveragedDrive {
    /// Switch duty cycle $d$.
    pub duty: f64,
    /// Input rail voltage $V_{in}$ (V).
    pub v_in: f64,
    /// Load resistance $R_{load}$ ($\Omega$).
    pub r_load: f64,
}

/// Evaluates continuous-conduction mode (CCM) averaged circuit state derivatives:
///
/// $$\frac{di_L}{dt} = \frac{d \cdot V_{in} - v_o}{L}$$
/// $$\frac{dv_o}{dt} = \frac{i_L - v_o / R_{load}}{C}$$
#[must_use]
pub fn nonlinear_dynamics(
    state: AveragedCircuit,
    drive: AveragedDrive,
) -> (f64, f64) {
    let d_il_dt =
        (drive.duty * drive.v_in - state.output_voltage_v) / INDUCTANCE;
    let d_vo_dt = (state.inductor_current_a
        - state.output_voltage_v / drive.r_load)
        / CAPACITANCE;
    (d_il_dt, d_vo_dt)
}

#[cfg(test)]
mod tests {
    use super::{
        AveragedCircuit, AveragedDrive, damping_ratio, natural_frequency,
        nonlinear_dynamics, operating_duty_cycle, plant, state_space_plant,
    };
    use control_rs::matrix::Owned;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn matches_design_doc_closed_form() {
        assert!((natural_frequency() - 1.0e4).abs() < 1.0);
        assert!((damping_ratio() - 0.5).abs() < 1e-9);
        assert!((operating_duty_cycle() - 5.0 / 12.0).abs() < 1e-12);

        let tf = plant();
        let den = tf.den_slice();
        let num = tf.num_slice();
        assert!((den[0] - 1.0e8).abs() < 1.0);
        assert!((den[1] - 1.0e4).abs() < 1e-6);
        assert!((den[2] - 1.0).abs() < 1e-12);
        assert!((num[0] - 1.2e9).abs() < 10.0);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn state_space_plant_matches_transfer_function_derivative() {
        let ss = state_space_plant();
        let x0 = Owned::<f64, 2, 1>::zero();
        let u0 = Owned::<f64, 1, 1>::scalar(0.1); // d_hat = 0.1
        let (x_dot, y) = ss.derivative(&x0, &u0);

        // At x = 0, u = 0.1: di_L/dt = 0.1 * 12 / 100e-6 = 12000 A/s, dvo/dt = 0
        assert!((x_dot.get(0, 0).copied().unwrap() - 12000.0).abs() < 1e-9);
        assert!((x_dot.get(1, 0).copied().unwrap() - 0.0).abs() < 1e-9);
        assert!((y.get(0, 0).copied().unwrap() - 0.0).abs() < 1e-9);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-1, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn nonlinear_dynamics_steady_state_equilibrium() {
        // At steady state D = 5/12, V_in = 12, V_out = 5, R_L = 1:
        // I_L = V_out / R_L = 5 A.
        let (d_il, d_vo) = nonlinear_dynamics(
            AveragedCircuit {
                inductor_current_a: 5.0,
                output_voltage_v: 5.0,
            },
            AveragedDrive {
                duty: 5.0 / 12.0,
                v_in: 12.0,
                r_load: 1.0,
            },
        );
        assert!(d_il.abs() < 1e-9);
        assert!(d_vo.abs() < 1e-9);
    }
}
