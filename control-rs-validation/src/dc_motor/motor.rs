//! src/motor.rs
//!
//! Physical model of a DC motor armature position servo.
//!
//! Electromechanical dynamics:
//! 1. Electrical circuit:
//!    $$V_a(t) = R_a i_a(t) + L_a \frac{di_a(t)}{dt} + K_b \omega(t)$$
//! 2. Mechanical rotor:
//!    $$J \frac{d\omega(t)}{dt} + b \omega(t) = K_t i_a(t) - \tau_L(t)$$
//! 3. Shaft kinematics:
//!    $$\frac{d\theta(t)}{dt} = \omega(t)$$
//!
//! Control-to-position continuous transfer function $G_{v\theta}(s)$:
//! $$G_{v\theta}(s) = \frac{\Theta(s)}{V_a(s)}
//!   = \frac{K_t}{s \left[ L_a J s^2 + (R_a J + L_a b)s + (R_a b + K_t K_b) \right]}$$

use control_rs::transfer_function::ArrayTransferFunction;

/// Armature electrical resistance $R_a$ ($\Omega$).
pub const R_A: f64 = 2.0;

/// Armature electrical inductance $L_a$ (H).
pub const L_A: f64 = 0.5e-3;

/// Motor torque constant $K_t$ ($\text{N}\cdot\text{m/A}$).
pub const K_T: f64 = 0.05;

/// Motor back-EMF constant $K_b$ ($\text{V}\cdot\text{s/rad}$).
pub const K_B: f64 = 0.05;

/// Rotor moment of inertia $J$ ($\text{kg}\cdot\text{m}^2$).
pub const J: f64 = 2.0e-4;

/// Viscous damping friction coefficient $b$ ($\text{N}\cdot\text{m}\cdot\text{s/rad}$).
pub const B: f64 = 1.0e-4;

/// Maximum rated terminal supply voltage $V_{\max}$ (V).
pub const V_MAX: f64 = 12.0;

/// Returns the 3rd-order continuous control-to-position transfer function
/// $G_{v\theta}(s) = \frac{\Theta(s)}{V_a(s)}$.
///
/// Polynomial coefficients are in ascending order of powers of $s$:
/// - Numerator: $[K_t]$
/// - Denominator: $[0.0, R_a b + K_t K_b, R_a J + L_a b, L_a J]$
#[must_use]
pub fn plant_position() -> ArrayTransferFunction<f64, 1, 4> {
    let d0 = 0.0;
    let d1 = R_A * B + K_T * K_B;
    let d2 = R_A * J + L_A * B;
    let d3 = L_A * J;

    ArrayTransferFunction::<f64, 1, 4>::continuous([K_T], [d0, d1, d2, d3])
}

/// Returns the 2nd-order continuous control-to-angular-velocity transfer function
/// $G_{v\omega}(s) = \frac{\Omega(s)}{V_a(s)}$.
#[cfg(test)]
#[must_use]
pub fn plant_velocity() -> ArrayTransferFunction<f64, 1, 3> {
    let d0 = R_A * B + K_T * K_B;
    let d1 = R_A * J + L_A * B;
    let d2 = L_A * J;

    ArrayTransferFunction::<f64, 1, 3>::continuous([K_T], [d0, d1, d2])
}

/// Computes the continuous time derivatives $\dot{x} = f(x, v_a, \tau_L)$ for the
/// state vector $x = [i_a, \omega, \theta]^T$.
#[must_use]
pub fn state_derivatives(x: [f64; 3], v_a: f64, tau_l: f64) -> [f64; 3] {
    let i_a = x[0];
    let omega = x[1];
    // x[2] is theta

    let d_ia = (v_a - R_A * i_a - K_B * omega) / L_A;
    let d_omega = (K_T * i_a - B * omega - tau_l) / J;
    let d_theta = omega;

    [d_ia, d_omega, d_theta]
}

/// Propagates the state vector $x = [i_a, \omega, \theta]^T$ across a time step
/// `dt` using 4th-order Runge-Kutta (RK4) numerical integration.
#[must_use]
pub fn rk4_step(x: [f64; 3], v_a: f64, tau_l: f64, dt: f64) -> [f64; 3] {
    let k1 = state_derivatives(x, v_a, tau_l);

    let x_k2 = [
        x[0] + 0.5 * dt * k1[0],
        x[1] + 0.5 * dt * k1[1],
        x[2] + 0.5 * dt * k1[2],
    ];
    let k2 = state_derivatives(x_k2, v_a, tau_l);

    let x_k3 = [
        x[0] + 0.5 * dt * k2[0],
        x[1] + 0.5 * dt * k2[1],
        x[2] + 0.5 * dt * k2[2],
    ];
    let k3 = state_derivatives(x_k3, v_a, tau_l);

    let x_k4 = [x[0] + dt * k3[0], x[1] + dt * k3[1], x[2] + dt * k3[2]];
    let k4 = state_derivatives(x_k4, v_a, tau_l);

    [
        x[0] + (dt / 6.0) * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]),
        x[1] + (dt / 6.0) * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]),
        x[2] + (dt / 6.0) * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]),
    ]
}

#[cfg(test)]
pub mod tests {
    use super::*;

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn test_motor_transfer_function_coefficients() {
        let plant = plant_position();
        let num = plant.num_slice();
        let den = plant.den_slice();

        assert_eq!(num.len(), 1);
        assert!((num[0] - 0.05).abs() < 1e-9);

        assert_eq!(den.len(), 4);
        assert!((den[0] - 0.0).abs() < 1e-9);
        // d1 = 2.0 * 1e-4 + 0.05 * 0.05 = 0.0002 + 0.0025 = 0.0027
        assert!((den[1] - 0.0027).abs() < 1e-9);
        // d2 = 2.0 * 2e-4 + 0.5e-3 * 1e-4 = 0.0004 + 0.00000005 = 0.00040005
        assert!((den[2] - 0.00040005).abs() < 1e-9);
        // d3 = 0.5e-3 * 2e-4 = 1e-7
        assert!((den[3] - 1.0e-7).abs() < 1e-12);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn test_motor_velocity_transfer_function_coefficients() {
        let plant = plant_velocity();
        let num = plant.num_slice();
        let den = plant.den_slice();

        assert_eq!(num.len(), 1);
        assert!((num[0] - K_T).abs() < 1e-9);

        assert_eq!(den.len(), 3);
        assert!((den[0] - (R_A * B + K_T * K_B)).abs() < 1e-9);
        assert!((den[1] - (R_A * J + L_A * B)).abs() < 1e-9);
        assert!((den[2] - (L_A * J)).abs() < 1e-12);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn test_steady_state_velocity_equilibrium() {
        let v_in = 12.0;
        let expected_omega = (K_T * v_in) / (R_A * B + K_T * K_B);
        let expected_ia = (B * v_in) / (R_A * B + K_T * K_B);

        // At equilibrium with tau_l = 0, derivatives must be zero
        let deriv =
            state_derivatives([expected_ia, expected_omega, 0.0], v_in, 0.0);
        assert!(
            deriv[0].abs() < 1e-9,
            "d_ia should be zero at equilibrium: {}",
            deriv[0]
        );
        assert!(
            deriv[1].abs() < 1e-9,
            "d_omega should be zero at equilibrium: {}",
            deriv[1]
        );
        assert!((deriv[2] - expected_omega).abs() < 1e-9);
    }

    #[test]
    /// # Verification
    /// Trace: classical-tools-examples#FR-2, classical-tools-examples#C-2
    /// Method: Requirements-based test
    fn test_rk4_step_integration_stability() {
        let mut x = [0.0, 0.0, 0.0];
        let dt = 1e-4;
        let v_in = 12.0;

        // Run for 2.5 seconds to reach full electromechanical steady state
        for _ in 0..25_000 {
            x = rk4_step(x, v_in, 0.0, dt);
        }

        let expected_omega = (K_T * v_in) / (R_A * B + K_T * K_B);
        let expected_ia = (B * v_in) / (R_A * B + K_T * K_B);

        assert!(
            (x[0] - expected_ia).abs() < 1e-3,
            "i_a mismatch: {} vs {}",
            x[0],
            expected_ia
        );
        assert!(
            (x[1] - expected_omega).abs() < 1e-3,
            "omega mismatch: {} vs {}",
            x[1],
            expected_omega
        );
        assert!(x[2] > 0.0, "position should be positive");
    }
}
