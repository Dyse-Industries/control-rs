//! DC Motor Servo Control Simulation Example
//!
//! Demonstrates modeling a permanent magnet DC motor using continuous-time state-space
//! representation, Tustin bilinear discretization, controllability analysis, and closed-loop
//! simulation.
//!
//! System Dynamics:
//! Rotor inertia: J = 0.01 kg·m²
//! Viscous friction: b = 0.1 N·m·s
//! Torque constant: Kt = 0.01 N·m/A
//! Back-EMF constant: Ke = 0.01 V/(rad/s)
//! Armature resistance: R = 1.0 Ω
//! Armature inductance: L = 0.5 H

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::imprecise_flops,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;
use control_rs::transfer_function::ArrayTransferFunction;

fn main() {
    println!("=== control-rs: DC Motor Speed Control Example ===");

    // Physical parameters
    let j = 0.01; // Rotor inertia (kg*m^2)
    let b = 0.1; // Viscous damping (N*m*s)
    let kt = 0.01; // Torque constant (N*m/A)
    let ke = 0.01; // Back-EMF constant (V*s/rad)
    let r = 1.0; // Resistance (Ohm)
    let l = 0.5; // Inductance (H)

    // State vector: x = [angular_velocity (rad/s), armature_current (A)]^T
    // Input: u = [armature_voltage (V)]
    // Output: y = [angular_velocity (rad/s)]
    //
    // dx/dt = [ -b/J    Kt/J ] x + [  0  ] u
    //         [ -Ke/L   -R/L ]     [ 1/L ]
    //
    // y = [ 1  0 ] x + [ 0 ] u
    let a = Owned::<f64, 2, 2>::from_row_arrays([
        [-b / j, kt / j],
        [-ke / l, -r / l],
    ]);
    let b_mat = Owned::<f64, 2, 1>::from_row_arrays([[0.0], [1.0 / l]]);
    let c = Owned::<f64, 1, 2>::from_row_arrays([[1.0, 0.0]]);
    let d = Owned::<f64, 1, 1>::from_row_arrays([[0.0]]);

    let motor_ss = ArrayStateSpace::continuous(a, b_mat, c, d);

    // Controllability matrix: Mc = [ B | A*B ] (2x2)
    let mc = motor_ss.controllability_matrix::<2>();
    println!("Controllability Matrix Mc:");
    println!(
        "  [{:8.4}, {:8.4}]",
        mc.get(0, 0).copied().unwrap_or(0.0),
        mc.get(0, 1).copied().unwrap_or(0.0)
    );
    println!(
        "  [{:8.4}, {:8.4}]",
        mc.get(1, 0).copied().unwrap_or(0.0),
        mc.get(1, 1).copied().unwrap_or(0.0)
    );

    // Determinant of 2x2 controllability matrix
    let m00 = mc.get(0, 0).copied().unwrap_or(0.0);
    let m01 = mc.get(0, 1).copied().unwrap_or(0.0);
    let m10 = mc.get(1, 0).copied().unwrap_or(0.0);
    let m11 = mc.get(1, 1).copied().unwrap_or(0.0);
    let det = m00 * m11 - m01 * m10;
    println!(
        "det(Mc) = {det:.6} (system is controllable: {})",
        det.abs() > 1e-9
    );

    // Discretize via Tustin bilinear transform at Ts = 10 ms
    let sample_time = 0.01;
    let motor_dt = match motor_ss.to_discrete_tustin(sample_time) {
        Ok(sys) => sys,
        Err(e) => {
            eprintln!("Discretization error: {e:?}");
            return;
        }
    };

    println!("\nDiscretized State-Space (Tustin, Ts = {sample_time}s):");
    println!(
        "  A_d = [[{:.4}, {:.4}], [{:.4}, {:.4}]]",
        motor_dt.a().get(0, 0).copied().unwrap_or(0.0),
        motor_dt.a().get(0, 1).copied().unwrap_or(0.0),
        motor_dt.a().get(1, 0).copied().unwrap_or(0.0),
        motor_dt.a().get(1, 1).copied().unwrap_or(0.0)
    );

    // Equivalent transfer function representation:
    // H(s) = Kt / ( (J*s + b)*(L*s + R) + Kt*Ke )
    //      = Kt / ( (J*L)*s^2 + (J*R + b*L)*s + (b*R + Kt*Ke) )
    // Ascending polynomial coefficients:
    // num = [Kt]
    // den = [b*R + Kt*Ke, J*R + b*L, J*L]
    let den_0 = b * r + kt * ke;
    let den_1 = j * r + b * l;
    let den_2 = j * l;
    let motor_tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
        [kt],
        [den_0, den_1, den_2],
    );

    println!("\nTransfer Function Frequency Response (Bode evaluation):");
    let frequencies = [0.1, 1.0, 10.0, 100.0];
    for &omega in &frequencies {
        let resp = motor_tf.eval_frequency(omega);
        let mag = (resp.re * resp.re + resp.im * resp.im).sqrt();
        let mag_db = 20.0 * mag.log10();
        let phase_deg = resp.im.atan2(resp.re).to_degrees();
        println!(
            "  w = {:6.1} rad/s -> Mag = {:7.2} dB, Phase = {:7.2} deg",
            omega, mag_db, phase_deg
        );
    }

    // Discrete-time step simulation: Unit step voltage input (12V)
    println!("\nTransient Step Response (12V step input, 50 steps / 0.5s):");
    let mut x = Owned::<f64, 2, 1>::zero();
    let u = Owned::<f64, 1, 1>::from_row_arrays([[12.0]]);

    println!("  Time [s] | Speed w [rad/s] | Current i [A]");
    println!("  ---------+-----------------+--------------");
    for k in 0..=50 {
        let t = (k as f64) * sample_time;
        if k % 5 == 0 {
            let speed = x.get(0, 0).copied().unwrap_or(0.0);
            let current = x.get(1, 0).copied().unwrap_or(0.0);
            println!("  {:8.2} | {:15.4} | {:12.4}", t, speed, current);
        }
        let (x_next, _) = motor_dt.step(&x, &u);
        x = x_next;
    }

    println!("\nSimulation completed successfully.");
}
