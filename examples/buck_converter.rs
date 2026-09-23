//! Synchronous Buck Converter Small-Signal Dynamics & Frequency Analysis Example
//!
//! Demonstrates small-signal average modeling of a DC-DC buck converter,
//! continuous-to-discrete Tustin transformation, resonance peak analysis,
//! and load-transient simulation.
//!
//! Converter Parameters:
//! Input Voltage: Vin = 12.0 V
//! Nominal Duty Cycle: D = 0.4167 (Vo = 5.0 V)
//! Inductor: L = 100 μH, ESR rL = 0.05 Ω
//! Capacitor: C = 220 μF
//! Nominal Load: R = 5.0 Ω (1 A output)

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
    println!("=== control-rs: Buck Converter Dynamics Example ===");

    let vin: f64 = 12.0; // Input supply voltage (V)
    let d_nom: f64 = 5.0 / vin; // Nominal duty cycle for 5V output
    let l: f64 = 100e-6; // Inductance (H)
    let rl: f64 = 0.05; // Inductor parasitic resistance (Ohm)
    let c: f64 = 220e-6; // Output filter capacitance (F)
    let r_load: f64 = 5.0; // Load resistance (Ohm)

    println!("Operating Point:");
    println!(
        "  Vin = {vin:.1} V, Vout = {:.1} V, D = {d_nom:.4}",
        vin * d_nom
    );
    println!(
        "  L = {:.1} uH, C = {:.1} uF, R_load = {:.1} Ohm",
        l * 1e6,
        c * 1e6,
        r_load
    );

    // Natural resonant frequency: w0 = 1 / sqrt(L*C)
    let w0 = 1.0 / (l * c).sqrt();
    let f0 = w0 / (2.0 * std::f64::consts::PI);
    println!("  LC Resonant Frequency f0 = {f0:.1} Hz (w0 = {w0:.1} rad/s)");

    // Small-signal state-space model:
    // States: x = [i_L (A), v_C (V)]^T
    // Input: u = [duty_perturbation d_hat]
    // Output: y = [v_out_perturbation (V)]
    //
    // dx/dt = [ -rL/L    -1/L   ] x + [ Vin/L ] d_hat
    //         [  1/C   -1/(R*C) ]     [   0   ]
    let a = Owned::<f64, 2, 2>::from_row_arrays([
        [-rl / l, -1.0 / l],
        [1.0 / c, -1.0 / (r_load * c)],
    ]);
    let b = Owned::<f64, 2, 1>::from_row_arrays([[vin / l], [0.0]]);
    let c_mat = Owned::<f64, 1, 2>::from_row_arrays([[0.0, 1.0]]);
    let d = Owned::<f64, 1, 1>::from_row_arrays([[0.0]]);

    let buck_ss = ArrayStateSpace::continuous(a, b, c_mat, d);

    // Control-to-output transfer function Gvd(s) = v_o(s) / d(s):
    // Gvd(s) = Vin / ( L*C*s^2 + (L/R + rL*C)*s + (1 + rL/R) )
    // In ascending polynomial coefficients:
    let a0 = 1.0 + rl / r_load;
    let a1 = l / r_load + rl * c;
    let a2 = l * c;
    let gvd =
        ArrayTransferFunction::<f64, 1, 3>::continuous([vin], [a0, a1, a2]);

    println!("\nFrequency Response of Control-to-Output Gvd(s):");
    let test_freqs_hz = [100.0, 500.0, f0, 2000.0, 10000.0];
    for &f in &test_freqs_hz {
        let w = 2.0 * std::f64::consts::PI * f;
        let resp = gvd.eval_frequency(w);
        let mag = (resp.re * resp.re + resp.im * resp.im).sqrt();
        let mag_db = 20.0 * mag.log10();
        let phase_deg = resp.im.atan2(resp.re).to_degrees();
        println!(
            "  f = {:7.1} Hz -> Mag = {:6.2} dB, Phase = {:7.2} deg",
            f, mag_db, phase_deg
        );
    }

    // Discretize for digital control loop at 100 kHz (Ts = 10 μs)
    let ts = 10e-6;
    let buck_dt = match buck_ss.to_discrete_tustin(ts) {
        Ok(sys) => sys,
        Err(e) => {
            eprintln!("Discretization error: {e:?}");
            return;
        }
    };

    println!(
        "\nDiscretized Model for 100 kHz Digital Controller (Ts = {:.0} us):",
        ts * 1e6
    );
    // Transient response to a +5% duty cycle step perturbation (d_hat = 0.05)
    let delta_d = 0.05;
    let u_step = Owned::<f64, 1, 1>::from_row_arrays([[delta_d]]);
    let mut x = Owned::<f64, 2, 1>::zero();

    println!(
        "Simulating +5% duty step response (delta_Vo expected ~= +{:.2} V):",
        vin * delta_d
    );
    println!("  Time [us] | Inductor Current [A] | Output Perturbation [V]");
    println!("  ----------+----------------------+--------------------------");

    for k in 0..=80 {
        let t_us = (k as f64) * ts * 1e6;
        if k % 8 == 0 {
            let i_l = x.get(0, 0).copied().unwrap_or(0.0);
            let v_o = x.get(1, 0).copied().unwrap_or(0.0);
            println!("  {:9.1} | {:20.4} | {:24.4}", t_us, i_l, v_o);
        }
        let (x_next, _) = buck_dt.step(&x, &u_step);
        x = x_next;
    }

    println!("\nSimulation completed successfully.");
}
