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

use control_rs::matrix::Owned;
use control_rs::state_space::ArrayStateSpace;
use control_rs::transfer_function::ArrayTransferFunction;

fn main() {
    println!("=== control-rs: Buck Converter Dynamics Example ===");

    let vin: f64 = 12.0; // Input supply voltage (V)
    let d_nom: f64 = 5.0 / vin; // Nominal duty cycle for 5V output
    let inductance: f64 = 100e-6; // Inductance L (H)
    let r_l: f64 = 0.05; // Inductor parasitic resistance rL (Ohm)
    let capacitance: f64 = 220e-6; // Output filter capacitance C (F)
    let r_load: f64 = 5.0; // Load resistance (Ohm)

    println!("Operating Point:");
    println!(
        "  Vin = {vin:.1} V, Vout = {:.1} V, D = {d_nom:.4}",
        vin * d_nom
    );
    println!(
        "  L = {:.1} uH, C = {:.1} uF, R_load = {r_load:.1} Ohm",
        inductance * 1e6,
        capacitance * 1e6,
    );

    // Natural resonant frequency: w0 = 1 / sqrt(L*C)
    let w0 = 1.0 / (inductance * capacitance).sqrt();
    let f0 = w0 / (2.0 * std::f64::consts::PI);
    println!("  LC Resonant Frequency f0 = {f0:.1} Hz (w0 = {w0:.1} rad/s)");

    // Small-signal state-space model:
    // States: x = [i_L (A), v_C (V)]^T
    // Input: u = [duty_perturbation d_hat]
    // Output: y = [v_out_perturbation (V)]
    //
    // dx/dt = [ -rL/L    -1/L   ] x + [ Vin/L ] d_hat
    //         [  1/C   -1/(R*C) ]     [   0   ]
    let a_mat = Owned::<f64, 2, 2>::from_row_arrays([
        [-r_l / inductance, -1.0 / inductance],
        [1.0 / capacitance, -1.0 / (r_load * capacitance)],
    ]);
    let b_mat =
        Owned::<f64, 2, 1>::from_row_arrays([[vin / inductance], [0.0]]);
    let c_mat = Owned::<f64, 1, 2>::from_row_arrays([[0.0, 1.0]]);
    let d_mat = Owned::<f64, 1, 1>::from_row_arrays([[0.0]]);

    let buck_ss = ArrayStateSpace::continuous(a_mat, b_mat, c_mat, d_mat);

    // Control-to-output transfer function Gvd(s) = v_o(s) / d(s):
    // Gvd(s) = Vin / ( L*C*s^2 + (L/R + rL*C)*s + (1 + rL/R) )
    // In ascending polynomial coefficients:
    let a0 = 1.0 + r_l / r_load;
    let a1 = r_l.mul_add(capacitance, inductance / r_load);
    let a2 = inductance * capacitance;
    let gvd =
        ArrayTransferFunction::<f64, 1, 3>::continuous([vin], [a0, a1, a2]);
    print_frequency_response(&gvd, f0);

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
    simulate_duty_step(&buck_dt, vin, ts);

    println!("\nSimulation completed successfully.");
}

/// Prints magnitude and phase of `Gvd(s)` around the LC resonance `f0`.
fn print_frequency_response(gvd: &ArrayTransferFunction<f64, 1, 3>, f0: f64) {
    println!("\nFrequency Response of Control-to-Output Gvd(s):");
    let test_freqs_hz = [100.0, 500.0, f0, 2000.0, 10000.0];
    for &f in &test_freqs_hz {
        let w = 2.0 * std::f64::consts::PI * f;
        let resp = gvd.eval_frequency(w);
        let mag_db = 20.0 * resp.re.hypot(resp.im).log10();
        let phase_deg = resp.im.atan2(resp.re).to_degrees();
        println!(
            "  f = {f:7.1} Hz -> Mag = {mag_db:6.2} dB, Phase = {phase_deg:7.2} deg"
        );
    }
}

/// Transient response to a +5% duty cycle step perturbation (`d_hat = 0.05`).
fn simulate_duty_step(
    buck_dt: &ArrayStateSpace<f64, 2, 1, 1>,
    vin: f64,
    ts: f64,
) {
    let delta_d = 0.05;
    let u_step = Owned::<f64, 1, 1>::from_row_arrays([[delta_d]]);
    let mut x = Owned::<f64, 2, 1>::zero();

    println!(
        "Simulating +5% duty step response (delta_Vo expected ~= +{:.2} V):",
        vin * delta_d
    );
    println!("  Time [us] | Inductor Current [A] | Output Perturbation [V]");
    println!("  ----------+----------------------+--------------------------");

    for k in 0..=80_u8 {
        let t_us = f64::from(k) * ts * 1e6;
        if k % 8 == 0 {
            let i_l = x.get(0, 0).copied().unwrap_or(0.0);
            let v_o = x.get(1, 0).copied().unwrap_or(0.0);
            println!("  {t_us:9.1} | {i_l:20.4} | {v_o:24.4}");
        }
        let (x_next, _) = buck_dt.step(&x, &u_step);
        x = x_next;
    }
}
