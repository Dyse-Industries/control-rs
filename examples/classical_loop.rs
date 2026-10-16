//! Classical Loop Design: From Plant to Firmware
//!
//! Follows one controller from design to deployment:
//!
//! 1. Build a PID compensator as a `TransferFunction`, close the loop with a
//!    plant and read the gain and phase margin in one call.
//! 2. Run the discrete `Pid` per sample against a simulated plant with a
//!    saturating actuator, the code that runs unchanged on the target.
//! 3. Factor a discrete filter into second-order sections and run it.
//!
//! Plant: `G(s) = 1 / (s + 1)^3`.

use control_rs::classical_control::{
    AntiWindup, BiquadCascade, ClassicalError, Df1, Pid, PidForm, PidParams,
    margin, pid, section_count, to_sections,
};
use control_rs::transfer_function::ArrayTransferFunction;

/// Sample time of the discrete controller and filter (s).
const H: f64 = 0.01;

fn main() -> Result<(), ClassicalError> {
    println!("=== control-rs: Classical Loop Example ===");
    loop_margins()?;
    run_pid()?;
    run_filter()
}

/// Closes a PID around the plant and prints the margins of the open loop.
fn loop_margins() -> Result<(), ClassicalError> {
    let plant = ArrayTransferFunction::<f64, 1, 4>::continuous(
        [1.0],
        [1.0, 3.0, 3.0, 1.0],
    );
    let controller = pid::<f64, 3, 3>(
        PidForm::Parallel {
            kp: 1.5,
            ki: 0.8,
            kd: 0.4,
        },
        Some(0.05),
    )?;
    let open_loop = controller.series::<1, 4, 3, 6>(&plant);

    // A logarithmic sweep from 0.01 to 100 rad/s.
    let sweep: [f64; 200] = core::array::from_fn(|k| {
        let exponent =
            -2.0 + 4.0 * f64::from(u16::try_from(k).unwrap_or(0)) / 199.0;
        10f64.powf(exponent)
    });
    let m = margin(&open_loop, &sweep);
    match m.phase_crossing {
        Some(p) => println!(
            "Gain margin:  {:6.2} dB at {:5.3} rad/s",
            p.gain_margin_db(),
            p.omega
        ),
        None => println!("Gain margin:  none in the sweep"),
    }
    if let Some(g) = m.gain_crossing {
        println!(
            "Phase margin: {:6.2} deg at {:5.3} rad/s (delay margin {:.3} s)",
            g.phase_margin_deg, g.omega, g.delay_margin
        );
    }
    Ok(())
}

/// Runs the discrete PID with output limits and anti-windup.
fn run_pid() -> Result<(), ClassicalError> {
    let mut ctrl = Pid::new(PidParams {
        kp: 1.5,
        ki: 0.8,
        kd: 0.4,
        tf: 0.05,
        h: H,
        u_min: -1.0,
        u_max: 1.0,
        anti_windup: AntiWindup::Clamping,
    })?;
    let (mut y, setpoint) = (0.0_f64, 0.8);
    for k in 0..=600u16 {
        let u = ctrl.step(setpoint, y);
        // First-order plant with a 0.5 s time constant.
        y += H * (u - y) / 0.5;
        if k % 100 == 0 {
            println!(
                "t = {:4.2} s  u = {u:6.3}  y = {y:6.3}",
                f64::from(k) * H
            );
        }
    }
    Ok(())
}

/// Factors `4 / (s^2 + 2 s + 4)` after Tustin discretization into sections.
fn run_filter() -> Result<(), ClassicalError> {
    let lowpass =
        ArrayTransferFunction::<f64, 1, 3>::continuous([4.0], [4.0, 2.0, 1.0])
            .to_discrete_tustin(H, None);
    let sections = to_sections::<f64, 3, 3, { section_count(3) }>(&lowpass)?;
    let mut filter = BiquadCascade::<Df1<f64>, 1>::from_coefficients(&sections);
    let settled = (0..2000).fold(0.0, |_, _| filter.update(1.0));
    println!("Low-pass step response settles at {settled:.4} (DC gain 1)");
    Ok(())
}
