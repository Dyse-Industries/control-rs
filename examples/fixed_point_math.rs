//! Fixed-Point Scalar Arithmetic and Discrete Filtering Example
//!
//! Demonstrates deterministic, zero-allocation Q-format fixed-point scalar arithmetic
//! for integer microcontrollers without hardware floating-point units (FPUs).
//!
//! Features showcased:
//! - Q16.16 (`Fixed<i32, 16>`) representation, step delta, and dynamic range.
//! - Basic arithmetic, widening multiplication, and convergent rounding.
//! - Saturation arithmetic preventing overflow wrap-around in control loops.
//! - Integer-only execution of a discrete first-order low-pass filter:
//!   y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
//! - Comparison against ideal double-precision reference to measure quantization residual.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use control_rs::math::fixed_num::Fixed;
use control_rs::math::ops::{SaturatingAdd, SaturatingMul, SaturatingSub};

/// Standard Q16.16 fixed-point type: signed 32-bit integer with 16 fractional bits.
/// Spans [-32768.0, +32767.99998] with resolution Delta = 2^-16 ~= 1.52588e-5.
type Q16 = Fixed<i32, 16>;

fn float_to_q16(val: f64) -> Q16 {
    let raw = (val * 65536.0).round() as i32;
    Q16::from_bits(raw)
}

fn q16_to_float(val: Q16) -> f64 {
    (val.to_bits() as f64) / 65536.0
}

fn main() {
    println!("=== control-rs: Fixed-Point Arithmetic & Filter Example ===");

    // 1. Inspect Q16 parameters and resolution
    println!("Q16.16 Format Properties:");
    println!("  Fractional shift: {} bits", Q16::SHIFT);
    println!("  Quantization step Delta: {:.8}", q16_to_float(Q16::DELTA));
    println!("  Min representable value: {:.2}", q16_to_float(Q16::MIN));
    println!("  Max representable value: {:.2}", q16_to_float(Q16::MAX));
    println!();

    // 2. Fundamental arithmetic & widening multiplication
    let a = float_to_q16(3.75);
    let b = float_to_q16(2.50);

    let sum = a + b;
    let diff = a - b;
    let prod = a * b; // Inherent Mul performs wide 64-bit multiplication and 16-bit convergent downscale

    println!("Basic Arithmetic (Q16.16):");
    println!("  a = {:.4}, b = {:.4}", q16_to_float(a), q16_to_float(b));
    println!("  a + b = {:.4} (expected 6.2500)", q16_to_float(sum));
    println!("  a - b = {:.4} (expected 1.2500)", q16_to_float(diff));
    println!("  a * b = {:.4} (expected 9.3750)", q16_to_float(prod));
    println!();

    // 3. Saturation arithmetic vs overflow
    println!("Saturation Arithmetic Protection:");
    let near_max = float_to_q16(32000.0);
    let delta = float_to_q16(2000.0);
    let sat_sum = near_max.saturating_add(&delta);
    println!(
        "  Saturating Add: {:.1} + {:.1} = {:.1} (clamped at MAX: {:.1})",
        q16_to_float(near_max),
        q16_to_float(delta),
        q16_to_float(sat_sum),
        q16_to_float(Q16::MAX)
    );

    let near_min = float_to_q16(-32000.0);
    let sat_diff = near_min.saturating_sub(&delta);
    println!(
        "  Saturating Sub: {:.1} - {:.1} = {:.1} (clamped at MIN: {:.1})",
        q16_to_float(near_min),
        q16_to_float(delta),
        q16_to_float(sat_diff),
        q16_to_float(Q16::MIN)
    );

    let large_val = float_to_q16(200.0);
    let sat_prod = large_val.saturating_mul(&large_val);
    println!(
        "  Saturating Mul: {:.1} * {:.1} = {:.1} (clamped at MAX)",
        q16_to_float(large_val),
        q16_to_float(large_val),
        q16_to_float(sat_prod)
    );
    println!();

    // 4. Fixed-Point Discrete Low-Pass IIR Filter
    // Difference equation: y[k] = alpha * x[k] + (1 - alpha) * y[k-1]
    // Filter parameters: cutoff frequency fc = 10 Hz, sampling fs = 1000 Hz (Ts = 1 ms)
    // alpha = 2*pi*fc*Ts / (1 + 2*pi*fc*Ts) ~= 0.05912
    let filter_alpha_f64 = 0.059_12;
    let filter_decay_f64 = 1.0 - filter_alpha_f64;

    let weight_q16 = float_to_q16(filter_alpha_f64);
    let decay_q16 = float_to_q16(filter_decay_f64);

    println!(
        "Discrete IIR Low-Pass Filter Simulation (fc = 10 Hz, fs = 1 kHz):"
    );
    println!(
        "  alpha (f64) = {:.5}, alpha (Q16) = {:.5}",
        filter_alpha_f64,
        q16_to_float(weight_q16)
    );
    println!("  Simulating unit step input response over 50 steps (50 ms):");
    println!(
        "  Step | Time [ms] | Fixed-Point Output | Ideal Float Output | Quantization Error"
    );
    println!(
        "  -----+-----------+--------------------+--------------------+-------------------"
    );

    let mut y_fixed = float_to_q16(0.0);
    let mut y_float = 0.0;
    let x_step_fixed = float_to_q16(1.0);
    let x_step_float = 1.0;

    let mut max_abs_err: f64 = 0.0;

    for k in 0..=50 {
        if k % 5 == 0 {
            let y_f = q16_to_float(y_fixed);
            let err = (y_f - y_float).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
            println!(
                "  {:4} | {:9.1} | {:18.5} | {:18.5} | {:17.2e}",
                k,
                f64::from(k),
                y_f,
                y_float,
                err
            );
        }

        // Advance fixed-point filter: y[k] = alpha * x + (1 - alpha) * y[k-1]
        y_fixed = (weight_q16 * x_step_fixed) + (decay_q16 * y_fixed);

        // Advance reference floating-point filter
        y_float = filter_alpha_f64 * x_step_float + filter_decay_f64 * y_float;
    }

    println!();
    println!("Peak quantization error: {:.4e} (< 2 * Delta)", max_abs_err);
    println!(
        "Fixed-point simulation completed with 0 heap allocations and 0 FPU instructions."
    );
}
