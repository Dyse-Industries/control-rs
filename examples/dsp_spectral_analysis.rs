//! Digital Signal Processing (DSP) Spectral Analysis Example
//!
//! Demonstrates frequency-domain spectral analysis and time-frequency signal reconstruction
//! using zero-allocation Radix-2 Fast Fourier Transform (FFT) routines.
//!
//! Features showcased:
//! - Time-domain signal synthesis with multiple harmonic tones and synthetic measurement noise.
//! - Forward Cooley-Tukey Radix-2 FFT via `DefaultDsp::fft`.
//! - Complex number algebra (`Complex<f64>`) and two-sided to single-sided power spectrum conversion.
//! - Peak frequency bin identification matching injected signal frequencies.
//! - Inverse FFT (`DefaultDsp::ifft`) verifying exact time-domain signal reconstruction.

#![allow(
    clippy::arithmetic_side_effects,
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::many_single_char_names,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::uninlined_format_args
)]

use control_rs::math::complex_num::Complex;
use control_rs::math::dsp::{DefaultDsp, FFT};

const BIN_WIDTH: f64 = FS / (N as f64); // Frequency resolution: 8.0 Hz per bin
const DT: f64 = 1.0 / FS; // Sampling interval: 1.953 ms
const FS: f64 = 512.0; // Sampling frequency: 512 Hz
const N: usize = 64; // Power-of-two FFT length

fn main() {
    println!("=== control-rs: DSP Spectral Analysis & FFT Example ===");
    println!("Signal Acquisition Configuration:");
    println!("  Sample count (N): {}", N);
    println!("  Sampling rate (Fs): {:.1} Hz", FS);
    println!("  Time window duration: {:.3} s", (N as f64) * DT);
    println!("  Frequency bin resolution: {:.2} Hz/bin", BIN_WIDTH);
    println!("  Nyquist folding frequency: {:.1} Hz", FS / 2.0);
    println!();

    // 1. Synthesize multi-tone signal:
    // f1 = 40.0 Hz (bin 5: 40/8 = 5), amplitude A1 = 2.5 V
    // f2 = 120.0 Hz (bin 15: 120/8 = 15), amplitude A2 = 1.2 V
    let f1 = 40.0;
    let a1 = 2.5;
    let f2 = 120.0;
    let a2 = 1.2;

    let mut time_signal = [0.0f64; N];
    for (n, sample) in time_signal.iter_mut().enumerate() {
        let t = (n as f64) * DT;
        let tone1 = a1 * (2.0 * core::f64::consts::PI * f1 * t).sin();
        let tone2 = a2 * (2.0 * core::f64::consts::PI * f2 * t).cos();
        // Slight pseudorandom disturbance
        let noise = 0.05 * (((n * 17 + 31) % 100) as f64 - 50.0) / 50.0;
        *sample = tone1 + tone2 + noise;
    }

    println!("Synthesized Injected Tones:");
    println!(
        "  Tone 1: {:.1} Hz, Amplitude = {:.2} V (expected bin 5)",
        f1, a1
    );
    println!(
        "  Tone 2: {:.1} Hz, Amplitude = {:.2} V (expected bin 15)",
        f2, a2
    );
    println!();

    // 2. Compute Forward FFT
    let mut frequency_spectrum = [Complex::<f64>::default(); N];
    DefaultDsp::fft(&time_signal, &mut frequency_spectrum);

    // 3. Compute single-sided magnitude spectrum
    // For real inputs, bins 1..(N/2 - 1) carry half the energy; scale by 2/N
    let half_n = N / 2;
    let mut magnitudes = [0.0f64; N / 2];

    println!("Single-Sided Frequency Spectrum (First 16 Bins):");
    println!(
        "  Bin | Center Freq [Hz] | Real Part | Imag Part | Magnitude [V]"
    );
    println!(
        "  ----+------------------+-----------+-----------+--------------"
    );

    for k in 0..half_n {
        let freq = (k as f64) * BIN_WIDTH;
        let c = frequency_spectrum[k];
        // Magnitude normalization: bin 0 (DC) is 1/N, AC bins are 2/N
        let scale = if k == 0 {
            1.0 / (N as f64)
        } else {
            2.0 / (N as f64)
        };
        let mag = c.re.hypot(c.im) * scale;
        magnitudes[k] = mag;

        if k <= 16 {
            let marker = if (freq - f1).abs() < 1e-3 || (freq - f2).abs() < 1e-3
            {
                " <-- PEAK"
            } else {
                ""
            };
            println!(
                "  {:3} | {:16.1} | {:9.4} | {:9.4} | {:13.4}{}",
                k, freq, c.re, c.im, mag, marker
            );
        }
    }
    println!();

    // 4. Identify spectral peak locations
    let mut detected_peaks = Vec::new();
    for k in 1..(half_n - 1) {
        if magnitudes[k] > magnitudes[k - 1]
            && magnitudes[k] > magnitudes[k + 1]
            && magnitudes[k] > 0.5
        {
            detected_peaks.push((k, (k as f64) * BIN_WIDTH, magnitudes[k]));
        }
    }

    println!("Detected Spectral Peaks:");
    for (bin, freq, mag) in &detected_peaks {
        println!(
            "  Bin {:2}: Frequency = {:5.1} Hz, Peak Magnitude = {:.3} V",
            bin, freq, mag
        );
    }
    assert!(
        detected_peaks.len() >= 2,
        "Expected at least 2 spectral peaks"
    );
    println!();

    // 5. Reconstruct Time-Domain Signal via Inverse FFT (IFFT)
    let mut reconstructed_signal = [0.0f64; N];
    DefaultDsp::ifft(&frequency_spectrum, &mut reconstructed_signal);

    // Compute maximum reconstruction residual ||x - x_rec||_inf
    let max_residual = time_signal
        .iter()
        .zip(&reconstructed_signal)
        .map(|(x, x_hat)| (x - x_hat).abs())
        .fold(0.0f64, f64::max);

    println!("Inverse FFT Signal Reconstruction Fidelity:");
    println!(
        "  Maximum reconstruction residual ||x - x_hat||_inf: {:.4e}",
        max_residual
    );
    assert!(
        max_residual < 1e-12,
        "IFFT reconstruction residual exceeded machine precision bound"
    );
    println!(
        "  Signal perfectly reconstructed within numerical roundoff tolerance."
    );
    println!();
    println!("DSP spectral analysis completed successfully.");
}
