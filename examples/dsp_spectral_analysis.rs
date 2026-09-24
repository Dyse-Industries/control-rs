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

use control_rs::math::complex_num::Complex;
use control_rs::math::dsp::{DefaultDsp, FFT};

const BIN_WIDTH: f64 = FS / N_F64; // Frequency resolution: 8.0 Hz per bin
const DT: f64 = 1.0 / FS; // Sampling interval: 1.953 ms
const F1: f64 = 40.0; // Tone 1 frequency (bin 5: 40/8 = 5)
const F2: f64 = 120.0; // Tone 2 frequency (bin 15: 120/8 = 15)
const FS: f64 = 512.0; // Sampling frequency: 512 Hz
const HALF_N: usize = N / 2; // Single-sided spectrum length
const N: usize = 64; // Power-of-two FFT length
const N_F64: f64 = 64.0; // `N` as a float

/// A detected spectral peak: bin index, center frequency and magnitude.
type Peak = (usize, f64, f64);

/// Two-sided complex FFT spectrum.
type Spectrum = [Complex<f64>; N];

fn main() {
    println!("=== control-rs: DSP Spectral Analysis & FFT Example ===");
    println!("Signal Acquisition Configuration:");
    println!("  Sample count (N): {N}");
    println!("  Sampling rate (Fs): {FS:.1} Hz");
    println!("  Time window duration: {:.3} s", N_F64 * DT);
    println!("  Frequency bin resolution: {BIN_WIDTH:.2} Hz/bin");
    println!("  Nyquist folding frequency: {:.1} Hz", FS / 2.0);
    println!();

    let time_signal = synthesize_signal();

    // 2. Compute Forward FFT
    let mut frequency_spectrum: Spectrum = [Complex::default(); N];
    DefaultDsp::fft(&time_signal, &mut frequency_spectrum);

    let magnitudes = print_spectrum(&frequency_spectrum);

    // 4. Identify spectral peak locations
    let detected_peaks = detect_peaks(&magnitudes);
    println!("Detected Spectral Peaks:");
    for (bin, freq, mag) in &detected_peaks {
        println!(
            "  Bin {bin:2}: Frequency = {freq:5.1} Hz, Peak Magnitude = {mag:.3} V"
        );
    }
    assert!(
        detected_peaks.len() >= 2,
        "Expected at least 2 spectral peaks"
    );
    println!();

    check_reconstruction(&time_signal, &frequency_spectrum);
    println!();
    println!("DSP spectral analysis completed successfully.");
}

/// Converts a sample or bin index to `f64` (exact for every index here).
fn index_f64(k: usize) -> f64 {
    f64::from(u32::try_from(k).unwrap_or(u32::MAX))
}

/// Step 1: synthesizes the two-tone test signal with a slight pseudorandom
/// disturbance.
fn synthesize_signal() -> [f64; N] {
    let a1 = 2.5; // Amplitude of tone 1 (V)
    let a2 = 1.2; // Amplitude of tone 2 (V)

    let mut time_signal = [0.0f64; N];
    for (n, sample) in (0_u32..).zip(time_signal.iter_mut()) {
        let t = f64::from(n) * DT;
        let tone1 = a1 * (2.0 * core::f64::consts::PI * F1 * t).sin();
        let tone2 = a2 * (2.0 * core::f64::consts::PI * F2 * t).cos();
        // Slight pseudorandom disturbance
        let jitter = n.wrapping_mul(17).wrapping_add(31) % 100;
        let noise = 0.05 * (f64::from(jitter) - 50.0) / 50.0;
        *sample = tone1 + tone2 + noise;
    }

    println!("Synthesized Injected Tones:");
    println!("  Tone 1: {F1:.1} Hz, Amplitude = {a1:.2} V (expected bin 5)");
    println!("  Tone 2: {F2:.1} Hz, Amplitude = {a2:.2} V (expected bin 15)");
    println!();
    time_signal
}

/// Step 3: prints the first bins of the single-sided magnitude spectrum and
/// returns all single-sided magnitudes.
///
/// For real inputs, bins 1..(N/2 - 1) carry half the energy; scale by 2/N.
fn print_spectrum(frequency_spectrum: &Spectrum) -> [f64; HALF_N] {
    let mut magnitudes = [0.0f64; HALF_N];

    println!("Single-Sided Frequency Spectrum (First 16 Bins):");
    println!(
        "  Bin | Center Freq [Hz] | Real Part | Imag Part | Magnitude [V]"
    );
    println!(
        "  ----+------------------+-----------+-----------+--------------"
    );

    for (k, (c, slot)) in frequency_spectrum
        .iter()
        .zip(magnitudes.iter_mut())
        .enumerate()
    {
        let freq = index_f64(k) * BIN_WIDTH;
        // Magnitude normalization: bin 0 (DC) is 1/N, AC bins are 2/N
        let scale = if k == 0 { 1.0 / N_F64 } else { 2.0 / N_F64 };
        let mag = c.re.hypot(c.im) * scale;
        *slot = mag;

        if k <= 16 {
            let marker = if (freq - F1).abs() < 1e-3 || (freq - F2).abs() < 1e-3
            {
                " <-- PEAK"
            } else {
                ""
            };
            println!(
                "  {k:3} | {freq:16.1} | {:9.4} | {:9.4} | {mag:13.4}{marker}",
                c.re, c.im
            );
        }
    }
    println!();
    magnitudes
}

/// Local maxima above 0.5 V, excluding the first and last bin.
fn detect_peaks(magnitudes: &[f64]) -> Vec<Peak> {
    magnitudes
        .windows(3)
        .zip(1_usize..)
        .filter_map(|(window, k)| match *window {
            [prev, cur, next] if cur > prev && cur > next && cur > 0.5 => {
                Some((k, index_f64(k) * BIN_WIDTH, cur))
            }
            _ => None,
        })
        .collect()
}

/// Step 5: reconstructs the time-domain signal via inverse FFT and checks the
/// residual.
fn check_reconstruction(time_signal: &[f64; N], frequency_spectrum: &Spectrum) {
    let mut reconstructed_signal = [0.0f64; N];
    DefaultDsp::ifft(frequency_spectrum, &mut reconstructed_signal);

    // Compute maximum reconstruction residual ||x - x_rec||_inf
    let max_residual = time_signal
        .iter()
        .zip(&reconstructed_signal)
        .map(|(x, x_hat)| (x - x_hat).abs())
        .fold(0.0f64, f64::max);

    println!("Inverse FFT Signal Reconstruction Fidelity:");
    println!(
        "  Maximum reconstruction residual ||x - x_hat||_inf: {max_residual:.4e}"
    );
    assert!(
        max_residual < 1e-12,
        "IFFT reconstruction residual exceeded machine precision bound"
    );
    println!(
        "  Signal perfectly reconstructed within numerical roundoff tolerance."
    );
}
