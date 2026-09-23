//! Transfer function numerical model control-rs-verification suite.
//!
//! Evaluates continuous and discrete frequency response kernels:
//! 1. Resonant notch filter frequency response (Bode magnitude & phase)
//! 2. Tustin bilinear transform frequency warping
//! 3. Zero-Order Hold (ZOH) discretization consistency
//! 4. Nyquist stability contour and gain/phase margins
//! 5. Controllable canonical form realization

#![allow(missing_docs)]

use std::path::Path;

use control_rs::transfer_function::ArrayTransferFunction;

use crate::h5_writer::H5Writer;
use crate::numeric::{KernelResult, column, index_f64, row_major};

type Tf<const N: usize, const D: usize> = ArrayTransferFunction<f64, N, D>;

/// Magnitude (dB) and phase (degrees) sampled on a frequency grid.
#[derive(Default)]
struct BodeTrace {
    mag_db: Vec<f64>,
    phase_deg: Vec<f64>,
}

/// Bode responses of the continuous notch system and its two discretizations.
struct BodeResponses {
    freqs_hz: Vec<f64>,
    continuous: BodeTrace,
    tustin: BodeTrace,
    zoh: BodeTrace,
}

/// Nyquist contour of the open loop and the margins read from it.
struct NyquistResult {
    h_re: Vec<f64>,
    h_im: Vec<f64>,
    phase_margin_deg: f64,
    gain_margin_db: f64,
}

/// Row-major `(A, B, C)` of a controllable canonical realization.
struct CcfRealization {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
}

/// Parameters of the resonant notch in series with a first-order low-pass.
struct NotchSpec {
    notch_hz: f64,
    zeta_zero: f64,
    zeta_pole: f64,
    cutoff_hz: f64,
}

impl BodeTrace {
    /// Appends one `(magnitude, phase_rad)` sample.
    fn push(&mut self, (mag, phase_rad): (f64, f64)) {
        self.mag_db.push(20.0 * mag.log10());
        self.phase_deg.push(phase_rad.to_degrees());
    }
}

fn synthesize_resonant_notch_system(spec: &NotchSpec) -> Tf<3, 4> {
    let wn = 2.0 * std::f64::consts::PI * spec.notch_hz;
    let wc = 2.0 * std::f64::consts::PI * spec.cutoff_hz;

    let notch = Tf::<3, 3>::continuous(
        [wn * wn, 2.0 * spec.zeta_zero * wn, 1.0],
        [wn * wn, 2.0 * spec.zeta_pole * wn, 1.0],
    );
    let lowpass = Tf::<1, 2>::continuous([wc], [wc, 1.0]);

    notch.series::<1, 2, 3, 4>(&lowpass)
}

fn compute_bode_responses() -> KernelResult<BodeResponses> {
    let tf_cont = synthesize_resonant_notch_system(&NotchSpec {
        notch_hz: 25.0,
        zeta_zero: 0.01,
        zeta_pole: 0.25,
        cutoff_hz: 40.0,
    });

    let dt = 0.005; // 200 Hz
    let tf_tustin = tf_cont.to_discrete_tustin(dt, None);
    let tf_zoh = tf_cont
        .to_discrete_zoh::<3>(dt)
        .map_err(|e| format!("ZOH conversion failed: {e:?}"))?;

    let num_freqs = 200_usize;
    let last = index_f64(num_freqs.saturating_sub(1));
    let freqs_hz: Vec<f64> = (0..num_freqs)
        .map(|i| 0.1 + 99.4 * index_f64(i) / last)
        .collect();

    let mut out = BodeResponses {
        freqs_hz: Vec::new(),
        continuous: BodeTrace::default(),
        tustin: BodeTrace::default(),
        zoh: BodeTrace::default(),
    };
    for &f_hz in &freqs_hz {
        let w = 2.0 * std::f64::consts::PI * f_hz;
        out.continuous.push(tf_cont.bode_point(w));
        out.tustin.push(tf_tustin.bode_point(w));
        out.zoh.push(tf_zoh.bode_point(w));
    }
    out.freqs_hz = freqs_hz;
    Ok(out)
}

fn compute_nyquist() -> NyquistResult {
    let tf_open = Tf::<2, 4>::continuous([100.0, 50.0], [0.0, 25.0, 2.0, 1.0]);
    let n = 200_usize;
    let last = index_f64(n.saturating_sub(1));
    let freqs: Vec<f64> = (0..n)
        .map(|i| 10.0_f64.powf(-2.0 + 5.0 * index_f64(i) / last))
        .collect();

    let mut h_re = Vec::with_capacity(n);
    let mut h_im = Vec::with_capacity(n);
    let mut gain_crossover_w = 0.0;
    let mut phase_crossover_w = 0.0;
    let mut min_mag_diff = f64::MAX;
    let mut min_im_diff = f64::MAX;

    for &w in &freqs {
        let resp = tf_open.eval_frequency(w);
        h_re.push(resp.re);
        h_im.push(resp.im);

        let mag = resp.re.hypot(resp.im);
        if (mag - 1.0).abs() < min_mag_diff {
            min_mag_diff = (mag - 1.0).abs();
            gain_crossover_w = w;
        }

        if resp.im.abs() < min_im_diff && resp.re < 0.0 {
            min_im_diff = resp.im.abs();
            phase_crossover_w = w;
        }
    }

    let gc_resp = tf_open.eval_frequency(gain_crossover_w);
    let phase_margin_rad = std::f64::consts::PI + gc_resp.im.atan2(gc_resp.re);

    let pc_resp = tf_open.eval_frequency(phase_crossover_w);
    let pc_mag = pc_resp.re.hypot(pc_resp.im);

    NyquistResult {
        h_re,
        h_im,
        phase_margin_deg: phase_margin_rad.to_degrees(),
        gain_margin_db: -20.0 * pc_mag.log10(),
    }
}

/// Controllable canonical form of H(s) = (2s + 3) / (s^2 + 5s + 4), row-major `(A, B, C)`.
fn compute_ccf_realization() -> KernelResult<CcfRealization> {
    let tf = Tf::<2, 3>::continuous([3.0, 2.0], [4.0, 5.0, 1.0]);
    let ss = tf
        .to_controllable_canonical_form::<2>()
        .map_err(|e| format!("CCF realization failed: {e:?}"))?;
    Ok(CcfRealization {
        a: row_major(&ss.a())?,
        b: column(&ss.b(), 0)?,
        c: row_major(&ss.c())?,
    })
}

/// Executes the transfer function control-rs-verification kernel and writes `target/verification/transfer_function.rust.h5`.
///
/// # Errors
///
/// Returns an error if a discretization or realization fails or the
/// container cannot be written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
    let mut writer = H5Writer::new();

    let bode = compute_bode_responses()?;
    for (name, data, tol) in [
        ("bode/freqs_hz", &bode.freqs_hz, 1e-6),
        ("bode/cont_mag_db", &bode.continuous.mag_db, 0.1),
        ("bode/cont_phase_deg", &bode.continuous.phase_deg, 0.5),
        ("bode/tustin_mag_db", &bode.tustin.mag_db, 0.1),
        ("bode/tustin_phase_deg", &bode.tustin.phase_deg, 0.5),
        ("bode/zoh_mag_db", &bode.zoh.mag_db, 0.2),
        ("bode/zoh_phase_deg", &bode.zoh.phase_deg, 1.0),
    ] {
        writer.add_dataset(name, data);
        writer.set_tolerance(name, "abs", tol);
    }

    let nyquist = compute_nyquist();
    writer.add_dataset("nyquist/h_re", &nyquist.h_re);
    writer.set_tolerance("nyquist/h_re", "abs", 0.05);

    writer.add_dataset("nyquist/h_im", &nyquist.h_im);
    writer.set_tolerance("nyquist/h_im", "abs", 0.05);

    writer.add_dataset("nyquist/phase_margin_deg", &[nyquist.phase_margin_deg]);
    writer.set_tolerance("nyquist/phase_margin_deg", "abs", 1.0);

    writer.add_dataset("nyquist/gain_margin_db", &[nyquist.gain_margin_db]);
    writer.set_tolerance("nyquist/gain_margin_db", "abs", 1.0);

    let ccf = compute_ccf_realization()?;
    for (name, data) in [
        ("realization/ccf_a", &ccf.a),
        ("realization/ccf_b", &ccf.b),
        ("realization/ccf_c", &ccf.c),
    ] {
        writer.add_dataset(name, data);
        writer.set_tolerance(name, "abs", 1e-12);
    }

    writer.write_to_file(output_path)
}
