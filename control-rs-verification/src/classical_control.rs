//! Classical control toolbox control-rs-verification suite.
//!
//! Emits the cross-check cases of `classical-control-design.md`:
//! 1. `DirectForm2T` output sequences of orders 1 to 4 (`df2t_order<n>/output`)
//! 2. `Df1` and `Df2t` cascade output sequences (`cascade_<form>/output`)
//! 3. Responses of factored cascades of orders 1 to 8, normalized by the
//!    peak source response (`sections_order<n>/re`, `/im`)
//! 4. Gain and phase margins with their crossovers (`margins_<case>/<margin>`)
//! 5. `Df1<Fixed>` cascade outputs at two scales (`df1_q<shift>/fixed`)

#![allow(missing_docs)]

use std::path::Path;

use control_rs::classical_control::{
    BiquadCascade, Df1, Df2t, DirectForm2T, SectionCoefficients, quantize,
    stability_margins, to_sections,
};
use control_rs::math::complex_num::Complex;
use control_rs::math::fixed_num::Fixed;
use control_rs::math::num_types::{Const, Dim, DimMax};
use control_rs::transfer_function::ArrayTransferFunction;

use crate::h5_writer::H5Writer;
use crate::numeric::{KernelResult, index_f64};

/// Frequencies per factored-cascade response.
const FREQS: usize = 64;

/// Samples per output sequence.
const SAMPLES: usize = 256;

type Coeffs = SectionCoefficients<f64>;

/// Conjugate pairs `(rho, theta)`.
type Pairs = [(f64, f64)];

/// Real and imaginary parts of a response.
type Parts = KernelResult<(Vec<f64>, Vec<f64>)>;

/// Gain margin, phase crossover, phase margin and gain crossover.
type MarginRow = KernelResult<[f64; 4]>;

/// Canonical type-level encoding of `Const<N>`.
type TypeNum<const N: usize> = <Const<N> as Dim>::TypeNum;

/// Shared excitation: `0.5 + 0.3 sin(0.2 k) + 0.2 cos(1.3 k)`, scaled.
fn excitation(scale: f64) -> Vec<f64> {
    (0..SAMPLES)
        .map(|k| {
            let t = index_f64(k);
            scale
                * 0.2f64.mul_add(
                    (1.3 * t).cos(),
                    0.3f64.mul_add((0.2 * t).sin(), 0.5),
                )
        })
        .collect()
}

/// Ascending coefficients of `prod (z - r)` over real roots and conjugate
/// pairs `rho e^(+/- j theta)`.
fn poly(real: &[f64], pairs: &Pairs) -> Vec<f64> {
    let factors = real.iter().map(|r| vec![-r, 1.0]).chain(
        pairs
            .iter()
            .map(|(rho, th)| vec![rho * rho, -2.0 * rho * th.cos(), 1.0]),
    );
    factors.fold(vec![1.0], |acc, f| {
        let mut out =
            vec![0.0; acc.len().saturating_add(f.len()).saturating_sub(1)];
        for (i, a) in acc.iter().enumerate() {
            for (j, b) in f.iter().enumerate() {
                if let Some(o) = out.get_mut(i.saturating_add(j)) {
                    *o += a * b;
                }
            }
        }
        out
    })
}

/// Copies `src` into a zero-padded array of capacity `M`.
fn pad<const M: usize>(src: &[f64]) -> [f64; M] {
    std::array::from_fn(|i| src.get(i).copied().unwrap_or(0.0))
}

/// `DirectForm2T` output of a discrete `(num, den)` of order `ORDER`,
/// with `D = ORDER + 1` coefficients each (ascending powers of `z`).
fn df2t_case<const ORDER: usize, const D: usize>(
    num: &[f64],
    den: &[f64],
) -> Vec<f64> {
    let lead = pad::<D>(den).last().copied().unwrap_or(1.0);
    let desc = |c: &[f64]| -> Vec<f64> {
        pad::<D>(c).iter().rev().map(|v| v / lead).collect()
    };
    let (b, a) = (desc(num), desc(den));
    let tail = |c: &[f64]| pad::<ORDER>(c.get(1..).unwrap_or(&[]));
    let b0 = b.first().copied().unwrap_or(0.0);
    let mut f = DirectForm2T::<f64, ORDER>::new(b0, tail(&b), tail(&a));
    excitation(1.0).into_iter().map(|u| f.update(u)).collect()
}

/// Section with poles `rho e^(+/- j theta)`, unit-circle zeros at
/// `+/- phi` and gain `g`.
fn biquad(rho: f64, theta: f64, phi: f64, g: f64) -> Coeffs {
    SectionCoefficients {
        b0: g,
        b1: -2.0 * g * phi.cos(),
        b2: g,
        a1: -2.0 * rho * theta.cos(),
        a2: rho * rho,
    }
}

/// The three-section cascade of the cascade and fixed-point cases.
fn cascade_case() -> [Coeffs; 3] {
    [
        biquad(0.9, 0.4, 1.5, 0.2),
        biquad(0.8, 1.1, 2.2, 0.2),
        biquad(0.6, 2.0, 2.8, 0.2),
    ]
}

/// Response of a factored cascade at `FREQS` frequencies, normalized by the
/// peak source response, as `(re, im)`.
fn sections_case<const D: usize, const L: usize>(
    num: &[f64],
    den: &[f64],
) -> Parts
where
    Const<D>: Dim,
    TypeNum<D>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    let tf =
        ArrayTransferFunction::<f64, D, D>::discrete(pad(num), pad(den), 1.0);
    let sections = to_sections::<f64, D, D, L>(&tf).map_err(|e| {
        format!("factorization of {D} coefficients failed: {e}")
    })?;
    let w: Vec<f64> = (0..FREQS)
        .map(|k| 0.049f64.mul_add(index_f64(k), 0.01))
        .collect();
    let peak = w
        .iter()
        .map(|wk| tf.eval_frequency(*wk).magnitude())
        .fold(0.0, f64::max);
    let resp: Vec<_> = w
        .iter()
        .map(|wk| cascade_response(&sections, *wk))
        .collect();
    Ok((
        resp.iter().map(|z| z.re / peak).collect(),
        resp.iter().map(|z| z.im / peak).collect(),
    ))
}

/// Product of section responses at `q = e^(-j w)`.
fn cascade_response(sections: &[Coeffs], w: f64) -> Complex<f64> {
    let (q_re, q_im) = (w.cos(), -w.sin());
    let eval = |c0: f64, c1: f64, c2: f64| {
        // Horner in q: c0 + q (c1 + q c2).
        let (in_re, in_im) = (q_re.mul_add(c2, c1), q_im * c2);
        (
            q_re.mul_add(in_re, -(q_im * in_im)) + c0,
            q_re.mul_add(in_im, q_im * in_re),
        )
    };
    sections.iter().fold(Complex::new(1.0, 0.0), |acc, sec| {
        let (num_re, num_im) = eval(sec.b0, sec.b1, sec.b2);
        let (den_re, den_im) = eval(1.0, sec.a1, sec.a2);
        let norm = den_re.mul_add(den_re, den_im * den_im);
        let ratio_re = num_re.mul_add(den_re, num_im * den_im) / norm;
        let ratio_im = num_im.mul_add(den_re, -(num_re * den_im)) / norm;
        Complex::new(
            acc.re.mul_add(ratio_re, -(acc.im * ratio_im)),
            acc.re.mul_add(ratio_im, acc.im * ratio_re),
        )
    })
}

/// `Df1<Fixed<i16, 13>>` and `Df1<Fixed<i32, 29>>` cascade outputs as `f64`.
fn fixed_cases() -> Parts {
    let q13 = quantize::<i16, 13, 3>(&cascade_case())
        .map_err(|e| format!("Q13 quantization failed: {e}"))?;
    let q29 = quantize::<i32, 29, 3>(&cascade_case())
        .map_err(|e| format!("Q29 quantization failed: {e}"))?;
    let mut c13 =
        BiquadCascade::<Df1<Fixed<i16, 13>>, 3>::from_coefficients(&q13);
    let mut c29 =
        BiquadCascade::<Df1<Fixed<i32, 29>>, 3>::from_coefficients(&q29);
    let u = excitation(0.25);
    Ok((
        u.iter()
            .map(|x| c13.update(Fixed::<i16, 13>::from_num(*x)).to_num())
            .collect(),
        u.iter()
            .map(|x| c29.update(Fixed::<i32, 29>::from_num(*x)).to_num())
            .collect(),
    ))
}

/// Margins of a continuous loop over `0.01 + 0.05 k`, `k < M`.
fn margins_case<const N: usize, const D: usize, const M: usize>(
    num: [f64; N],
    den: [f64; D],
) -> MarginRow
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let sys = ArrayTransferFunction::<f64, N, D>::continuous(num, den);
    let w: [f64; M] =
        std::array::from_fn(|k| 0.05f64.mul_add(index_f64(k), 0.01));
    let m = stability_margins::<f64, N, D, M, 1, 60>(&sys, &w);
    let [Some(gc)] = m.gain_crossings else {
        return Err("no gain crossover".to_string());
    };
    let (gm, w_pc) = m.phase_crossings[0]
        .map_or((f64::INFINITY, f64::NAN), |p| (p.gain_margin, p.omega));
    Ok([gm, w_pc, gc.phase_margin_deg, gc.omega])
}

/// Adds the four margin datasets of one case.
fn add_margins(writer: &mut H5Writer, case: &str, m: [f64; 4]) {
    let [gain_margin, phase_cross, phase_margin, gain_cross] = m;
    if gain_margin.is_finite() {
        writer.add_dataset(&format!("{case}/gain_margin"), &[gain_margin]);
        writer.add_dataset(&format!("{case}/w_pc"), &[phase_cross]);
    }
    writer.add_dataset(&format!("{case}/phase_margin"), &[phase_margin]);
    writer.add_dataset(&format!("{case}/w_gc"), &[gain_cross]);
}

/// Adds the factored-cascade response datasets of order `D - 1`.
fn add_sections<const D: usize, const L: usize>(
    writer: &mut H5Writer,
    num: &[f64],
    den: &[f64],
) -> KernelResult<()>
where
    Const<D>: Dim,
    TypeNum<D>: DimMax<TypeNum<D>, Output = TypeNum<D>>,
{
    let (re, im) = sections_case::<D, L>(num, den)?;
    let order = D.saturating_sub(1);
    writer.add_dataset(&format!("sections_order{order}/re"), &re);
    writer.add_dataset(&format!("sections_order{order}/im"), &im);
    Ok(())
}

/// Adds the `DirectForm2T` and cascade output datasets.
fn add_outputs(writer: &mut H5Writer) {
    let doubled: Vec<f64> = poly(&[0.8], &[]).iter().map(|v| 2.0 * v).collect();
    for (name, data) in [
        (
            "df2t_order1/output",
            df2t_case::<1, 2>(&poly(&[-0.5], &[]), &doubled),
        ),
        (
            "df2t_order2/output",
            df2t_case::<2, 3>(
                &poly(&[], &[(1.0, 1.0)]),
                &poly(&[], &[(0.95, 0.3)]),
            ),
        ),
        (
            "df2t_order3/output",
            df2t_case::<3, 4>(
                &poly(&[-1.0, 0.5, 0.1], &[]),
                &poly(&[0.7, 0.2, -0.5], &[]),
            ),
        ),
        (
            "df2t_order4/output",
            df2t_case::<4, 5>(
                &poly(&[-1.0, 0.2], &[(1.0, 2.0)]),
                &poly(&[0.7, -0.4], &[(0.9, 0.6)]),
            ),
        ),
    ] {
        writer.add_dataset(name, &data);
    }
    let u = excitation(1.0);
    let mut df1 =
        BiquadCascade::<Df1<f64>, 3>::from_coefficients(&cascade_case());
    let mut df2t =
        BiquadCascade::<Df2t<f64>, 3>::from_coefficients(&cascade_case());
    let out1: Vec<f64> = u.iter().map(|x| df1.update(*x)).collect();
    let out2: Vec<f64> = u.iter().map(|x| df2t.update(*x)).collect();
    writer.add_dataset("cascade_df1/output", &out1);
    writer.add_dataset("cascade_df2t/output", &out2);
}

/// Adds the factored-cascade responses of orders 1 to 8.
fn add_all_sections(writer: &mut H5Writer) -> KernelResult<()> {
    let p = [(0.95, 0.3), (0.8, 1.0), (0.6, 2.0), (0.5, 2.7)];
    let z = [(1.0, 0.9), (1.0, 1.7), (0.9, 2.5), (1.2, 0.2)];
    let pp = |n: usize| p.get(..n).unwrap_or(&[]);
    let zz = |n: usize| z.get(..n).unwrap_or(&[]);
    add_sections::<2, 1>(writer, &poly(&[-1.0], &[]), &poly(&[0.7], &[]))?;
    add_sections::<3, 1>(writer, &poly(&[], zz(1)), &poly(&[], pp(1)))?;
    add_sections::<4, 2>(writer, &poly(&[], zz(1)), &poly(&[0.7], pp(1)))?;
    add_sections::<5, 2>(
        writer,
        &poly(&[0.0, -1.0], zz(1)),
        &poly(&[], pp(2)),
    )?;
    add_sections::<6, 3>(writer, &poly(&[-1.0], zz(2)), &poly(&[-0.3], pp(2)))?;
    add_sections::<7, 3>(writer, &poly(&[0.5], zz(2)), &poly(&[], pp(3)))?;
    add_sections::<8, 4>(
        writer,
        &poly(&[-1.0, 0.4], zz(2)),
        &poly(&[0.2], pp(3)),
    )?;
    add_sections::<9, 4>(writer, &poly(&[], zz(4)), &poly(&[], pp(4)))
}

/// Executes the classical control kernels and writes
/// `target/verification/classical_control.rust.h5`.
///
/// # Errors
///
/// Returns an error if a factorization, quantization or margin case fails
/// or the container cannot be written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
    let mut writer = H5Writer::new();
    add_outputs(&mut writer);
    add_all_sections(&mut writer)?;
    let third = margins_case::<1, 4, 61>([4.0], [1.0, 3.0, 3.0, 1.0])?;
    add_margins(&mut writer, "margins_third_order", third);
    let integrator = margins_case::<1, 3, 201>([10.0], [0.0, 1.0, 1.0])?;
    add_margins(&mut writer, "margins_integrator", integrator);
    let (f13, f29) = fixed_cases()?;
    writer.add_dataset("df1_q13/fixed", &f13);
    writer.add_dataset("df1_q29/fixed", &f29);
    writer.write_to_file(output_path)
}
