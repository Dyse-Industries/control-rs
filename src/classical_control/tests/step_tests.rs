//! Unit tests for `step`.

use super::*;
use crate::classical_control::step::*;

/// Sample period; coarser under Miri, where the interpreter dominates.
const H: f64 = if cfg!(miri) { 0.004 } else { 0.001 };
/// Samples over `[0, 8]`.
const LEN: usize = if cfg!(miri) { 2001 } else { 8001 };
const WN: f64 = 2.0;
const ZETA: f64 = 0.3;

/// Sample times and values.
type Trace = ([f64; LEN], [f64; LEN]);

fn response() -> Trace {
    let wd = WN * (1.0 - ZETA * ZETA).sqrt();
    let phi = (1.0 - ZETA * ZETA).sqrt().atan2(ZETA);
    let t: [f64; LEN] =
        core::array::from_fn(|k| H * f64::from(u16::try_from(k).unwrap()));
    let y = t.map(|tk| {
        1.0 - (-ZETA * WN * tk).exp() * (wd * tk + phi).sin()
            / (1.0 - ZETA * ZETA).sqrt()
    });
    (t, y)
}

fn settle_time(theta: f64, t: &[f64], y: &[f64]) -> f64 {
    let last_out = y.iter().rposition(|v| (v - 1.0).abs() > theta).unwrap();
    *t.get(last_out.saturating_add(1)).unwrap()
}

#[test]
fn second_order_closed_form() {
    let (t, y) = response();
    let opts = StepOptions {
        y_final: Some(1.0),
        ..StepOptions::default()
    };
    let info = step_info(&t, &y, &opts).unwrap();
    let wd = WN * (1.0 - ZETA * ZETA).sqrt();
    let tp = core::f64::consts::PI / wd;
    let os = 100.0
        * (-ZETA * core::f64::consts::PI / (1.0 - ZETA * ZETA).sqrt()).exp();
    assert!((info.peak_time - tp).abs() <= H);
    let ddy = WN * WN * (os / 100.0);
    assert!((info.overshoot - os).abs() <= 100.0 * ddy * H * H / 8.0 + 1e-9);
    assert!((info.peak - (1.0 + os / 100.0)).abs() <= ddy * H * H);
    // Rise and settling times against a fine-grid evaluation of the
    // closed form.
    let wd_t = |tk: f64| {
        let phi = (1.0 - ZETA * ZETA).sqrt().atan2(ZETA);
        1.0 - (-ZETA * WN * tk).exp() * (wd * tk + phi).sin()
            / (1.0 - ZETA * ZETA).sqrt()
    };
    let cross = |level: f64| {
        let mut lo = 0.0;
        let mut hi = tp;
        for _ in 0..60 {
            let mid = f64::midpoint(lo, hi);
            if wd_t(mid) < level {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        lo
    };
    let rise = cross(0.9) - cross(0.1);
    assert!((info.rise_time.unwrap() - rise).abs() <= H);
    let st = info.settling_time.unwrap();
    assert!((st - settle_time(0.02, &t, &y)).abs() <= H);
}

#[test]
fn defaults_and_threshold() {
    let (t, y) = response();
    let default = step_info(&t, &y, &StepOptions::default()).unwrap();
    let explicit = StepOptions {
        y_final: y.last().copied(),
        ..StepOptions::default()
    };
    assert_eq!(default, step_info(&t, &y, &explicit).unwrap());
    let five = StepOptions {
        y_final: Some(1.0),
        threshold: 0.05,
        ..StepOptions::default()
    };
    let info = step_info(&t, &y, &five).unwrap();
    // Envelope bound of the 5% settling time.
    let ts_env = -(0.05 * (1.0 - ZETA * ZETA).sqrt()).ln() / (ZETA * WN);
    let st = info.settling_time.unwrap();
    assert!(st <= ts_env + H);
    assert!((st - settle_time(0.05, &t, &y)).abs() <= H);
    let flat = [1.0; 4];
    assert_eq!(
        step_info(
            &[0.0, 1.0, 2.0, 3.0],
            &flat,
            &StepOptions {
                y_init: 1.0,
                ..StepOptions::default()
            }
        ),
        Err(ClassicalError::InvalidParameter)
    );
}

#[test]
fn threshold_bounds() {
    let t = [0.0, 1.0, 2.0];
    let y = [0.0, 0.5, 1.0];
    let with = |threshold| StepOptions {
        threshold,
        ..StepOptions::default()
    };
    assert!(step_info(&t, &y, &with(0.0)).is_ok());
    assert_eq!(
        step_info(&t, &y, &with(-0.1)),
        Err(ClassicalError::InvalidParameter)
    );
}

#[test]
fn sample_on_band_edge_is_inside() {
    // |1.5 - 1.0| equals the band exactly.
    let got = settling(&[0.0, 1.0, 2.0], &[1.5, 1.0, 1.0], 1.0, 0.5);
    assert_eq!(got, Some(0.0));
}

#[test]
fn equal_deviation_keeps_first_peak_time() {
    let (peak, time, max_norm) =
        peak_of(&[0.0, 1.0, 2.0], &[0.0, 1.0, 1.0], 0.0, |v| v);
    assert_eq!((peak, time, max_norm), (1.0, 1.0, 1.0));
}
