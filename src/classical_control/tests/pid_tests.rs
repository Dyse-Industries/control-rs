//! Unit tests for `pid`.

use super::*;
use crate::classical_control::pid::*;

use crate::math::num_traits::Trig;

/// Parallel gains with a filter, wide limits and the given mode.
fn params(anti_windup: AntiWindup<f64>, limit: f64) -> PidParams<f64> {
    PidParams {
        kp: 2.0,
        ki: 1.5,
        kd: 0.2,
        tf: 0.05,
        h: 0.01,
        u_min: -limit,
        u_max: limit,
        anti_windup,
    }
}

/// Unsaturated output `v` recomputed from the last sample.
fn unsaturated(c: &Pid<f64>) -> f64 {
    c.params.kp * c.e_last + c.integral + c.params.kd * c.delta
}

#[test]
/// Within limits, `f32` outputs follow the §4.10 recurrence evaluated
/// in `f64` with relative error at most `10 k u` at sample `k` (FR-11).
fn unsaturated_matches_recurrence() {
    let p32 = PidParams {
        kp: 2.0f32,
        ki: 1.5,
        kd: 0.2,
        tf: 0.05,
        h: 0.01,
        u_min: -1e6,
        u_max: 1e6,
        anti_windup: AntiWindup::Clamping,
    };
    let mut ctl = Pid::new(p32).unwrap();
    let (kp, ki, kd, tf, period) = (
        2.0f64,
        f64::from(1.5f32),
        f64::from(0.2f32),
        f64::from(0.05f32),
        f64::from(0.01f32),
    );
    let (ad, bd) = (tf / (tf + period), 1.0 / (tf + period));
    let (mut integ, mut filt, mut y_prev) = (0.0f64, 0.0f64, None::<f64>);
    let u32_round = f64::from(f32::EPSILON) / 2.0;
    for step in 1..=200u16 {
        let time = f32::from(step) * 0.01;
        let (sp, meas) = (10.0f32, 0.5 * Trig::sin(3.0 * time));
        let (sp64, meas64) = (f64::from(sp), f64::from(meas));
        let err = sp64 - meas64;
        filt = ad * filt - y_prev.map_or(0.0, |yp| meas64 - yp) * bd;
        let raw = kp * err + integ + kd * filt;
        integ += ki * period * err;
        y_prev = Some(meas64);
        let out = f64::from(ctl.step(sp, meas));
        let bound = 10.0 * f64::from(step) * u32_round * raw.abs();
        assert!((out - raw).abs() <= bound);
    }
}

#[test]
/// A setpoint step at constant measurement leaves the derivative state
/// unchanged exactly (FR-11).
fn setpoint_step_no_derivative_kick() {
    let mut a = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
    let mut b = a;
    for k in 0..50u8 {
        let r = if k < 10 { 0.0 } else { 5.0 };
        a.step(r, 1.0);
        b.step(0.0, 1.0);
        assert_eq!(a.delta.to_bits(), b.delta.to_bits());
    }
}

#[test]
/// Outputs stay within the limits and invalid parameters are rejected
/// (FR-11).
fn output_within_limits() {
    let mut c = Pid::new(params(AntiWindup::Clamping, 0.5)).unwrap();
    for k in 0..100u8 {
        let r = if k % 20 < 10 { 10.0 } else { -10.0 };
        let u = c.step(r, 0.0);
        assert!((-0.5..=0.5).contains(&u));
    }
    let base = params(AntiWindup::Clamping, 1.0);
    for bad in [
        PidParams { u_min: 2.0, ..base },
        PidParams { h: 0.0, ..base },
        PidParams { tf: -1.0, ..base },
        PidParams {
            anti_windup: AntiWindup::BackCalculation { tt: 0.0 },
            ..base
        },
        PidParams {
            kp: f64::NAN,
            ..base
        },
        PidParams {
            ki: f64::INFINITY,
            ..base
        },
        PidParams {
            kd: f64::NEG_INFINITY,
            ..base
        },
        PidParams {
            u_max: f64::NAN,
            ..base
        },
        PidParams {
            h: f64::INFINITY,
            ..base
        },
        PidParams {
            tf: f64::INFINITY,
            ..base
        },
        PidParams {
            anti_windup: AntiWindup::BackCalculation { tt: f64::INFINITY },
            ..base
        },
    ] {
        assert_eq!(Pid::new(bad), Err(ClassicalError::InvalidParameter));
    }
}

#[test]
/// Under sustained saturation in `Clamping` the integrator is constant
/// exactly and integrates again once the error changes sign (FR-12).
fn clamping_holds_integrator() {
    let mut c = Pid::new(params(AntiWindup::Clamping, 1.0)).unwrap();
    c.step(0.4, 0.0);
    let held = c.integral;
    for _ in 0..100 {
        assert_eq!(c.step(10.0, 0.0).to_bits(), 1.0f64.to_bits());
        assert_eq!(c.integral.to_bits(), held.to_bits());
    }
    c.step(-0.1, 0.0);
    assert!(c.integral < held);
}

#[test]
/// Under sustained saturation in `BackCalculation` the integrator
/// converges to `u_lim - K_p e - K_d delta + T_t K_i e` (FR-12).
fn back_calculation_fixed_point() {
    let tt = 0.2;
    let p = params(AntiWindup::BackCalculation { tt }, 1.0);
    let mut c = Pid::new(p).unwrap();
    let e = 3.0;
    for _ in 0..400 {
        c.step(e, 0.0);
    }
    let fixed = p.u_max - p.kp * e - p.kd * c.delta + tt * p.ki * e;
    assert!(((c.integral - fixed) / fixed).abs() <= 1e-6);
}

#[test]
/// `set_gains` keeps the unsaturated output of the last sample within
/// `4u (|K_p e| + |I| + |K_d delta|)` (FR-13).
fn bumpless_gain_change() {
    let mut c = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
    for k in 0..30u8 {
        c.step(1.0, 0.1 * Trig::sin(f64::from(k)));
    }
    let v = unsaturated(&c);
    let scale = (c.params.kp * c.e_last).abs()
        + c.integral.abs()
        + (c.params.kd * c.delta).abs();
    c.set_gains(5.0, 0.7, 0.9);
    let v2 = unsaturated(&c);
    assert!((v2 - v).abs() <= 4.0 * (f64::EPSILON / 2.0) * scale);
}

#[test]
/// Each gain alone, and a change before the first sample, keeps the
/// recomputed unsaturated output (FR-13). `K_i` needs no correction
/// because `I` already holds `K_i h e` increments; the terms the
/// requirement excludes are the next sample's `Delta K_p (e' - e)`.
fn bumpless_each_gain_alone() {
    let changes = [(5.0, 0.5, 0.2), (2.0, 0.9, 0.2), (2.0, 0.5, 0.9)];
    for (kp, ki, kd) in changes {
        let mut c = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
        for k in 0..30u8 {
            c.step(1.0, 0.1 * Trig::sin(f64::from(k)));
        }
        let v = unsaturated(&c);
        let scale = (c.params.kp * c.e_last).abs()
            + c.integral.abs()
            + (c.params.kd * c.delta).abs();
        c.set_gains(kp, ki, kd);
        let v2 = unsaturated(&c);
        assert!((v2 - v).abs() <= 4.0 * (f64::EPSILON / 2.0) * scale);
    }
    let mut fresh = Pid::new(params(AntiWindup::Clamping, 1e6)).unwrap();
    let v = unsaturated(&fresh);
    fresh.set_gains(9.0, 0.1, 3.0);
    assert_eq!(unsaturated(&fresh).to_bits(), v.to_bits());
}

#[test]
/// `to_transfer_function` equals the parallel form with filter exactly
/// (FR-14).
fn analysis_model_matches_compensator() {
    let p = params(AntiWindup::Clamping, 1.0);
    let c = Pid::new(p).unwrap();
    let got = c.to_transfer_function::<3, 3>().unwrap();
    let form = PidForm::Parallel {
        kp: p.kp,
        ki: p.ki,
        kd: p.kd,
    };
    assert_eq!(got, pid::<f64, 3, 3>(form, Some(p.tf)).unwrap());
}

#[test]
/// A zero derivative filter with `K_d = 0` yields the unfiltered PI model
/// (FR-14).
fn analysis_model_without_filter() {
    let p = PidParams {
        kd: 0.0,
        tf: 0.0,
        ..params(AntiWindup::Clamping, 1.0)
    };
    let c = Pid::new(p).unwrap();
    let got = c.to_transfer_function::<2, 2>().unwrap();
    assert_eq!(got.num_slice(), [p.ki, p.kp]);
    assert_eq!(got.den_slice(), [0.0, 1.0]);
}

#[test]
/// `reset` returns the controller to its constructed state (FR-11).
fn reset_restores_constructed_state() {
    let p = params(AntiWindup::Clamping, 1.0);
    let fresh = Pid::new(p).unwrap();
    let mut c = fresh;
    for k in 0..20u8 {
        c.step(1.0, 0.1 * Trig::sin(f64::from(k)));
    }
    assert_ne!(c, fresh);
    c.reset();
    assert_eq!(c, fresh);
}

#[test]
/// Reverse-acting `Clamping` holds while `K_i e` drives further into
/// saturation (FR-12).
fn clamping_reverse_acting_holds() {
    let p = PidParams::<f64> {
        kp: -1.0,
        ki: -1.0,
        kd: 0.0,
        tf: 0.0,
        h: 0.1,
        u_min: -1.0,
        u_max: 1.0,
        anti_windup: AntiWindup::Clamping,
    };
    let mut rev = Pid::new(p).unwrap();
    let mut fwd = Pid::new(PidParams {
        kp: 1.0,
        ki: 1.0,
        ..p
    })
    .unwrap();
    for _ in 0..50 {
        assert_eq!(rev.step(10.0, 0.0).to_bits(), (-1.0f64).to_bits());
        assert_eq!(fwd.step(10.0, 0.0).to_bits(), 1.0f64.to_bits());
    }
    assert_eq!(fwd.integral.to_bits(), 0.0f64.to_bits());
    assert_eq!(rev.integral.to_bits(), 0.0f64.to_bits());
    assert_eq!(fwd.step(0.0, 0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(rev.step(0.0, 0.0).to_bits(), 0.0f64.to_bits());
}

#[test]
/// `Clamping` holds the integrator at the lower limit while the error is
/// negative, and unwinds it while the error is negative but the output
/// is pushed above the upper limit (FR-12).
fn clamping_negative_error() {
    let p = PidParams::<f64> {
        kp: 1.0,
        ki: 1.0,
        kd: 1.0,
        tf: 0.0,
        h: 0.01,
        u_min: -1.0,
        u_max: 1.0,
        anti_windup: AntiWindup::Clamping,
    };
    let mut c = Pid::new(p).unwrap();
    // e = -11, v = -11 < u_min: held.
    c.step(-1.0, 10.0);
    assert_eq!(c.integral.to_bits(), 0.0f64.to_bits());
    // e = -6, derivative term +500 pushes v above u_max: integrates.
    c.step(-1.0, 5.0);
    assert!(c.integral < 0.0);
}
