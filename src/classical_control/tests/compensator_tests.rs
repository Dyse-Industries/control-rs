//! Unit tests for `compensator`.

use super::*;
use crate::classical_control::compensator::*;

#[test]
fn pid_forms() {
    let (kp, ti, td, tf) = (2.0, 0.5, 0.25, 0.01);
    let std =
        pid::<f64, 3, 3>(PidForm::Standard { kp, ti, td }, Some(tf)).unwrap();
    assert_eq!(std.num_slice(), [kp, kp * (ti + tf), kp * (ti * (tf + td))]);
    assert_eq!(std.den_slice(), [0.0, ti, ti * tf]);

    let (kc, tau_i, tau_d) = (3.0, 0.4, 0.1);
    let ser = pid::<f64, 3, 3>(PidForm::Series { kc, tau_i, tau_d }, Some(tf))
        .unwrap();
    assert_eq!(
        ser.num_slice(),
        [kc, kc * (tau_i + (tau_d + tf)), kc * (tau_i * (tau_d + tf))]
    );
    assert_eq!(ser.den_slice(), [0.0, tau_i, tau_i * tf]);

    let (ki, kd) = (4.0, 0.3);
    let par =
        pid::<f64, 3, 3>(PidForm::Parallel { kp, ki, kd }, Some(tf)).unwrap();
    assert_eq!(par.num_slice(), [ki, kp + ki * tf, kp * tf + kd]);
    assert_eq!(par.den_slice(), [0.0, 1.0, tf]);
    assert!(par.is_continuous());
}

#[test]
fn pid_filtered_proper() {
    let form = PidForm::Parallel {
        kp: 1.0,
        ki: 1.0,
        kd: 1.0,
    };
    let filtered = pid::<f64, 3, 3>(form, Some(0.1)).unwrap();
    assert!(filtered.num_slice().len() <= filtered.den_slice().len());
    assert_eq!(pid::<f64, 3, 3>(form, None), Err(ClassicalError::Improper));
    let pi = PidForm::Parallel {
        kp: 2.0,
        ki: 3.0,
        kd: 0.0,
    };
    let tf = pid::<f64, 2, 2>(pi, None).unwrap();
    assert_eq!(tf.num_slice(), [3.0, 2.0]);
    assert_eq!(tf.den_slice(), [0.0, 1.0]);
    assert_eq!(
        pid::<f64, 3, 3>(form, Some(0.0)),
        Err(ClassicalError::InvalidParameter)
    );
}

#[test]
fn lead_lag() {
    let (gain, corner, ratio) = (2.0, 3.0, 10.0);
    let ld = lead(gain, corner, ratio).unwrap();
    assert_eq!(ld.num_slice(), [gain, gain / corner]);
    assert_eq!(ld.den_slice(), [1.0, 1.0 / (corner * ratio)]);
    let (lag_gain, lag_corner) = (5.0, 0.2);
    let lg = lag(lag_gain, lag_corner).unwrap();
    assert_eq!(lg.num_slice(), [lag_gain, lag_gain / lag_corner]);
    assert_eq!(lg.den_slice(), [1.0, lag_gain / lag_corner]);
    let bad = Err(ClassicalError::InvalidParameter);
    assert_eq!(lead(gain, 0.0, ratio), bad);
    assert_eq!(lead(gain, corner, -1.0), bad);
    assert_eq!(lag(0.0, lag_corner), bad);
    assert_eq!(lag(lag_gain, -lag_corner), bad);
    assert_eq!(lead(gain, corner, 0.0), bad);
    assert_eq!(lag(lag_gain, 0.0), bad);
}

#[test]
fn pid_time_bounds() {
    let bad = Err(ClassicalError::InvalidParameter);
    let std = |ti, td| PidForm::Standard { kp: 1.0, ti, td };
    assert_eq!(pid::<f64, 3, 3>(std(0.0, 0.25), Some(0.1)), bad);
    assert_eq!(pid::<f64, 3, 3>(std(0.5, -0.25), Some(0.1)), bad);
    let ser = PidForm::Series {
        kc: 1.0,
        tau_i: 0.0,
        tau_d: 0.1,
    };
    assert_eq!(pid::<f64, 3, 3>(ser, Some(0.1)), bad);
}

#[test]
fn pid_fit_conditions() {
    let form = PidForm::Parallel {
        kp: 1.0,
        ki: 1.0,
        kd: 1.0,
    };
    let improper = ClassicalError::Improper;
    assert_eq!(pid::<f64, 3, 4>(form, Some(0.1)).unwrap_err(), improper);
    assert_eq!(pid::<f64, 2, 3>(form, Some(0.1)).unwrap_err(), improper);
    assert_eq!(pid::<f64, 4, 3>(form, Some(0.1)).unwrap_err(), improper);
    let low = PidForm::Parallel {
        kp: 0.0,
        ki: 1.0,
        kd: 0.0,
    };
    assert!(pid::<f64, 3, 3>(low, Some(0.1)).is_ok());
}
