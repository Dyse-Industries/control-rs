//! Unit tests for `response`.

use super::*;
use crate::classical_control::response::*;

const U: f64 = f64::EPSILON / 2.0;

fn grid() -> [f64; 41] {
    core::array::from_fn(|k| {
        10f64.powf(-2.0 + 0.1 * f64::from(u8::try_from(k).unwrap()))
    })
}

#[test]
fn first_second_order_closed_form() {
    let w = grid();
    let first =
        ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [2.0, 1.0]);
    for p in frequency_response(&first, &w) {
        let mag = 2.0 / (p.omega * p.omega + 4.0).sqrt();
        let phase = -(p.omega / 2.0).atan().to_degrees();
        assert!(
            (10f64.powf(p.magnitude_db / 20.0) - mag).abs()
                <= 10.0 * U * mag * 4.0
        );
        assert!((p.phase_deg - phase).abs() <= 1e-9);
    }
    let (wn, zeta) = (3.0, 0.2);
    let second = ArrayTransferFunction::<f64, 1, 3>::continuous(
        [wn * wn],
        [wn * wn, 2.0 * zeta * wn, 1.0],
    );
    for p in frequency_response(&second, &w) {
        let re = wn * wn - p.omega * p.omega;
        let im = 2.0 * zeta * wn * p.omega;
        let mag = wn * wn / re.hypot(im);
        let phase = -im.atan2(re).to_degrees();
        assert!(
            (10f64.powf(p.magnitude_db / 20.0) - mag).abs()
                <= 10.0 * U * mag * 4.0
        );
        assert!((p.phase_deg - phase).abs() <= 1e-9);
    }
}

#[test]
fn discrete_unit_circle() {
    let ts = 0.1;
    let sys =
        ArrayTransferFunction::<f64, 1, 2>::discrete([0.5], [-0.5, 1.0], ts);
    for p in frequency_response(&sys, &grid()) {
        let theta = p.omega * ts;
        let z = Complex::new(theta.cos(), theta.sin());
        let expected = Complex::new(0.5, 0.0) / (z - Complex::new(0.5, 0.0));
        let err = (p.value - expected).magnitude();
        assert!(err <= 10.0 * U * expected.magnitude() * 4.0);
    }
}

#[test]
fn phase_unwrapped() {
    let sys = ArrayTransferFunction::<f64, 1, 5>::continuous(
        [1.0],
        [1.0, 4.0, 6.0, 4.0, 1.0],
    );
    let pts = frequency_response(&sys, &grid());
    for pair in pts.windows(2) {
        if let [a, b] = pair {
            assert!((b.phase_deg - a.phase_deg).abs() < 180.0);
        }
    }
    let last = pts.last().unwrap().phase_deg;
    assert!(last < -300.0);
}

#[test]
fn unwrap_threshold() {
    let mut up = PhaseUnwrap::<f64>::new();
    up.next(0.0);
    assert!((up.next(180.0) - 180.0).abs() < 1e-12);
    let mut down = PhaseUnwrap::<f64>::new();
    down.next(0.0);
    assert!((down.next(-180.0) + 180.0).abs() < 1e-12);
    let mut wrap = PhaseUnwrap::<f64>::new();
    wrap.next(100.0);
    assert!((wrap.next(-100.0) - 260.0).abs() < 1e-12);
}
