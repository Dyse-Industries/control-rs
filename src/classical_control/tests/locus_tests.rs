//! Unit tests for `locus`.

use super::*;

const U: f64 = f64::EPSILON / 2.0;

/// `L(s) = (s + 2) / ((s + 1)(s + 3)(s + 5))`.
fn plant() -> ArrayTransferFunction<f64, 2, 4> {
    ArrayTransferFunction::continuous([2.0, 1.0], [15.0, 23.0, 9.0, 1.0])
}

#[test]
fn locus_roots_satisfy_characteristic() {
    let gains: [f64; 20] =
        core::array::from_fn(|k| 2.5 * f64::from(u8::try_from(k).unwrap()));
    let rows = root_locus(&plant(), &gains).unwrap();
    for (row, &k) in rows.iter().zip(&gains) {
        let c = [15.0 + 2.0 * k, 23.0 + k, 9.0, 1.0];
        let poly = ArrayPolynomial::<f64, 4>::from_coefficients(c);
        for &p in &row[..3] {
            let residual = poly.evaluate_complex(p).magnitude();
            let scale: f64 = c
                .iter()
                .zip(0..)
                .map(|(cj, j)| cj.abs() * p.magnitude().powi(j))
                .sum();
            assert!(residual / scale <= 10.0 * 3.0 * U, "k={k} r={residual}");
        }
    }
}

#[test]
fn locus_zero_gain() {
    let rows = root_locus(&plant(), &[0.0]).unwrap();
    let poles = plant().poles().unwrap();
    // Every open-loop pole is a locus point at k = 0.
    for q in &poles[..3] {
        let nearest = rows[0][..3]
            .iter()
            .map(|p| (*p - *q).magnitude())
            .fold(f64::INFINITY, f64::min);
        assert!(nearest <= 10.0 * 3.0 * U);
    }
}
