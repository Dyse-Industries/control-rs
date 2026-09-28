//! Boundary and dispatch tests for `polynomial/mod.rs`.

use crate::polynomial::{
    ArrayPolynomial, ConversionError, DivisionError, RootError,
};

fn assert_coeffs(got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len());
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 1e-12, "got {g} want {w}");
    }
}

#[test]
fn test_div_rem_leading_coefficient_threshold() {
    let num = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 2.0, 3.0]);
    let below =
        ArrayPolynomial::<f64, 2>::from_coefficients([1.0, f64::EPSILON / 2.0]);
    assert_eq!(
        num.div_rem::<2, 2, 1>(&below),
        Err(DivisionError::ZeroLeadingCoefficient)
    );
    let at = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, f64::EPSILON]);
    assert!(num.div_rem::<2, 2, 1>(&at).is_ok());
}

#[test]
fn test_div_rem_truncates_quotient_and_pads_remainder() {
    // (x^3 + 2x^2 + 3x + 4) / (x + 1): Q = x^2 + x + 2, R = 2.
    let num =
        ArrayPolynomial::<f64, 4>::from_coefficients([4.0, 3.0, 2.0, 1.0]);
    let den = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 1.0]);
    // Quotient capacity smaller than the full quotient degree + 1.
    let (quot, _) = num.div_rem::<2, 2, 1>(&den).unwrap();
    assert_coeffs(&quot.to_coefficients(), &[2.0, 1.0]);
    // Remainder capacity larger than the dividend capacity.
    let (quot, rem) = num.div_rem::<2, 3, 5>(&den).unwrap();
    assert_coeffs(&quot.to_coefficients(), &[2.0, 1.0, 1.0]);
    assert_coeffs(&rem.to_coefficients(), &[2.0, 0.0, 0.0, 0.0, 0.0]);
}

#[test]
fn test_companion_matrix_dimension_guard() {
    let p = ArrayPolynomial::<f64, 1>::from_coefficients([1.0]);
    assert_eq!(
        p.companion_matrix::<0>().err(),
        Some(ConversionError::DimensionMismatch)
    );
}

#[test]
fn test_companion_matrix_monic_tolerance_is_inclusive() {
    let eps = f64::EPSILON;
    let within =
        ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 1.0 + 2.0 * eps]);
    assert!(within.companion_matrix::<1>().is_ok());
    let outside =
        ArrayPolynomial::<f64, 2>::from_coefficients([3.0, 1.0 + 4.0 * eps]);
    assert_eq!(
        outside.companion_matrix::<1>().err(),
        Some(ConversionError::NonMonicPolynomial)
    );
}

#[test]
fn test_line_intercept_and_quadratic_size_guards() {
    let short = ArrayPolynomial::<f64, 1>::from_coefficients([1.0]);
    assert_eq!(short.line_intercept(), Err(RootError::DimensionMismatch));
    let linear = ArrayPolynomial::<f64, 2>::from_coefficients([1.0, 1.0]);
    assert_eq!(linear.quadratic_roots(), Err(RootError::DimensionMismatch));

    // Larger capacities use only the leading terms each solver needs.
    let cubic_cap =
        ArrayPolynomial::<f64, 4>::from_coefficients([6.0, 2.0, 0.0, 0.0]);
    assert_coeffs(&[cubic_cap.line_intercept().unwrap().re], &[-3.0]);
    let quartic_cap =
        ArrayPolynomial::<f64, 4>::from_coefficients([6.0, -5.0, 1.0, 0.0]);
    let mut re = quartic_cap.quadratic_roots().unwrap().map(|r| r.re);
    re.sort_by(f64::total_cmp);
    assert_coeffs(&re, &[2.0, 3.0]);
}

#[test]
fn test_aberth_seed_radius_uses_largest_coefficient_ratio() {
    // |c_i / lead| = 3, 8, 2: seeds lie on radius 1 + 8.
    let p = ArrayPolynomial::<f64, 4>::from_coefficients([3.0, -8.0, 2.0, 1.0]);
    let seeds = p.aberth_initial_seeds(3, 1.0);
    for seed in seeds.iter().take(3) {
        let r = (seed.re * seed.re + seed.im * seed.im).sqrt();
        assert!((r - 9.0).abs() < 1e-12, "radius {r}");
    }
}

#[test]
fn test_roots_dispatch_matches_dedicated_solvers() {
    let z1 = ArrayPolynomial::<f64, 1>::from_coefficients([0.0]);
    assert!(z1.roots().is_ok());

    // Iterative Aberth refinement lands one unit in the last place away for this input, so exact
    // equality proves the closed-form solver was used.
    let lin = ArrayPolynomial::<f64, 2>::from_coefficients([2.0, 3.0]);
    assert_eq!(
        lin.roots().unwrap().first().copied(),
        Some(lin.line_intercept().unwrap())
    );

    let quad = ArrayPolynomial::<f64, 3>::from_coefficients([1.0, 3.0, 7.0]);
    let expect = quad.quadratic_roots().unwrap();
    let got = quad.roots().unwrap();
    assert_eq!(got.get(..2), Some(&expect[..]));
}
