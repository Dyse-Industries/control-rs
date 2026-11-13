//! Unit tests for `nyquist`.

use super::*;
use crate::classical_control::nyquist::*;

/// Contour samples; fewer under Miri, where the interpreter dominates.
const SAMPLES: usize = if cfg!(miri) { 201 } else { 1001 };

fn count<const N: usize, const D: usize>(
    sys: &ArrayTransferFunction<f64, N, D>,
) -> Result<NyquistCount, ClassicalError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let mut contour = [Complex::new(0.0, 0.0); SAMPLES];
    nyquist_encirclements(sys, 100.0, 1e-3, &mut contour)
}

#[test]
fn encirclements_known_cases() {
    // Closed loop (s + 1)^3 + 2: stable.
    let stable = ArrayTransferFunction::<f64, 1, 4>::continuous(
        [2.0],
        [1.0, 3.0, 3.0, 1.0],
    );
    let got = count(&stable).unwrap();
    assert_eq!((got.encirclements, got.open_loop_rhp), (0, 0));
    // Closed loop (s + 1)^3 + 10: two right-half-plane poles.
    let unstable = ArrayTransferFunction::<f64, 1, 4>::continuous(
        [10.0],
        [1.0, 3.0, 3.0, 1.0],
    );
    let got = count(&unstable).unwrap();
    assert_eq!((got.encirclements, got.open_loop_rhp), (2, 0));
    // L = 2 / (s - 1): closed loop s + 1 is stable, one CCW encirclement.
    let open_unstable =
        ArrayTransferFunction::<f64, 1, 2>::continuous([2.0], [-1.0, 1.0]);
    let got = count(&open_unstable).unwrap();
    assert_eq!((got.encirclements, got.open_loop_rhp), (-1, 1));
    // Integrating plant L = 1 / (s (s + 1)): closed loop stable.
    let integrating =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 1.0, 1.0]);
    let got = count(&integrating).unwrap();
    assert_eq!((got.encirclements, got.open_loop_rhp), (0, 0));
    let discrete =
        ArrayTransferFunction::<f64, 1, 2>::discrete([1.0], [0.5, 1.0], 0.1);
    assert_eq!(count(&discrete), Err(ClassicalError::NotContinuous));
}

#[test]
fn improper_open_loop_is_rejected() {
    // L = s^3: 1 + s^3 has two right-half-plane roots, and a jω-only
    // contour would report (N, P) = (0, 0).
    let improper = ArrayTransferFunction::<f64, 4, 1>::continuous(
        [0.0, 0.0, 0.0, 1.0],
        [1.0],
    );
    assert_eq!(count(&improper), Err(ClassicalError::Improper));
    let proper =
        ArrayTransferFunction::<f64, 2, 2>::continuous([1.0, 2.0], [3.0, 4.0]);
    assert!(count(&proper).is_ok());
}

#[test]
fn contour_through_critical_point() {
    // L(j0) = -1 exactly at the contour's middle sample.
    let sys =
        ArrayTransferFunction::<f64, 1, 2>::continuous([-1.0], [1.0, 1.0]);
    assert_eq!(
        count(&sys),
        Err(ClassicalError::ContourThroughCriticalPoint)
    );
}

#[test]
fn parameter_bounds() {
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([2.0], [1.0, 3.0, 3.0]);
    let bad = Err(ClassicalError::InvalidParameter);
    let mut c5 = [Complex::new(0.0, 0.0); 5];
    assert_eq!(nyquist_encirclements(&sys, 0.0, 1e-3, &mut c5), bad);
    assert_eq!(nyquist_encirclements(&sys, 10.0, 0.0, &mut c5), bad);
    assert_eq!(nyquist_encirclements(&sys, -1.0, 1e-3, &mut c5), bad);
    let mut c2 = [Complex::new(0.0, 0.0); 2];
    assert_eq!(nyquist_encirclements(&sys, 10.0, 1e-3, &mut c2), bad);
    let mut c3 = [Complex::new(0.0, 0.0); 3];
    assert!(nyquist_encirclements(&sys, 10.0, 1e-3, &mut c3).is_ok());
}

#[test]
fn pole_on_indent_radius_is_contour_error() {
    // Pole at `s = r` lies on the rightmost indent sample `s = r`, so
    // `L` is NaN and the contour is rejected (FR-4).
    let r = 1e-3;
    let sys = ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [-r, 1.0]);
    assert_eq!(
        count(&sys),
        Err(ClassicalError::ContourThroughCriticalPoint)
    );
}

#[test]
fn contour_indents_around_axis_pole() {
    let r = 1e-3;
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 1.0, 1.0]);
    let mut contour = [Complex::new(0.0, 0.0); 3];
    nyquist_encirclements(&sys, 100.0, r, &mut contour).unwrap();
    // Middle sample w = 0 moves to s = r: L = 1 / (r (r + 1)).
    let expected = 1.0 / (r * (r + 1.0));
    assert!(((contour[1].re - expected) / expected).abs() < 1e-9);
    assert!(contour[1].im.abs() < 1e-9 * expected);
}

#[test]
fn crossing_through_critical_point_is_not_counted() {
    // The segment passes exactly through the origin of `1 + L`.
    assert_eq!(crossing(-1.0, 1.0, 1.0, -1.0), 0);
    assert_eq!(crossing(-2.0, 1.0, 0.0, -1.0), 1);
    assert_eq!(crossing(-2.0, -1.0, 0.0, 1.0), -1);
}

#[test]
fn padded_proper_loop_is_accepted() {
    // Capacity above the degree leaves trailing zero coefficients; the
    // degree is the last nonzero one (FR-4).
    let sys = ArrayTransferFunction::<f64, 4, 3>::continuous(
        [1.0, 2.0, 0.0, 0.0],
        [1.0, 2.0, 1.0],
    );
    assert!(count(&sys).is_ok());
}

#[test]
fn zero_denominator_is_improper() {
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 0.0, 0.0]);
    assert_eq!(count(&sys), Err(ClassicalError::Improper));
}

#[test]
fn pole_on_indent_radius_is_not_counted() {
    // P counts poles with Re p > r: a pole at Re p = r lies on the
    // indentation arc, as `indent` includes `|Re p| <= r`. An even
    // sample count keeps every sample off the pole itself.
    let r = 1e-3;
    let mut contour = [Complex::new(0.0, 0.0); 1000];
    let on_arc =
        ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [-r, 1.0]);
    let got = nyquist_encirclements(&on_arc, 100.0, r, &mut contour);
    assert_eq!(got.map(|c| c.open_loop_rhp), Ok(0));
    let beyond =
        ArrayTransferFunction::<f64, 1, 2>::continuous([1.0], [-1.01 * r, 1.0]);
    let got = nyquist_encirclements(&beyond, 100.0, r, &mut contour);
    assert_eq!(got.map(|c| c.open_loop_rhp), Ok(1));
}

#[test]
fn critical_point_between_samples_is_rejected() {
    // L = 1 / s^2 is real and negative on the axis and passes through
    // -1 at w = 1, which is not a sample of the cubic grid. A segment
    // that passes through -1 must be rejected, not counted.
    let sys =
        ArrayTransferFunction::<f64, 1, 3>::continuous([1.0], [0.0, 0.0, 1.0]);
    let mut contour = [Complex::new(0.0, 0.0); SAMPLES];
    assert_eq!(
        nyquist_encirclements(&sys, 100.0, 1e-3, &mut contour),
        Err(ClassicalError::ContourThroughCriticalPoint)
    );
}

#[test]
fn sparse_grid_still_indents_near_axis_pole() {
    // L = 1 / ((s^2 + 1/4)(s + 1)) has poles at +/- j/2 on the axis.
    // The closed loop s^3 + s^2 + s/4 + 5/4 has two right-half-plane
    // roots, so N = 2 and P = 0 for any grid. 101 samples are spaced
    // about 0.17 apart near w = 1/2, far more than the indentation
    // diameter, so no sample lies on an arc.
    let sys = ArrayTransferFunction::<f64, 1, 4>::continuous(
        [1.0],
        [0.25, 0.25, 1.0, 1.0],
    );
    let r = 0.02;
    let mut sparse = [Complex::new(0.0, 0.0); 101];
    let mut dense = [Complex::new(0.0, 0.0); 1001];
    let a = nyquist_encirclements(&sys, 100.0, r, &mut dense).unwrap();
    let b = nyquist_encirclements(&sys, 100.0, r, &mut sparse).unwrap();
    assert_eq!((a.encirclements, a.open_loop_rhp), (2, 0));
    assert_eq!(a, b);
}

#[test]
fn next_arc_point_is_the_smallest_strictly_inside_the_gap() {
    // Two near-axis poles at 0.5j and 0.51j with r = 0.02: the lowest
    // arc point is 0.5 - r, from the first pole.
    let r = 0.02;
    let poles = [Complex::new(0.0, 0.5), Complex::new(0.0, 0.51)];
    let first = 0.5 - r;
    assert_eq!(
        next_arc_point(&poles, r, 0.4, 1.0).map(f64::to_bits),
        Some(first.to_bits())
    );
    // The bounds are exclusive: a candidate equal to `hi` is outside the
    // gap, and one equal to `lo` has been visited.
    assert_eq!(next_arc_point(&poles, r, 0.4, first), None);
    let after = next_arc_point(&poles, r, first, 1.0).unwrap();
    assert!(after > first);
    // Poles farther than `r` from the axis have no arc.
    let off = [Complex::new(0.5, 0.5)];
    assert_eq!(next_arc_point(&off, r, 0.0, 1.0), None);
}

#[test]
fn chord_tolerance_scales_with_the_larger_endpoint() {
    // In `1 + L` coordinates the chord (1e-4, -1e6) -> (1e-4, 1)
    // passes 1e-4 from the origin. The bound for the larger endpoint,
    // 1e-8 (1 + 1e6) = 1e-2, rejects it; the smaller endpoint's bound,
    // about 2e-8, would accept it.
    let mut w = Walker::new(1e-8);
    w.push(Complex::new(-1.0 + 1e-4, -1e6)).unwrap();
    assert_eq!(
        w.push(Complex::new(-1.0 + 1e-4, 1.0)),
        Err(ClassicalError::ContourThroughCriticalPoint)
    );
}

#[test]
fn arc_point_is_strictly_inside_and_minimal() {
    let pole = |re, im| Complex::new(re, im);
    let poles = [pole(0.0, 5.0), pole(0.0, 0.0)];
    // Arc ends at -1 and 1 for the pole at 0 are excluded as bounds.
    assert_eq!(next_arc_point(&poles, 1.0, 1.0, 2.0), None);
    assert_eq!(next_arc_point(&poles, 1.0, -2.0, -1.0), None);
    // The minimum over all poles, not the first found.
    assert_eq!(next_arc_point(&poles, 1.0, -2.0, 9.0), Some(-1.0));
}
