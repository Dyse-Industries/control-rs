//! # Root Locus Sweep Unit and Verification Tests
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::float_cmp
)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod root_locus_test_suite {
    use crate::classical_tools::root_locus::{
        RootLocusError, sweep, sweep_adaptive,
    };
    use crate::math::complex_num::Complex;
    use crate::polynomial::ArrayPolynomial;
    use crate::transfer_function::ArrayTransferFunction;

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_roots_satisfy_characteristic_equation() {
        // G(s) = 1 / (s (s + 2)): num = [1], den = [0, 2, 1] ascending.
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        let gains = [0.0_f64, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0];
        const DEGREE: usize = 2; // D - 1
        let mut out = [Complex::new(0.0, 0.0); 7 * DEGREE];

        sweep(&tf, &gains, &mut out).unwrap();

        for (g, &gain) in gains.iter().enumerate() {
            // Independent characteristic-equation residual check: does not
            // reuse `sweep`'s coefficient assembly.
            let poly =
                ArrayPolynomial::<f64, 3>::from_coefficients([gain, 2.0, 1.0]);
            for &root in &out[g * DEGREE..(g + 1) * DEGREE] {
                let residual = poly.evaluate_complex(root).magnitude();
                assert!(
                    residual < 1e-6,
                    "gain {gain}: root {root:?} has residual {residual}"
                );
            }
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_zero_gain_matches_open_loop_poles() {
        // At k = 0 the closed-loop characteristic equation is just D(s), so
        // the swept roots must equal the open-loop poles.
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [2.0, 3.0, 1.0],
        );
        let gains = [0.0_f64];
        let mut out = [Complex::new(0.0, 0.0); 2];
        sweep(&tf, &gains, &mut out).unwrap();

        let mut poles = tf.poles().unwrap();
        // `slice::sort_by` requires `alloc`; `sort_unstable_by` is in `core`.
        poles[..2].sort_unstable_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        let mut roots = out;
        roots.sort_unstable_by(|a, b| a.re.partial_cmp(&b.re).unwrap());
        for (root, pole) in roots.iter().zip(poles[..2].iter()) {
            assert!((*root - *pole).magnitude() < 1e-9);
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_improper_system_errors() {
        let tf = ArrayTransferFunction::<f64, 3, 2>::continuous(
            [1.0, 1.0, 1.0],
            [1.0, 1.0],
        );
        let mut out = [Complex::new(0.0, 0.0); 1];
        assert_eq!(
            sweep(&tf, &[0.0], &mut out),
            Err(RootLocusError::ImproperSystem)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_buffer_size_mismatch_errors() {
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        let mut wrong_size = [Complex::new(0.0, 0.0); 1];
        assert_eq!(
            sweep(&tf, &[0.0, 1.0], &mut wrong_size),
            Err(RootLocusError::BufferSizeMismatch)
        );
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_branch_continuity_nearest_neighbor() {
        // G(s) = 1 / (s (s + 2)): open-loop poles at s = -2 and s = 0.
        // As K increases:
        // - Branch 0 moves from -2.0 towards -1.0, then into the complex plane (-1 +/- j*omega)
        // - Branch 1 moves from 0.0 towards -1.0, then into the complex plane (-1 -/+ j*omega)
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        const N_GAINS: usize = 50;
        let mut gains = [0.0_f64; N_GAINS];
        for (i, g) in gains.iter_mut().enumerate() {
            *g = 10.0 * (i as f64) / ((N_GAINS - 1) as f64);
        }

        const DEGREE: usize = 2;
        let mut out = [Complex::new(0.0, 0.0); N_GAINS * DEGREE];
        sweep(&tf, &gains, &mut out).unwrap();

        // Check canonical starting order at g = 0
        assert!((out[0].re - (-2.0)).abs() < 1e-6);
        assert!(out[0].im.abs() < 1e-6);
        assert!((out[1].re - 0.0).abs() < 1e-6);
        assert!(out[1].im.abs() < 1e-6);

        // Check continuity along each branch: step-to-step displacement is bounded
        for g in 1..N_GAINS {
            for b in 0..DEGREE {
                let curr = out[g * DEGREE + b];
                let prev = out[(g - 1) * DEGREE + b];
                let disp = (curr - prev).magnitude();
                assert!(
                    disp < 0.6,
                    "branch {b} jumped by {disp} between steps {} and {}",
                    g - 1,
                    g
                );
            }
        }

        // At final gain K = 10.0: roots are -1 +/- j*3
        let final_0 = out[(N_GAINS - 1) * DEGREE];
        let final_1 = out[(N_GAINS - 1) * DEGREE + 1];
        assert!((final_0.re - (-1.0)).abs() < 1e-4);
        assert!((final_1.re - (-1.0)).abs() < 1e-4);
        assert!((final_0.im.abs() - 3.0).abs() < 1e-4);
        assert!((final_1.im.abs() - 3.0).abs() < 1e-4);
        // The two branches must have opposite imaginary signs
        assert!((final_0.im + final_1.im).abs() < 1e-4);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_substepping_large_gain_step() {
        // Jumping directly from K = 0 to K = 10 with no intermediate gains
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        let gains = [0.0_f64, 10.0];
        const DEGREE: usize = 2;
        let mut out = [Complex::new(0.0, 0.0); 2 * DEGREE];
        sweep(&tf, &gains, &mut out).unwrap();

        // End roots must satisfy the characteristic equation at K = 10
        let poly =
            ArrayPolynomial::<f64, 3>::from_coefficients([10.0, 2.0, 1.0]);
        for &root in &out[DEGREE..2 * DEGREE] {
            let residual = poly.evaluate_complex(root).magnitude();
            assert!(residual < 1e-6);
        }
        // Roots are -1 +/- 3j
        assert!((out[DEGREE].re - (-1.0)).abs() < 1e-4);
        assert!((out[DEGREE + 1].re - (-1.0)).abs() < 1e-4);
        assert!((out[DEGREE].im.abs() - 3.0).abs() < 1e-4);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_sweep_adaptive_reaches_k_max() {
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        const MAX_POINTS: usize = 200;
        const DEGREE: usize = 2;
        let mut gains = [0.0_f64; MAX_POINTS];
        let mut roots = [Complex::new(0.0, 0.0); MAX_POINTS * DEGREE];
        let max_disp = 0.2_f64;

        let count =
            sweep_adaptive(&tf, (0.0, 10.0), max_disp, &mut gains, &mut roots)
                .unwrap();

        assert!(count > 1, "must record multiple points");
        assert!(count <= MAX_POINTS, "must not exceed buffer capacity");
        assert!(
            (gains[count - 1] - 10.0).abs() < 1e-9,
            "final gain must equal k_max"
        );

        // Check monotonically increasing gains and bounded displacement
        for i in 1..count {
            assert!(
                gains[i] > gains[i - 1],
                "gains must strictly increase: gains[{i}] = {} <= gains[{}] = {}",
                gains[i],
                i - 1,
                gains[i - 1]
            );
            for b in 0..DEGREE {
                let prev = roots[(i - 1) * DEGREE + b];
                let curr = roots[i * DEGREE + b];
                let disp = (curr - prev).magnitude();
                assert!(
                    disp <= max_disp + 1e-4,
                    "displacement {disp} exceeds max_disp {max_disp} at step {i}"
                );
            }
        }

        // Verify characteristic equation residual for all recorded roots
        for i in 0..count {
            let poly = ArrayPolynomial::<f64, 3>::from_coefficients([
                gains[i], 2.0, 1.0,
            ]);
            for b in 0..DEGREE {
                let root = roots[i * DEGREE + b];
                let residual = poly.evaluate_complex(root).magnitude();
                assert!(
                    residual < 1e-6,
                    "gain {}: root {root:?} residual {residual} >= 1e-6",
                    gains[i]
                );
            }
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_sweep_adaptive_buffer_fill_termination() {
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        const SMALL_CAP: usize = 6;
        const DEGREE: usize = 2;
        let mut gains = [0.0_f64; SMALL_CAP];
        let mut roots = [Complex::new(0.0, 0.0); SMALL_CAP * DEGREE];

        // Very small max_displacement so 6 points cannot possibly span K = 0 to 10
        let count =
            sweep_adaptive(&tf, (0.0, 10.0), 0.02, &mut gains, &mut roots)
                .unwrap();

        assert_eq!(count, SMALL_CAP, "buffer capacity must be reached");
        assert!(
            gains[count - 1] < 10.0,
            "k_max should not be reached with small buffer and fine steps"
        );

        for i in 0..count {
            let poly = ArrayPolynomial::<f64, 3>::from_coefficients([
                gains[i], 2.0, 1.0,
            ]);
            for b in 0..DEGREE {
                let root = roots[i * DEGREE + b];
                let residual = poly.evaluate_complex(root).magnitude();
                assert!(residual < 1e-6);
            }
        }
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_sweep_adaptive_breakaway_dense_sampling() {
        // G(s) = 1 / (s(s+2)), breakaway at s = -1, K = 1.
        let tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        const CAPACITY: usize = 150;
        const DEGREE: usize = 2;
        let mut gains = [0.0_f64; CAPACITY];
        let mut roots = [Complex::new(0.0, 0.0); CAPACITY * DEGREE];
        let max_disp = 0.08_f64;

        let count =
            sweep_adaptive(&tf, (0.0, 2.0), max_disp, &mut gains, &mut roots)
                .unwrap();

        assert!((gains[count - 1] - 2.0).abs() < 1e-9);

        // Verify that step-to-step displacement across the breakaway region is strictly bounded
        for i in 1..count {
            for b in 0..DEGREE {
                let prev = roots[(i - 1) * DEGREE + b];
                let curr = roots[i * DEGREE + b];
                let disp = (curr - prev).magnitude();
                assert!(
                    disp <= max_disp + 1e-4,
                    "at step {i} (gain {}): displacement {disp} exceeds max_disp {max_disp}",
                    gains[i]
                );
            }
        }

        // Verify continuity: roots at K=0 are at -2 and 0
        assert!((roots[0].re - (-2.0)).abs() < 1e-4);
        assert!((roots[1].re - 0.0).abs() < 1e-4);

        // At K=2, roots are -1 +/- j
        let final_0 = roots[(count - 1) * DEGREE];
        let final_1 = roots[(count - 1) * DEGREE + 1];
        assert!((final_0.re - (-1.0)).abs() < 1e-3);
        assert!((final_1.re - (-1.0)).abs() < 1e-3);
        assert!((final_0.im.abs() - 1.0).abs() < 1e-3);
        assert!((final_1.im.abs() - 1.0).abs() < 1e-3);
    }

    #[cfg_attr(test, test)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_sweep_adaptive_error_cases() {
        let improper_tf = ArrayTransferFunction::<f64, 3, 2>::continuous(
            [1.0, 1.0, 1.0],
            [1.0, 1.0],
        );
        let mut g = [0.0_f64; 10];
        let mut r = [Complex::new(0.0, 0.0); 10];
        assert_eq!(
            sweep_adaptive(&improper_tf, (0.0, 1.0), 0.1, &mut g, &mut r),
            Err(RootLocusError::ImproperSystem)
        );

        let proper_tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );
        let mut wrong_r = [Complex::new(0.0, 0.0); 5]; // should be 10 * 2 = 20
        assert_eq!(
            sweep_adaptive(&proper_tf, (0.0, 1.0), 0.1, &mut g, &mut wrong_r),
            Err(RootLocusError::BufferSizeMismatch)
        );

        // Inverted range
        let mut valid_r = [Complex::new(0.0, 0.0); 20];
        assert_eq!(
            sweep_adaptive(&proper_tf, (2.0, 1.0), 0.1, &mut g, &mut valid_r),
            Err(RootLocusError::InvalidParameter)
        );

        // Non-positive max_displacement
        assert_eq!(
            sweep_adaptive(&proper_tf, (0.0, 1.0), 0.0, &mut g, &mut valid_r),
            Err(RootLocusError::InvalidParameter)
        );
        assert_eq!(
            sweep_adaptive(&proper_tf, (0.0, 1.0), -1.0, &mut g, &mut valid_r),
            Err(RootLocusError::InvalidParameter)
        );

        // NaN range
        assert_eq!(
            sweep_adaptive(
                &proper_tf,
                (f64::NAN, 1.0),
                0.1,
                &mut g,
                &mut valid_r
            ),
            Err(RootLocusError::InvalidParameter)
        );
    }

    #[cfg_attr(test, test)]
    #[allow(clippy::type_complexity)]
    /// # Verification
    /// Trace: classical-tools#FR-3
    /// Method: Requirements-based test
    fn test_sweep_adaptive_boundary_cases() {
        let proper_tf = ArrayTransferFunction::<f64, 1, 3>::continuous(
            [1.0],
            [0.0, 2.0, 1.0],
        );

        // Empty buffer returns Ok(0)
        let mut empty_g = [0.0_f64; 0];
        let mut empty_r: [Complex<f64>; 0] = [];
        assert_eq!(
            sweep_adaptive(
                &proper_tf,
                (0.0, 1.0),
                0.1,
                &mut empty_g,
                &mut empty_r
            ),
            Ok(0)
        );

        // k_min == k_max returns Ok(1) with initial roots
        let mut single_g = [0.0_f64; 5];
        let mut single_r = [Complex::new(0.0, 0.0); 10];
        let single_count = sweep_adaptive(
            &proper_tf,
            (2.0, 2.0),
            0.1,
            &mut single_g,
            &mut single_r,
        )
        .unwrap();
        assert_eq!(single_count, 1);
        assert_eq!(single_g[0], 2.0);
    }
}
