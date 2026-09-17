//! # Root Locus Sweep Engine
//!
//! Evaluates closed-loop pole trajectories for a swept feedback gain.
//!
//! Given open-loop transfer function $G(s) = N(s)/D(s)$ and a gain
//! array $K = [k_0, \dots, k_M]$, forms the closed-loop characteristic
//! polynomials $P_j(s) = D(s) + k_j N(s)$ and solves each for its complex
//! roots via companion-matrix eigensolves (`control_rs::polynomial`,
//! NumPy Developers, 2024), writing them into a caller-supplied
//! pre-allocated buffer.
#![allow(
    clippy::indexing_slicing,
    clippy::arithmetic_side_effects,
    clippy::doc_markdown,
    clippy::type_complexity,
    clippy::too_long_first_doc_paragraph
)]

use crate::math::complex_num::Complex;
use crate::math::num_traits::Float;
use crate::math::num_types::{Const, Dim};
use crate::polynomial::{ArrayPolynomial, RootError};
use crate::transfer_function::ArrayTransferFunction;
use core::fmt;

/// Errors from [`sweep`] and [`sweep_adaptive`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RootLocusError {
    /// The numerator's degree exceeds the denominator's: not a proper
    /// transfer function.
    ImproperSystem,
    /// `out`'s length does not equal `gains.len() * (D - 1)`, the number of
    /// gains times the characteristic polynomial's degree.
    BufferSizeMismatch,
    /// Root-finding failed to converge for one of the swept gains.
    RootFinding(RootError),
    /// Invalid parameters (e.g. `k_min > k_max`, `max_displacement <= 0`, or NaN).
    InvalidParameter,
}

impl fmt::Display for RootLocusError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ImproperSystem => {
                write!(f, "numerator degree exceeds denominator degree (N > D)")
            }
            Self::BufferSizeMismatch => {
                write!(
                    f,
                    "output buffer length must equal gains.len() * (D - 1)"
                )
            }
            Self::RootFinding(e) => write!(f, "root finding failed: {e}"),
            Self::InvalidParameter => {
                write!(
                    f,
                    "invalid parameter: k_min <= k_max and max_displacement > 0 required"
                )
            }
        }
    }
}

/// Solves the closed-loop characteristic polynomial roots for a single gain.
fn solve_closed_loop_roots<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    gain: T,
) -> Result<[Complex<T>; D], RootLocusError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let den = tf.den_slice();
    let num = tf.num_slice();
    let mut coeffs = [T::ZERO; D];
    for (i, coeff) in coeffs.iter_mut().enumerate() {
        let n_i = if i < N { num[i] } else { T::ZERO };
        *coeff = den[i] + gain * n_i;
    }
    let poly = ArrayPolynomial::<T, D>::from_coefficients(coeffs);
    let (roots, converged) = poly
        .roots_best_effort()
        .map_err(RootLocusError::RootFinding)?;
    if converged
        || max_backward_error::<T, D>(&poly, &roots) <= backward_error_bound()
    {
        return Ok(roots);
    }
    Err(RootLocusError::RootFinding(RootError::ConvergenceFailure))
}

/// Backward-error bound below which an unconverged Aberth iterate is still
/// accepted as the closed-loop pole set.
///
/// A breakaway makes two closed-loop poles coincide, and simultaneous
/// iteration converges only linearly on a multiple root, so the step bound
/// is unreachable there while the iterate itself already satisfies the
/// characteristic equation. $10^{-10}$ leaves four orders above the f64
/// backward error of a well-separated sweep and five below the residual
/// scale of an iterate that has not found the roots at all.
// Case-by-case: Arithmetic side effects are unavoidable for a generic
// constant built from repeated multiplication.
#[allow(clippy::arithmetic_side_effects)]
fn backward_error_bound<T: Float + Copy>() -> T {
    let ten = T::from_usize(10);
    let hundred = ten * ten;
    let hundred_million = hundred * hundred * hundred * hundred;
    T::ONE / (hundred_million * hundred)
}

/// Largest relative backward error $|P(\hat s)| / \sum_k |c_k| |\hat s|^k$
/// over `roots`, which is the standard scale-free residual for a computed
/// polynomial root (Higham, 2002).
fn max_backward_error<T: Float + Copy, const D: usize>(
    poly: &ArrayPolynomial<T, D>,
    roots: &[Complex<T>; D],
) -> T
where
    Const<D>: Dim,
{
    let degree = D.saturating_sub(1);
    let coeffs = poly.to_coefficients();
    let mut worst = T::ZERO;
    for root in roots.iter().take(degree) {
        let residual = poly.evaluate_complex(*root).magnitude();
        let magnitude = root.magnitude();
        let mut scale = T::ZERO;
        let mut coeff_sum = T::ZERO;
        let mut power = T::ONE;
        for coeff in &coeffs {
            scale = scale + coeff.abs() * power;
            coeff_sum = coeff_sum + coeff.abs();
            power = power * magnitude;
        }
        // A root at the origin of a polynomial with no constant term drives
        // both residual and scale to zero together, so the ratio is
        // undefined there. Floor the scale at the polynomial's own
        // evaluation rounding level.
        let floor = coeff_sum * T::epsilon();
        if scale < floor {
            scale = floor;
        }
        if scale <= T::ZERO {
            continue;
        }
        let relative = residual / scale;
        if relative > worst {
            worst = relative;
        }
    }
    worst
}

/// True when the numerator's actual degree exceeds the denominator's.
fn is_improper<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
) -> bool
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    slice_degree(tf.num_slice()) > slice_degree(tf.den_slice())
}

fn slice_degree<T: Float + Copy>(coeffs: &[T]) -> Option<usize> {
    coeffs.iter().rposition(|&c| c != T::ZERO)
}

/// Sorts roots deterministically in-place by real part ascending, then imaginary part ascending.
fn sort_roots_canonically<T: Float + Copy>(roots: &mut [Complex<T>]) {
    for i in 1..roots.len() {
        let key = roots[i];
        let mut j = i;
        while j > 0 {
            let prev = roots[j - 1];
            let should_swap = if prev.re > key.re {
                true
            } else if prev.re < key.re {
                false
            } else {
                prev.im > key.im
            };
            if should_swap {
                roots[j] = prev;
                j -= 1;
            } else {
                break;
            }
        }
        roots[j] = key;
    }
}

/// Pairs candidate roots to previous poles via global greedy nearest-neighbor matching
/// to maintain branch continuity across gains.
fn match_nearest_neighbors<T: Float + Copy, const D: usize>(
    prev: &[Complex<T>],
    curr: &[Complex<T>],
    matched: &mut [Complex<T>; D],
) {
    let deg = prev.len();
    let mut used_prev = [false; D];
    let mut used_curr = [false; D];

    for _ in 0..deg {
        let mut best_dist_sq: Option<T> = None;
        let mut best_i = 0;
        let mut best_j = 0;

        for (i, &p) in prev.iter().enumerate().take(deg) {
            if used_prev[i] {
                continue;
            }
            for (j, &c) in curr.iter().enumerate().take(deg) {
                if used_curr[j] {
                    continue;
                }
                let dre = p.re - c.re;
                let dim = p.im - c.im;
                let dist_sq = dre * dre + dim * dim;
                if best_dist_sq.is_none_or(|best| dist_sq < best) {
                    best_dist_sq = Some(dist_sq);
                    best_i = i;
                    best_j = j;
                }
            }
        }

        matched[best_i] = curr[best_j];
        used_prev[best_i] = true;
        used_curr[best_j] = true;
    }
}

/// Computes the minimum squared distance between distinct branches in `tracked`.
fn min_branch_separation_sq<T: Float + Copy>(
    tracked: &[Complex<T>],
) -> Option<T> {
    let mut min_sq: Option<T> = None;
    for (i, p1) in tracked.iter().enumerate() {
        for p2 in tracked.iter().skip(i + 1) {
            let dre = p1.re - p2.re;
            let dim = p1.im - p2.im;
            let dist_sq = dre * dre + dim * dim;
            if min_sq.is_none_or(|curr| dist_sq < curr) {
                min_sq = Some(dist_sq);
            }
        }
    }
    min_sq
}

/// Computes the maximum squared displacement between corresponding elements of `prev` and `curr`.
fn max_displacement_sq<T: Float + Copy>(
    prev: &[Complex<T>],
    curr: &[Complex<T>],
) -> T {
    let mut max_sq = T::ZERO;
    for (&p, &c) in prev.iter().zip(curr.iter()) {
        let dre = p.re - c.re;
        let dim = p.im - c.im;
        let dist_sq = dre * dre + dim * dim;
        if dist_sq > max_sq {
            max_sq = dist_sq;
        }
    }
    max_sq
}

/// Performs intermediate gain sub-steps between `k_interval.0` and `k_interval.1` to bound pole displacements.
fn substep_advance_poles<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    k_interval: (T, T),
    num_substeps: usize,
    tracked: &mut [Complex<T>; D],
) -> Result<(), RootLocusError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let degree = D.saturating_sub(1);
    let (k_prev, k_curr) = k_interval;
    let dk = k_curr - k_prev;
    let s_total = T::from_usize(num_substeps);

    let mut advanced = false;
    let mut reached_k_curr = false;
    let mut last_error = None;
    for s in 1..=num_substeps {
        let frac = T::from_usize(s) / s_total;
        let k_sub = k_prev + dk * frac;
        // A mid-interval sub-step that lands on a breakaway is rejected
        // rather than fatal: the gain is skipped and the next sub-step
        // carries the branch across. The terminal gain `k_curr` must still
        // succeed — otherwise poles from an earlier sub-step would be
        // attributed to `k_curr` by the caller.
        let sub_roots = match solve_closed_loop_roots::<T, N, D>(tf, k_sub) {
            Ok(roots) => roots,
            Err(e) => {
                last_error = Some(e);
                continue;
            }
        };
        let mut sub_matched = [Complex::new(T::ZERO, T::ZERO); D];
        match_nearest_neighbors::<T, D>(
            &tracked[..degree],
            &sub_roots[..degree],
            &mut sub_matched,
        );
        tracked[..degree].copy_from_slice(&sub_matched[..degree]);
        advanced = true;
        if s == num_substeps {
            reached_k_curr = true;
        }
    }
    if advanced && reached_k_curr {
        Ok(())
    } else {
        Err(last_error.unwrap_or(RootLocusError::RootFinding(
            RootError::ConvergenceFailure,
        )))
    }
}

/// Advances tracked poles from `k_prev` to `k_curr` using nearest-neighbor matching,
/// adaptively sub-stepping if pole displacement exceeds half the minimum branch separation.
fn advance_poles_step<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    k_prev: T,
    k_curr: T,
    tracked: &mut [Complex<T>; D],
) -> Result<(), RootLocusError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    let degree = D.saturating_sub(1);
    if degree <= 1 {
        let roots = solve_closed_loop_roots::<T, N, D>(tf, k_curr)?;
        tracked[..degree].copy_from_slice(&roots[..degree]);
        return Ok(());
    }

    let Some(min_inter_sq) = min_branch_separation_sq(&tracked[..degree])
    else {
        return Ok(());
    };

    // When the candidate gain itself is unsolvable (e.g. leading-coefficient
    // cancellation), sub-stepping still advances through intermediate gains
    // but must fail if the terminal `k_curr` never resolves — otherwise the
    // caller would attribute earlier poles to `k_curr`.
    let Ok(candidate) = solve_closed_loop_roots::<T, N, D>(tf, k_curr) else {
        return substep_advance_poles(tf, (k_prev, k_curr), 16, tracked);
    };
    let mut matched = [Complex::new(T::ZERO, T::ZERO); D];
    match_nearest_neighbors::<T, D>(
        &tracked[..degree],
        &candidate[..degree],
        &mut matched,
    );

    let max_disp_sq =
        max_displacement_sq(&tracked[..degree], &matched[..degree]);
    let threshold = min_inter_sq / T::from_usize(4);

    if max_disp_sq <= threshold || min_inter_sq < T::epsilon() {
        tracked[..degree].copy_from_slice(&matched[..degree]);
        return Ok(());
    }

    let ratio = (max_disp_sq / min_inter_sq).sqrt();
    let two = T::from_usize(2);
    let mut num_substeps = 2_usize;
    while num_substeps < 16 && T::from_usize(num_substeps) < two * ratio {
        num_substeps += 1;
    }

    substep_advance_poles(tf, (k_prev, k_curr), num_substeps, tracked)
}

/// Sweeps `tf`'s closed-loop poles over `gains`, writing
/// `gains.len() * (D - 1)` complex roots into `out` in gain-major order:
/// `out[g * (D - 1) .. (g + 1) * (D - 1)]` holds the roots of
/// $D(s) + \text{gains}\[g\] \cdot N(s) = 0$.
///
/// Branch trajectories are tracked continuously across swept gains using
/// nearest-neighbor matching against preceding poles, with adaptive gain
/// sub-stepping to guarantee pole displacements remain within branch
/// separation boundaries.
///
/// # Errors
/// - [`RootLocusError::ImproperSystem`] if the numerator degree exceeds the
///   denominator degree.
/// - [`RootLocusError::BufferSizeMismatch`] if `out.len() != gains.len() * (D - 1)`.
/// - [`RootLocusError::RootFinding`] if root-finding fails for any gain.
pub fn sweep<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    gains: &[T],
    out: &mut [Complex<T>],
) -> Result<(), RootLocusError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    if is_improper(tf) {
        return Err(RootLocusError::ImproperSystem);
    }
    let degree = D.saturating_sub(1);
    if out.len() != gains.len() * degree {
        return Err(RootLocusError::BufferSizeMismatch);
    }
    if degree == 0 || gains.is_empty() {
        return Ok(());
    }

    let mut tracked = [Complex::new(T::ZERO, T::ZERO); D];

    // Gain 0: Initial poles sorted canonically
    let init_roots = solve_closed_loop_roots::<T, N, D>(tf, gains[0])?;
    tracked[..degree].copy_from_slice(&init_roots[..degree]);
    sort_roots_canonically(&mut tracked[..degree]);
    out[..degree].copy_from_slice(&tracked[..degree]);

    for g in 1..gains.len() {
        let k_prev = gains[g - 1];
        let k_curr = gains[g];
        advance_poles_step::<T, N, D>(tf, k_prev, k_curr, &mut tracked)?;
        out[g * degree..(g + 1) * degree].copy_from_slice(&tracked[..degree]);
    }

    Ok(())
}

/// Sweeps `tf`'s closed-loop poles adaptively over the gain range `k_range = (k_min, k_max)`,
/// bounding the maximum pole displacement between consecutive recorded steps to `max_displacement`.
///
/// Unlike [`sweep`], which evaluates a caller-fixed gain grid and discards intermediate sub-steps,
/// `sweep_adaptive` dynamically selects gain steps to follow pole trajectories smoothly—allocating
/// dense steps across high-sensitivity regions (such as breakaway/break-in points) and broader steps
/// across flat regions—writing every accepted sub-step into `gains_out` and `roots_out`.
///
/// Sweeping terminates when either `k_max` is reached or the output buffer capacity is exhausted.
/// Returns the number of trajectory points written.
///
/// # Errors
/// - [`RootLocusError::ImproperSystem`] if the numerator degree exceeds the
///   denominator degree.
/// - [`RootLocusError::BufferSizeMismatch`] if `roots_out.len() != gains_out.len() * (D - 1)`.
/// - [`RootLocusError::InvalidParameter`] if `k_range.0 > k_range.1`, `max_displacement <= T::ZERO`,
///   or any parameter is non-finite / NaN.
/// - [`RootLocusError::RootFinding`] if root-finding fails for any gain.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
pub fn sweep_adaptive<T: Float + Copy, const N: usize, const D: usize>(
    tf: &ArrayTransferFunction<T, N, D>,
    k_range: (T, T),
    max_displacement: T,
    gains_out: &mut [T],
    roots_out: &mut [Complex<T>],
) -> Result<usize, RootLocusError>
where
    Const<N>: Dim,
    Const<D>: Dim,
{
    if is_improper(tf) {
        return Err(RootLocusError::ImproperSystem);
    }
    let degree = D.saturating_sub(1);
    if roots_out.len() != gains_out.len() * degree {
        return Err(RootLocusError::BufferSizeMismatch);
    }
    let (k_min, k_max) = k_range;
    let valid_range = matches!(
        k_min.partial_cmp(&k_max),
        Some(core::cmp::Ordering::Less | core::cmp::Ordering::Equal)
    );
    let valid_disp = matches!(
        max_displacement.partial_cmp(&T::ZERO),
        Some(core::cmp::Ordering::Greater)
    );
    if !valid_range || !valid_disp {
        return Err(RootLocusError::InvalidParameter);
    }
    if gains_out.is_empty() || degree == 0 {
        return Ok(0);
    }

    let mut tracked = [Complex::new(T::ZERO, T::ZERO); D];

    // Gain 0: Initial poles at k_min sorted canonically
    let init_roots = solve_closed_loop_roots::<T, N, D>(tf, k_min)?;
    tracked[..degree].copy_from_slice(&init_roots[..degree]);
    sort_roots_canonically(&mut tracked[..degree]);
    gains_out[0] = k_min;
    roots_out[..degree].copy_from_slice(&tracked[..degree]);

    let mut count = 1_usize;
    let capacity = gains_out.len();
    if count == capacity || k_min == k_max {
        return Ok(count);
    }

    let span = k_max - k_min;
    let mut dk = span / T::from_usize((capacity - 1).max(1));
    if dk <= T::ZERO {
        dk = span;
    }
    let min_dk = if span * T::epsilon() > T::ZERO {
        span * T::epsilon()
    } else {
        T::epsilon()
    };

    let mut k_curr = k_min;
    let target_disp_sq = max_displacement * max_displacement;
    let two = T::from_usize(2);
    let half = T::ONE / two;
    let four = T::from_usize(4);

    while count < capacity && k_curr < k_max {
        if dk < min_dk {
            dk = min_dk;
        }
        let mut k_cand = k_curr + dk;
        if k_cand >= k_max {
            k_cand = k_max;
            dk = k_max - k_curr;
        }

        let candidate = solve_closed_loop_roots::<T, N, D>(tf, k_cand)?;
        let mut matched = [Complex::new(T::ZERO, T::ZERO); D];
        match_nearest_neighbors::<T, D>(
            &tracked[..degree],
            &candidate[..degree],
            &mut matched,
        );

        let max_disp_sq =
            max_displacement_sq(&tracked[..degree], &matched[..degree]);
        let min_inter_sq = min_branch_separation_sq(&tracked[..degree]);

        // Branch identity threshold: 1/4 of minimum branch separation
        let branch_thresh = match min_inter_sq {
            Some(sq) if sq >= T::epsilon() => sq / four,
            _ => target_disp_sq,
        };

        let accepted =
            max_disp_sq <= target_disp_sq && max_disp_sq <= branch_thresh;

        if accepted || dk <= min_dk {
            k_curr = k_cand;
            tracked[..degree].copy_from_slice(&matched[..degree]);
            gains_out[count] = k_curr;
            roots_out[count * degree..(count + 1) * degree]
                .copy_from_slice(&tracked[..degree]);
            count += 1;

            if max_disp_sq > T::ZERO {
                let ratio = (target_disp_sq / max_disp_sq).sqrt();
                let step_factor = if ratio > two { two } else { ratio };
                dk = dk * step_factor;
            } else {
                dk = dk * two;
            }
        } else {
            let new_dk = dk * half;
            dk = if new_dk < min_dk { min_dk } else { new_dk };
        }
    }

    Ok(count)
}
