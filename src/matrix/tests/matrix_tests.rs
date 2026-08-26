//! # Matrix / Linear Algebra Tests
//!
//! ## Functional Requirement Coverage (`matrix-design.md`)
//!
//! - **FR-1** (compile-time dimension enforcement): every test below is
//!   parameterized on `Const<N>`/plain `const usize` shapes; a dimension
//!   mismatch (e.g. adding a `2x2` to a `3x3`) is a *compile* error, not a
//!   runtime one, so it isn't exercised as a passing test case (§4.9.1).
//! - **FR-2** (static constructors): `test_zero_identity_diagonal_const_eval`.
//! - **FR-3** (core arithmetic): `test_add_sub_neg`, `test_mul_matrix_matrix`,
//!   `test_mul_matrix_vector`.
//! - **FR-4** (transposition, determinant, inversion):
//!   `test_transpose_*`, `test_lu_determinant`, `test_lu_invert_round_trip`.
//! - **FR-5** (structural specializations): `test_upper_lower_symmetric_*`,
//!   `test_ldlt_*`, `test_cholesky_*`, `test_qr_*` (§4.7 decomposition
//!   objects).
//! - **FR-6** (coordinate-based instantiation): `test_from_fn`.
//! - **FR-7** (concatenation): not implemented in this pass (not in scope —
//!   see the design doc's Development Plan Step 4/7 split).
//! - **FR-8** (Polynomial/Tensor interop): covered in sibling polynomial and tensor test suites.
//!
//! Companion-matrix construction is implemented in `Polynomial::companion_matrix`.
//!
//! §6.1.4's `127 x 127` stack-bounds ceiling is a compile-time type bound
//! (`Const<N>: Dim` for `N` up to 127), not something a runtime unit test
//! exercises — no test instantiates near that size.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::cast_precision_loss
)]

#[cfg_attr(not(test), control_rs_macros::ets_suite)]
pub mod matrix_test_suite {
    use crate::assert_almost_eq;
    use crate::math::LinAlgError;
    use crate::math::num_types::{Const, Dim};
    use crate::matrix::specialized::{
        solve_lower_triangular, solve_upper_triangular,
    };
    use crate::matrix::{
        LowerTriangular, Matrix, Owned, Symmetric, UpperTriangular,
    };

    /// Asserts every element of two same-shaped owned matrices is almost
    /// equal.
    fn _assert_matrix_almost_eq<const R: usize, const C: usize>(
        a: &Owned<f64, R, C>,
        b: &Owned<f64, R, C>,
    ) where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        for i in 0..R {
            for j in 0..C {
                assert_almost_eq!(*a.get(i, j).unwrap(), *b.get(i, j).unwrap());
            }
        }
    }

    // --- Constructors (FR-2, FR-6) ---

    #[cfg_attr(test, test)]
    /// `zero`/`identity`/`diagonal` are usable in a `const` context and
    /// produce the expected elements (FR-2).
    fn test_zero_identity_diagonal_const_eval() {
        const ZERO: Owned<f64, 2, 3> = Owned::<f64, 2, 3>::zero();
        for i in 0..2 {
            for j in 0..3 {
                assert_almost_eq!(*ZERO.get(i, j).unwrap(), 0.0);
            }
        }

        const IDENTITY: Owned<f64, 3, 3> = Owned::<f64, 3, 3>::identity();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert_almost_eq!(*IDENTITY.get(i, j).unwrap(), expected);
            }
        }

        const DIAG: Owned<f64, 3, 3> =
            Owned::<f64, 3, 3>::diagonal([1.0, 2.0, 3.0]);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { [1.0, 2.0, 3.0][i] } else { 0.0 };
                assert_almost_eq!(*DIAG.get(i, j).unwrap(), expected);
            }
        }
    }

    #[cfg_attr(test, test)]
    /// `from_fn` builds a matrix element-by-element from row/column indices
    /// (FR-6).
    fn test_from_fn() {
        let m: Owned<f64, 2, 3> = Matrix::from_fn(|i, j| (i * 10 + j) as f64);
        assert_almost_eq!(*m.get(0, 0).unwrap(), 0.0);
        assert_almost_eq!(*m.get(0, 2).unwrap(), 2.0);
        assert_almost_eq!(*m.get(1, 0).unwrap(), 10.0);
        assert_almost_eq!(*m.get(1, 2).unwrap(), 12.0);
    }

    // --- Operators (FR-3) ---

    #[cfg_attr(test, test)]
    /// `Add`/`Sub`/`Neg` are element-wise (FR-3).
    fn test_add_sub_neg() {
        let a: Owned<f64, 2, 2> = Matrix::from_fn(|i, j| (i * 2 + j) as f64);
        let b: Owned<f64, 2, 2> = Owned::identity();

        let sum = &a + &b;
        assert_almost_eq!(*sum.get(0, 0).unwrap(), 1.0);
        assert_almost_eq!(*sum.get(0, 1).unwrap(), 1.0);
        assert_almost_eq!(*sum.get(1, 0).unwrap(), 2.0);
        assert_almost_eq!(*sum.get(1, 1).unwrap(), 4.0);

        let diff = &a - &b;
        assert_almost_eq!(*diff.get(0, 0).unwrap(), -1.0);
        assert_almost_eq!(*diff.get(1, 1).unwrap(), 2.0);

        let neg = -&a;
        assert_almost_eq!(*neg.get(0, 0).unwrap(), 0.0);
        assert_almost_eq!(*neg.get(0, 1).unwrap(), -1.0);
        assert_almost_eq!(*neg.get(1, 1).unwrap(), -3.0);
    }

    #[cfg_attr(test, test)]
    /// Matrix-matrix multiplication statically enforces `(M x N) * (N x P)
    /// -> (M x P)` and computes the standard product (FR-3).
    fn test_mul_matrix_matrix() {
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| (i * 2 + j + 1) as f64); // [[1,2],[3,4]]
        let identity: Owned<f64, 2, 2> = Owned::identity();

        let product = &a * &identity;
        _assert_matrix_almost_eq(&product, &a);

        let b: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| if i == j { 2.0 } else { 0.0 });
        let scaled = &a * &b;
        assert_almost_eq!(*scaled.get(0, 0).unwrap(), 2.0);
        assert_almost_eq!(*scaled.get(0, 1).unwrap(), 4.0);
        assert_almost_eq!(*scaled.get(1, 0).unwrap(), 6.0);
        assert_almost_eq!(*scaled.get(1, 1).unwrap(), 8.0);
    }

    #[cfg_attr(test, test)]
    /// Matrix-vector multiplication is the `P == 1` case of the same `Mul`
    /// impl (FR-3, matrix-design.md §4.5's "Matrix-Vector Multiplication").
    fn test_mul_matrix_vector() {
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| (i * 2 + j + 1) as f64); // [[1,2],[3,4]]
        let x: Owned<f64, 2, 1> = Matrix::from_fn(|_, _| 1.0);
        let y = &a * &x;
        assert_almost_eq!(*y.get(0, 0).unwrap(), 3.0);
        assert_almost_eq!(*y.get(1, 0).unwrap(), 7.0);
    }

    // --- Transposition (FR-4) ---

    #[cfg_attr(test, test)]
    /// `transpose_view`/`transpose_into`/`transpose` agree on a non-square
    /// matrix, and `transpose_view` is a true zero-copy reinterpretation
    /// (reads the same flat data `transpose()` copies).
    fn test_transpose_non_square() {
        let a: Owned<f64, 2, 3> = Matrix::from_fn(|i, j| (i * 3 + j) as f64);

        let view = a.transpose_view();
        for i in 0..3 {
            for j in 0..2 {
                assert_almost_eq!(
                    *view.get(i, j).unwrap(),
                    *a.get(j, i).unwrap()
                );
            }
        }

        let owned = a.transpose();
        for i in 0..3 {
            for j in 0..2 {
                assert_almost_eq!(
                    *owned.get(i, j).unwrap(),
                    *a.get(j, i).unwrap()
                );
            }
        }

        let mut dest: Owned<f64, 3, 2> = Owned::zero();
        a.transpose_into(&mut dest);
        _assert_matrix_almost_eq(&dest, &owned);
    }

    #[cfg_attr(test, test)]
    /// `transpose_mut` performs an in-place transposition for square
    /// matrices (FR-4).
    fn test_transpose_mut_square() {
        let mut a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| (i * 2 + j) as f64); // [[0,1],[2,3]]
        a.transpose_mut();
        assert_almost_eq!(*a.get(0, 0).unwrap(), 0.0);
        assert_almost_eq!(*a.get(0, 1).unwrap(), 2.0);
        assert_almost_eq!(*a.get(1, 0).unwrap(), 1.0);
        assert_almost_eq!(*a.get(1, 1).unwrap(), 3.0);
    }

    // --- LU: determinant, solve, inversion (FR-4) ---

    #[cfg_attr(test, test)]
    /// `LuDecomposition::determinant` matches the closed-form 2x2
    /// determinant, including the sign flip from a row exchange.
    fn test_lu_determinant() {
        // [[2, 1], [1, 3]], det = 5, no pivoting needed.
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        let lu = a.into_lu().unwrap();
        assert_almost_eq!(lu.determinant(), 5.0);

        // [[0, 1], [1, 0]], det = -1: forces one row exchange.
        let swapped: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[0.0, 1.0], [1.0, 0.0]][i][j]);
        let lu_swapped = swapped.into_lu().unwrap();
        assert_almost_eq!(lu_swapped.determinant(), -1.0);
    }

    #[cfg_attr(test, test)]
    /// `LuDecomposition::solve_mut` solves `A * x = b` against a
    /// known-answer system.
    fn test_lu_solve_mut() {
        // A = [[2, 1], [1, 3]], b = [3, 5] -> x = [0.8, 1.4]
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        let lu = a.into_lu().unwrap();
        let mut b: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [3.0, 5.0][i]);
        lu.solve_mut(&mut b);
        assert_almost_eq!(*b.get(0, 0).unwrap(), 0.8, 1e-9);
        assert_almost_eq!(*b.get(1, 0).unwrap(), 1.4, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// `invert_mut`/`invert_into` round-trip: `A * A^-1 == I` within
    /// tolerance (FR-4).
    fn test_lu_invert_round_trip() {
        let a: Owned<f64, 3, 3> = Matrix::from_fn(|i, j| {
            [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]][i][j]
        });

        let mut inv_mut = a;
        let mut pivots = [0usize; 3];
        inv_mut.invert_mut(&mut pivots).unwrap();
        let product = &a * &inv_mut;
        _assert_matrix_almost_eq(&product, &Owned::<f64, 3, 3>::identity());

        let mut inv_into: Owned<f64, 3, 3> = Owned::zero();
        let mut pivots2 = [0usize; 3];
        a.invert_into(&mut inv_into, &mut pivots2).unwrap();
        _assert_matrix_almost_eq(&inv_into, &inv_mut);
    }

    #[cfg_attr(test, test)]
    /// A singular matrix fails `LU` decomposition with
    /// `LinAlgError::SingularMatrix` rather than panicking (§4.9.3).
    fn test_lu_singular_matrix_errors() {
        // Row 2 is a multiple of row 1 -> singular.
        let singular: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[1.0, 2.0], [2.0, 4.0]][i][j]);
        let result = singular.into_lu();
        assert_eq!(result.err(), Some(LinAlgError::SingularMatrix));
    }

    // --- Structural specializations (FR-5) ---

    #[cfg_attr(test, test)]
    /// `UpperTriangular`/`LowerTriangular` accept matrices that satisfy the
    /// invariant and reject matrices that don't (FR-5).
    fn test_upper_lower_construction() {
        let upper: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| if i <= j { 1.0 } else { 0.0 });
        assert!(UpperTriangular::from_owned(upper).is_some());

        let not_upper: Owned<f64, 2, 2> = Matrix::from_fn(|_, _| 1.0);
        assert!(UpperTriangular::from_owned(not_upper).is_none());

        let lower: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| if i >= j { 1.0 } else { 0.0 });
        assert!(LowerTriangular::from_owned(lower).is_some());

        let not_lower: Owned<f64, 2, 2> = Matrix::from_fn(|_, _| 1.0);
        assert!(LowerTriangular::from_owned(not_lower).is_none());
    }

    #[cfg_attr(test, test)]
    /// `solve_lower_triangular`/`solve_upper_triangular` match §4.10.1's
    /// forward/back-substitution example against a known-answer system.
    fn test_solve_triangular() {
        let l: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 0.0], [3.0, 4.0]][i][j]);
        let lower = LowerTriangular::from_owned(l).unwrap();
        let b: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [4.0, 23.0][i]);
        // 2*x0 = 4 -> x0 = 2; 3*x0 + 4*x1 = 23 -> x1 = 4.25
        let x = solve_lower_triangular(&lower, &b).unwrap();
        assert_almost_eq!(*x.get(0, 0).unwrap(), 2.0, 1e-9);
        assert_almost_eq!(*x.get(1, 0).unwrap(), 4.25, 1e-9);

        let u: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 3.0], [0.0, 4.0]][i][j]);
        let upper = UpperTriangular::from_owned(u).unwrap();
        let b2: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [23.0, 17.0][i]);
        // 4*x1 = 17 -> x1 = 4.25; 2*x0 + 3*4.25 = 23 -> x0 = 5.125
        let x2 = solve_upper_triangular(&upper, &b2).unwrap();
        assert_almost_eq!(*x2.get(1, 0).unwrap(), 4.25, 1e-9);
        assert_almost_eq!(*x2.get(0, 0).unwrap(), 5.125, 1e-9);
    }

    // --- LDLT (FR-5) ---

    #[cfg_attr(test, test)]
    /// `Symmetric` accepts symmetric matrices and rejects asymmetric ones.
    fn test_symmetric_construction() {
        let sym: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        assert!(Symmetric::from_owned(sym).is_some());

        let asym: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 1.0], [5.0, 3.0]][i][j]);
        assert!(Symmetric::from_owned(asym).is_none());
    }

    #[cfg_attr(test, test)]
    /// `LdltDecomposition::determinant`/`solve_mut` match `LuDecomposition`
    /// on a symmetric positive-definite system.
    fn test_ldlt_determinant_and_solve() {
        // A = [[4, 2], [2, 3]], det = 8, solve A*x = [6, 5] -> x = [1, 1].
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
        let sym = Symmetric::from_owned(a).unwrap();
        let ldlt = sym.into_ldlt().unwrap();
        assert_almost_eq!(ldlt.determinant(), 8.0, 1e-9);

        let mut b: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [6.0, 5.0][i]);
        ldlt.solve_mut(&mut b);
        assert_almost_eq!(*b.get(0, 0).unwrap(), 1.0, 1e-9);
        assert_almost_eq!(*b.get(1, 0).unwrap(), 1.0, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// A singular symmetric matrix fails `LDL^T` decomposition with
    /// `LinAlgError::SingularMatrix`.
    fn test_ldlt_singular_matrix_errors() {
        let singular: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[0.0, 0.0], [0.0, 1.0]][i][j]);
        let sym = Symmetric::from_owned(singular).unwrap();
        let result = sym.into_ldlt();
        assert_eq!(result.err(), Some(LinAlgError::SingularMatrix));
    }

    // --- Cholesky (FR-5) ---

    #[cfg_attr(test, test)]
    /// `CholeskyDecomposition::solve_mut` matches `LdltDecomposition` on the
    /// same symmetric positive-definite system as
    /// `test_ldlt_determinant_and_solve` (`matrix-design.md` §5.5).
    fn test_cholesky_solve() {
        // A = [[4, 2], [2, 3]], solve A*x = [6, 5] -> x = [1, 1].
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
        let sym = Symmetric::from_owned(a).unwrap();
        let chol = sym.into_cholesky().unwrap();

        let mut b: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [6.0, 5.0][i]);
        chol.solve_mut(&mut b);
        assert_almost_eq!(*b.get(0, 0).unwrap(), 1.0, 1e-9);
        assert_almost_eq!(*b.get(1, 0).unwrap(), 1.0, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// A symmetric but indefinite matrix (eigenvalues `3` and `-1`, not
    /// positive definite) fails Cholesky decomposition with
    /// `LinAlgError::NotPositiveDefinite` rather than panicking on a negative
    /// `sqrt` argument (§4.9.3).
    fn test_cholesky_not_positive_definite_errors() {
        let indefinite: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[1.0, 2.0], [2.0, 1.0]][i][j]);
        let sym = Symmetric::from_owned(indefinite).unwrap();
        let result = sym.into_cholesky();
        assert_eq!(result.err(), Some(LinAlgError::NotPositiveDefinite));
    }

    // --- QR (FR-5) ---

    #[cfg_attr(test, test)]
    /// `QrDecomposition::solve_mut` produces an `x` satisfying `A * x == b`
    /// (residual check) on the same system as `test_lu_invert_round_trip`
    /// (§4.7, Householder reflections).
    fn test_qr_solve_residual() {
        let a: Owned<f64, 3, 3> = Matrix::from_fn(|i, j| {
            [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]][i][j]
        });
        let b: Owned<f64, 3, 1> = Matrix::from_fn(|i, _| [1.0, 2.0, 3.0][i]);

        let qr = a.into_qr();
        let mut x = b;
        qr.solve_mut(&mut x).unwrap();

        let residual = &a * &x;
        _assert_matrix_almost_eq(&residual, &b);
    }

    #[cfg_attr(test, test)]
    /// Unlike `LU`/`LDL^T`, `into_qr` itself never fails on a rank-deficient
    /// matrix (`matrix-design.md` §5.5) — the resulting `R` factor simply
    /// carries a near-zero pivot, which `solve_mut` reports as
    /// `LinAlgError::SingularMatrix` rather than dividing by it.
    fn test_qr_solve_singular_matrix_errors() {
        // Row 2 is a multiple of row 1 -> rank-deficient.
        let singular: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[1.0, 2.0], [2.0, 4.0]][i][j]);
        let qr = singular.into_qr();
        let mut b: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [1.0, 2.0][i]);
        let result = qr.solve_mut(&mut b);
        assert_eq!(result.err(), Some(LinAlgError::SingularMatrix));
    }

    // --- Kalman filter covariance update (validation, §6.2.1) ---

    #[cfg_attr(test, test)]
    /// End-to-end numeric integrity check mirroring §6.2.1's discrete
    /// Kalman filter covariance update `P = (I - K*H) * P_pred`.
    fn test_kalman_covariance_update_validation() {
        let p_pred: Owned<f64, 2, 2> = Owned::identity();
        let k: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [0.5, 0.25][i]);
        let h: Owned<f64, 1, 2> = Matrix::from_fn(|_, j| [1.0, 0.0][j]);

        let identity: Owned<f64, 2, 2> = Owned::identity();
        let k_h = &k * &h;
        let diff = &identity - &k_h;
        let updated = &diff * &p_pred;

        assert_almost_eq!(*updated.get(0, 0).unwrap(), 0.5);
        assert_almost_eq!(*updated.get(0, 1).unwrap(), 0.0);
        assert_almost_eq!(*updated.get(1, 0).unwrap(), -0.25);
        assert_almost_eq!(*updated.get(1, 1).unwrap(), 1.0);
    }
}

// Property-based coverage of algebraic matrix identities (§6.1.2 of
// `matrix-design.md`). Kept outside the `#[ets_suite]`-wrapped module above:
// `proptest` is a host-only dev-dependency, unavailable to the `no_std`/
// on-target `ets` feature build (matches `storage_tests.rs`'s convention).
#[cfg(test)]
mod matrix_property_tests {
    use crate::math::num_types::{Const, Dim};
    use crate::matrix::{Matrix, Owned};
    use proptest::prelude::*;

    fn matrix_from_vec<const R: usize, const C: usize>(
        vals: &[f64],
    ) -> Owned<f64, R, C>
    where
        Const<R>: Dim,
        Const<C>: Dim,
    {
        Matrix::from_fn(|i, j| vals[j * R + i])
    }

    proptest! {
        /// `(A * B)^T == B^T * A^T` for arbitrary 2x3 * 3x2 matrices.
        #[test]
        fn prop_transpose_of_product(
            a_vals in proptest::collection::vec(-100.0..100.0_f64, 6),
            b_vals in proptest::collection::vec(-100.0..100.0_f64, 6),
        ) {
            let a: Owned<f64, 2, 3> = matrix_from_vec(&a_vals);
            let b: Owned<f64, 3, 2> = matrix_from_vec(&b_vals);

            let ab = &a * &b;
            let lhs = ab.transpose();
            let rhs = &b.transpose() * &a.transpose();

            for i in 0..2 {
                for j in 0..2 {
                    let l = *lhs.get(i, j).unwrap();
                    let r = *rhs.get(i, j).unwrap();
                    prop_assert!((l - r).abs() < 1e-6);
                }
            }
        }

        /// `A * (B + C) == A*B + A*C` (left distributivity) for arbitrary
        /// 2x2 matrices.
        #[test]
        fn prop_distributivity(
            a_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
            b_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
            c_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
        ) {
            let a: Owned<f64, 2, 2> = matrix_from_vec(&a_vals);
            let b: Owned<f64, 2, 2> = matrix_from_vec(&b_vals);
            let c: Owned<f64, 2, 2> = matrix_from_vec(&c_vals);

            let sum = &b + &c;
            let lhs = &a * &sum;
            let rhs = &(&a * &b) + &(&a * &c);

            for i in 0..2 {
                for j in 0..2 {
                    let l = *lhs.get(i, j).unwrap();
                    let r = *rhs.get(i, j).unwrap();
                    prop_assert!((l - r).abs() < 1e-6);
                }
            }
        }

        /// `Add` is associative: `(A + B) + C == A + (B + C)`.
        #[test]
        fn prop_add_associativity(
            a_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
            b_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
            c_vals in proptest::collection::vec(-100.0..100.0_f64, 4),
        ) {
            let a: Owned<f64, 2, 2> = matrix_from_vec(&a_vals);
            let b: Owned<f64, 2, 2> = matrix_from_vec(&b_vals);
            let c: Owned<f64, 2, 2> = matrix_from_vec(&c_vals);

            let lhs = &(&a + &b) + &c;
            let rhs = &a + &(&b + &c);

            for i in 0..2 {
                for j in 0..2 {
                    let l = *lhs.get(i, j).unwrap();
                    let r = *rhs.get(i, j).unwrap();
                    prop_assert!((l - r).abs() < 1e-6);
                }
            }
        }
    }
}
