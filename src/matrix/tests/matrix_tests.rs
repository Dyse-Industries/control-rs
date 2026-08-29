//! # Matrix / Linear Algebra Tests
//!
//! ## Functional Requirement Coverage (`matrix-design.md`)
//!
//! - **FR-1** (compile-time shape verification): tests are parameterized on
//!   `Const<N>`; a dimension mismatch is a compile error (`compile_fail` in
//!   `src/matrix/mod.rs`).
//! - **FR-2** (matrix algebra): `test_add_sub_neg`, `test_mul_matrix_matrix`,
//!   `test_mul_matrix_vector`, `test_kalman_covariance_update`,
//!   `prop_add_associativity`.
//! - **FR-3** (fallible factorizations and solvers): `test_lu_*`,
//!   `test_cholesky_*`, `test_qr_*`, `test_lu_factor_residual`,
//!   `test_cholesky_factor_residual`, `test_qr_orthogonality`.
//! - **FR-4** (coordinate element access): `test_coordinate_access`.
//! - **FR-5** (structural specializations): `test_symmetric_construction`,
//!   `test_upper_lower_construction`, `test_ldlt_*`.
//! - **FR-6** (zero-copy submatrix views): `test_strided_submatrix`.
//!
//! Companion-matrix construction is implemented in `Polynomial::companion_matrix`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::similar_names,
    clippy::unwrap_used,
    clippy::items_after_statements,
    clippy::cast_precision_loss,
    clippy::float_cmp
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

    /// Residual-ratio factor $\tau$ in $\lVert AA^{-1}-I\rVert_\infty \le \tau\kappa\varepsilon$.
    const INV_ROUNDTRIP_TAU: f64 = 20.0;
    /// LAPACK residual-ratio threshold $\tau$ for $A-LU$ / $A-LL^T$ (`matrix-design.md` §6.3).
    const FACTOR_RESIDUAL_TAU: f64 = 20.0;

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

    /// $\lVert M - I\rVert_\infty$ for a square matrix.
    fn _inf_norm_from_identity<const N: usize>(m: &Owned<f64, N, N>) -> f64
    where
        Const<N>: Dim,
    {
        let ident = Owned::<f64, N, N>::identity();
        let mut best = 0.0_f64;
        for i in 0..N {
            let mut row = 0.0_f64;
            for j in 0..N {
                row +=
                    (*m.get(i, j).unwrap() - *ident.get(i, j).unwrap()).abs();
            }
            best = best.max(row);
        }
        best
    }

    /// $A A^{-1} = I$ with $\varepsilon$ scaled by $\kappa_\infty(A)$.
    fn _assert_inv_identity_roundtrip<const N: usize>(a: &Owned<f64, N, N>)
    where
        Const<N>: Dim,
    {
        let lu = a.into_lu().unwrap();
        let inv = lu.inverse().unwrap();
        let product = a * &inv;
        let err = _inf_norm_from_identity(&product);
        let kappa = a.inf_norm() * inv.inf_norm();
        let bound = INV_ROUNDTRIP_TAU * kappa * f64::EPSILON;
        assert!(
            err <= bound.max(f64::EPSILON),
            "||A Ainv - I||_inf={err} exceeds tau*kappa*eps={bound} (kappa={kappa})"
        );
    }

    /// $\lVert A - B\rVert_\infty / (N \lVert A\rVert_\infty \varepsilon)$.
    fn _factor_residual_ratio<const N: usize>(
        a: &Owned<f64, N, N>,
        recon: &Owned<f64, N, N>,
    ) -> f64
    where
        Const<N>: Dim,
    {
        let mut num = 0.0_f64;
        for i in 0..N {
            let mut row = 0.0_f64;
            for j in 0..N {
                row +=
                    (*a.get(i, j).unwrap() - *recon.get(i, j).unwrap()).abs();
            }
            num = num.max(row);
        }
        let den = (N as f64) * a.inf_norm() * f64::EPSILON;
        if den == 0.0 { 0.0 } else { num / den }
    }

    fn _swap_rows<const N: usize>(m: &mut Owned<f64, N, N>, i: usize, j: usize)
    where
        Const<N>: Dim,
    {
        if i == j {
            return;
        }
        for c in 0..N {
            let vi = *m.get(i, c).unwrap();
            let vj = *m.get(j, c).unwrap();
            *m.get_mut(i, c).unwrap() = vj;
            *m.get_mut(j, c).unwrap() = vi;
        }
    }

    /// Exact $M = M^T$ (bitwise), for operators that return a symmetric matrix.
    fn _assert_exactly_symmetric<const N: usize>(m: &Owned<f64, N, N>)
    where
        Const<N>: Dim,
    {
        for i in 0..N {
            for j in 0..N {
                assert_eq!(
                    *m.get(i, j).unwrap(),
                    *m.get(j, i).unwrap(),
                    "symmetry ({i},{j})"
                );
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
        lu.solve_mut(&mut b).unwrap();
        assert_almost_eq!(*b.get(0, 0).unwrap(), 0.8, 1e-9);
        assert_almost_eq!(*b.get(1, 0).unwrap(), 1.4, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// Packed $PA = LU$ residual ratio $< 20$ (`matrix-design.md` §6.3).
    fn test_lu_factor_residual() {
        let a: Owned<f64, 3, 3> = Matrix::from_fn(|i, j| {
            [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]][i][j]
        });
        let mut packed = a;
        let mut pivots = [0usize; 3];
        packed.lu_decompose_mut(&mut pivots).unwrap();

        let mut pa = a;
        for k in 0..3 {
            _swap_rows(&mut pa, k, pivots[k]);
        }

        let mut l = Owned::<f64, 3, 3>::identity();
        let mut u = Owned::<f64, 3, 3>::zero();
        for i in 0..3 {
            for j in 0..3 {
                let v = *packed.get(i, j).unwrap();
                if i > j {
                    *l.get_mut(i, j).unwrap() = v;
                } else {
                    *u.get_mut(i, j).unwrap() = v;
                }
            }
        }
        let lu = &l * &u;
        let ratio = _factor_residual_ratio(&pa, &lu);
        assert!(
            ratio < FACTOR_RESIDUAL_TAU,
            "||PA - LU|| residual ratio {ratio} >= {FACTOR_RESIDUAL_TAU}"
        );
    }

    #[cfg_attr(test, test)]
    /// Undersized pivot scratch returns `WorkspaceTooSmall` instead of panicking.
    fn test_lu_decompose_undersized_pivots() {
        let mut a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, 1.0], [1.0, 3.0]][i][j]);
        let mut pivots = [0usize; 1];
        assert_eq!(
            a.lu_decompose_mut(&mut pivots),
            Err(LinAlgError::WorkspaceTooSmall)
        );
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
    /// $A A^{-1} = I$ with $\varepsilon$ scaled by $\kappa_\infty(A)$
    /// (`matrix-design.md` §6.3 linear-solve / inversion residual).
    fn test_inverse_identity_roundtrip_cond_scaled() {
        let well: Owned<f64, 3, 3> = Matrix::from_fn(|i, j| {
            [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]][i][j]
        });
        _assert_inv_identity_roundtrip(&well);

        let hilbert: Owned<f64, 4, 4> =
            Matrix::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
        _assert_inv_identity_roundtrip(&hilbert);
    }

    #[cfg_attr(test, test)]
    /// Induced $\infty$-norm is sub-multiplicative: $\lVert AB\rVert_\infty
    /// \le \lVert A\rVert_\infty \lVert B\rVert_\infty$ (FR-2).
    fn test_inf_norm_submultiplicative() {
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[2.0, -1.0], [0.5, 3.0]][i][j]);
        let b: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[-1.0, 4.0], [2.0, 0.5]][i][j]);
        let ab = &a * &b;
        let lhs = ab.inf_norm();
        let rhs = a.inf_norm() * b.inf_norm();
        assert!(lhs <= rhs, "||AB||_∞={lhs} exceeds ||A||_∞||B||_∞={rhs}");
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
        ldlt.solve_mut(&mut b).unwrap();
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
        chol.solve_mut(&mut b).unwrap();
        assert_almost_eq!(*b.get(0, 0).unwrap(), 1.0, 1e-9);
        assert_almost_eq!(*b.get(1, 0).unwrap(), 1.0, 1e-9);
    }

    #[cfg_attr(test, test)]
    /// $A - L L^T$ residual ratio $< 20$ (`matrix-design.md` §6.3).
    fn test_cholesky_factor_residual() {
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[4.0, 2.0], [2.0, 3.0]][i][j]);
        let mut sym = Symmetric::from_owned(a).unwrap();
        sym.cholesky_decompose_mut().unwrap();
        let l = *sym.as_matrix();
        let llt = &l * &l.transpose();
        let ratio = _factor_residual_ratio(&a, &llt);
        assert!(
            ratio < FACTOR_RESIDUAL_TAU,
            "||A - LL^T|| residual ratio {ratio} >= {FACTOR_RESIDUAL_TAU}"
        );
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
    /// $\lVert Q^T Q - I\rVert_\infty < N\varepsilon$ (`matrix-design.md` §6.3).
    fn test_qr_orthogonality() {
        let a: Owned<f64, 3, 3> = Matrix::from_fn(|i, j| {
            [[4.0, 3.0, 2.0], [1.0, 5.0, 3.0], [2.0, 1.0, 6.0]][i][j]
        });
        let mut r = a;
        let mut q = Owned::<f64, 3, 3>::zero();
        r.qr_decompose_mut(&mut q);
        let qtq = &q.transpose() * &q;
        let err = _inf_norm_from_identity(&qtq);
        let bound = 3.0 * f64::EPSILON;
        assert!(err < bound, "||Q^T Q - I||_inf={err} exceeds N eps={bound}");
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
    fn test_kalman_covariance_update() {
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

    #[cfg_attr(test, test)]
    /// Covariance time-update $P^- = A P A^T + Q$ and Joseph measurement
    /// update return bitwise-symmetric $P$ (`matrix-design.md` §6.2.1).
    fn test_covariance_predict_symmetry() {
        let a: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[1.0, 0.5], [0.0, 1.0]][i][j]);
        let p: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[4.0, 1.0], [1.0, 2.0]][i][j]);
        let q: Owned<f64, 2, 2> =
            Matrix::from_fn(|i, j| [[0.25, 0.0], [0.0, 0.5]][i][j]);
        let ap = &a * &p;
        let predicted = &(&ap * &a.transpose()) + &q;
        _assert_exactly_symmetric(&predicted);

        let ident: Owned<f64, 2, 2> = Owned::identity();
        let k: Owned<f64, 2, 1> = Matrix::from_fn(|i, _| [0.5, 0.25][i]);
        let h: Owned<f64, 1, 2> = Matrix::from_fn(|_, j| [1.0, 0.0][j]);
        let r: Owned<f64, 1, 1> = Matrix::from_fn(|_, _| 1.0);
        let i_kh = &ident - &(&k * &h);
        let joseph_left = &(&i_kh * &predicted) * &i_kh.transpose();
        let kr = &k * &r;
        let joseph = &joseph_left + &(&kr * &k.transpose());
        _assert_exactly_symmetric(&joseph);
    }

    #[cfg_attr(test, test)]
    fn test_coordinate_access() {
        let mut m: Owned<f64, 2, 2> = Owned::zero();
        m.set(0, 1, 3.0).unwrap();
        assert_eq!(*m.get(0, 1).unwrap(), 3.0);
        assert!(m.set(2, 0, 1.0).is_err());
        assert!(m.get(2, 0).is_none());
    }

    #[cfg_attr(test, test)]
    fn test_strided_submatrix() {
        let m: Owned<f64, 3, 3> = Owned::from_fn(|i, j| (i * 3 + j) as f64);
        let sub = m.submatrix::<2, 2>(1, 1).unwrap();
        assert_eq!(*sub.get(0, 0).unwrap(), *m.get(1, 1).unwrap());
        assert_eq!(*sub.get(1, 1).unwrap(), *m.get(2, 2).unwrap());
        assert!(m.submatrix::<2, 2>(2, 2).is_none());
        let rev = m.reverse_view();
        assert_eq!(*rev.get(0, 0).unwrap(), *m.get(2, 2).unwrap());
        let sl = m.slice();
        assert_eq!(sl.as_slice().len(), 9);
    }

    #[cfg_attr(test, test)]
    fn test_c_abi_layout() {
        let m: Owned<f64, 2, 2> = Owned::from_array([[1.0, 2.0], [3.0, 4.0]]);
        let slice = m.as_slice();
        assert_eq!(slice.len(), 4);
        assert_eq!(
            core::mem::size_of_val(slice),
            4 * core::mem::size_of::<f64>()
        );
    }

    #[cfg_attr(test, test)]
    fn test_lu_workspace_too_small() {
        let mut a: Owned<f64, 2, 2> = Owned::identity();
        let mut pivots = [0usize; 1];
        assert_eq!(
            a.lu_decompose_mut(&mut pivots),
            Err(LinAlgError::WorkspaceTooSmall)
        );
    }

    #[cfg_attr(test, test)]
    fn test_qr_rect_tall() {
        let a: Owned<f64, 3, 2> =
            Owned::from_fn(|i, j| [[1.0, 0.0], [1.0, 1.0], [0.0, 1.0]][i][j]);
        let qr = a.into_qr_rect().unwrap();
        let recon = &qr.q * &qr.r;
        for i in 0..3 {
            for j in 0..2 {
                assert_almost_eq!(
                    *recon.get(i, j).unwrap(),
                    *a.get(i, j).unwrap(),
                    1e-9
                );
            }
        }
    }

    #[cfg_attr(test, test)]
    fn test_matrix_storage_views_and_mul_into() {
        let mut m = Owned::<f64, 2, 2>::diagonal([3.0, 4.0]);
        assert_eq!(m.cols(), 2);
        assert_eq!(m.rows(), 2);
        assert_eq!(m.storage().as_slice()[0], 3.0);
        *m.get_mut(0, 1).unwrap() = 1.0;
        {
            let mut sl = m.slice_mut();
            *sl.get_mut(1, 0).unwrap() = 2.0;
        }
        let v = m.view();
        assert_eq!(v.get(0, 0), Some(&3.0));
        let rv = m.reverse_view();
        assert!(rv.get(0, 0).is_some());
        {
            let mut vm = m.view_mut();
            *vm.get_mut(1, 1).unwrap() = 5.0;
        }
        assert!(m.submatrix::<2, 2>(1, 0).is_none());
        let ident = Owned::<f64, 2, 2>::identity();
        let mut out = Owned::<f64, 2, 2>::zero();
        m.mul_into(&ident, &mut out);
        assert_almost_eq!(*out.get(0, 0).unwrap(), 3.0);
        let stored = m.into_storage();
        assert_eq!(stored.as_slice()[3], 5.0);
    }

    #[cfg_attr(test, test)]
    fn test_scalar_mul_trace_expm_write_block() {
        let ident = Owned::<f64, 2, 2>::identity();
        let scaled = &ident * 3.0;
        assert_almost_eq!(*scaled.get(0, 0).unwrap(), 3.0);
        assert_almost_eq!(*scaled.get(1, 1).unwrap(), 3.0);
        let scaled_ref = &ident * &2.0;
        assert_almost_eq!(*scaled_ref.get(0, 0).unwrap(), 2.0);
        assert_almost_eq!(ident.trace(), 2.0);
        assert_almost_eq!(ident.inf_norm(), 1.0);

        let zero = Owned::<f64, 2, 2>::zero();
        let e0 = zero.expm();
        assert_almost_eq!(*e0.get(0, 0).unwrap(), 1.0);
        assert_almost_eq!(*e0.get(1, 1).unwrap(), 1.0);
        assert_almost_eq!(*e0.get(0, 1).unwrap(), 0.0);

        let mut dest = Owned::<f64, 3, 3>::zero();
        let src = Owned::<f64, 2, 2>::identity();
        dest.write_block(1, 1, &src);
        assert_almost_eq!(*dest.get(1, 1).unwrap(), 1.0);
        assert_almost_eq!(*dest.get(2, 2).unwrap(), 1.0);
        assert_almost_eq!(*dest.get(0, 0).unwrap(), 0.0);
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

        /// $A A^{-1} = I$ for non-singular $A$, with $\varepsilon$ scaled by
        /// $\kappa_\infty(A)$.
        #[test]
        fn prop_inverse_identity_roundtrip(
            vals in proptest::collection::vec(-20.0..20.0_f64, 4),
        ) {
            let a: Owned<f64, 2, 2> = matrix_from_vec(&vals);
            let Ok(lu) = a.into_lu() else {
                return Ok(());
            };
            let Ok(inv) = lu.inverse() else {
                return Ok(());
            };
            let product = &a * &inv;
            let ident = Owned::<f64, 2, 2>::identity();
            let mut err = 0.0_f64;
            for i in 0..2 {
                let mut row = 0.0_f64;
                for j in 0..2 {
                    row += (*product.get(i, j).unwrap()
                        - *ident.get(i, j).unwrap())
                    .abs();
                }
                err = err.max(row);
            }
            let kappa = a.inf_norm() * inv.inf_norm();
            if !kappa.is_finite() {
                return Ok(());
            }
            let bound = 20.0 * kappa * f64::EPSILON;
            prop_assert!(
                err <= bound.max(f64::EPSILON),
                "||AAinv-I||_inf={err} exceeds tau*kappa*eps={bound} (kappa={kappa})"
            );
        }

        /// $\lVert AB\rVert_\infty \le \lVert A\rVert_\infty \lVert B\rVert_\infty$
        /// up to a $\gamma_n$ rounding factor on the computed product.
        #[test]
        fn prop_inf_norm_submultiplicative(
            a_vals in proptest::collection::vec(-50.0..50.0_f64, 4),
            b_vals in proptest::collection::vec(-50.0..50.0_f64, 4),
        ) {
            let a: Owned<f64, 2, 2> = matrix_from_vec(&a_vals);
            let b: Owned<f64, 2, 2> = matrix_from_vec(&b_vals);
            let ab = &a * &b;
            let lhs = ab.inf_norm();
            let rhs = a.inf_norm() * b.inf_norm();
            let n = 2.0_f64;
            let ke = n * f64::EPSILON;
            let gamma = if ke >= 1.0 { f64::INFINITY } else { ke / (1.0 - ke) };
            prop_assert!(
                lhs <= rhs * (1.0 + gamma) || rhs == 0.0,
                "||AB||_∞={lhs} exceeds ||A||||B||(1+γ₂)={}",
                rhs * (1.0 + gamma)
            );
        }
    }
}
