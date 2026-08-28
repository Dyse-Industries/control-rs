//! Dense `Matrix` demo. Copy this file and point `Store` / `Blas` at your backends.
//!
//! `Store::from_array` / `Owned::from_array` use **column-major** literals: each
//! inner array is one column. NumPy `[[a, b], [c, d]]` is `[[a, c], [b, d]]`.

use crate::{
    ABS_F64, SOLVE_RESIDUAL_TAU, native_artifact, owned_to_rows, print_matrix,
    save, solve_residual_ratio,
};
use control_rs::math::num_types::Const;
use control_rs::math::storage::ArrayStorage;
use control_rs::math::subprograms::DefaultBlas;
use control_rs::matrix::{LuDecomposition, Matrix, Owned};
use serde_json::json;

/// Swap this for a custom dense backend.
type Store<const R: usize, const C: usize> = ArrayStorage<f64, R, C>;
/// Swap this for a hardware BLAS (`Gemm` / `Getrf` / `Getrs`).
type Blas = DefaultBlas;
type Mat<const R: usize, const C: usize> =
    Matrix<f64, Const<R>, Const<C>, Store<R, C>>;

pub fn main() {
    println!("=== Matrix Numerical Model Example ===");

    let m1 =
        Mat::<2, 2>::from_storage(Store::from_array([[1.0, 3.0], [2.0, 4.0]]));
    let m2 =
        Mat::<2, 2>::from_storage(Store::from_array([[5.0, 7.0], [6.0, 8.0]]));

    println!("\n--- Matrix Construction & Basic Arithmetic ---");
    print_matrix("M1", &m1);
    print_matrix("M2", &m2);

    let sum: Mat<2, 2> = &m1 + &m2;
    let diff: Mat<2, 2> = &m2 - &m1;
    let mut prod = Mat::<2, 2>::zero();
    m1.mul_into_with::<Blas, Const<2>, _, _>(&m2, &mut prod);
    let trans: Mat<2, 2> = m1.transpose();

    print_matrix("M1 + M2", &sum);
    print_matrix("M2 - M1", &diff);
    print_matrix("M1 * M2", &prod);
    print_matrix("M1^T (Transpose)", &trans);

    let a_ref = Mat::<3, 3>::from_storage(Store::from_array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 0.5],
        [-1.0, 4.0, -1.0],
    ]));
    let b_ref =
        Mat::<3, 1>::from_storage(Store::from_array([[1.0, -2.0, 0.0]]));
    let a = Mat::<3, 3>::from_storage(Store::from_array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 0.5],
        [-1.0, 4.0, -1.0],
    ]));
    let mut x =
        Mat::<3, 1>::from_storage(Store::from_array([[1.0, -2.0, 0.0]]));

    println!("\n--- Linear System A * x = b ---");
    print_matrix("A", &a_ref);
    print_matrix("b", &b_ref);

    let lu = LuDecomposition::decompose_with::<Blas>(a).expect("LU decompose");
    lu.solve_mut_with::<Blas, 1>(&mut x).expect("LU solve");
    print_matrix("Solution x", &x);

    let a_owned: Owned<f64, 3, 3> = Owned::from_array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 0.5],
        [-1.0, 4.0, -1.0],
    ]);
    let b_owned: Owned<f64, 3, 1> = Owned::from_array([[1.0, -2.0, 0.0]]);
    let x_owned = Owned::<f64, 3, 1>::from_array([[
        x.get(0, 0).copied().unwrap(),
        x.get(1, 0).copied().unwrap(),
        x.get(2, 0).copied().unwrap(),
    ]]);
    let residual_ratio = solve_residual_ratio(&a_owned, &x_owned, &b_owned);
    println!("Solve residual ratio: {residual_ratio:.6e}");
    assert!(
        residual_ratio < SOLVE_RESIDUAL_TAU,
        "solve residual ratio {residual_ratio} exceeds {SOLVE_RESIDUAL_TAU}"
    );

    let a_inv = lu.inverse_with::<Blas>().expect("inverse");
    println!("\n--- Matrix Inversion & Identity Check ---");
    print_matrix("A^-1", &a_inv);
    let mut ident = Mat::<3, 3>::zero();
    a_ref.mul_into_with::<Blas, Const<3>, _, _>(&a_inv, &mut ident);
    print_matrix("A * A^-1 (Identity check)", &ident);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            let got = ident.get(i, j).copied().unwrap();
            assert!(
                (got - expected).abs() <= ABS_F64,
                "identity ({i},{j}): {got} vs {expected}"
            );
        }
    }

    let values = json!({
        "SUM": owned_to_rows(&sum),
        "DIFF": owned_to_rows(&diff),
        "PROD": owned_to_rows(&prod),
        "TRANSPOSE": owned_to_rows(&trans),
        "X": owned_to_rows(&x),
        "A_INV": owned_to_rows(&a_inv),
        "residual_ratio": residual_ratio,
    });
    save(
        "results/matrix/native.json",
        &native_artifact("matrix", values, json!({})),
    );
}
