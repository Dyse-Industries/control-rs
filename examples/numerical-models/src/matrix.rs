//! Dense `Matrix` demo. Copy this file and point `Store` / `Blas` at your backends.
//!
//! `Store::from_array` / `Owned::from_array` use **column-major** literals: each
//! inner array is one column. NumPy `[[a, b], [c, d]]` is `[[a, c], [b, d]]`.

use crate::{
    ABS_F64, SOLVE_RESIDUAL_TAU, col0, frobenius, inf_norm_mat,
    native_artifact, owned_to_rows, print_matrix, save, solve_residual_ratio,
    time_kernel, timing_entry,
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

const GEMM_N: usize = 64;
const GEMM_ITERS: u32 = 200;
const HILBERT_N: usize = 8;
const SE3_N: usize = 40;
const SE3_THETA: f64 = 0.15;
const SE3_DX: f64 = 0.04;
const SE3_DY: f64 = 0.01;
const SE3_DZ: f64 = 0.03;

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

    println!("\n--- Hilbert n={HILBERT_N} ---");
    let h = Mat::<8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let mut b_h = Mat::<8, 1>::zero();
    let x_true = Mat::<8, 1>::from_fn(|_, _| 1.0);
    h.mul_into_with::<Blas, Const<1>, _, _>(&x_true, &mut b_h);
    let h_for_lu = Mat::<8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let lu_h =
        LuDecomposition::decompose_with::<Blas>(h_for_lu).expect("Hilbert LU");
    let mut x_h = b_h;
    lu_h.solve_mut_with::<Blas, 1>(&mut x_h)
        .expect("Hilbert solve");
    let h_inv = lu_h.inverse_with::<Blas>().expect("Hilbert inverse");
    let h_owned =
        Owned::<f64, 8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let mut b_h_owned = Owned::<f64, 8, 1>::zero();
    h_owned.mul_into(&Owned::<f64, 8, 1>::from_fn(|_, _| 1.0), &mut b_h_owned);
    let x_h_owned =
        Owned::<f64, 8, 1>::from_fn(|i, _| x_h.get(i, 0).copied().unwrap());
    let residual_ratio_hilbert =
        solve_residual_ratio(&h_owned, &x_h_owned, &b_h_owned);
    let kappa_hilbert = inf_norm_mat(&h_owned) * inf_norm_mat(&h_inv);
    println!("Hilbert residual ratio: {residual_ratio_hilbert:.6e}");
    println!("Hilbert kappa_inf: {kappa_hilbert:.6e}");
    assert!(
        residual_ratio_hilbert < SOLVE_RESIDUAL_TAU,
        "Hilbert residual ratio {residual_ratio_hilbert} exceeds {SOLVE_RESIDUAL_TAU}"
    );

    println!("\n--- Timed GEMM n={GEMM_N} ---");
    let ga = Mat::<64, 64>::from_fn(|i, j| {
        0.01 * ((i + 1) as f64) * ((j + 3) as f64) / 64.0
    });
    let gb = Mat::<64, 64>::from_fn(|i, j| {
        0.02 * ((i + 2) as f64) * ((j + 1) as f64) / 64.0
    });
    let mut gc = Mat::<64, 64>::zero();
    ga.mul_into_with::<Blas, Const<64>, _, _>(&gb, &mut gc);
    let gemm00 = gc.get(0, 0).copied().unwrap();
    let gemm_frob = frobenius(&gc);
    let gemm_ns = time_kernel(GEMM_ITERS, || {
        ga.mul_into_with::<Blas, Const<64>, _, _>(&gb, &mut gc);
        gc.get(0, 0).copied()
    });
    println!("GEMM C[0,0] = {gemm00:.10e}");
    println!("GEMM ||C||_F = {gemm_frob:.6e}");
    println!("GEMM min ns ({GEMM_ITERS} iters): {gemm_ns}");

    println!("\n--- SE(3) GEMM chain n={SE3_N} ---");
    let (c_th, s_th) = (SE3_THETA.cos(), SE3_THETA.sin());
    let t_se3 = Mat::<4, 4>::from_storage(Store::from_array([
        [c_th, s_th, 0.0, 0.0],
        [-s_th, c_th, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [SE3_DX, SE3_DY, SE3_DZ, 1.0],
    ]));
    let mut pose = Mat::<4, 4>::identity();
    let mut se3_xyz = Vec::with_capacity(SE3_N);
    let mut se3_r = Vec::with_capacity(SE3_N);
    for _ in 0..SE3_N {
        se3_xyz.push(vec![
            pose.get(0, 3).copied().unwrap(),
            pose.get(1, 3).copied().unwrap(),
            pose.get(2, 3).copied().unwrap(),
        ]);
        let mut rot_rows = Vec::with_capacity(3);
        for i in 0..3 {
            rot_rows.push(vec![
                pose.get(i, 0).copied().unwrap(),
                pose.get(i, 1).copied().unwrap(),
                pose.get(i, 2).copied().unwrap(),
            ]);
        }
        se3_r.push(rot_rows);
        let mut next = Mat::<4, 4>::zero();
        t_se3.mul_into_with::<Blas, Const<4>, _, _>(&pose, &mut next);
        pose = next;
    }
    println!(
        "T^0 xyz = [{:.6}, {:.6}, {:.6}]",
        se3_xyz[0][0], se3_xyz[0][1], se3_xyz[0][2]
    );
    println!(
        "T^{} xyz = [{:.6}, {:.6}, {:.6}]",
        SE3_N - 1,
        se3_xyz[SE3_N - 1][0],
        se3_xyz[SE3_N - 1][1],
        se3_xyz[SE3_N - 1][2]
    );

    let hilbert_x = col0(&x_h);
    let idx: Vec<f64> = (0..HILBERT_N).map(|i| i as f64).collect();
    let values = json!({
        "SUM": owned_to_rows(&sum),
        "DIFF": owned_to_rows(&diff),
        "PROD": owned_to_rows(&prod),
        "TRANSPOSE": owned_to_rows(&trans),
        "X": owned_to_rows(&x),
        "A_INV": owned_to_rows(&a_inv),
        "HILBERT_X": hilbert_x,
        "HILBERT_A_INV": owned_to_rows(&h_inv),
        "GEMM00": gemm00,
        "SE3_T": owned_to_rows(&t_se3),
        "SE3_XYZ": se3_xyz,
        "SE3_R": se3_r,
    });
    let series = json!({
        "hilbert_x": { "x": idx, "y": hilbert_x },
    });
    let metrics = json!({
        "residual_ratio": residual_ratio,
        "residual_ratio_hilbert": residual_ratio_hilbert,
        "kappa_hilbert": kappa_hilbert,
        "gemm_frob": gemm_frob,
    });
    let timings = json!({
        "gemm": timing_entry(GEMM_ITERS, gemm_ns),
    });
    save(
        "results/matrix/native.json",
        &native_artifact("matrix", values, series, metrics, timings),
    );
}
