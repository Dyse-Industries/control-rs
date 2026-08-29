//! Dense `Matrix` demo. Copy this file and point `Store` / `Blas` at your backends.
//!
//! Default `run` uses column-major `ArrayStorage` and `DefaultBlas`. `run_row`
//! swaps in `RowArrayStorage`. `run_accelerate` (feature `accelerate`) uses
//! `RowArrayStorage` plus Apple Accelerate `Gemm`. LU / Hilbert stay on
//! `Owned` + `DefaultBlas` (`Getrf` is not on `AccelerateBlas`).

use crate::suite::{
    case_inputs, emit_stdout, json_f64, json_f64_vec, json_usize,
    require_usize, storage_from_col, storage_from_rows, storage_from_rows_init,
};
use crate::{
    ABS_F64, SOLVE_RESIDUAL_TAU, col0, frobenius, inf_norm_mat,
    native_artifact, owned_to_rows, print_matrix, solve_residual_ratio,
    time_kernel, timing_entry,
};
use control_rs::math::num_types::Const;
use control_rs::math::storage::{
    ArrayStorage, DenseStorageMut, RowArrayStorage, StorageInit,
};
use control_rs::math::subprograms::DefaultBlas;
use control_rs::math::subprograms::level3::Gemm;
use control_rs::matrix::{LuDecomposition, Matrix, Owned};
use serde_json::{Value, json};

/// Dense owning leaf used by a matrix specialization.
pub trait DenseLeaf {
    /// Stack storage for an `R × C` `f64` matrix.
    type Store<const R: usize, const C: usize>: Clone
        + DenseStorageMut<f64, R = Const<R>, C = Const<C>>
        + StorageInit<f64, Const<R>, Const<C>>
    where
        Const<R>: control_rs::math::num_types::Dim,
        Const<C>: control_rs::math::num_types::Dim;
}

/// Column-major `ArrayStorage` leaf (default tutorial).
pub struct ColMajorLeaf;
/// Row-major `RowArrayStorage` leaf (NumPy C-order / Accelerate GEMM).
pub struct RowMajorLeaf;

impl DenseLeaf for ColMajorLeaf {
    type Store<const R: usize, const C: usize>
        = ArrayStorage<f64, R, C>
    where
        Const<R>: control_rs::math::num_types::Dim,
        Const<C>: control_rs::math::num_types::Dim;
}

impl DenseLeaf for RowMajorLeaf {
    type Store<const R: usize, const C: usize>
        = RowArrayStorage<f64, R, C>
    where
        Const<R>: control_rs::math::num_types::Dim,
        Const<C>: control_rs::math::num_types::Dim;
}

const GEMM_N: usize = 64;
const HILBERT_N: usize = 8;
const SE3_N: usize = 40;

fn mat_from_rows<L, const R: usize, const C: usize>(
    v: &Value,
) -> Matrix<f64, Const<R>, Const<C>, L::Store<R, C>>
where
    L: DenseLeaf,
    Const<R>: control_rs::math::num_types::Dim,
    Const<C>: control_rs::math::num_types::Dim,
{
    Matrix::from_storage(storage_from_rows_init::<L::Store<R, C>, R, C>(v))
}

fn mat_zero<L, const R: usize, const C: usize>()
-> Matrix<f64, Const<R>, Const<C>, L::Store<R, C>>
where
    L: DenseLeaf,
    Const<R>: control_rs::math::num_types::Dim,
    Const<C>: control_rs::math::num_types::Dim,
{
    Matrix::from_storage(L::Store::<R, C>::zeros())
}

/// Column-major `ArrayStorage` + `DefaultBlas`.
pub fn run(suite: &Value) {
    run_leaf::<ColMajorLeaf, DefaultBlas>(suite, "rust");
}

/// Row-major `RowArrayStorage` + `DefaultBlas`.
pub fn run_row(suite: &Value) {
    run_leaf::<RowMajorLeaf, DefaultBlas>(suite, "rust-row");
}

/// Row-major storage + Apple Accelerate `Gemm`.
#[cfg(feature = "accelerate")]
pub fn run_accelerate(suite: &Value) {
    run_leaf::<RowMajorLeaf, aarch64_subprograms::AccelerateBlas>(
        suite,
        "rust-accelerate",
    );
}

fn run_leaf<L, B>(suite: &Value, source: &str)
where
    L: DenseLeaf,
    B: Gemm<f64, L::Store<2, 2>, L::Store<2, 2>, L::Store<2, 2>>
        + Gemm<f64, L::Store<64, 64>, L::Store<64, 64>, L::Store<64, 64>>
        + Gemm<f64, L::Store<4, 4>, L::Store<4, 4>, L::Store<4, 4>>,
{
    eprintln!("=== Matrix Numerical Model Example ({source}) ===");

    let arith = case_inputs(suite, "matrix.host.arithmetic");
    let m1 = mat_from_rows::<L, 2, 2>(&arith["M1"]);
    let m2 = mat_from_rows::<L, 2, 2>(&arith["M2"]);

    eprintln!("\n--- Matrix Construction & Basic Arithmetic ---");
    print_matrix("M1", &m1);
    print_matrix("M2", &m2);

    let sum = &m1 + &m2;
    let diff = &m2 - &m1;
    let mut prod = mat_zero::<L, 2, 2>();
    m1.mul_into_with::<B, Const<2>, _, _>(&m2, &mut prod);
    let trans =
        Owned::<f64, 2, 2>::from_fn(|i, j| m1.get(j, i).copied().unwrap());

    print_matrix("M1 + M2", &sum);
    print_matrix("M2 - M1", &diff);
    print_matrix("M1 * M2", &prod);
    print_matrix("M1^T (Transpose)", &trans);

    let lu_in = case_inputs(suite, "matrix.host.lu_solve_inverse");
    let a_ref =
        Owned::<f64, 3, 3>::from_storage(storage_from_rows(&lu_in["A"]));
    let b_ref = Owned::<f64, 3, 1>::from_storage(storage_from_col(&lu_in["b"]));
    let a = Owned::<f64, 3, 3>::from_storage(storage_from_rows(&lu_in["A"]));
    let mut x = Owned::<f64, 3, 1>::from_storage(storage_from_col(&lu_in["b"]));

    eprintln!("\n--- Linear System A * x = b ---");
    print_matrix("A", &a_ref);
    print_matrix("b", &b_ref);

    let lu = LuDecomposition::decompose_with::<DefaultBlas>(a)
        .expect("LU decompose");
    lu.solve_mut_with::<DefaultBlas, 1>(&mut x)
        .expect("LU solve");
    print_matrix("Solution x", &x);

    let residual_ratio = solve_residual_ratio(&a_ref, &x, &b_ref);
    eprintln!("Solve residual ratio: {residual_ratio:.6e}");
    assert!(
        residual_ratio < SOLVE_RESIDUAL_TAU,
        "solve residual ratio {residual_ratio} exceeds {SOLVE_RESIDUAL_TAU}"
    );

    let a_inv = lu.inverse_with::<DefaultBlas>().expect("inverse");
    eprintln!("\n--- Matrix Inversion & Identity Check ---");
    print_matrix("A^-1", &a_inv);
    let mut ident = Owned::<f64, 3, 3>::zero();
    a_ref.mul_into_with::<DefaultBlas, Const<3>, _, _>(&a_inv, &mut ident);
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

    let hilbert = case_inputs(suite, "matrix.host.hilbert");
    require_usize(hilbert, "n", HILBERT_N);
    eprintln!("\n--- Hilbert n={HILBERT_N} ---");
    let h = Owned::<f64, 8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let mut b_h = Owned::<f64, 8, 1>::zero();
    let x_true = Owned::<f64, 8, 1>::from_fn(|_, _| 1.0);
    h.mul_into_with::<DefaultBlas, Const<1>, _, _>(&x_true, &mut b_h);
    let h_for_lu =
        Owned::<f64, 8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let lu_h = LuDecomposition::decompose_with::<DefaultBlas>(h_for_lu)
        .expect("Hilbert LU");
    let mut x_h = b_h;
    lu_h.solve_mut_with::<DefaultBlas, 1>(&mut x_h)
        .expect("Hilbert solve");
    let h_inv = lu_h.inverse_with::<DefaultBlas>().expect("Hilbert inverse");
    let h_owned =
        Owned::<f64, 8, 8>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let mut b_h_owned = Owned::<f64, 8, 1>::zero();
    h_owned.mul_into(&Owned::<f64, 8, 1>::from_fn(|_, _| 1.0), &mut b_h_owned);
    let residual_ratio_hilbert =
        solve_residual_ratio(&h_owned, &x_h, &b_h_owned);
    let kappa_hilbert = inf_norm_mat(&h_owned) * inf_norm_mat(&h_inv);
    eprintln!("Hilbert residual ratio: {residual_ratio_hilbert:.6e}");
    eprintln!("Hilbert kappa_inf: {kappa_hilbert:.6e}");
    assert!(
        residual_ratio_hilbert < SOLVE_RESIDUAL_TAU,
        "Hilbert residual ratio {residual_ratio_hilbert} exceeds {SOLVE_RESIDUAL_TAU}"
    );

    let gemm = case_inputs(suite, "matrix.host.gemm");
    require_usize(gemm, "n", GEMM_N);
    let gemm_iters = json_usize(&gemm["iters"]).unwrap_or(200) as u32;
    eprintln!("\n--- Timed GEMM n={GEMM_N} ---");
    let ga = Matrix::<f64, Const<64>, Const<64>, L::Store<64, 64>>::from_fn(
        |i, j| 0.01 * ((i + 1) as f64) * ((j + 3) as f64) / 64.0,
    );
    let gb = Matrix::<f64, Const<64>, Const<64>, L::Store<64, 64>>::from_fn(
        |i, j| 0.02 * ((i + 2) as f64) * ((j + 1) as f64) / 64.0,
    );
    let mut gc = mat_zero::<L, 64, 64>();
    ga.mul_into_with::<B, Const<64>, _, _>(&gb, &mut gc);
    let gemm00 = gc.get(0, 0).copied().unwrap();
    let gemm_frob = frobenius(&gc);
    let gemm_ns = time_kernel(gemm_iters, || {
        ga.mul_into_with::<B, Const<64>, _, _>(&gb, &mut gc);
        gc.get(0, 0).copied()
    });
    eprintln!("GEMM C[0,0] = {gemm00:.10e}");
    eprintln!("GEMM ||C||_F = {gemm_frob:.6e}");
    eprintln!("GEMM min ns ({gemm_iters} iters): {gemm_ns}");

    let se3 = case_inputs(suite, "matrix.host.se3_chain");
    require_usize(se3, "n", SE3_N);
    let theta = json_f64(&se3["theta"]);
    let tvec = json_f64_vec(&se3["t"]);
    eprintln!("\n--- SE(3) GEMM chain n={SE3_N} ---");
    let (c_th, s_th) = (theta.cos(), theta.sin());
    let t_se3 = Matrix::<f64, Const<4>, Const<4>, L::Store<4, 4>>::from_fn(
        |i, j| match (i, j) {
            (0, 0) => c_th,
            (0, 1) => -s_th,
            (0, 3) => tvec[0],
            (1, 0) => s_th,
            (1, 1) => c_th,
            (1, 3) => tvec[1],
            (2, 2) => 1.0,
            (2, 3) => tvec[2],
            (3, 3) => 1.0,
            _ => 0.0,
        },
    );
    let mut pose =
        Matrix::<f64, Const<4>, Const<4>, L::Store<4, 4>>::from_storage(
            L::Store::<4, 4>::identity(),
        );
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
        let mut next = mat_zero::<L, 4, 4>();
        t_se3.mul_into_with::<B, Const<4>, _, _>(&pose, &mut next);
        pose = next;
    }
    eprintln!(
        "T^0 xyz = [{:.6}, {:.6}, {:.6}]",
        se3_xyz[0][0], se3_xyz[0][1], se3_xyz[0][2]
    );
    eprintln!(
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
        "gemm": timing_entry(gemm_iters, gemm_ns),
    });
    emit_stdout(&native_artifact(
        "matrix", source, values, series, metrics, timings,
    ));
}
