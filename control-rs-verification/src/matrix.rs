//! Matrix numerical model control-rs-verification suite.
//!
//! Evaluates linear algebra kernels:
//! 1. 8x8 EKF state covariance update recursion
//! 2. Ill-conditioned Hilbert solve ($H_{10} x = b$) & backward error residual
//! 3. Vandermonde polynomial interpolation solve
//! 4. Cholesky solve across graded eigenvalue spread
//! 5. QR decomposition orthogonality loss

#![allow(
    missing_docs,
    clippy::arithmetic_side_effects,
    clippy::cast_precision_loss,
    clippy::indexing_slicing,
    clippy::suboptimal_flops,
    clippy::too_many_lines,
    clippy::unwrap_used
)]

use std::path::Path;

use control_rs::matrix::{LuDecomposition, Owned, Symmetric};

use crate::h5_writer::H5Writer;

/// Generates 8x8 EKF covariance recursion matrix.
fn generate_ekf_covariance() -> Vec<f64> {
    const N: usize = 8;
    let p_0 = Owned::<f64, N, N>::from_fn(|i, j| {
        let diff = (i as f64) - (j as f64);
        (-0.25 * diff * diff).exp() + if i == j { 0.1 } else { 0.0 }
    });

    let h = Owned::<f64, N, N>::identity();
    let k = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.4 } else { 0.01 });
    let r = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.05 } else { 0.0 });

    let kh = &k * &h;
    let eye = Owned::<f64, N, N>::identity();
    let i_minus_kh = &eye - &kh;

    let k_t = k.transpose();
    let kr = &k * &r;
    let krk_t = &kr * &k_t;
    let i_minus_kh_t = i_minus_kh.transpose();

    let mut p_current = p_0;
    for _ in 0..100 {
        let p_temp = &i_minus_kh * &p_current;
        let p_update1 = &p_temp * &i_minus_kh_t;
        p_current = &p_update1 + &krk_t;
    }

    let mut out = Vec::with_capacity(N * N);
    for i in 0..N {
        for j in 0..N {
            out.push(*p_current.get(i, j).expect("in bounds"));
        }
    }
    out
}

fn hilbert_solve() -> (Vec<f64>, f64) {
    const N: usize = 10;
    let h = Owned::<f64, N, N>::from_fn(|i, j| 1.0 / ((i + j + 1) as f64));
    let ones = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    let b = &h * &ones;

    let lu = LuDecomposition::decompose(h).expect("Hilbert LU");
    let mut x = b;
    lu.solve_mut(&mut x).expect("Hilbert solve");

    let hx = &h * &x;
    let mut max_res = 0.0_f64;
    let mut x_vec = Vec::with_capacity(N);

    for i in 0..N {
        let val_x = *x.get(i, 0).expect("in bounds");
        let val_hx = *hx.get(i, 0).expect("in bounds");
        let val_b = *b.get(i, 0).expect("in bounds");
        let res = (val_hx - val_b).abs();
        if res > max_res {
            max_res = res;
        }
        x_vec.push(val_x);
    }

    (x_vec, max_res)
}

fn vandermonde_solve() -> Vec<f64> {
    const N: usize = 8;
    let v = Owned::<f64, N, N>::from_fn(|i, j| {
        let x = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        let power = i32::try_from(j).expect("column index fits in i32");
        x.powi(power)
    });
    let y = Owned::<f64, N, 1>::from_fn(|i, _| {
        let x = -1.0 + 2.0 * (i as f64) / ((N - 1) as f64);
        1.0 / (1.0 + 25.0 * x * x)
    });

    let lu = LuDecomposition::decompose(v).expect("Vandermonde LU");
    let mut c = y;
    lu.solve_mut(&mut c).expect("Vandermonde solve");

    (0..N).map(|i| *c.get(i, 0).expect("in bounds")).collect()
}

fn cholesky_spread_solve() -> Vec<f64> {
    const N: usize = 8;
    let a = Owned::<f64, N, N>::from_fn(|i, j| {
        let scale = |k: usize| 10f64.powf(-10.0 * (k as f64) / 7.0);
        if i == j {
            scale(i) + 1e-3
        } else {
            1e-3 * (scale(i) * scale(j)).sqrt()
        }
    });
    let spd = Symmetric::<f64, N>::from_owned(a).expect("symmetric matrix");
    let chol = spd.into_cholesky().expect("Cholesky decomposition");

    let mut x = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    chol.solve_mut(&mut x).expect("Cholesky solve");

    (0..N).map(|i| *x.get(i, 0).expect("in bounds")).collect()
}

fn qr_orthogonality_loss() -> f64 {
    const N: usize = 6;
    let mut a = Owned::<f64, N, N>::from_fn(|i, j| {
        let base = 1.0 / ((i + j + 1) as f64);
        if j == 4 {
            1.0 / ((i + 3 + 1) as f64) + 1e-9
        } else {
            base
        }
    });
    let mut q = Owned::<f64, N, N>::zero();
    a.qr_decompose_mut(&mut q);

    let qt_q = &q.transpose() * &q;
    let eye = Owned::<f64, N, N>::identity();
    let diff = &qt_q - &eye;

    let mut sum_sq = 0.0;
    for i in 0..N {
        for j in 0..N {
            let val = *diff.get(i, j).expect("in bounds");
            sum_sq += val * val;
        }
    }
    sum_sq.sqrt()
}

/// Executes the matrix control-rs-verification kernel and writes `results/matrix.rust.h5`.
pub fn emit_container(output_path: &Path) -> Result<(), String> {
    let mut writer = H5Writer::new();

    let cov = generate_ekf_covariance();
    writer.add_dataset("covariance_heatmap/matrix", &cov);
    writer.set_tolerance("covariance_heatmap/matrix", "abs", 1e-4);

    let (hx, h_res) = hilbert_solve();
    writer.add_dataset("hilbert_solve/x", &hx);
    writer.set_tolerance("hilbert_solve/x", "abs", 0.05);
    writer.add_dataset("hilbert_solve/residual", &[h_res]);
    writer.set_tolerance("hilbert_solve/residual", "abs", 1e-12);

    let v_c = vandermonde_solve();
    writer.add_dataset("vandermonde_solve/x", &v_c);
    writer.set_tolerance("vandermonde_solve/x", "abs", 1e-3);

    let chol_x = cholesky_spread_solve();
    writer.add_dataset("cholesky_spread/x", &chol_x);
    writer.set_tolerance("cholesky_spread/x", "abs", 1e-3);

    let qr_loss = qr_orthogonality_loss();
    writer.add_dataset("qr_orthogonality/residual", &[qr_loss]);
    writer.set_tolerance("qr_orthogonality/residual", "abs", 1e-6);

    writer.write_to_file(output_path)
}
