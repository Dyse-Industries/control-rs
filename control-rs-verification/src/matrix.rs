//! Matrix numerical model control-rs-verification suite.
//!
//! Evaluates linear algebra kernels:
//! 1. 8x8 EKF state covariance update recursion
//! 2. Ill-conditioned Hilbert solve ($H_{10} x = b$) & backward error residual
//! 3. Vandermonde polynomial interpolation solve
//! 4. Cholesky solve across graded eigenvalue spread
//! 5. QR decomposition orthogonality loss

use std::path::Path;

use control_rs::matrix::{LuDecomposition, Owned, Symmetric};

use crate::h5_writer::H5Writer;
use crate::numeric::{
    KernelResult, SeriesResult, add, column, entry, index_f64, matmul,
    row_major, sub,
};

/// Solution and backward-error residual of the Hilbert system.
struct HilbertSolve {
    /// Solution vector `x` of `H x = b`.
    x: Vec<f64>,
    /// `max_i |(H x - b)_i|`.
    residual: f64,
}

/// Generates 8x8 EKF covariance recursion matrix.
fn generate_ekf_covariance() -> SeriesResult {
    const N: usize = 8;
    let p_0 = Owned::<f64, N, N>::from_fn(|i, j| {
        let diff = index_f64(i) - index_f64(j);
        (-0.25 * diff * diff).exp() + if i == j { 0.1 } else { 0.0 }
    });

    let h = Owned::<f64, N, N>::identity();
    let k = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.4 } else { 0.01 });
    let r = Owned::<f64, N, N>::from_fn(|i, j| if i == j { 0.05 } else { 0.0 });

    let kh = matmul(&k, &h);
    let eye = Owned::<f64, N, N>::identity();
    let i_minus_kh = sub(&eye, &kh);

    let k_t = k.transpose();
    let kr = matmul(&k, &r);
    let krk_t = matmul(&kr, &k_t);
    let i_minus_kh_t = i_minus_kh.transpose();

    let mut p_current = p_0;
    for _ in 0..100 {
        let p_temp = matmul(&i_minus_kh, &p_current);
        let p_update = matmul(&p_temp, &i_minus_kh_t);
        p_current = add(&p_update, &krk_t);
    }

    row_major(&p_current)
}

fn hilbert_solve() -> KernelResult<HilbertSolve> {
    const N: usize = 10;
    let h = Owned::<f64, N, N>::from_fn(|i, j| {
        1.0 / (index_f64(i) + index_f64(j) + 1.0)
    });
    let ones = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    let b = matmul(&h, &ones);

    let lu = LuDecomposition::decompose(h)
        .map_err(|e| format!("Hilbert LU: {e:?}"))?;
    let mut x = b;
    lu.solve_mut(&mut x)
        .map_err(|e| format!("Hilbert solve: {e:?}"))?;

    let product = matmul(&h, &x);
    let mut residual = 0.0_f64;
    for i in 0..N {
        residual =
            residual.max((entry(&product, i, 0)? - entry(&b, i, 0)?).abs());
    }

    Ok(HilbertSolve {
        x: column(&x, 0)?,
        residual,
    })
}

/// Equispaced sample `i` of `N` points on `[-1, 1]`.
fn equispaced_node<const N: usize>(i: usize) -> f64 {
    -1.0 + 2.0 * index_f64(i) / index_f64(N.saturating_sub(1))
}

fn vandermonde_solve() -> SeriesResult {
    const N: usize = 8;
    let v = Owned::<f64, N, N>::from_fn(|i, j| {
        let power = i32::try_from(j).unwrap_or(i32::MAX);
        equispaced_node::<N>(i).powi(power)
    });
    let y = Owned::<f64, N, 1>::from_fn(|i, _| {
        let x = equispaced_node::<N>(i);
        1.0 / (25.0 * x).mul_add(x, 1.0)
    });

    let lu = LuDecomposition::decompose(v)
        .map_err(|e| format!("Vandermonde LU: {e:?}"))?;
    let mut c = y;
    lu.solve_mut(&mut c)
        .map_err(|e| format!("Vandermonde solve: {e:?}"))?;

    column(&c, 0)
}

fn cholesky_spread_solve() -> SeriesResult {
    const N: usize = 8;
    let a = Owned::<f64, N, N>::from_fn(|i, j| {
        let scale = |k: usize| 10f64.powf(-10.0 * index_f64(k) / 7.0);
        if i == j {
            scale(i) + 1e-3
        } else {
            1e-3 * (scale(i) * scale(j)).sqrt()
        }
    });
    let spd = Symmetric::<f64, N>::from_owned(a)
        .ok_or_else(|| "graded-spread matrix is not symmetric".to_string())?;
    let chol = spd
        .into_cholesky()
        .map_err(|e| format!("Cholesky decomposition: {e:?}"))?;

    let mut x = Owned::<f64, N, 1>::from_fn(|_, _| 1.0);
    chol.solve_mut(&mut x)
        .map_err(|e| format!("Cholesky solve: {e:?}"))?;

    column(&x, 0)
}

fn qr_orthogonality_loss() -> KernelResult<f64> {
    const N: usize = 6;
    let mut a = Owned::<f64, N, N>::from_fn(|i, j| {
        if j == 4 {
            1.0 / (index_f64(i) + 4.0) + 1e-9
        } else {
            1.0 / (index_f64(i) + index_f64(j) + 1.0)
        }
    });
    let mut q = Owned::<f64, N, N>::zero();
    a.qr_decompose_mut(&mut q);

    let qt_q = matmul(&q.transpose(), &q);
    let eye = Owned::<f64, N, N>::identity();
    let diff = sub(&qt_q, &eye);

    let mut sum_sq = 0.0_f64;
    for v in row_major(&diff)? {
        sum_sq = v.mul_add(v, sum_sq);
    }
    Ok(sum_sq.sqrt())
}

/// Executes the matrix control-rs-verification kernel and writes `target/verification/matrix.rust.h5`.
///
/// # Errors
///
/// Returns an error if a decomposition fails or the container cannot be
/// written.
pub fn emit_container(output_path: &Path) -> KernelResult<()> {
    let mut writer = H5Writer::new();

    let cov = generate_ekf_covariance()?;
    writer.add_dataset("covariance_heatmap/matrix", &cov);
    writer.set_tolerance("covariance_heatmap/matrix", "abs", 1e-4);

    let hilbert = hilbert_solve()?;
    writer.add_dataset("hilbert_solve/x", &hilbert.x);
    writer.set_tolerance("hilbert_solve/x", "abs", 0.05);
    writer.add_dataset("hilbert_solve/residual", &[hilbert.residual]);
    writer.set_tolerance("hilbert_solve/residual", "abs", 1e-12);

    let v_c = vandermonde_solve()?;
    writer.add_dataset("vandermonde_solve/x", &v_c);
    writer.set_tolerance("vandermonde_solve/x", "abs", 1e-3);

    let chol_x = cholesky_spread_solve()?;
    writer.add_dataset("cholesky_spread/x", &chol_x);
    writer.set_tolerance("cholesky_spread/x", "abs", 1e-3);

    let qr_loss = qr_orthogonality_loss()?;
    writer.add_dataset("qr_orthogonality/residual", &[qr_loss]);
    writer.set_tolerance("qr_orthogonality/residual", "abs", 1e-6);

    writer.write_to_file(output_path)
}
