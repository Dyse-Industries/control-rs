import warnings
from pathlib import Path
import numpy as np
from scipy.linalg import cholesky, hilbert, qr, solve_triangular

warnings.filterwarnings("ignore", category=RuntimeWarning)

from h5_writer import get_results_dir, write_h5


def generate_datasets():
    # 1. EKF 8x8 Covariance Recursion
    n = 8
    i_idx, j_idx = np.ogrid[:n, :n]
    diff = i_idx - j_idx
    p_0 = np.exp(-0.25 * (diff**2)) + 0.1 * np.eye(n)

    h = np.eye(n)
    k = 0.01 * np.ones((n, n))
    np.fill_diagonal(k, 0.4)
    r = 0.05 * np.eye(n)

    kh = k @ h
    eye = np.eye(n)
    i_minus_kh = eye - kh
    krk_t = k @ r @ k.T
    p_current = p_0.copy()

    for _ in range(100):
        p_update1 = i_minus_kh @ p_current @ i_minus_kh.T
        p_current = p_update1 + krk_t

    # 2. Hilbert Solve
    h_n = 10
    h_mat = hilbert(h_n)
    b = h_mat @ np.ones(h_n)
    lu_x = np.linalg.solve(h_mat, b)
    lu_res = float(np.max(np.abs(h_mat @ lu_x - b)))

    # 3. Vandermonde Solve
    v_n = 8
    nodes = np.linspace(-1.0, 1.0, v_n)
    v_mat = np.vander(nodes, increasing=True)
    y_runge = 1.0 / (1.0 + 25.0 * (nodes**2))
    v_c = np.linalg.solve(v_mat, y_runge)

    # 4. Cholesky Graded Eigenvalue Spread
    c_n = 8
    scale = 10.0 ** (-10.0 * np.arange(c_n) / 7.0)
    a_spd = np.zeros((c_n, c_n))
    for i in range(c_n):
        for j in range(c_n):
            if i == j:
                a_spd[i, j] = scale[i] + 1e-3
            else:
                a_spd[i, j] = 1e-3 * np.sqrt(scale[i] * scale[j])
    chol = cholesky(a_spd, lower=True)
    rhs = np.ones(c_n)
    chol_x = solve_triangular(chol.T, solve_triangular(chol, rhs, lower=True))

    # 5. QR Orthogonality Loss
    qr_n = 6
    a_qr = np.zeros((qr_n, qr_n))
    for i in range(qr_n):
        for j in range(qr_n):
            base = 1.0 / (i + j + 1.0)
            a_qr[i, j] = 1.0 / (i + 3 + 1.0) + 1e-9 if j == 4 else base
    q, _ = qr(a_qr)
    qr_diff = q.T @ q - np.eye(qr_n)
    qr_loss = float(np.sqrt(np.sum(qr_diff**2)))

    datasets = {
        "covariance_heatmap/matrix": p_current.flatten(),
        "hilbert_solve/x": lu_x,
        "hilbert_solve/residual": [lu_res],
        "vandermonde_solve/x": v_c,
        "cholesky_spread/x": chol_x,
        "qr_orthogonality/residual": [qr_loss],
    }

    tolerances = {
        "covariance_heatmap/matrix": ("abs", 1e-4, {"jax": 1e-6}),
        "hilbert_solve/x": ("abs", 0.05),
        "hilbert_solve/residual": ("abs", 1e-12),
        "vandermonde_solve/x": ("abs", 1e-3),
        "cholesky_spread/x": ("abs", 1e-3),
        "qr_orthogonality/residual": ("abs", 1e-6),
    }

    return datasets, tolerances


def main():
    datasets, tolerances = generate_datasets()
    out_file = get_results_dir() / "matrix.scipy.h5"
    write_h5(out_file, datasets, tolerances)


if __name__ == "__main__":
    main()
