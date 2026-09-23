#!/usr/bin/env python3
"""Matrix reference oracle generating target/verification/matrix.jax.h5 via JAX (x64)."""

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np

from h5_writer import get_results_dir, write_h5


def generate_datasets():
    # 1. EKF 8x8 covariance recursion
    n = 8
    idx = jnp.arange(n)
    diff = idx[:, None] - idx[None, :]
    p_current = jnp.exp(-0.25 * diff.astype(jnp.float64) ** 2) + 0.1 * jnp.eye(n)
    k = 0.01 * jnp.ones((n, n)) + (0.4 - 0.01) * jnp.eye(n)
    r = 0.05 * jnp.eye(n)
    i_minus_kh = jnp.eye(n) - k
    krk_t = k @ r @ k.T
    for _ in range(100):
        p_current = i_minus_kh @ p_current @ i_minus_kh.T + krk_t

    # 2. Hilbert solve
    h_n = 10
    hi = jnp.arange(h_n, dtype=jnp.float64)
    h_mat = 1.0 / (hi[:, None] + hi[None, :] + 1.0)
    b = h_mat @ jnp.ones(h_n)
    lu_x = jnp.linalg.solve(h_mat, b)
    lu_res = float(jnp.max(jnp.abs(h_mat @ lu_x - b)))

    # 3. Vandermonde solve
    nodes = jnp.linspace(-1.0, 1.0, 8)
    v_mat = jnp.vander(nodes, increasing=True)
    v_c = jnp.linalg.solve(v_mat, 1.0 / (1.0 + 25.0 * nodes**2))

    # 4. Cholesky graded eigenvalue spread
    c_n = 8
    scale = 10.0 ** (-10.0 * jnp.arange(c_n) / 7.0)
    off = 1e-3 * jnp.sqrt(scale[:, None] * scale[None, :])
    a_spd = off * (1.0 - jnp.eye(c_n)) + jnp.diag(scale + 1e-3)
    chol = jsl.cholesky(a_spd, lower=True)
    y = jsl.solve_triangular(chol, jnp.ones(c_n), lower=True)
    chol_x = jsl.solve_triangular(chol.T, y, lower=False)

    # 5. QR orthogonality loss
    qr_n = 6
    qi = jnp.arange(qr_n, dtype=jnp.float64)
    a_qr = 1.0 / (qi[:, None] + qi[None, :] + 1.0)
    a_qr = a_qr.at[:, 4].set(1.0 / (qi + 4.0) + 1e-9)
    q, _ = jnp.linalg.qr(a_qr)
    qr_loss = float(jnp.sqrt(jnp.sum((q.T @ q - jnp.eye(qr_n)) ** 2)))

    return {
        "covariance_heatmap/matrix": np.asarray(p_current).flatten(),
        "hilbert_solve/x": np.asarray(lu_x),
        "hilbert_solve/residual": [lu_res],
        "vandermonde_solve/x": np.asarray(v_c),
        "cholesky_spread/x": np.asarray(chol_x),
        "qr_orthogonality/residual": [qr_loss],
    }


def main():
    write_h5(get_results_dir() / "matrix.jax.h5", generate_datasets())


if __name__ == "__main__":
    main()
