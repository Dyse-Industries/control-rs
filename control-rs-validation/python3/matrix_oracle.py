#!/usr/bin/env python3
"""
python3/matrix_oracle.py

SciPy oracle for the matrix validation suite. Mirrors
`control-rs-validation/src/matrix.rs` kernel for kernel on the same
ill-conditioned inputs and writes `results/matrix.scipy.h5`.

Every kernel is chosen so that double precision loses digits: a Hilbert solve
at kappa ~ 1e13, a monomial Vandermonde solve on equispaced nodes, an explicit
inverse residual, a Cholesky solve with a 1e10 eigenvalue spread, and QR
orthogonality loss on a near-rank-deficient matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy.linalg import cho_factor, cho_solve, hilbert, lu_factor, lu_solve, qr

from h5_write import attr_specs_from_toml, write_variant_file

HILBERT_N = 10
VANDERMONDE_N = 8
SPREAD_N = 8
QR_N = 6


def hilbert_solve() -> dict:
    """Solve H x = b with b = H @ 1, exact x = 1."""
    h = hilbert(HILBERT_N)
    b = h @ np.ones(HILBERT_N)
    x = lu_solve(lu_factor(h), b)
    residual = float(np.max(np.abs(h @ x - b)))
    return {"x": x.tolist(), "residual": residual, "order": HILBERT_N}


def vandermonde_solve() -> dict:
    """Interpolate the Runge function on equispaced nodes in the monomial basis."""
    nodes = np.linspace(-1.0, 1.0, VANDERMONDE_N)
    v = np.vander(nodes, VANDERMONDE_N, increasing=True)
    y = 1.0 / (1.0 + 25.0 * nodes**2)
    c = lu_solve(lu_factor(v), y)
    return {"x": c.tolist(), "order": VANDERMONDE_N}


def inverse_residual() -> dict:
    """Residual of an explicitly formed Hilbert inverse."""
    h = hilbert(SPREAD_N)
    inv = lu_solve(lu_factor(h), np.eye(SPREAD_N))
    residual = h @ inv - np.eye(SPREAD_N)
    return {"matrix": residual.tolist(), "order": SPREAD_N}


def graded_spd() -> np.ndarray:
    """SPD matrix whose eigenvalues span ten decades."""
    scale = 10.0 ** (-10.0 * np.arange(SPREAD_N) / 7.0)
    a = 1e-3 * np.sqrt(np.outer(scale, scale))
    np.fill_diagonal(a, scale + 1e-3)
    return a


def cholesky_spread() -> dict:
    """Cholesky solve where the smallest eigenvalues decide the answer."""
    x = cho_solve(cho_factor(graded_spd(), lower=True), np.ones(SPREAD_N))
    return {"x": x.tolist(), "order": SPREAD_N}


def near_rank_deficient() -> np.ndarray:
    """Hilbert-like matrix whose columns 3 and 4 differ by 1e-9."""
    i_idx, j_idx = np.mgrid[:QR_N, :QR_N]
    a = 1.0 / (i_idx + j_idx + 1.0)
    a[:, 4] = 1.0 / (np.arange(QR_N) + 4.0) + 1e-9
    return a


def qr_orthogonality() -> dict:
    """Frobenius norm of Q^T Q - I for a near-rank-deficient matrix."""
    q, r = qr(near_rank_deficient())
    residual = float(np.linalg.norm(q.T @ q - np.eye(QR_N), "fro"))
    return {"residual": residual, "r": r.tolist(), "order": QR_N}


def run_python_oracle() -> dict:
    """Assemble the SciPy payload."""
    return {
        "hilbert_solve": hilbert_solve(),
        "vandermonde_solve": vandermonde_solve(),
        "inverse_residual": inverse_residual(),
        "cholesky_spread": cholesky_spread(),
        "qr_orthogonality": qr_orthogonality(),
    }


GATED = [
    "hilbert_solve/x",
    "hilbert_solve/residual",
    "vandermonde_solve/x",
    "inverse_residual/matrix",
    "cholesky_spread/x",
    "qr_orthogonality/residual",
]

SIGNAL_KEYS = {
    "hilbert_solve/x": "nmv.matrix.hilbert_solve.scipy",
    "hilbert_solve/residual": "nmv.matrix.hilbert_residual.scipy",
    "vandermonde_solve/x": "nmv.matrix.vandermonde_solve.scipy",
    "inverse_residual/matrix": "nmv.matrix.inverse_residual.scipy",
    "cholesky_spread/x": "nmv.matrix.cholesky_spread.scipy",
    "qr_orthogonality/residual": "nmv.matrix.qr_orthogonality.scipy",
}


if __name__ == "__main__":
    results = run_python_oracle()
    table = Path(__file__).resolve().parent.parent / "tolerances/numerical_models.toml"
    specs = attr_specs_from_toml(table, SIGNAL_KEYS)
    write_variant_file(
        Path("results") / "matrix.scipy.h5",
        results,
        gated_paths=GATED,
        meta=results,
        attr_specs=specs,
    )
