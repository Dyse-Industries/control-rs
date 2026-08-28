#!/usr/bin/env python3
"""Host-scale oracle JSON artifacts under results/<slug>/."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from numpy.polynomial.polynomial import polyfromroots, polyval
from scipy import linalg, signal
from scipy.interpolate import RegularGridInterpolator

CRATE_ROOT = Path(__file__).resolve().parents[2]
TAU = 10.0
EPS = float(np.finfo(np.float64).eps)


def save(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")


def _eval_kappa(coeffs: np.ndarray, x: float) -> float:
    ax = abs(x)
    tilde = float(np.polynomial.polynomial.polyval(ax, np.abs(coeffs)))
    val = abs(float(np.polynomial.polynomial.polyval(x, coeffs)))
    if val == 0.0:
        return float("inf")
    return tilde / val


def _well_conditioned(n: int, scale: float) -> np.ndarray:
    idx = np.arange(n, dtype=np.float64)
    return np.eye(n) + scale / (idx[:, None] + idx[None, :] + 1.0)


def emit_hilbert(n: int) -> None:
    a = linalg.hilbert(n)
    b = np.ones(n, dtype=np.float64)
    x = linalg.solve(a, b)
    kappa = float(np.linalg.cond(a))
    save(
        CRATE_ROOT / "results" / "matrix" / f"host_hilbert{n}_python.json",
        {
            "slug": "matrix",
            "source": "python",
            "fixture": f"hilbert{n}",
            "values": {
                "A": a.tolist(),
                "b": b.reshape(n, 1).tolist(),
                "x_gold": x.reshape(n, 1).tolist(),
                "kappa": kappa,
                "keps": kappa * EPS,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )


def emit_gemm_lu1024() -> None:
    a = _well_conditioned(1024, 0.01)
    b = _well_conditioned(1024, 0.02)
    c = a @ b
    rhs = np.ones(1024, dtype=np.float64)
    x = linalg.solve(a, rhs)
    kappa = float(np.linalg.cond(a))
    save(
        CRATE_ROOT / "results" / "matrix" / "host_gemm1024_python.json",
        {
            "slug": "matrix",
            "source": "python",
            "fixture": "gemm1024",
            "values": {
                "C_gold": c.tolist(),
                "kappa": kappa,
                "keps": kappa * EPS,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )
    save(
        CRATE_ROOT / "results" / "matrix" / "host_lu1024_python.json",
        {
            "slug": "matrix",
            "source": "python",
            "fixture": "lu1024",
            "values": {
                "b": rhs.reshape(1024, 1).tolist(),
                "x_gold": x.reshape(1024, 1).tolist(),
                "kappa": kappa,
                "keps": kappa * EPS,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )


def emit_poly52() -> None:
    roots = np.arange(1.0, 52.0, dtype=np.float64)
    coeffs = np.asarray(polyfromroots(roots), dtype=np.float64)
    poly_x = 20.5
    poly_val = float(polyval(poly_x, coeffs))
    poly_kappa = _eval_kappa(coeffs, poly_x)
    save(
        CRATE_ROOT / "results" / "polynomial" / "host_poly52_python.json",
        {
            "slug": "polynomial",
            "source": "python",
            "fixture": "poly52",
            "values": {
                "coeffs": coeffs.tolist(),
                "poly_x": poly_x,
                "poly_val": poly_val,
                "kappa": poly_kappa,
                "keps": poly_kappa * EPS,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )


def emit_tf_clustered() -> None:
    roots = np.arange(1.0, 52.0, dtype=np.float64)
    coeffs = np.asarray(polyfromroots(roots), dtype=np.float64)
    tf_num = np.array([1.0], dtype=np.float64)
    tf_w = np.array([0.5, 1.0, 2.0], dtype=np.float64)
    _w, h = signal.freqs(np.flip(tf_num), np.flip(coeffs), worN=tf_w)
    tf_kappa = max(_eval_kappa(coeffs, float(w)) for w in tf_w)
    save(
        CRATE_ROOT / "results" / "transfer_function" / "host_tf_clustered_python.json",
        {
            "slug": "transfer_function",
            "source": "python",
            "fixture": "tf_clustered",
            "values": {
                "den": coeffs.tolist(),
                "omegas": tf_w.tolist(),
                "h_re": np.asarray(h.real, dtype=np.float64).tolist(),
                "h_im": np.asarray(h.imag, dtype=np.float64).tolist(),
                "kappa": tf_kappa,
                "keps": tf_kappa * EPS,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )


def emit_stiff_zoh() -> None:
    a_stiff = np.diag([-1.0, -1.0e6])
    tmat = np.array([[1.0, 1.0], [0.0, 1.0]], dtype=np.float64)
    a_scr = tmat @ a_stiff @ np.linalg.inv(tmat)
    b_c = np.array([[0.0], [1.0]], dtype=np.float64)
    c_c = np.array([[1.0, 0.0]], dtype=np.float64)
    d_c = np.array([[0.0]], dtype=np.float64)
    dt = 1.0e-6
    ad, _bd, _cd, _dd, _ = signal.cont2discrete(
        (a_scr, b_c, c_c, d_c), dt, method="zoh"
    )
    stiff_kappa = float(np.linalg.cond(a_scr))
    save(
        CRATE_ROOT / "results" / "state_space" / "host_stiff_zoh_python.json",
        {
            "slug": "state_space",
            "source": "python",
            "fixture": "stiff_zoh",
            "values": {
                "A": a_scr.tolist(),
                "Ad_gold": np.asarray(ad, dtype=np.float64).tolist(),
                "dt": dt,
                "kappa": stiff_kappa,
                "tau": TAU,
                "eps": EPS,
            },
            "series": {},
        },
    )


def emit_tensor1024() -> None:
    n_g = 1024
    idx = np.arange(n_g, dtype=np.float32)
    grid = idx[:, None] + idx[None, :]
    axes = (np.arange(n_g, dtype=np.float32), np.arange(n_g, dtype=np.float32))
    interp = RegularGridInterpolator(
        axes, grid, method="linear", bounds_error=False, fill_value=None
    )
    points = np.array(
        [[0.0, 0.0], [10.5, 20.25], [100.0, 200.0], [512.5, 512.5]],
        dtype=np.float32,
    )
    samples = np.asarray(interp(points), dtype=np.float32)
    save(
        CRATE_ROOT / "results" / "tensor" / "host_tensor1024_python.json",
        {
            "slug": "tensor",
            "source": "python",
            "fixture": "tensor1024",
            "values": {
                "points": points.tolist(),
                "samples_gold": samples.tolist(),
            },
            "series": {},
        },
    )


def main() -> None:
    for n in (12, 16, 128):
        emit_hilbert(n)
    emit_gemm_lu1024()
    emit_poly52()
    emit_tf_clustered()
    emit_stiff_zoh()
    emit_tensor1024()


if __name__ == "__main__":
    main()
