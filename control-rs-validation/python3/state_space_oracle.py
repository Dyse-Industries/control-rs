#!/usr/bin/env python3
"""
python3/state_space_oracle.py

SciPy (and optional harold) oracle for the state-space validation suite.
Mirrors `control-rs-validation/src/state_space.rs`: a 2000-step trajectory that
accumulates per-step rounding, a 5e3 stiffness ratio pushed through ZOH, a
similarity transform by an ill-conditioned T, and controllability and
observability matrices of a graded system that are numerically rank-deficient.

Writes `results/state_space.scipy.h5`, plus `results/state_space.harold.h5`
when harold is installed.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import signal

from h5_write import attr_specs_from_toml, present_paths, write_variant_file

HORIZON = 2000
PENDULUM_TS = 0.05
STIFF_TS = 1e-3
GRADED_N = 6


def pendulum_discrete(omega0: float, b: float, dt: float):
    """ZOH discretization of a lightly damped pendulum linearization."""
    a = np.array([[0.0, 1.0], [-omega0 * omega0, -b]], dtype=np.float64)
    b_mat = np.array([[0.0], [1.0]], dtype=np.float64)
    c = np.array([[1.0, 0.0]], dtype=np.float64)
    d = np.array([[0.0]], dtype=np.float64)
    ad, bd, cd, dd, _ = signal.cont2discrete((a, b_mat, c, d), dt, method="zoh")
    return ad, bd, cd, dd


def phase_portrait() -> dict:
    """2000-step free response released near the inverted equilibrium."""
    ad, _bd, _cd, _dd = pendulum_discrete(2.0, 0.02, PENDULUM_TS)
    x = np.array([np.pi - 0.15, 0.5], dtype=np.float64)

    theta = []
    theta_dot = []
    for _ in range(HORIZON):
        theta.append(float(x[0]))
        theta_dot.append(float(x[1]))
        x = ad @ x

    return {
        "theta": theta,
        "theta_dot": theta_dot,
        "ts": PENDULUM_TS,
        "steps": HORIZON,
    }


def stiff_plant():
    """Continuous stiff plant with eigenvalues -1 and -5e3."""
    a = np.array([[-1.0, 0.0], [0.0, -5.0e3]], dtype=np.float64)
    b = np.array([[1.0], [1.0]], dtype=np.float64)
    c = np.array([[1.0, 1.0]], dtype=np.float64)
    d = np.array([[0.0]], dtype=np.float64)
    return a, b, c, d


def stiff_discrete():
    """ZOH discretization of the stiff plant at STIFF_TS."""
    return signal.cont2discrete(stiff_plant(), STIFF_TS, method="zoh")


def stiff_zoh() -> dict:
    """Discrete A and B of the stiff plant."""
    ad, bd, _cd, _dd, _ = stiff_discrete()
    return {"ad": ad.tolist(), "bd": bd.tolist(), "ts": STIFF_TS}


def stiff_step() -> dict:
    """500-sample unit-step response of the stiff discrete realization."""
    ad, bd, cd, dd, _ = stiff_discrete()
    x = np.zeros((2, 1), dtype=np.float64)
    y = []
    for _ in range(500):
        y.append(float((cd @ x + dd)[0, 0]))
        x = ad @ x + bd
    return {"y": y}


def similarity() -> dict:
    """a_tilde = T A T^-1 with kappa_2(T) ~ 1e8, matching control-rs."""
    a = np.array([[0.0, 1.0], [-4.0, -0.8]], dtype=np.float64)
    t = np.array([[1.0, 1.0e8], [1.0e-8, 2.0]], dtype=np.float64)
    a_tilde = t @ a @ np.linalg.inv(t)
    return {"a_tilde": a_tilde.tolist(), "t": t.tolist()}


def graded_system():
    """Graded 6-state system whose modes span six decades."""
    i_idx, j_idx = np.mgrid[:GRADED_N, :GRADED_N]
    a = 0.01 / (i_idx + j_idx + 1.0)
    np.fill_diagonal(a, -(10.0 ** (np.arange(GRADED_N) - 3)))
    b = (1.0 / (np.arange(GRADED_N) + 1.0)).reshape(GRADED_N, 1)
    c = (1.0 / (np.arange(GRADED_N) + 1.0)).reshape(1, GRADED_N)
    return a, b, c


def ctrb() -> dict:
    """[B, AB, ..., A^(n-1) B] of the graded system."""
    a, b, _c = graded_system()
    cols = [b]
    for _ in range(GRADED_N - 1):
        cols.append(a @ cols[-1])
    return {"matrix": np.hstack(cols).tolist()}


def obsv() -> dict:
    """[C; CA; ...; CA^(n-1)] of the graded system."""
    a, _b, c = graded_system()
    rows = [c]
    for _ in range(GRADED_N - 1):
        rows.append(rows[-1] @ a)
    return {"matrix": np.vstack(rows).tolist()}


def run_scipy_oracle() -> dict:
    """Assemble the SciPy payload."""
    return {
        "phase_portrait": phase_portrait(),
        "stiff_zoh": stiff_zoh(),
        "stiff_step": stiff_step(),
        "similarity": similarity(),
        "ctrb": ctrb(),
        "obsv": obsv(),
    }


def run_harold_oracle() -> dict | None:
    """Independent discretization of the stiff plant, or None when unavailable."""
    try:
        from harold import State, discretize
    except ImportError:
        return None

    a, b, c, d = stiff_plant()
    discrete = discretize(State(a, b, c, d), STIFF_TS, method="zoh")
    return {"stiff_zoh": {"ad": np.asarray(discrete.a).tolist()}}


GATED = [
    "phase_portrait/theta",
    "phase_portrait/theta_dot",
    "stiff_zoh/ad",
    "stiff_zoh/bd",
    "stiff_step/y",
    "similarity/a_tilde",
    "ctrb/matrix",
    "obsv/matrix",
]

SIGNAL_KEYS = {
    "phase_portrait/theta": "nmv.state_space.phase_portrait.theta.scipy",
    "phase_portrait/theta_dot": "nmv.state_space.phase_portrait.theta_dot.scipy",
    "stiff_zoh/ad": "nmv.state_space.stiff_zoh.ad",
    "stiff_zoh/bd": "nmv.state_space.stiff_zoh.bd",
    "stiff_step/y": "nmv.state_space.stiff_step.y",
    "similarity/a_tilde": "nmv.state_space.similarity.a_tilde",
    "ctrb/matrix": "nmv.state_space.ctrb.matrix",
    "obsv/matrix": "nmv.state_space.obsv.matrix",
}

HAROLD_KEYS = {"stiff_zoh/ad": "nmv.state_space.stiff_zoh.ad_harold"}


if __name__ == "__main__":
    scipy_results = run_scipy_oracle()
    harold_results = run_harold_oracle()
    table = Path(__file__).resolve().parent.parent / "tolerances/numerical_models.toml"
    specs = attr_specs_from_toml(
        table, SIGNAL_KEYS, {"harold": HAROLD_KEYS} if harold_results else None
    )
    results = Path("results")
    write_variant_file(
        results / "state_space.scipy.h5",
        scipy_results,
        gated_paths=GATED,
        meta=scipy_results,
        attr_specs=specs,
    )
    if harold_results:
        harold_paths = present_paths(harold_results, GATED)
        if harold_paths:
            write_variant_file(
                results / "state_space.harold.h5",
                harold_results,
                gated_paths=harold_paths,
                meta=harold_results,
            )
