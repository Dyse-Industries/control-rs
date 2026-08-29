#!/usr/bin/env python3
"""Polynomial suite plots from results/polynomial/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import SymLogNorm

from plot_util import (
    BOUND,
    PY,
    RS,
    abs_poly_eval,
    apply_style,
    as_f64,
    flatten_numbers,
    gamma,
    heatmap,
    load_artifacts,
    pair,
    relative_error_matrix,
    results_dir_from_argv,
    save_fig,
    series_xy,
    timings_figure,
)


def plot_horner(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    x, y_py = series_xy(py, "horner")
    _, y_rs = series_xy(rs, "horner")
    if y_py.size == 0:
        x = as_f64(py_vals.get("CLUSTER_X"))
        y_py = as_f64(py_vals.get("CLUSTER_Y"))
        y_rs = as_f64(rs_vals.get("CLUSTER_Y"))
    fig, (ax, ax_e) = plt.subplots(
        2, 1, figsize=(8.2, 7.0), sharex=True, layout="constrained"
    )
    n = min(x.size, y_py.size)
    ax.scatter(x[:n], y_py[:n], s=16, color=PY, zorder=3, label="python Horner")
    if y_rs.size:
        m = min(n, y_rs.size)
        ax.scatter(
            x[:m], y_rs[:m], s=22, marker="x", color=RS, zorder=4, label="rust Horner"
        )
        err = np.abs(y_py[:m] - y_rs[:m])
        ax_e.scatter(
            x[:m],
            np.maximum(err, 1e-18),
            s=16,
            color=PY,
            zorder=3,
            label=r"$|$python $-$ rust$|$",
        )
        ax_e.set_yscale("log")
    coeffs = flatten_numbers(rs_vals.get("CLUSTER_COEFFS") or py_vals.get("CLUSTER_COEFFS"))
    if coeffs:
        g = gamma(32.0)
        bound = np.array([g * abs_poly_eval(coeffs, abs(float(v))) for v in x[:n]])
        ax.fill_between(
            x[:n],
            -bound,
            bound,
            color=BOUND,
            alpha=0.22,
            label=r"Higham $\gamma_{32}\tilde{p}$",
        )
        ax_e.semilogy(
            x[:n],
            np.maximum(bound, 1e-18),
            "--",
            color=BOUND,
            lw=1.4,
            label=r"Higham $\gamma_{32}\tilde{p}$",
        )
    ax.set_ylabel(r"$p(x)$")
    ax.set_title(r"clustered-root Horner, $\mathrm{roots}=\{1\}^8\cup\{1.01\}^8$")
    ax.legend(frameon=False)
    ax_e.set_xlabel(r"$x$")
    ax_e.set_ylabel("abs error")
    ax_e.set_title("Horner disagreement vs Higham bound")
    ax_e.legend(frameon=False)
    save_fig(fig, results_dir, "horner")


def plot_companion(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    c_py = as_f64(py_vals.get("COMPANION") or [])
    c_rs = as_f64(rs_vals.get("COMPANION") or [])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.2), layout="constrained")
    if c_py.ndim != 2:
        axes[0].set_title("companion (missing)")
        axes[1].set_title("companion relative error")
        save_fig(fig, results_dir, "companion")
        return
    im = axes[0].imshow(
        c_py,
        cmap="coolwarm",
        origin="upper",
        norm=SymLogNorm(
            linthresh=1.0,
            vmin=min(float(np.min(c_py)), -1.0),
            vmax=max(float(np.max(c_py)), 1.0),
        ),
    )
    axes[0].set_title(r"companion of $(x+1)^{12}$")
    axes[0].set_xlabel("column")
    axes[0].set_ylabel("row")
    axes[0].set_xticks(range(c_py.shape[1]))
    axes[0].set_yticks(range(c_py.shape[0]))
    axes[0].grid(False)
    fig.colorbar(im, ax=axes[0], pad=0.02, fraction=0.046)
    heatmap(axes[1], relative_error_matrix(c_py, c_rs), "companion relative error")
    save_fig(fig, results_dir, "companion")


def main() -> None:
    apply_style()
    results_dir = results_dir_from_argv("plot_polynomial.py")
    artifacts = load_artifacts(results_dir)
    py, rs = pair(artifacts, "polynomial")
    plot_horner(results_dir, py, rs)
    plot_companion(results_dir, py, rs)
    timings_figure(results_dir, artifacts, "polynomial")


if __name__ == "__main__":
    main()
