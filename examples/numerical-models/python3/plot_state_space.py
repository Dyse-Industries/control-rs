#!/usr/bin/env python3
"""State-space suite plots from results/state_space/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_util import (
    PY,
    RS,
    apply_style,
    as_f64,
    load_artifacts,
    overlay,
    pair,
    results_dir_from_argv,
    save_fig,
    series_xy,
    timings_figure,
)


def plot_free_response(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    x1_py = as_f64(py_vals.get("FREE_X1") or [])
    x2_py = as_f64(py_vals.get("FREE_X2") or [])
    x1_rs = as_f64(rs_vals.get("FREE_X1") or [])
    x2_rs = as_f64(rs_vals.get("FREE_X2") or [])
    t, _ = series_xy(py, "free_x1")
    fig, (ax_ph, ax_t) = plt.subplots(
        1, 2, figsize=(11.2, 5.2), layout="constrained"
    )
    ax_ph.scatter(x1_py, x2_py, s=14, color=PY, zorder=3, label="python")
    n = min(x1_rs.size, x2_rs.size)
    if n:
        ax_ph.scatter(
            x1_rs[:n], x2_rs[:n], s=22, marker="x", color=RS, zorder=4, label="rust"
        )
    if x1_py.size:
        ax_ph.plot(x1_py[0], x2_py[0], "ko", ms=6, label=r"$x_0$")
    ax_ph.plot(0.0, 0.0, "k+", ms=12, mew=1.6, label="origin")
    ax_ph.set_aspect("equal", adjustable="datalim")
    ax_ph.set_xlabel(r"$x_1$ (position)")
    ax_ph.set_ylabel(r"$x_2$ (velocity)")
    ax_ph.set_title(r"free response phase portrait ($u=0$)")
    ax_ph.legend(frameon=False)
    if t.size == 0:
        t = np.arange(x1_py.size, dtype=np.float64)
    ax_t.scatter(t[: x1_py.size], x1_py, s=14, color=PY, zorder=3, label=r"python $x_1$")
    ax_t.scatter(
        t[: x2_py.size], x2_py, s=14, color=PY, alpha=0.45, zorder=3, label=r"python $x_2$"
    )
    if x1_rs.size:
        ax_t.scatter(
            t[: x1_rs.size], x1_rs, s=22, marker="x", color=RS, zorder=4, label=r"rust $x_1$"
        )
    if x2_rs.size:
        ax_t.scatter(
            t[: x2_rs.size],
            x2_rs,
            s=22,
            marker="x",
            color=RS,
            alpha=0.7,
            zorder=4,
            label=r"rust $x_2$",
        )
    ax_t.set_xlabel(r"$t$ (s)")
    ax_t.set_ylabel("state")
    ax_t.set_title("free-response states")
    ax_t.legend(frameon=False)
    save_fig(fig, results_dir, "free_response")


def plot_step(results_dir, py: dict, rs: dict) -> None:
    t, y_py = series_xy(py, "step_y")
    _, y_rs = series_xy(rs, "step_y")
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    fig, axes = plt.subplots(
        3, 1, figsize=(8.2, 7.4), sharex=True, layout="constrained"
    )
    overlay(
        axes[0],
        t,
        y_py,
        y_rs,
        xlabel="",
        ylabel=r"$y$",
        title="unit-step output",
        kind="scatter",
    )
    overlay(
        axes[1],
        t,
        py_vals.get("STEP_X1"),
        rs_vals.get("STEP_X1"),
        xlabel="",
        ylabel=r"$x_1$",
        title="unit-step $x_1$",
        kind="scatter",
    )
    overlay(
        axes[2],
        t,
        py_vals.get("STEP_X2"),
        rs_vals.get("STEP_X2"),
        xlabel=r"$t$ (s)",
        ylabel=r"$x_2$",
        title="unit-step $x_2$",
        kind="scatter",
    )
    save_fig(fig, results_dir, "step_response")


def plot_stiff(results_dir, py: dict, rs: dict) -> None:
    t, y_py = series_xy(py, "stiff_y")
    _, y_rs = series_xy(rs, "stiff_y")
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    overlay(
        ax,
        t,
        y_py,
        y_rs,
        xlabel=r"$t$ (s)",
        ylabel=r"$y$",
        title=r"stiff ZOH plant $A=\mathrm{diag}(-200,-0.5)$",
        kind="scatter",
    )
    save_fig(fig, results_dir, "stiff_zoh")


def main() -> None:
    apply_style()
    results_dir = results_dir_from_argv("plot_state_space.py")
    artifacts = load_artifacts(results_dir)
    py, rs = pair(artifacts, "state_space")
    plot_free_response(results_dir, py, rs)
    plot_step(results_dir, py, rs)
    plot_stiff(results_dir, py, rs)
    timings_figure(results_dir, artifacts, "state_space")


if __name__ == "__main__":
    main()
