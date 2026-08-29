#!/usr/bin/env python3
"""Tensor suite plots from results/tensor/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_util import (
    BOUND,
    PY,
    RS,
    apply_style,
    as_f64,
    heatmap,
    load_artifacts,
    overlay,
    pair,
    relative_error_matrix,
    results_dir_from_argv,
    save_fig,
    series_xy,
    timings_figure,
)

QUANT_LSB = 1.0 / 256.0


def plot_curved_surface(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    tab = as_f64(py_vals.get("CURVED_TABLE") or [])
    fig = plt.figure(figsize=(7.4, 6.4))
    ax = fig.add_subplot(111, projection="3d")
    if tab.ndim != 2:
        ax.set_title("saddle table (missing)")
        save_fig(fig, results_dir, "curved_surface")
        return
    n, m = tab.shape
    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    surf = ax.plot_surface(
        ii,
        jj,
        tab,
        cmap="coolwarm",
        linewidth=0,
        antialiased=True,
        rstride=1,
        cstride=1,
        alpha=0.92,
        vmin=-1.0,
        vmax=1.0,
    )
    ax.scatter([7.5], [7.5], [0.0], color="k", s=36, zorder=6, label="saddle")
    fig.colorbar(surf, ax=ax, shrink=0.62, pad=0.08, label="table value")
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("value")
    ax.set_title(r"$16\times 16$ saddle $u^{2}-v^{2}$ at $(7.5,7.5)$")
    ax.view_init(elev=24, azim=-60)
    try:
        ax.set_box_aspect((1, 1, 0.55))
    except AttributeError:
        pass
    save_fig(fig, results_dir, "curved_surface")
    fig, ax = plt.subplots(figsize=(6.6, 5.6), layout="constrained")
    rs_vals = rs.get("values") or {}
    heatmap(
        ax,
        relative_error_matrix(
            py_vals.get("CURVED_TABLE") or [],
            rs_vals.get("CURVED_TABLE") or [],
        ),
        r"saddle table relative error",
    )
    save_fig(fig, results_dir, "curved_error")


def plot_curved_cut(results_dir, py: dict, rs: dict) -> None:
    x, y_py = series_xy(py, "curved")
    _, y_rs = series_xy(rs, "curved")
    fig, (ax, ax_e) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True, layout="constrained"
    )
    overlay(
        ax,
        x,
        y_py,
        y_rs,
        xlabel="",
        ylabel="sample",
        title=r"cut $j=7.5$ through the saddle",
        kind="scatter",
    )
    if x.size:
        true = ((x - 7.5) / 7.5) ** 2
        ax.plot(x, true, color="0.45", lw=1.2, zorder=2, label=r"$u^{2}$ ($v=0$)")
        ax.legend(frameon=False)
    weiser = float(
        (rs.get("metrics") or {}).get("weiser_bound")
        or (py.get("metrics") or {}).get("weiser_bound")
        or 0.0
    )
    if y_rs.size and y_py.size:
        n = min(x.size, y_py.size, y_rs.size)
        ax_e.scatter(
            x[:n],
            np.maximum(np.abs(y_py[:n] - y_rs[:n]), 1e-18),
            s=16,
            color=PY,
            zorder=3,
            label=r"$|$python $-$ rust$|$",
        )
        ax_e.set_yscale("log")
    if weiser > 0.0:
        ax_e.axhline(weiser, color=BOUND, ls="--", lw=1.4, label="Weiser bound")
    ax_e.set_xlabel(r"cut coordinate $s$")
    ax_e.set_ylabel("abs error")
    ax_e.set_title("interpolation disagreement vs Weiser bound")
    ax_e.legend(frameon=False)
    save_fig(fig, results_dir, "curved_cut")


def plot_q7(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    de_py = as_f64(py_vals.get("DEQUANT") or [])
    relu_py = as_f64(py_vals.get("RELU_DEQUANT") or [])
    de_rs = as_f64(rs_vals.get("DEQUANT") or [])
    relu_rs = as_f64(rs_vals.get("RELU_DEQUANT") or [])
    idx = np.arange(max(de_py.size, relu_py.size), dtype=np.float64)
    fig, (ax, ax_e) = plt.subplots(
        2, 1, figsize=(8.0, 6.6), layout="constrained"
    )
    overlay(
        ax,
        idx,
        de_py,
        de_rs,
        xlabel="",
        ylabel="value",
        title="Q7 dequantized inputs",
        kind="scatter",
    )
    ax.scatter(
        idx[: relu_py.size], relu_py, s=16, marker="D", color=PY, alpha=0.7, label="python ReLU"
    )
    if relu_rs.size:
        ax.scatter(
            idx[: relu_rs.size],
            relu_rs,
            s=22,
            marker="x",
            color=RS,
            alpha=0.8,
            label="rust ReLU",
        )
    ax.legend(frameon=False)
    q_py = float((py.get("metrics") or {}).get("quant_roundtrip_max") or 0.0)
    q_rs = float((rs.get("metrics") or {}).get("quant_roundtrip_max") or 0.0)
    ax_e.axhline(QUANT_LSB, color=BOUND, ls="--", lw=1.4, label=r"$2^{-8}$")
    ax_e.bar([0], [max(q_py, 1e-18)], 0.4, color=PY, label="python")
    ax_e.bar([1], [max(q_rs, 1e-18)], 0.4, color=RS, label="rust")
    ax_e.set_xticks([0, 1], ["python", "rust"])
    ax_e.set_ylabel("max round-trip error")
    ax_e.set_title(rf"Q7 round-trip vs LSB ${QUANT_LSB:.4g}$")
    ax_e.legend(frameon=False)
    save_fig(fig, results_dir, "q7_relu")


def plot_affine(results_dir, py: dict, rs: dict) -> None:
    x, y_py = series_xy(py, "interp")
    _, y_rs = series_xy(rs, "interp")
    fig, ax = plt.subplots(figsize=(7.6, 4.4), layout="constrained")
    overlay(
        ax,
        x,
        y_py,
        y_rs,
        xlabel=r"query $\alpha$ (first axis)",
        ylabel="interpolated sample",
        title=r"affine $3\times 3$ table samples",
        kind="scatter",
    )
    save_fig(fig, results_dir, "affine_interp")


def main() -> None:
    apply_style()
    results_dir = results_dir_from_argv("plot_tensor.py")
    py, rs = pair(load_artifacts(results_dir), "tensor")
    plot_curved_surface(results_dir, py, rs)
    plot_curved_cut(results_dir, py, rs)
    plot_affine(results_dir, py, rs)
    plot_q7(results_dir, py, rs)
    timings_figure(results_dir, py, rs, "tensor")


if __name__ == "__main__":
    main()
