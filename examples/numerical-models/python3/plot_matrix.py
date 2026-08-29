#!/usr/bin/env python3
"""Matrix suite plots from results/matrix/."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from plot_util import (
    BOUND,
    EPS,
    GRAY,
    PY,
    RS,
    TAU,
    apply_style,
    as_f64,
    heatmap,
    load_artifacts,
    native_docs,
    pair,
    relative_error_matrix,
    results_dir_from_argv,
    save_fig,
    series_xy,
    source_style,
    timings_figure,
)


def plot_hilbert_inverse(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    err = relative_error_matrix(
        py_vals.get("HILBERT_A_INV") or [],
        rs_vals.get("HILBERT_A_INV") or [],
    )
    kappa = max(
        float((py.get("metrics") or {}).get("kappa_hilbert") or 0.0),
        float((rs.get("metrics") or {}).get("kappa_hilbert") or 0.0),
    )
    fig, ax = plt.subplots(figsize=(6.8, 5.8), layout="constrained")
    heatmap(
        ax,
        err,
        rf"Hilbert $H^{{-1}}$ relative error ($\kappa_\infty$={kappa:.2e})",
    )
    save_fig(fig, results_dir, "hilbert_inverse")


def plot_hilbert_solve(results_dir, artifacts: dict, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    x_idx, y_py = series_xy(py, "hilbert_x")
    if y_py.size == 0:
        y_py = as_f64(py_vals.get("HILBERT_X"))
        x_idx = np.arange(y_py.size, dtype=np.float64)
    y_rs = as_f64(rs_vals.get("HILBERT_X"))
    kappa = max(
        float((py.get("metrics") or {}).get("kappa_hilbert") or 0.0),
        float((rs.get("metrics") or {}).get("kappa_hilbert") or 0.0),
    )
    bound = TAU * kappa * EPS
    fig, (ax, ax_e) = plt.subplots(
        2, 1, figsize=(7.6, 6.6), sharex=True, layout="constrained"
    )
    ax.axhline(0.0, color=GRAY, lw=1.0, label="true")
    n = x_idx.size
    ax.plot(x_idx, y_py[:n] - 1.0, "o", color=PY, ms=6, label="python")
    if y_rs.size:
        ax.plot(
            x_idx[: min(n, y_rs.size)],
            y_rs[:n] - 1.0,
            "x",
            color=RS,
            ms=7,
            label="rust",
        )
    for src, doc in native_docs(artifacts):
        if src == "rust":
            continue
        y = as_f64((doc.get("values") or {}).get("HILBERT_X"))
        if y.size == 0:
            continue
        st = source_style(src)
        m = min(n, y.size)
        ax.plot(
            x_idx[:m],
            y[:m] - 1.0,
            marker=st["marker"],
            ls=st["ls"],
            color=st["color"],
            ms=6,
            label=src,
        )
    ax.set_ylabel(r"$\hat{x}_i - 1$")
    ax.set_title("Hilbert $n=8$ manufactured solve")
    ax.legend(frameon=False)
    err_py = np.maximum(np.abs(y_py[:n] - 1.0), 1e-18)
    ax_e.semilogy(x_idx, err_py, "o-", color=PY, label=r"python $|\hat{x}-1|$")
    if y_rs.size:
        m = min(n, y_rs.size)
        ax_e.semilogy(
            x_idx[:m],
            np.maximum(np.abs(y_rs[:m] - 1.0), 1e-18),
            "x--",
            color=RS,
            label=r"rust $|\hat{x}-1|$",
        )
    for src, doc in native_docs(artifacts):
        if src == "rust":
            continue
        y = as_f64((doc.get("values") or {}).get("HILBERT_X"))
        if y.size == 0:
            continue
        st = source_style(src)
        m = min(n, y.size)
        ax_e.semilogy(
            x_idx[:m],
            np.maximum(np.abs(y[:m] - 1.0), 1e-18),
            color=st["color"],
            ls=st["ls"],
            marker=st["marker"],
            label=rf"{src} $|\hat{{x}}-1|$",
        )
    if bound > 0.0:
        ax_e.axhline(bound, color=BOUND, ls="--", lw=1.4, label=r"$\tau\kappa\varepsilon$")
    ax_e.set_xlabel("component")
    ax_e.set_ylabel("forward error")
    ax_e.set_title(rf"forward error vs $\tau\kappa\varepsilon={bound:.2e}$")
    ax_e.legend(frameon=False)
    save_fig(fig, results_dir, "hilbert_solve")


def plot_se3(results_dir, artifacts: dict, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    xyz_py = as_f64(py_vals.get("SE3_XYZ") or [])
    xyz_rs = as_f64(rs_vals.get("SE3_XYZ") or [])
    r_py = as_f64(py_vals.get("SE3_R") or [])
    r_rs = as_f64(rs_vals.get("SE3_R") or [])
    fig = plt.figure(figsize=(7.4, 6.8))
    ax = fig.add_subplot(111, projection="3d")
    if xyz_py.ndim != 2 or xyz_py.shape[1] != 3:
        ax.set_title("SE(3) (missing XYZ)")
        save_fig(fig, results_dir, "se3_chain")
        return
    ax.plot(
        xyz_py[:, 0],
        xyz_py[:, 1],
        xyz_py[:, 2],
        "-",
        color=PY,
        lw=1.8,
        label="python",
    )
    if xyz_rs.ndim == 2 and xyz_rs.shape[1] == 3:
        ax.plot(
            xyz_rs[:, 0],
            xyz_rs[:, 1],
            xyz_rs[:, 2],
            "--",
            color=RS,
            lw=1.4,
            label="rust",
        )
    for src, doc in native_docs(artifacts):
        if src == "rust":
            continue
        xyz = as_f64((doc.get("values") or {}).get("SE3_XYZ") or [])
        if xyz.ndim == 2 and xyz.shape[1] == 3:
            st = source_style(src)
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                st["ls"],
                color=st["color"],
                lw=1.2,
                label=src,
            )
    ax.scatter(*xyz_py[0], color=PY, s=28, zorder=5)
    ax.scatter(*xyz_py[-1], color=PY, s=36, marker="s", zorder=5)
    span = float(np.ptp(xyz_py, axis=0).max()) if xyz_py.size else 0.05
    scale = 0.10 * max(span, 0.05)
    n = xyz_py.shape[0]
    for k in (0, n // 2, n - 1):
        origin = xyz_py[k]
        rot = r_py[k] if r_py.ndim == 3 and k < r_py.shape[0] else np.eye(3)
        for i, color in enumerate(("#E69F00", "#009E73", "#56B4E9")):
            d = rot[:, i] * scale
            ax.quiver(*origin, *d, color=color, lw=1.1, arrow_length_ratio=0.18)
        if xyz_rs.ndim == 2 and k < xyz_rs.shape[0]:
            origin_rs = xyz_rs[k]
            rot_rs = r_rs[k] if r_rs.ndim == 3 and k < r_rs.shape[0] else np.eye(3)
            for i, color in enumerate(("#E69F00", "#009E73", "#56B4E9")):
                d = rot_rs[:, i] * scale
                ax.quiver(
                    *origin_rs,
                    *d,
                    color=color,
                    lw=0.7,
                    arrow_length_ratio=0.18,
                    alpha=0.55,
                )
    ax.set_title(r"SE(3) end-effector chain $T^{k}$, $k=0\ldots 39$")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(frameon=False, loc="upper left")
    ax.view_init(elev=22, azim=-58)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass
    save_fig(fig, results_dir, "se3_chain")


def main() -> None:
    apply_style()
    results_dir = results_dir_from_argv("plot_matrix.py")
    artifacts = load_artifacts(results_dir)
    py, rs = pair(artifacts, "matrix")
    plot_hilbert_inverse(results_dir, py, rs)
    plot_hilbert_solve(results_dir, artifacts, py, rs)
    plot_se3(results_dir, artifacts, py, rs)
    timings_figure(results_dir, artifacts, "matrix")


if __name__ == "__main__":
    main()
