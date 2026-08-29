#!/usr/bin/env python3
"""Transfer-function suite plots from results/transfer_function/."""

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


def plot_bode(results_dir, py: dict, rs: dict) -> None:
    w, mag_py = series_xy(py, "bode_mag")
    _, mag_rs = series_xy(rs, "bode_mag")
    _, ph_py = series_xy(py, "bode_phase")
    _, ph_rs = series_xy(rs, "bode_phase")
    fig, (ax_m, ax_p) = plt.subplots(
        2, 1, figsize=(8.2, 6.8), sharex=True, layout="constrained"
    )
    overlay(
        ax_m,
        w,
        mag_py,
        mag_rs,
        xlabel="",
        ylabel=r"$|H(j\omega)|$",
        title="underdamped complex-pair Bode magnitude",
        yscale="log",
    )
    ax_m.set_xscale("log")
    overlay(
        ax_p,
        w,
        np.degrees(as_f64(ph_py)),
        np.degrees(as_f64(ph_rs)),
        xlabel=r"$\omega$ (rad/s)",
        ylabel=r"$\angle H(j\omega)$ (deg)",
        title="underdamped complex-pair Bode phase",
    )
    ax_p.set_xscale("log")
    omega_n = float((py.get("values") or {}).get("OMEGA_N") or 0.0)
    if omega_n > 0.0:
        ax_m.axvline(omega_n, color="0.45", ls=":", lw=1.2, label=r"$\omega_n$")
        ax_p.axvline(omega_n, color="0.45", ls=":", lw=1.2)
        ax_m.legend(frameon=False)
    save_fig(fig, results_dir, "bode")


def plot_nyquist(ax, re_py, im_py, re_rs, im_rs, title: str) -> None:
    re_p = as_f64(re_py)
    im_p = as_f64(im_py)
    re_r = as_f64(re_rs)
    im_r = as_f64(im_rs)
    ax.plot(re_p, im_p, color=PY, label="python")
    n = min(re_r.size, im_r.size)
    if n:
        ax.plot(re_r[:n], im_r[:n], "--", color=RS, label="rust")
    ax.plot(-1.0, 0.0, "k+", ms=12, mew=1.8, label=r"$-1$")
    ax.axhline(0.0, color="0.65", lw=0.7)
    ax.axvline(0.0, color="0.65", lw=0.7)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(r"$\mathrm{Re}\,H(j\omega)$")
    ax.set_ylabel(r"$\mathrm{Im}\,H(j\omega)$")
    ax.set_title(title)
    ax.legend(frameon=False)


def plot_nyquist_pair(results_dir, py: dict, rs: dict) -> None:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    fig, ax = plt.subplots(figsize=(6.6, 6.4), layout="constrained")
    plot_nyquist(
        ax,
        py_vals.get("H_RE"),
        py_vals.get("H_IM"),
        rs_vals.get("H_RE"),
        rs_vals.get("H_IM"),
        "underdamped complex-pair Nyquist",
    )
    save_fig(fig, results_dir, "nyquist_complex_pair")
    fig, ax = plt.subplots(figsize=(6.6, 6.4), layout="constrained")
    plot_nyquist(
        ax,
        py_vals.get("CLUSTER_H_RE"),
        py_vals.get("CLUSTER_H_IM"),
        rs_vals.get("CLUSTER_H_RE"),
        rs_vals.get("CLUSTER_H_IM"),
        r"clustered-pole Nyquist $1/[(s+1)^4(s+1.01)^4]$",
    )
    save_fig(fig, results_dir, "nyquist_clustered")


def plot_cluster_mag(results_dir, py: dict, rs: dict) -> None:
    w, mag_py = series_xy(py, "cluster_mag")
    _, mag_rs = series_xy(rs, "cluster_mag")
    fig, ax = plt.subplots(figsize=(8.0, 4.6), layout="constrained")
    overlay(
        ax,
        w,
        mag_py,
        mag_rs,
        xlabel=r"$\omega$ (rad/s)",
        ylabel=r"$|H(j\omega)|$",
        title="clustered-pole magnitude",
        yscale="log",
    )
    ax.set_xscale("log")
    save_fig(fig, results_dir, "cluster_bode")


def main() -> None:
    apply_style()
    results_dir = results_dir_from_argv("plot_transfer_function.py")
    py, rs = pair(load_artifacts(results_dir), "transfer_function")
    plot_bode(results_dir, py, rs)
    plot_nyquist_pair(results_dir, py, rs)
    plot_cluster_mag(results_dir, py, rs)
    timings_figure(results_dir, py, rs, "transfer_function")


if __name__ == "__main__":
    main()
