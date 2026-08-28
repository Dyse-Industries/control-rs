#!/usr/bin/env python3
"""Read generator JSON and write comparison plots.

Default: read-only. Pass ``--force`` to run every Python and Rust generator
first, then plot.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec

CRATE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = CRATE_ROOT / "results"
SLUGS = ("matrix", "polynomial", "state_space", "transfer_function", "tensor")
EPS = float(np.finfo(np.float64).eps)
TAU = 20.0
ABS_F64 = 1e-12
ABS_F32 = 1e-6
QUANT_LSB = 1.0 / 256.0
STRESS_KEYS = {
    "matrix": ("HILBERT_X",),
    "polynomial": ("CLUSTER_Y",),
    "state_space": ("STIFF_Y", "STIFF_BD"),
    "transfer_function": ("CLUSTER_H_RE", "CLUSTER_H_IM"),
    "tensor": ("CURVED_SAMPLES",),
}


def load(path: Path, hint: str) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing artifact: {path}\nrun: {hint}")
    return json.loads(path.read_text(encoding="utf-8"))


def flatten_numbers(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    return []


def as_f64(value) -> np.ndarray:
    return np.asarray(value, dtype=np.float64)


def max_abs_diff(a, b) -> float:
    fa = np.asarray(flatten_numbers(a), dtype=np.float64)
    fb = np.asarray(flatten_numbers(b), dtype=np.float64)
    n = min(fa.size, fb.size)
    if n == 0:
        return 0.0
    return float(np.max(np.abs(fa[:n] - fb[:n])))


def max_rel_diff(a, b, floor: float | None = None) -> float:
    fa = np.asarray(flatten_numbers(a), dtype=np.float64)
    fb = np.asarray(flatten_numbers(b), dtype=np.float64)
    n = min(fa.size, fb.size)
    if n == 0:
        return 0.0
    scale = floor if floor is not None else EPS
    denom = np.maximum(np.maximum(np.abs(fa[:n]), np.abs(fb[:n])), scale)
    return float(np.max(np.abs(fa[:n] - fb[:n]) / denom))


def gamma(k: float) -> float:
    ke = k * EPS
    if ke >= 1.0:
        return float("inf")
    return ke / (1.0 - ke)


def abs_poly_eval(coeffs: list[float], x_abs: float) -> float:
    acc = 0.0
    for c in reversed(coeffs):
        acc = acc * x_abs + abs(c)
    return acc


def stress_max(
    py_vals: dict, rs_vals: dict, keys: tuple[str, ...], floor: float | None = None
) -> tuple[float, float]:
    worst_abs = 0.0
    worst_rel = 0.0
    for k in keys:
        if k not in py_vals:
            continue
        worst_abs = max(worst_abs, max_abs_diff(py_vals[k], rs_vals.get(k)))
        worst_rel = max(worst_rel, max_rel_diff(py_vals[k], rs_vals.get(k), floor=floor))
    return worst_abs, worst_rel


def flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def relative_error_matrix(a_py, a_rs) -> np.ndarray:
    a = as_f64(a_py)
    b = as_f64(a_rs)
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        m = min(a.shape[1], b.shape[1]) if a.ndim == 2 and b.ndim == 2 else n
        a = a[:n, :m] if a.ndim == 2 else a[:n]
        b = b[:n, :m] if b.ndim == 2 else b[:n]
    denom = np.maximum(np.abs(a), EPS)
    return np.abs(a - b) / denom


def claim_lines(slug: str, py: dict, rs: dict) -> list[str]:
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    py_m = py.get("metrics") or {}
    rs_m = rs.get("metrics") or {}
    lines: list[str] = []
    if slug == "matrix":
        hx_p = np.asarray(flatten_numbers(py_vals.get("HILBERT_X")), dtype=np.float64)
        hx_r = np.asarray(flatten_numbers(rs_vals.get("HILBERT_X")), dtype=np.float64)
        kappa = max(float(py_m.get("kappa_hilbert") or 0.0), float(rs_m.get("kappa_hilbert") or 0.0))
        bound = TAU * kappa * EPS
        native_fwd = float(np.max(np.abs(hx_r - 1.0))) if hx_r.size else 0.0
        python_fwd = float(np.max(np.abs(hx_p - 1.0))) if hx_p.size else 0.0
        inv_p = np.asarray(flatten_numbers(py_vals.get("HILBERT_A_INV")), dtype=np.float64)
        inv_r = np.asarray(flatten_numbers(rs_vals.get("HILBERT_A_INV")), dtype=np.float64)
        n_inv = min(inv_p.size, inv_r.size)
        inv_abs = float(np.max(np.abs(inv_p[:n_inv] - inv_r[:n_inv]))) if n_inv else 0.0
        inv_scale = float(np.max(np.abs(inv_p[:n_inv]))) if n_inv else 1.0
        inv_rel = inv_abs / max(inv_scale, EPS)
        rr = float(rs_m.get("residual_ratio_hilbert") or 0.0)
        lines.append(
            f"  - Hilbert |x̂−1|_∞ native={native_fwd:.3e} python={python_fwd:.3e}"
            f" vs τκε={bound:.3e} ({flag(native_fwd < bound and python_fwd < bound)})"
        )
        lines.append(
            f"  - residual_ratio_hilbert={rr:.3e} vs τ={TAU:g}"
            f" ({flag(rr < TAU)}); κ_∞(H)={kappa:.3e} (Hilbert n=8)"
        )
        lines.append(
            f"  - |ΔH⁻¹|_∞={inv_abs:.3e} (rel {inv_rel:.3e}; not a τκε key)"
        )
        lines.append("  - figure: Hilbert relative-error heatmap and SE(3) GEMM chain")
    elif slug == "polynomial":
        xs = np.asarray(flatten_numbers(rs_vals.get("CLUSTER_X")), dtype=np.float64)
        yp = np.asarray(flatten_numbers(py_vals.get("CLUSTER_Y")), dtype=np.float64)
        yr = np.asarray(flatten_numbers(rs_vals.get("CLUSTER_Y")), dtype=np.float64)
        coeffs = flatten_numbers(rs_vals.get("CLUSTER_COEFFS") or py_vals.get("CLUSTER_COEFFS"))
        g = gamma(32.0)
        n = min(yp.size, yr.size, xs.size)
        err = float(np.max(np.abs(yp[:n] - yr[:n]))) if n else 0.0
        bound = 0.0
        for i in range(n):
            bound = max(bound, g * abs_poly_eval(coeffs, abs(float(xs[i]))))
        cdiff = max_abs_diff(py_vals.get("CLUSTER_COEFFS"), rs_vals.get("CLUSTER_COEFFS"))
        ok = err <= max(bound, ABS_F64)
        lines.append(
            f"  - clustered Horner |py−rs|_∞={err:.3e} vs γ₃₂ p̃={bound:.3e} ({flag(ok)})"
        )
        lines.append(
            f"  - |Δcoeffs|_∞={cdiff:.3e}; true p(x)≲ε on [0.9,1.1]"
            f" (cancellation residues, both ≪ γ₃₂ p̃)"
        )
        lines.append("  - figure: Horner overlay with Higham γ₃₂ p̃ band")
    elif slug == "state_space":
        lines.append("  - figure: free-response phase portrait (x2 vs x1)")
    elif slug == "transfer_function":
        lines.append("  - figure: Butterworth and clustered-pole Nyquist contours")
    elif slug == "tensor":
        curved_abs = max_abs_diff(py_vals.get("CURVED_SAMPLES"), rs_vals.get("CURVED_SAMPLES"))
        weiser = float(rs_m.get("weiser_bound") or py_m.get("weiser_bound") or 0.0)
        q_py = float(py_m.get("quant_roundtrip_max") or 0.0)
        q_rs = float(rs_m.get("quant_roundtrip_max") or 0.0)
        curved_ok = curved_abs <= max(weiser, ABS_F32 * 10.0)
        q_ok = q_py <= QUANT_LSB and q_rs <= QUANT_LSB
        lines.append(
            f"  - curved |py−rs|_∞={curved_abs:.3e} vs Weiser={weiser:.3e}"
            f" ({flag(curved_ok)})"
        )
        lines.append(
            f"  - Q7 round-trip max py={q_py:.3e} rust={q_rs:.3e}"
            f" vs 2⁻⁸={QUANT_LSB:.3e} ({flag(q_ok)})"
        )
        lines.append("  - figure: 16×16 curved-table surface and relative-error heatmap")
    return lines


def plot_heatmap(ax, err: np.ndarray, title: str) -> None:
    data = np.maximum(err, 1e-18)
    vmax = max(float(data.max()), 1e-12)
    im = ax.imshow(data, norm=LogNorm(vmin=1e-18, vmax=vmax), cmap="magma", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def plot_se3(ax, py_vals: dict, rs_vals: dict) -> None:
    xyz_py = as_f64(py_vals.get("SE3_XYZ") or [])
    xyz_rs = as_f64(rs_vals.get("SE3_XYZ") or [])
    r_py = as_f64(py_vals.get("SE3_R") or [])
    r_rs = as_f64(rs_vals.get("SE3_R") or [])
    if xyz_py.ndim != 2 or xyz_py.shape[1] != 3:
        ax.set_title("SE(3) (missing XYZ)")
        return
    ax.plot(xyz_py[:, 0], xyz_py[:, 1], xyz_py[:, 2], "-", lw=1.5, label="python")
    if xyz_rs.ndim == 2 and xyz_rs.shape[1] == 3:
        ax.plot(xyz_rs[:, 0], xyz_rs[:, 1], xyz_rs[:, 2], "--", lw=1.2, label="rust")
    span = float(np.ptp(xyz_py, axis=0).max()) if xyz_py.size else 0.05
    scale = 0.08 * max(span, 0.05)
    n = xyz_py.shape[0]
    ks = (0, n // 2, n - 1)
    for k in ks:
        origin_py = xyz_py[k]
        rot = r_py[k] if r_py.ndim == 3 and k < r_py.shape[0] else np.eye(3)
        for i, color in enumerate(("r", "g", "b")):
            d = rot[:, i] * scale
            ax.quiver(*origin_py, *d, color=color, lw=1.0, arrow_length_ratio=0.2)
        if xyz_rs.ndim == 2 and k < xyz_rs.shape[0]:
            origin_rs = xyz_rs[k]
            rot_rs = r_rs[k] if r_rs.ndim == 3 and k < r_rs.shape[0] else np.eye(3)
            for i, color in enumerate(("r", "g", "b")):
                d = rot_rs[:, i] * scale
                ax.quiver(
                    *origin_rs, *d, color=color, lw=0.8, arrow_length_ratio=0.2, alpha=0.7
                )
    ax.set_title("SE(3) end-effector chain")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=7)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def plot_phase(ax, py_vals: dict, rs_vals: dict) -> None:
    x1_py = as_f64(py_vals.get("FREE_X1") or [])
    x2_py = as_f64(py_vals.get("FREE_X2") or [])
    x1_rs = as_f64(rs_vals.get("FREE_X1") or [])
    x2_rs = as_f64(rs_vals.get("FREE_X2") or [])
    ax.plot(x1_py, x2_py, "-", lw=1.5, label="python")
    n = min(x1_rs.size, x2_rs.size)
    if n:
        ax.plot(x1_rs[:n], x2_rs[:n], "--", lw=1.2, label="rust")
    if x1_py.size:
        ax.plot(x1_py[0], x2_py[0], "ko", ms=5, label="x0")
    ax.plot(0.0, 0.0, "k+", ms=10, label="origin")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x1 (position)")
    ax.set_ylabel("x2 (velocity)")
    ax.set_title("phase portrait (u=0)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_step_y(ax, py_series: dict, rs_series: dict) -> None:
    py_xy = py_series.get("step_y") or {}
    rs_xy = rs_series.get("step_y") or {}
    if not py_xy:
        ax.set_title("step y(t) (none)")
        return
    x = as_f64(py_xy["x"])
    py_y = as_f64(py_xy["y"])
    ax.plot(x, py_y, "-", lw=1.5, label="python")
    if rs_xy:
        rs_y = as_f64(rs_xy.get("y", []))
        n = min(x.size, rs_y.size)
        ax.plot(x[:n], rs_y[:n], "--", lw=1.2, label="rust")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.set_title("unit-step y(t)")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_nyquist(ax, re_py, im_py, re_rs, im_rs, title: str) -> None:
    re_p = as_f64(re_py)
    im_p = as_f64(im_py)
    re_r = as_f64(re_rs)
    im_r = as_f64(im_rs)
    ax.plot(re_p, im_p, "-", lw=1.5, label="python")
    n = min(re_r.size, im_r.size)
    if n:
        ax.plot(re_r[:n], im_r[:n], "--", lw=1.2, label="rust")
    ax.plot(-1.0, 0.0, "k+", ms=12, mew=2, label="−1")
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(0.0, color="0.6", lw=0.6)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("Re H(jω)")
    ax.set_ylabel("Im H(jω)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)


def plot_tensor_surface(ax, py_vals: dict, rs_vals: dict) -> None:
    tab_py = as_f64(py_vals.get("CURVED_TABLE") or [])
    tab_rs = as_f64(rs_vals.get("CURVED_TABLE") or [])
    if tab_py.ndim != 2:
        ax.set_title("curved table (missing)")
        return
    n, m = tab_py.shape
    ii, jj = np.meshgrid(np.arange(n), np.arange(m), indexing="ij")
    ax.plot_surface(ii, jj, tab_py, cmap="Blues", alpha=0.55, linewidth=0, antialiased=True)
    if tab_rs.shape == tab_py.shape:
        ax.plot_surface(
            ii, jj, tab_rs, cmap="Oranges", alpha=0.45, linewidth=0, antialiased=True
        )
    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("value")
    ax.set_title("16×16 curved table")
    try:
        ax.set_box_aspect((1, 1, 0.6))
    except AttributeError:
        pass


def plot_horner(ax_ov, ax_err, py_series: dict, rs_series: dict, py_vals: dict, rs_vals: dict) -> None:
    py_xy = py_series.get("horner") or {}
    rs_xy = rs_series.get("horner") or {}
    if not py_xy:
        ax_ov.set_title("polynomial (no series)")
        ax_err.set_title("polynomial error")
        return
    x = as_f64(py_xy["x"])
    py_y = as_f64(py_xy["y"])
    rs_y = as_f64(rs_xy.get("y", [])) if rs_xy else np.array([])
    ax_ov.plot(x, py_y, "o-", ms=3, label="python horner")
    if rs_y.size:
        n = min(py_y.size, rs_y.size, x.size)
        ax_ov.plot(x[:n], rs_y[:n], "x--", ms=3, label="rust horner")
        err = np.abs(py_y[:n] - rs_y[:n])
        ax_err.semilogy(x[:n], np.maximum(err, 1e-18), label="horner")
    coeffs = flatten_numbers(rs_vals.get("CLUSTER_COEFFS") or py_vals.get("CLUSTER_COEFFS"))
    if coeffs:
        g = gamma(32.0)
        bound = np.array([g * abs_poly_eval(coeffs, abs(float(v))) for v in x])
        ax_ov.fill_between(x, -bound, bound, color="0.7", alpha=0.35, label="Higham γ₃₂ p̃")
        ax_err.semilogy(x, np.maximum(bound, 1e-18), "k--", lw=1, label="Higham bound")
    ax_ov.set_title("polynomial overlay")
    ax_ov.legend(fontsize=7)
    ax_ov.grid(True, alpha=0.3)
    ax_err.set_title("polynomial |python − rust|")
    ax_err.set_ylabel("abs err")
    ax_err.legend(fontsize=7)
    ax_err.grid(True, which="both", alpha=0.3)


def plot_timings(ax, py_t: dict, rs_t: dict, title: str) -> None:
    names = sorted(set(py_t) | set(rs_t))
    if not names:
        ax.set_title(f"{title} timings (none)")
        return
    x = np.arange(len(names))
    py_ns = [float(py_t.get(k, {}).get("ns", 0)) for k in names]
    rs_ns = [float(rs_t.get(k, {}).get("ns", 0)) for k in names]
    ax.bar(x - 0.2, py_ns, 0.4, label="python")
    ax.bar(x + 0.2, rs_ns, 0.4, label="rust")
    ax.set_xticks(x, names, rotation=20, ha="right")
    ax.set_ylabel("min ns")
    ax.set_title(f"{title} kernel time")
    ax.legend(fontsize=7)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_yscale("log")


def metric_line(metrics: dict) -> str:
    parts = []
    for key in (
        "residual_ratio",
        "residual_ratio_hilbert",
        "kappa_hilbert",
        "quant_roundtrip_max",
    ):
        if key in metrics and isinstance(metrics[key], (int, float)):
            parts.append(f"{key}={metrics[key]:.3e}")
    return "; ".join(parts)


def timing_line(py_t: dict, rs_t: dict) -> str:
    parts = []
    for k in sorted(set(py_t) | set(rs_t)):
        p = py_t.get(k, {}).get("ns")
        r = rs_t.get(k, {}).get("ns")
        if p is None and r is None:
            continue
        parts.append(f"{k} py={p} ns rust={r} ns")
    return "; ".join(parts)


def add_slug_axes(fig, gs, row: int, slug: str):
    if slug == "matrix":
        return (
            fig.add_subplot(gs[row, 0]),
            fig.add_subplot(gs[row, 1], projection="3d"),
            fig.add_subplot(gs[row, 2]),
        )
    if slug == "tensor":
        return (
            fig.add_subplot(gs[row, 0], projection="3d"),
            fig.add_subplot(gs[row, 1]),
            fig.add_subplot(gs[row, 2]),
        )
    return (
        fig.add_subplot(gs[row, 0]),
        fig.add_subplot(gs[row, 1]),
        fig.add_subplot(gs[row, 2]),
    )


def fill_slug_axes(axes, slug: str, py: dict, rs: dict) -> None:
    ax0, ax1, ax2 = axes
    py_vals = py.get("values") or {}
    rs_vals = rs.get("values") or {}
    py_series = py.get("series") or {}
    rs_series = rs.get("series") or {}
    py_t = py.get("timings") or {}
    rs_t = rs.get("timings") or {}
    if slug == "matrix":
        err = relative_error_matrix(
            py_vals.get("HILBERT_A_INV") or [],
            rs_vals.get("HILBERT_A_INV") or [],
        )
        kappa = max(
            float((py.get("metrics") or {}).get("kappa_hilbert") or 0.0),
            float((rs.get("metrics") or {}).get("kappa_hilbert") or 0.0),
        )
        plot_heatmap(ax0, err, f"Hilbert H⁻¹ relative error (κ∞={kappa:.2e})")
        plot_se3(ax1, py_vals, rs_vals)
        plot_timings(ax2, py_t, rs_t, slug)
    elif slug == "polynomial":
        plot_horner(ax0, ax1, py_series, rs_series, py_vals, rs_vals)
        plot_timings(ax2, py_t, rs_t, slug)
    elif slug == "state_space":
        plot_phase(ax0, py_vals, rs_vals)
        plot_step_y(ax1, py_series, rs_series)
        plot_timings(ax2, py_t, rs_t, slug)
    elif slug == "transfer_function":
        plot_nyquist(
            ax0,
            py_vals.get("H_RE"),
            py_vals.get("H_IM"),
            rs_vals.get("H_RE"),
            rs_vals.get("H_IM"),
            "Butterworth Nyquist",
        )
        plot_nyquist(
            ax1,
            py_vals.get("CLUSTER_H_RE"),
            py_vals.get("CLUSTER_H_IM"),
            rs_vals.get("CLUSTER_H_RE"),
            rs_vals.get("CLUSTER_H_IM"),
            "clustered-pole Nyquist",
        )
        plot_timings(ax2, py_t, rs_t, slug)
    elif slug == "tensor":
        plot_tensor_surface(ax0, py_vals, rs_vals)
        err = relative_error_matrix(
            py_vals.get("CURVED_TABLE") or [],
            rs_vals.get("CURVED_TABLE") or [],
        )
        plot_heatmap(ax1, err, "curved table relative error")
        plot_timings(ax2, py_t, rs_t, slug)


def generate_all() -> None:
    python = sys.executable
    for slug in SLUGS:
        script = Path(__file__).resolve().parent / f"{slug}.py"
        print(f"running {script.name}", flush=True)
        subprocess.run([python, str(script)], cwd=CRATE_ROOT, check=True)
    print("running cargo run --release", flush=True)
    subprocess.run(["cargo", "run", "--release"], cwd=CRATE_ROOT, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Python vs native JSON. "
            "With --force, regenerate all generator files first."
        )
    )
    parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="run all python3/<slug>.py and cargo run --release, then plot",
    )
    return parser.parse_args()


def load_pair(slug: str) -> tuple[dict, dict]:
    py = load(
        RESULTS / slug / "python.json",
        f"python3 python3/{slug}.py",
    )
    rs = load(
        RESULTS / slug / "native.json",
        f"cargo run --release --bin {slug}",
    )
    return py, rs


def main() -> None:
    args = parse_args()
    if args.force:
        generate_all()
    lines = ["# Numerical-model V&V report", ""]
    fig = plt.figure(figsize=(14, 3.4 * len(SLUGS)))
    gs = GridSpec(len(SLUGS), 3, figure=fig, hspace=0.45, wspace=0.35)
    for row, slug in enumerate(SLUGS):
        py, rs = load_pair(slug)
        py_vals = py.get("values", {})
        rs_vals = rs.get("values", {})
        py_t = py.get("timings") or {}
        rs_t = rs.get("timings") or {}
        rel_floor = ABS_F32 if slug == "tensor" else None
        if slug == "polynomial":
            xs = flatten_numbers(rs_vals.get("CLUSTER_X") or py_vals.get("CLUSTER_X"))
            coeffs = flatten_numbers(
                rs_vals.get("CLUSTER_COEFFS") or py_vals.get("CLUSTER_COEFFS")
            )
            g = gamma(32.0)
            rel_floor = max(
                (g * abs_poly_eval(coeffs, abs(float(x))) for x in xs),
                default=ABS_F64,
            )
        abs_e, rel_e = stress_max(
            py_vals, rs_vals, STRESS_KEYS[slug], floor=rel_floor
        )
        met = metric_line(rs.get("metrics") or {})
        times = timing_line(py_t, rs_t)
        lines.append(
            f"- **{slug}**: stress max |python − rust| = {abs_e:.3e}"
            f" (rel {rel_e:.3e})"
        )
        lines.extend(claim_lines(slug, py, rs))
        if met:
            lines.append(f"  - metrics: {met}")
        if times:
            lines.append(f"  - timings: {times}")
        axes = add_slug_axes(fig, gs, row, slug)
        fill_slug_axes(axes, slug, py, rs)
        out_png = RESULTS / slug / "plot.png"
        fig_s = plt.figure(figsize=(12, 3.6))
        gs_s = GridSpec(1, 3, figure=fig_s, wspace=0.35)
        axes_s = add_slug_axes(fig_s, gs_s, 0, slug)
        fill_slug_axes(axes_s, slug, py, rs)
        fig_s.savefig(out_png, dpi=120, bbox_inches="tight")
        plt.close(fig_s)
        lines.append(f"  - plot: `{out_png}`")

    overview = RESULTS / "report.png"
    fig.savefig(overview, dpi=120, bbox_inches="tight")
    plt.close(fig)
    report = RESULTS / "report.md"
    lines.append("")
    lines.append(f"Overview figure: `{overview}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    print(f"wrote {overview}")


if __name__ == "__main__":
    main()
