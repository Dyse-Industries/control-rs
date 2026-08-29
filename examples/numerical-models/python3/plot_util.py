"""Shared helpers for per-suite numerical-model plotters."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

EPS = float(np.finfo(np.float64).eps)
TAU = 20.0
PY = "#0072B2"
RS = "#D55E00"
BOUND = "#009E73"
GRAY = "#6C6C6C"


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.18,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
            "lines.linewidth": 1.8,
            "axes.axisbelow": True,
            "mathtext.fontset": "dejavusans",
            "axes.titleweight": "medium",
        }
    )


def results_dir_from_argv(script: str) -> Path:
    if len(sys.argv) != 2:
        print(f"usage: {script} <results/<slug>/>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    if not path.is_dir():
        print(f"not a directory: {path}", file=sys.stderr)
        sys.exit(1)
    return path


def load_artifacts(results_dir: Path) -> dict[str, dict]:
    artifacts: dict[str, dict] = {}
    for path in sorted(results_dir.glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SystemExit(f"invalid JSON {path}: {e}") from e
        source = doc.get("source")
        if not source:
            continue
        artifacts[str(source)] = doc
    if not artifacts:
        raise SystemExit(f"no result JSON in {results_dir}")
    return artifacts


def pair(artifacts: dict[str, dict], expected_slug: str) -> tuple[dict, dict]:
    py = artifacts.get("python") or next(iter(artifacts.values()))
    others = [a for src, a in artifacts.items() if src != "python"]
    rs = others[0] if others else py
    slug = str(py.get("slug") or rs.get("slug") or "")
    if slug != expected_slug:
        raise SystemExit(f"expected slug `{expected_slug}`, got `{slug}`")
    return py, rs


def as_f64(value) -> np.ndarray:
    return np.asarray(value if value is not None else [], dtype=np.float64)


def flatten_numbers(value) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numbers(item))
        return out
    return []


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


def relative_error_matrix(a_py, a_rs) -> np.ndarray:
    a = as_f64(a_py)
    b = as_f64(a_rs)
    if a.size == 0 or b.size == 0:
        return np.zeros((0, 0), dtype=np.float64)
    if a.shape != b.shape:
        n = min(a.shape[0], b.shape[0])
        if a.ndim == 2 and b.ndim == 2:
            m = min(a.shape[1], b.shape[1])
            a = a[:n, :m]
            b = b[:n, :m]
        else:
            a = a[:n]
            b = b[:n]
    denom = np.maximum(np.abs(a), EPS)
    return np.abs(a - b) / denom


def overlay(
    ax,
    x,
    y_py,
    y_rs,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    yscale: str = "linear",
    kind: str = "line",
) -> None:
    x = as_f64(x)
    y_py = as_f64(y_py)
    y_rs = as_f64(y_rs)
    n_py = min(x.size, y_py.size)
    n = min(x.size, y_rs.size)
    if kind == "scatter":
        ax.scatter(
            x[:n_py],
            y_py[:n_py],
            s=16,
            color=PY,
            zorder=3,
            label="python",
        )
        if n:
            ax.scatter(
                x[:n],
                y_rs[:n],
                s=22,
                marker="x",
                color=RS,
                zorder=4,
                label="rust",
            )
    else:
        ax.plot(x[:n_py], y_py[:n_py], color=PY, label="python")
        if n:
            ax.plot(x[:n], y_rs[:n], "--", color=RS, label="rust")
    if yscale == "log":
        ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, loc="best")


def heatmap(ax, err: np.ndarray, title: str, cbar_label: str = "relative error"):
    if err.size == 0:
        ax.set_title(f"{title} (missing)")
        return None
    data = np.maximum(np.abs(err), 1e-18)
    finite = data[np.isfinite(data)]
    vmin = float(np.percentile(finite, 5)) if finite.size else 1e-18
    vmax = float(np.max(finite)) if finite.size else 1e-12
    if vmax <= vmin:
        vmax = vmin * 10.0
    im = ax.imshow(
        data,
        norm=mpl.colors.LogNorm(vmin=max(vmin, 1e-18), vmax=vmax),
        cmap="magma",
        origin="upper",
        interpolation="nearest",
    )
    ax.set_title(title)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    if err.shape[1] <= 16:
        ax.set_xticks(range(err.shape[1]))
    if err.shape[0] <= 16:
        ax.set_yticks(range(err.shape[0]))
    ax.grid(False)
    cbar = ax.figure.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cbar.set_label(cbar_label)
    return im


def timings_figure(results_dir: Path, py: dict, rs: dict, title: str) -> None:
    py_t = py.get("timings") or {}
    rs_t = rs.get("timings") or {}
    names = sorted(set(py_t) | set(rs_t))
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    if not names:
        ax.set_title(f"{title}: no timings")
        save_fig(fig, results_dir, "timings")
        return
    x = np.arange(len(names))
    py_ns = [max(float(py_t.get(k, {}).get("ns", 0) or 0), 1.0) for k in names]
    rs_ns = [max(float(rs_t.get(k, {}).get("ns", 0) or 0), 1.0) for k in names]
    w = 0.36
    ax.bar(x - w / 2, py_ns, w, color=PY, label="python")
    ax.bar(x + w / 2, rs_ns, w, color=RS, label="rust")
    ax.set_xticks(x, names)
    ax.set_ylabel("min kernel time (ns)")
    ax.set_title(f"{title}: kernel wall-time (informational)")
    ax.set_yscale("log")
    ax.legend(frameon=False)
    save_fig(fig, results_dir, "timings")


def save_fig(fig, results_dir: Path, name: str) -> None:
    path = results_dir / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}", file=sys.stderr)


def series_xy(doc: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    xy = (doc.get("series") or {}).get(key) or {}
    return as_f64(xy.get("x")), as_f64(xy.get("y"))
