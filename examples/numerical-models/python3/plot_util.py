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
ROW = "#CC79A7"
ACC = "#E69F00"

SOURCE_STYLE = {
    "python": {"color": PY, "marker": "o", "ls": "-"},
    "rust": {"color": RS, "marker": "x", "ls": "--"},
    "rust-row": {"color": ROW, "marker": "+", "ls": ":"},
    "rust-accelerate": {"color": ACC, "marker": "s", "ls": "-."},
}
_FALLBACK_COLORS = ("#56B4E9", "#F0E442", "#009E73", "#000000")


def source_style(source: str) -> dict:
    if source in SOURCE_STYLE:
        return dict(SOURCE_STYLE[source])
    color = _FALLBACK_COLORS[hash(source) % len(_FALLBACK_COLORS)]
    return {"color": color, "marker": "d", "ls": "-."}


def native_docs(artifacts: dict[str, dict]) -> list[tuple[str, dict]]:
    return [(s, d) for s, d in artifacts.items() if s != "python"]


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
    others: dict | None = None,
) -> None:
    x = as_f64(x)
    y_py = as_f64(y_py)
    series = [("python", y_py), ("rust", as_f64(y_rs))]
    if others:
        for src, y in others.items():
            series.append((src, as_f64(y)))
    n_py = min(x.size, y_py.size)
    if kind == "scatter":
        for src, y in series:
            n = min(x.size, y.size)
            if n == 0 and src != "python":
                continue
            st = source_style(src)
            ax.scatter(
                x[: n if src != "python" else n_py],
                y[: n if src != "python" else n_py],
                s=16 if src == "python" else 22,
                marker=st["marker"],
                color=st["color"],
                zorder=3 if src == "python" else 4,
                label=src,
            )
    else:
        for src, y in series:
            n = min(x.size, y.size)
            if n == 0 and src != "python":
                continue
            st = source_style(src)
            take = n_py if src == "python" else n
            ax.plot(
                x[:take],
                y[:take],
                color=st["color"],
                ls=st["ls"],
                marker=st["marker"] if src != "python" else None,
                markevery=max(take // 16, 1) if src != "python" else None,
                label=src,
            )
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


def timings_figure(results_dir: Path, artifacts: dict, title: str) -> None:
    sources = list(artifacts)
    sources.sort(key=lambda s: (s != "python", s != "rust", s))
    names = sorted(
        set().union(
            *(set((artifacts[s].get("timings") or {})) for s in sources)
        )
        if sources
        else set()
    )
    fig, ax = plt.subplots(figsize=(6.6, 4.2), layout="constrained")
    if not names:
        ax.set_title(f"{title}: no timings")
        save_fig(fig, results_dir, "timings")
        return
    x = np.arange(len(names))
    nsrc = max(len(sources), 1)
    w = 0.8 / nsrc
    for i, src in enumerate(sources):
        t = artifacts[src].get("timings") or {}
        ns = [max(float((t.get(k) or {}).get("ns", 0) or 0), 1.0) for k in names]
        ax.bar(
            x + (i - (nsrc - 1) / 2) * w,
            ns,
            w,
            color=source_style(src)["color"],
            label=src,
        )
    ax.set_xticks(x, names)
    ax.set_ylabel("min kernel time (ns)")
    blas = ((artifacts.get("python") or {}).get("metrics") or {}).get(
        "numpy_blas"
    )
    caption = f"{title}: kernel wall-time (informational)"
    if blas:
        caption = f"{caption}; NumPy BLAS: {blas}"
    ax.set_title(caption)
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
