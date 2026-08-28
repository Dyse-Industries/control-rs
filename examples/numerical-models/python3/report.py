#!/usr/bin/env python3
"""Read generator JSON and write comparison plots. Does not recompute models."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CRATE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = CRATE_ROOT / "results"
SLUGS = ("matrix", "polynomial", "state_space", "transfer_function", "tensor")


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


def max_abs_diff(a, b) -> float:
    fa = np.asarray(flatten_numbers(a), dtype=np.float64)
    fb = np.asarray(flatten_numbers(b), dtype=np.float64)
    n = min(fa.size, fb.size)
    if n == 0:
        return 0.0
    return float(np.max(np.abs(fa[:n] - fb[:n])))


def plot_series(ax, py_series: dict, rs_series: dict, title: str) -> None:
    for name, py_xy in py_series.items():
        rs_xy = rs_series.get(name, {})
        ax.plot(py_xy["x"], py_xy["y"], "o-", label=f"python {name}")
        if rs_xy:
            ax.plot(rs_xy["x"], rs_xy["y"], "x--", label=f"rust {name}")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)


def plot_values_bar(ax, py_vals: dict, rs_vals: dict, title: str) -> None:
    keys = list(py_vals.keys())
    diffs = [max_abs_diff(py_vals[k], rs_vals.get(k)) for k in keys]
    ax.bar(range(len(keys)), diffs)
    ax.set_xticks(range(len(keys)), keys, rotation=45, ha="right")
    ax.set_ylabel("max |python − rust|")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)


def main() -> None:
    lines = ["# Numerical-model V&V report", ""]
    fig, axes = plt.subplots(len(SLUGS), 1, figsize=(8, 3 * len(SLUGS)))
    if len(SLUGS) == 1:
        axes = [axes]
    for ax, slug in zip(axes, SLUGS):
        py = load(
            RESULTS / slug / "python.json",
            f"python3 python3/{slug}.py",
        )
        rs = load(
            RESULTS / slug / "native.json",
            f"cargo run --bin {slug}",
        )
        py_vals = py.get("values", {})
        rs_vals = rs.get("values", {})
        py_series = py.get("series") or {}
        rs_series = rs.get("series") or {}
        worst = max(
            (max_abs_diff(py_vals[k], rs_vals.get(k)) for k in py_vals),
            default=0.0,
        )
        lines.append(f"- **{slug}**: max |python − rust| = {worst:.3e}")
        if py_series:
            plot_series(ax, py_series, rs_series, slug)
        else:
            plot_values_bar(ax, py_vals, rs_vals, slug)
        out_png = RESULTS / slug / "plot.png"
        # per-slug figure as well
        fig_s, ax_s = plt.subplots(figsize=(7, 4))
        if py_series:
            plot_series(ax_s, py_series, rs_series, slug)
        else:
            plot_values_bar(ax_s, py_vals, rs_vals, slug)
        fig_s.tight_layout()
        fig_s.savefig(out_png, dpi=120)
        plt.close(fig_s)
        lines.append(f"  - plot: `{out_png}`")

    fig.tight_layout()
    overview = RESULTS / "report.png"
    fig.savefig(overview, dpi=120)
    plt.close(fig)
    report = RESULTS / "report.md"
    lines.append("")
    lines.append(f"Overview figure: `{overview}`")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {report}")
    print(f"wrote {overview}")


if __name__ == "__main__":
    main()
