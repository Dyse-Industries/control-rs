#!/usr/bin/env python3
"""Overlay matrix python.json vs native.json → results/matrix/plot.png."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CRATE_ROOT / "results" / "matrix"
PYTHON_JSON = RESULTS / "python.json"
NATIVE_JSON = RESULTS / "native.json"
OUT_PNG = RESULTS / "plot.png"


def load(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing artifact: {path}\nrun: {hint}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    python = load(PYTHON_JSON, "python3 python/src/matrix.py")
    native = load(NATIVE_JSON, "cargo run --example matrix")

    keys = [
        k
        for k in python["values"]
        if k != "residual_ratio" and isinstance(python["values"][k], list)
    ]
    n = len(keys)
    if n == 0:
        print("no matrix arrays to plot", file=sys.stderr)
        raise SystemExit(1)

    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(10, 3 * rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for ax, key in zip(axes_flat, keys, strict=False):
        py_arr = np.asarray(python["values"][key], dtype=np.float64)
        rs_arr = np.asarray(native["values"][key], dtype=np.float64)
        err = np.abs(py_arr - rs_arr)
        im = ax.imshow(err, cmap="viridis")
        ax.set_title(f"|Δ| {key}")
        fig.colorbar(im, ax=ax, fraction=0.046)

    for ax in axes_flat[n:]:
        ax.axis("off")

    rr_py = python["values"].get("residual_ratio", 0.0)
    rr_rs = native["values"].get("residual_ratio", 0.0)
    fig.suptitle(f"matrix V&V  residual_ratio python={rr_py:.3e} rust={rr_rs:.3e}")
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"wrote {OUT_PNG}")


if __name__ == "__main__":
    main()
