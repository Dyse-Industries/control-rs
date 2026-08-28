#!/usr/bin/env python3
"""Polynomial V&V plot — results/polynomial/plot.png."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CRATE_ROOT / "results" / "polynomial"


def load(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing artifact: {path}\nrun: {hint}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    python = load(RESULTS / "python.json", "python3 python/src/polynomial.py")
    native = load(RESULTS / "native.json", "cargo run --example polynomial")
    keys = ["P_REAL", "P_DERIV", "P_INTEG", "REM"]
    py_vals = [python["values"][k] for k in keys]
    rs_vals = [native["values"][k] for k in keys]
    err = [abs(p - r) for p, r in zip(py_vals, rs_vals, strict=True)]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(keys, err, color="steelblue")
    ax.set_ylabel("|python − rust|")
    ax.set_title("polynomial scalar V&V")
    fig.tight_layout()
    out = RESULTS / "plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
