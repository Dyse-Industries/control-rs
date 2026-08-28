#!/usr/bin/env python3
"""Tensor V&V plot — results/tensor/plot.png."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CRATE_ROOT / "results" / "tensor"


def load(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing artifact: {path}\nrun: {hint}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    python = load(RESULTS / "python.json", "python3 python/src/tensor.py")
    native = load(RESULTS / "native.json", "cargo run --example tensor")
    fig, ax = plt.subplots(figsize=(8, 4))
    for doc, style in ((python, "C0o"), (native, "C1x")):
        s = doc["series"]["interp"]
        ax.plot(s["x"], s["y"], style, label=doc["source"])
    ax.set_xlabel("query x0")
    ax.set_ylabel("interpolated value")
    ax.set_title("tensor interpolation V&V")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
