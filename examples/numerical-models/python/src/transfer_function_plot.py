#!/usr/bin/env python3
"""Transfer-function V&V plot — results/transfer_function/plot.png."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CRATE_ROOT / "results" / "transfer_function"


def load(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing artifact: {path}\nrun: {hint}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    python = load(RESULTS / "python.json", "python3 python/src/transfer_function.py")
    native = load(RESULTS / "native.json", "cargo run --example transfer_function")
    fig, ax = plt.subplots(figsize=(8, 4))
    for doc, style in ((python, "C0--"), (native, "C1-")):
        s = doc["series"]["bode_mag"]
        ax.loglog(s["x"], s["y"], style, label=doc["source"])
    ax.set_xlabel("omega [rad/s]")
    ax.set_ylabel("|H(jω)|")
    ax.set_title("transfer_function Bode magnitude V&V")
    ax.legend()
    fig.tight_layout()
    out = RESULTS / "plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
