#!/usr/bin/env python3
"""State-space V&V plot — results/state_space/plot.png."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

CRATE_ROOT = Path(__file__).resolve().parents[2]
RESULTS = CRATE_ROOT / "results" / "state_space"


def load(path: Path, hint: str) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"missing artifact: {path}\nrun: {hint}", file=sys.stderr)
        raise SystemExit(1) from None


def main() -> None:
    python = load(RESULTS / "python.json", "python3 python/src/state_space.py")
    native = load(RESULTS / "native.json", "cargo run --example state_space")
    fig, axes = plt.subplots(3, 1, figsize=(8, 8), sharex=True)
    for ax, name in zip(axes, ["step_y", "step_x1", "step_x2"], strict=True):
        for doc, style in ((python, "C0--"), (native, "C1-")):
            s = doc["series"][name]
            ax.plot(s["x"], s["y"], style, label=doc["source"])
        ax.set_ylabel(name)
        ax.legend()
    axes[-1].set_xlabel("t [s]")
    fig.suptitle("state_space step response V&V")
    fig.tight_layout()
    out = RESULTS / "plot.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
