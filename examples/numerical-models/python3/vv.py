"""Shared JSON / timing helpers for numerical-model Python generators."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

CRATE_ROOT = Path(__file__).resolve().parents[1]
EPS = float(np.finfo(np.float64).eps)
TAU = 20.0


def time_kernel(iters: int, fn: Callable[[], object]) -> int:
    fn()
    best: int | None = None
    for _ in range(iters):
        t0 = time.perf_counter_ns()
        fn()
        dt = time.perf_counter_ns() - t0
        if best is None or dt < best:
            best = dt
    return int(best if best is not None else 0)


def timing_entry(iters: int, ns: int) -> dict:
    return {"iters": iters, "ns": ns}


def residual_ratio(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    ax = a @ x.ravel()
    num = float(np.linalg.norm(ax - b.ravel(), ord=np.inf))
    den = float(
        np.linalg.norm(a, ord=np.inf)
        * np.linalg.norm(x.ravel(), ord=np.inf)
        * EPS
    )
    return 0.0 if den == 0.0 else num / den


def gamma(k: float) -> float:
    ke = k * EPS
    if ke >= 1.0:
        return float("inf")
    return ke / (1.0 - ke)


def save_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {path}")
