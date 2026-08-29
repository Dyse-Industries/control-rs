"""Shared JSON / timing helpers for numerical-model Python validators."""

from __future__ import annotations

import json
import sys
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


def load_suite(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def case_inputs(suite: dict, case_id: str) -> dict:
    for case in suite.get("cases") or []:
        if case.get("id") == case_id:
            return case.get("inputs") or {}
    print(f"missing case `{case_id}`", file=sys.stderr)
    sys.exit(1)


def require_int(inputs: dict, key: str, expected: int) -> int:
    got = inputs.get(key)
    if got != expected:
        print(f"inputs.{key}={got} != compiled {expected}", file=sys.stderr)
        sys.exit(1)
    return int(got)


def emit_stdout(doc: dict) -> None:
    json.dump(doc, sys.stdout, indent=2)
    sys.stdout.write("\n")


def run_cli(slug: str, build: Callable[[dict], dict]) -> None:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <suite.json>", file=sys.stderr)
        sys.exit(2)
    path = Path(sys.argv[1])
    suite = load_suite(path)
    got = suite.get("slug")
    if got != slug:
        print(f"suite slug `{got}` does not match `{slug}`", file=sys.stderr)
        sys.exit(1)
    emit_stdout(build(suite))
