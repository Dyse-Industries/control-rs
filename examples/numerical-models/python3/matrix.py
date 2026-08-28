#!/usr/bin/env python3
"""Matrix numerical oracle — writes results/matrix/python.json."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import linalg

CRATE_ROOT = Path(__file__).resolve().parents[1]
RESULTS = CRATE_ROOT / "results" / "matrix"
OUT_PATH = RESULTS / "python.json"


def scenario() -> dict[str, np.ndarray]:
    m1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    m2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
    a = np.array(
        [[3.0, 2.0, -1.0], [2.0, -2.0, 4.0], [-1.0, 0.5, -1.0]],
        dtype=np.float64,
    )
    b = np.array([1.0, -2.0, 0.0], dtype=np.float64)
    x = linalg.solve(a, b)
    a_inv = linalg.inv(a)
    return {
        "m1": m1,
        "m2": m2,
        "sum": m1 + m2,
        "diff": m2 - m1,
        "prod": m1 @ m2,
        "transpose": m1.T.copy(),
        "a": a,
        "b": b.reshape(3, 1),
        "x": x.reshape(3, 1),
        "a_inv": a_inv,
    }


def residual_ratio(a: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    ax = a @ x.ravel()
    num = float(np.linalg.norm(ax - b.ravel(), ord=np.inf))
    den = float(
        np.linalg.norm(a, ord=np.inf)
        * np.linalg.norm(x.ravel(), ord=np.inf)
        * np.finfo(float).eps
    )
    return 0.0 if den == 0.0 else num / den


def build_artifact() -> dict:
    s = scenario()
    return {
        "slug": "matrix",
        "source": "python",
        "values": {
            "SUM": s["sum"].tolist(),
            "DIFF": s["diff"].tolist(),
            "PROD": s["prod"].tolist(),
            "TRANSPOSE": s["transpose"].tolist(),
            "X": s["x"].tolist(),
            "A_INV": s["a_inv"].tolist(),
            "residual_ratio": residual_ratio(s["a"], s["x"], s["b"].ravel()),
        },
        "series": {},
    }


def save_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    out = OUT_PATH
    save_json(out, build_artifact())
    print(f"wrote {out}")
