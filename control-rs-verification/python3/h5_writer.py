#!/usr/bin/env python3
"""Shared HDF5 writer for Python reference oracles."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np
import h5py


def write_h5(
    path: Path | str,
    datasets: Dict[str, Any],
    tolerances: Optional[Dict[str, Tuple[str, float]]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(path, "w") as f:
        for key, val in datasets.items():
            arr = np.asarray(val, dtype=np.float64)
            parts = [p for p in key.strip("/").split("/") if p]
            if len(parts) > 1:
                grp = f.require_group("/".join(parts[:-1]))
                ds = grp.create_dataset(parts[-1], data=arr)
            else:
                ds = f.create_dataset(parts[0], data=arr)

            if tolerances and key in tolerances:
                meas, bnd = tolerances[key]
                ds.attrs["measure"] = str(meas)
                ds.attrs["bound"] = float(bnd)

    print(f"Emitted reference container: {path}")


def get_results_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    return repo_root / "results"
