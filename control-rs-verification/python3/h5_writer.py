#!/usr/bin/env python3
"""Shared HDF5 writer for Python reference oracles."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import h5py


def write_h5(
    path: Path | str,
    datasets: Dict[str, Any],
    tolerances: Optional[Dict[str, Tuple]] = None,
    missing_ok: Optional[Dict[str, List[str]]] = None,
) -> None:
    """Writes `datasets` to `path`.

    Each tolerance is `(measure, bound)` or `(measure, bound, {peer: bound})`;
    the per-peer bounds are stored as `bound.<peer>` attributes. `missing_ok`
    maps a dataset to the peers allowed to omit it (`missing_ok.<peer> = 1`).
    """
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
                meas, bnd, *rest = tolerances[key]
                ds.attrs["measure"] = str(meas)
                ds.attrs["bound"] = float(bnd)
                for peer, peer_bnd in (rest[0] if rest else {}).items():
                    ds.attrs[f"bound.{peer}"] = float(peer_bnd)

            for peer in (missing_ok or {}).get(key, []):
                ds.attrs[f"missing_ok.{peer}"] = np.int64(1)

    print(f"Emitted reference container: {path}")


def get_results_dir() -> Path:
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent
    return repo_root / "target" / "verification"
