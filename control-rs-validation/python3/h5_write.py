"""Shared HDF5 writers for host companion oracles.

Each variant is a standalone file at ``results/<name>.<variant>.h5`` with
datasets at ``/<signal_path>``. True-oracle files attach ``measure`` / ``bound``
attributes. ``/_meta`` holds plot-only extras and is skipped by the gate.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def write_dict_to_h5(group, data):
    """Write a nested dict of numeric arrays as HDF5 groups and datasets."""
    for key, val in data.items():
        name = str(key)
        if isinstance(val, dict):
            subgroup = group.require_group(name)
            write_dict_to_h5(subgroup, val)
        elif val is not None:
            arr = np.atleast_1d(np.asarray(val, dtype=np.float64))
            if name in group:
                del group[name]
            group.create_dataset(name, data=arr)


def as_1d(arr) -> np.ndarray:
    """Squeeze frequency-response arrays to a 1D float64 vector."""
    return np.asarray(arr, dtype=np.float64).squeeze().reshape(-1)


def project_dict(data: Mapping[str, Any], paths: list[str]) -> dict:
    """Build a nested dict containing only slash-separated ``paths``."""
    out: dict[str, Any] = {}
    for path in paths:
        parts = [p for p in path.split("/") if p]
        src: Any = data
        for part in parts:
            if not isinstance(src, Mapping) or part not in src:
                raise KeyError(f"gated path '{path}' is missing from the payload")
            src = src[part]
        dst = out
        for part in parts[:-1]:
            dst = dst.setdefault(part, {})
        dst[parts[-1]] = src
    return out


def overlay_dict(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict:
    """Deep-merge ``overlay`` onto a copy of ``base``."""
    result = copy.deepcopy(dict(base))

    def merge(dst: dict, src: Mapping[str, Any]) -> None:
        for key, val in src.items():
            if isinstance(val, Mapping) and isinstance(dst.get(key), dict):
                merge(dst[key], val)
            else:
                dst[key] = copy.deepcopy(val)

    merge(result, overlay)
    return result


def _load_tolerances(table_path: Path) -> dict:
    import tomllib

    with table_path.open("rb") as handle:
        parsed = tomllib.load(handle)
    return parsed.get("tolerances", {})


def _apply_spec(dataset, spec: Mapping[str, Any], peer_bounds: Mapping[str, float] | None = None) -> None:
    dataset.attrs["measure"] = str(spec.get("measure", "abs"))
    if "bound" in spec:
        dataset.attrs["bound"] = float(spec["bound"])
    elif spec.get("measure") != "interval":
        dataset.attrs["bound"] = 0.0
    if "interval" in spec:
        dataset.attrs["interval"] = np.asarray(spec["interval"], dtype=np.float64)
    independent = spec.get("independent", True)
    dataset.attrs["independent"] = np.uint8(1 if independent else 0)
    if peer_bounds:
        for peer, bound in peer_bounds.items():
            dataset.attrs[f"bound.{peer}"] = float(bound)


def attr_specs_from_toml(
    table_path: Path | str,
    signal_to_key: Mapping[str, str],
    peer_key_maps: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Map signal paths to HDF5 attribute specs loaded from a human-doc TOML table."""
    table = _load_tolerances(Path(table_path))
    specs: dict[str, dict[str, Any]] = {}
    for signal, key in signal_to_key.items():
        spec = table.get(key)
        if spec is None:
            raise KeyError(f"tolerance key '{key}' missing from {table_path}")
        entry = dict(spec)
        peers: dict[str, float] = {}
        if peer_key_maps:
            for peer, mapping in peer_key_maps.items():
                peer_key = mapping.get(signal)
                if not peer_key:
                    continue
                peer_spec = table.get(peer_key)
                if peer_spec is None:
                    raise KeyError(f"tolerance key '{peer_key}' missing from {table_path}")
                if "bound" in peer_spec:
                    peers[peer] = float(peer_spec["bound"])
        if peers:
            entry["peer_bounds"] = peers
        specs[signal] = entry
    return specs


def apply_attrs(h5_file, specs: Mapping[str, Mapping[str, Any]]) -> None:
    """Write true-oracle attributes onto datasets at ``/<signal>``."""
    for signal, spec in specs.items():
        path = signal if signal.startswith("/") else f"/{signal}"
        dataset = h5_file[path]
        _apply_spec(dataset, spec, spec.get("peer_bounds"))


def write_variant_file(
    path: Path | str,
    payload: Mapping[str, Any],
    *,
    gated_paths: list[str] | None = None,
    meta: Mapping[str, Any] | None = None,
    attr_specs: Mapping[str, Mapping[str, Any]] | None = None,
) -> None:
    """Create ``results/<name>.<variant>.h5`` (truncate) with datasets at the file root."""
    import h5py

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    root_payload = project_dict(payload, gated_paths) if gated_paths else dict(payload)
    with h5py.File(output, "w") as handle:
        write_dict_to_h5(handle, root_payload)
        extras = dict(meta) if meta else {}
        if extras:
            write_dict_to_h5(handle.require_group("_meta"), extras)
        if attr_specs:
            apply_attrs(handle, attr_specs)
