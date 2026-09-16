"""
control_rs_plot

Shared plotting and theme infrastructure for control-rs validation examples.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence

from cycler import cycler
import matplotlib

matplotlib.use("Agg")  # Must be set before importing matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np

from .theme import (
    BG_COLOR,
    BRAND_DEEP_BLUE,
    BRAND_NAME,
    BRAND_NAVY,
    BRAND_STEEL,
    CMAP_CONTROL_RS,
    COLOR_AMBER,
    COLOR_ASYMPTOTE,
    COLOR_BLUE,
    COLOR_CRIT,
    COLOR_CYAN,
    COLOR_GREEN,
    COLOR_LIME,
    COLOR_ORANGE,
    COLOR_PRIMARY,
    COLOR_PURPLE,
    COLOR_QUATERNARY,
    COLOR_RED,
    COLOR_SECONDARY,
    COLOR_TARGET,
    COLOR_TERTIARY,
    COLOR_WHITE,
    GRID_COLOR,
    MATLAB_COLORS,
    PALETTE,
    PANEL_BG,
    SUBDUED_TEXT,
    TEXT_COLOR,
    apply_theme,
    generate_prop_cycle,
    is_matlab_theme,
)

__all__ = [
    "MissingData",
    "require",
    "require_arrays",
    "BG_COLOR",
    "PANEL_BG",
    "TEXT_COLOR",
    "GRID_COLOR",
    "SUBDUED_TEXT",
    "BRAND_NAME",
    "BRAND_DEEP_BLUE",
    "BRAND_NAVY",
    "BRAND_STEEL",
    "CMAP_CONTROL_RS",
    "COLOR_BLUE",
    "COLOR_AMBER",
    "COLOR_CYAN",
    "COLOR_ORANGE",
    "COLOR_PURPLE",
    "COLOR_WHITE",
    "COLOR_RED",
    "COLOR_GREEN",
    "COLOR_LIME",
    "COLOR_PRIMARY",
    "COLOR_SECONDARY",
    "COLOR_TERTIARY",
    "COLOR_QUATERNARY",
    "COLOR_TARGET",
    "COLOR_CRIT",
    "COLOR_ASYMPTOTE",
    "MATLAB_COLORS",
    "is_matlab_theme",
    "PALETTE",
    "apply_theme",
    "cycler",
    "generate_prop_cycle",
    "resolve_results_path",
    "setup_argument_parser",
    "save_or_show",
    "h5_group_to_dict",
    "load_plot_container",
]


class MissingData(Exception):
    """Raised when a payload lacks a key a figure needs.

    Validation dashboards must never substitute invented values for absent
    results: a rendered panel is a claim about what the code produced. Every
    lookup of plotted data goes through :func:`require`, so a missing key
    stops the figure and names itself instead of being silently replaced by a
    plausible-looking default.
    """


def require(payload, *path, context: str = ""):
    """Returns ``payload[path[0]][path[1]]...`` or raises :class:`MissingData`.

    A ``None`` or empty value is treated as absent, since an empty series
    plots as a blank panel that reads like a passing result.
    """
    node = payload
    walked: list[str] = []
    for key in path:
        walked.append(str(key))
        if not isinstance(node, dict) or key not in node:
            where = f" ({context})" if context else ""
            raise MissingData(f"missing payload key '{'.'.join(walked)}'{where}")
        node = node[key]
    if node is None or (hasattr(node, "__len__") and len(node) == 0):
        where = f" ({context})" if context else ""
        raise MissingData(f"empty payload value '{'.'.join(path)}'{where}")
    return node


def require_arrays(payload, *paths, context: str = "", equal_length: bool = True):
    """Requires several sibling series and checks they share a length."""
    values = [require(payload, *p, context=context) if isinstance(p, tuple)
              else require(payload, p, context=context) for p in paths]
    if equal_length:
        lengths = {len(v) for v in values}
        if len(lengths) > 1:
            named = ", ".join(
                f"{p if isinstance(p, str) else '.'.join(p)}={len(v)}"
                for p, v in zip(paths, values)
            )
            where = f" ({context})" if context else ""
            raise MissingData(f"series length mismatch{where}: {named}")
    return values


def h5_group_to_dict(group) -> dict:
    """Recursively converts an HDF5 group into nested lists and scalars."""
    res = {}
    for key, item in group.items():
        if hasattr(item, "keys"):
            res[key] = h5_group_to_dict(item)
        else:
            val = item[()]
            if isinstance(val, bytes):
                val = val.decode("utf-8")
            elif isinstance(val, np.ndarray):
                if val.ndim == 0 or val.shape == (1,):
                    val = val.item()
                else:
                    val = val.tolist()
            res[key] = val
    return res


def load_plot_container(path: Path) -> dict:
    """Loads a rust-baseline plot payload from an HDF5 container or JSON envelope."""
    path = Path(path)
    if path.suffix.lower() == ".h5":
        import h5py

        with h5py.File(path, "r") as f:
            rust = h5_group_to_dict(f["rust"]) if "rust" in f else {}
            scipy = h5_group_to_dict(f["scipy"]) if "scipy" in f else {}
            ngspice = h5_group_to_dict(f["ngspice"]) if "ngspice" in f else {}
            rust["oracle"] = dict(scipy)
            if ngspice:
                rust["oracle"]["ngspice"] = ngspice
            return rust

    with open(path, encoding="utf-8") as handle:
        loaded = json.load(handle)
    if "sources" in loaded and "rust" in loaded["sources"]:
        data = loaded["sources"]["rust"].get("payload", {})
        data["oracle"] = {}
        if "scipy" in loaded["sources"]:
            data["oracle"].update(loaded["sources"]["scipy"].get("payload", {}))
        if "ngspice" in loaded["sources"]:
            data["oracle"]["ngspice"] = loaded["sources"]["ngspice"].get(
                "payload", {}
            )
        return data
    return loaded


def resolve_results_path(
    specified_path: str | Path | None,
    candidate_paths: Sequence[str | Path],
    error_msg: str = "Could not locate results container file (.h5).",
) -> Path:
    """Resolves a results file path across specified and candidate paths."""
    if specified_path:
        p = Path(specified_path)
        if p.is_file():
            return p.resolve()

    for cand in candidate_paths:
        p = Path(cand)
        if p.is_file():
            return p.resolve()

    raise FileNotFoundError(error_msg)


def setup_argument_parser(
    description: str,
    default_results: str | Path | None = None,
    default_output: str | Path | None = None,
    default_json: str | Path | None = None,
) -> argparse.ArgumentParser:
    """Builds a standardized argparse CLI harness for control-rs plotting scripts."""
    parser = argparse.ArgumentParser(description=description)
    default_val = default_results if default_results is not None else default_json
    parser.add_argument(
        "--results",
        "--h5",
        "--json",
        dest="results",
        type=str,
        default=str(default_val) if default_val else None,
        help="Path to input validation results container (.h5)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(default_output) if default_output else None,
        help="Path to output PNG image file",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display interactive matplotlib window",
    )
    return parser


def save_or_show(
    fig: plt.Figure,
    output_path: str | Path,
    show: bool = False,
    dpi: int = 150,
) -> None:
    """Saves figure to disk and optionally opens interactive display window."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=dpi, bbox_inches="tight", facecolor=BG_COLOR, edgecolor="none")
    print(f"Saved validation dashboard figure to: {out}")
    if show:
        plt.show()
    else:
        plt.close(fig)
