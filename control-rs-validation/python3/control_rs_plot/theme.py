"""
control_rs_plot.theme

Canonical styling constants, semantic color palette, and matplotlib theme
configuration for control-rs validation examples and dashboards.
Strong high-contrast engineering palette with deep aerospace brand identity.
"""

from __future__ import annotations

import os
from typing import Sequence
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# Brand Identity (Deep Aerospace Navy)
BRAND_NAME = "control-rs"
BRAND_DEEP_BLUE = "#00205B"     # Authoritative Lockheed Aerospace Navy
BRAND_NAVY = "#0A2540"          # Midnight Navy
BRAND_STEEL = "#94A3B8"         # Precision Technical Steel

# Surface & Canvas Environment (Option A: Dark Aerospace Canvas)
BG_COLOR = "#0B111A"
PANEL_BG = "#111C2A"
GRID_COLOR = "#1E293B"
TEXT_COLOR = "#F1F5F9"
SUBDUED_TEXT = "#94A3B8"

# Semantic Reference & Status Tokens (Decoupled from Data Series)
COLOR_TARGET = "#10B981"        # Cold Mint / Emerald (setpoint, settling band, safe region)
COLOR_CRIT = "#EF4444"          # Signal Red (-180°, |z|=1, saturation, disturbance)
COLOR_ASYMPTOTE = "#94A3B8"     # Cool Slate Steel (0 dB crossover, zero axes, asymptotes)

# Strong Line Colors (Balanced Across the Spectrum)
COLOR_BLUE = "#3B82F6"          # Vivid Cobalt Blue
COLOR_AMBER = "#F59E0B"         # Luminous Amber / Gold
COLOR_CYAN = "#06B6D4"          # Electric Cyan
COLOR_ORANGE = "#F97316"        # Aerospace Safety Orange
COLOR_PURPLE = "#A855F7"        # Royal Purple
COLOR_WHITE = "#FFFFFF"         # Brilliant Stark White
COLOR_RED = "#EF4444"           # Signal Red
COLOR_GREEN = "#22C55E"         # Classic Python Green
COLOR_LIME = "#86EFAC"          # Harold Mint / Lime Green (Python offset)

# Functional Data Series Hierarchy (Alternating Contrast)
COLOR_PRIMARY = COLOR_BLUE
COLOR_SECONDARY = COLOR_AMBER
COLOR_TERTIARY = COLOR_CYAN
COLOR_QUATERNARY = COLOR_ORANGE

# Brand colormap: Canvas Void -> Deep Navy -> Royal Cobalt -> Sky Cyan -> Crisp White
CMAP_CONTROL_RS = LinearSegmentedColormap.from_list(
    "control_rs_brand",
    [BG_COLOR, BRAND_DEEP_BLUE, "#1D4ED8", "#38BDF8", "#FFFFFF"],
)

# Classic MATLAB default lines colormap
MATLAB_COLORS = [
    "#0072BD",  # blue
    "#D95319",  # orange
    "#EDB120",  # yellow
    "#7E2F8E",  # purple
    "#77AC30",  # green
    "#4DBEEE",  # cyan
    "#A2142F",  # dark red
]


def is_matlab_theme() -> bool:
    """Returns True if MATLAB theme is requested via environment variable."""
    return (
        os.environ.get("CONTROL_RS_THEME", "").strip().lower() in ("matlab", "1", "true")
        or os.environ.get("MATLAB_THEME", "").strip().lower() in ("1", "true", "yes")
    )


def generate_prop_cycle(
    base_colors: Sequence[str] | None = None,
) -> list[str]:
    """Generates an expanded data series color cycle with alternating contrast."""
    if is_matlab_theme():
        return list(MATLAB_COLORS)

    if base_colors is None:
        base_colors = [
            COLOR_BLUE,
            COLOR_AMBER,
            COLOR_CYAN,
            COLOR_ORANGE,
            COLOR_PURPLE,
            COLOR_WHITE,
            COLOR_RED,
        ]

    tier_base = list(base_colors)
    tier_light = ["#93C5FD", "#FDE68A", "#67E8F9", "#FDBA74", "#D8B4FE", "#F8FAFC", "#FCA5A5"]
    tier_deep = ["#1D4ED8", "#D97706", "#0891B2", "#C2410C", "#7E22CE", "#CBD5E1", "#B91C1C"]

    return tier_base + tier_light + tier_deep


PALETTE = {
    "bg": BG_COLOR,
    "panel": PANEL_BG,
    "grid": GRID_COLOR,
    "text": TEXT_COLOR,
    "subdued": SUBDUED_TEXT,
    "blue": COLOR_BLUE,
    "amber": COLOR_AMBER,
    "cyan": COLOR_CYAN,
    "orange": COLOR_ORANGE,
    "purple": COLOR_PURPLE,
    "white": COLOR_WHITE,
    "red": COLOR_RED,
    "green": COLOR_GREEN,
    "lime": COLOR_LIME,
    "primary": COLOR_PRIMARY,
    "secondary": COLOR_SECONDARY,
    "tertiary": COLOR_TERTIARY,
    "quaternary": COLOR_QUATERNARY,
    "target": COLOR_TARGET,
    "crit": COLOR_CRIT,
    "asymptote": COLOR_ASYMPTOTE,
    "brand_deep_blue": BRAND_DEEP_BLUE,
    "brand_navy": BRAND_NAVY,
    "brand_steel": BRAND_STEEL,
}


def apply_theme(font_size: float = 9.5) -> None:
    """Applies the canonical control-rs theme (or MATLAB theme if configured via env var)."""
    if is_matlab_theme():
        plt.rcParams.update(
            {
                "figure.facecolor": "#F0F0F0",
                "axes.facecolor": "#FFFFFF",
                "axes.edgecolor": "#1E1E1E",
                "axes.labelcolor": "#000000",
                "axes.titlecolor": "#000000",
                "axes.grid": True,
                "grid.color": "#E5E5E5",
                "grid.linestyle": "-",
                "grid.alpha": 0.8,
                "xtick.color": "#000000",
                "ytick.color": "#000000",
                "text.color": "#000000",
                "font.family": "sans-serif",
                "font.size": font_size,
                "lines.linewidth": 1.25,
                "legend.facecolor": "#FFFFFF",
                "legend.edgecolor": "#CCCCCC",
                "legend.labelcolor": "#000000",
                "legend.framealpha": 0.95,
                "legend.fontsize": font_size - 1.0,
                "axes.prop_cycle": cycler(color=MATLAB_COLORS),
            }
        )
        return

    plt.rcParams.update(
        {
            "figure.facecolor": BG_COLOR,
            "axes.facecolor": PANEL_BG,
            "axes.edgecolor": GRID_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.grid": True,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.alpha": 0.6,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "text.color": TEXT_COLOR,
            "font.family": "sans-serif",
            "font.size": font_size,
            "lines.linewidth": 1.25,
            "legend.facecolor": PANEL_BG,
            "legend.edgecolor": GRID_COLOR,
            "legend.labelcolor": TEXT_COLOR,
            "legend.framealpha": 0.9,
            "legend.fontsize": font_size - 1.0,
            "axes.prop_cycle": cycler(color=generate_prop_cycle()),
        }
    )
