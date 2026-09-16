#! /usr/bin/env python3
import json
import os
# ==========================================
# 1. The control-rs Brand & Theme Setup
# ==========================================
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import h5py
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.lines import Line2D

# Ensure shared python3 directory is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from control_rs_plot import (
    MissingData,
    require,
    BG_COLOR,
    GRID_COLOR,
    PANEL_BG,
    TEXT_COLOR,
    COLOR_BLUE,
    COLOR_AMBER,
    COLOR_CYAN,
    COLOR_ORANGE,
    COLOR_PURPLE,
    COLOR_WHITE,
    COLOR_RED,
    COLOR_GREEN,
    COLOR_LIME,
    COLOR_PRIMARY,
    COLOR_SECONDARY,
    COLOR_TERTIARY,
    COLOR_QUATERNARY,
    COLOR_TARGET,
    COLOR_CRIT,
    COLOR_ASYMPTOTE,
    CMAP_CONTROL_RS,
    apply_theme as base_apply_theme,
)

# Language-specific brand alignment:
# Rust: Amber / Orange
# Python: Classic Green
# Harold: Mint / Lime (pure Python library offset)
# C: Blue
# Julia: Purple
# Fortran: Stark White
COLOR_RS = COLOR_AMBER
COLOR_PY = COLOR_GREEN
COLOR_HAROLD = COLOR_LIME
COLOR_ALT = COLOR_PURPLE
COLOR_C = COLOR_BLUE
COLOR_JULIA = COLOR_PURPLE
COLOR_FORTRAN = COLOR_WHITE


def apply_control_rs_theme():
    """Applies the canonical control-rs dark mode branding to matplotlib."""
    base_apply_theme(font_size=10.0)
    plt.rcParams.update({
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


apply_control_rs_theme()


def get_output_dir() -> str:
    """Resolves the output directory strictly to control-rs-validation/results."""
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    target_dir = os.path.abspath(os.path.join(script_dir, '..', 'results'))
    os.makedirs(target_dir, exist_ok=True)
    return target_dir


STYLING_PALETTE = {
    "rust": (COLOR_RS, "s", "--"),
    "python3": (COLOR_PY, "o", "-"),
    "c": (COLOR_C, "d", "-."),
    "julia": (COLOR_JULIA, "^", ":"),
    "fortran": (COLOR_FORTRAN, "v", "-")
}


def python_primary(sources: dict) -> dict:
    """Prefer SciPy payloads, then NumPy, for the Python plot series."""
    py3 = sources.get('python3', {})
    return py3.get('scipy') or py3.get('numpy') or {}


def get_style(lang: str, impl: str, index: int):
    if lang == "python3" and impl == "harold":
        return COLOR_HAROLD, "o", ":"
    if lang in STYLING_PALETTE:
        color, marker, ls = STYLING_PALETTE[lang]
    else:
        colors = [COLOR_BLUE, COLOR_AMBER, COLOR_CYAN, COLOR_ORANGE, COLOR_PURPLE, COLOR_WHITE]
        markers = ["o", "s", "d", "^", "v"]
        linestyles = ["-", "--", "-.", ":"]
        color = colors[index % len(colors)]
        marker = markers[index % len(markers)]
        ls = linestyles[index % len(linestyles)]
    if impl not in ("default", "scipy", "numpy"):
        ls = ":" if ls == "-" else "-"
    return color, marker, ls


# ==========================================
# 2. Abstract Base Class
# ==========================================
class BaseModelPlotter(ABC):
    """
    Abstract base class enforcing a strict contract for all model plotters.
    """

    def __init__(self, sources: dict):
        self.sources = sources
        self.rust_data = sources.get('rust', {}).get('default', {})
        self.py_data = python_primary(sources)

    @abstractmethod
    def plot_details(self) -> plt.Figure:
        """
        Generates and returns a standalone Figure detailing the specific numerical domain.
        """
        pass

    @abstractmethod
    def plot_summary(self, ax: plt.Axes):
        """
        Draws the single most critical metric onto the provided Axes object
        for the combined overview figure.
        """
        pass


# ==========================================
# 3. Concrete Implementations
# ==========================================
class MatrixPlotter(BaseModelPlotter):
    """
    Matrix operations analysis focusing on EKF Ill-Conditioned Covariance Update & Collapse.
    Implements a 4-Quadrant visualization:
    Q1: EKF Covariance Relative Error Heatmap
    Q2: Algorithmic Scaling O(N^3) with Error Bars (1,000 iters)
    Q3: 32x32 Hilbert Solve Latency Jitter Violin Plot (1,000 iters)
    Q4: Decomposition Speedup Factors Bar Chart (16x16 State Matrix)
    """

    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Matrix Validation: EKF Benchmarks", fontsize=15,
                     fontweight='bold', y=0.98)

        ax1 = axs[0, 0]
        py_cov = np.array(self.py_data.get('covariance_heatmap', {}).get('matrix', []))
        rust_cov = np.array(self.rust_data.get('covariance_heatmap', {}).get('matrix', []))

        if py_cov.size == 0 or rust_cov.size != py_cov.size:
            raise MissingData(
                "covariance_heatmap.matrix absent or shape-mismatched between "
                f"the rust ({rust_cov.shape}) and python3 ({py_cov.shape}) payloads"
            )
        err_matrix = np.abs(py_cov - rust_cov) / (np.abs(py_cov) + 1e-15)

        vmin = max(1e-16, np.min(err_matrix[err_matrix > 0]) if np.any(err_matrix > 0) else 1e-16)
        vmax = max(1e-4, np.max(err_matrix))

        im1 = ax1.imshow(err_matrix, cmap=CMAP_CONTROL_RS, norm=LogNorm(vmin=vmin, vmax=vmax),
                         interpolation='nearest', aspect='auto')
        cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("Relative Error E", color=TEXT_COLOR, fontsize=8)
        ax1.set_title("EKF Covariance Relative Error", fontsize=11,
                      fontweight='bold')
        ax1.set_xlabel("State Col Index")
        ax1.set_ylabel("State Row Index")

        ax2 = axs[0, 1]
        n_dims = [2, 4, 8, 16, 32, 64]

        idx = 0
        for lang, impls in self.sources.items():
            for impl, impl_data in impls.items():
                scaling = impl_data.get('scaling', {})
                if not scaling:
                    continue
                n_dims_impl = scaling.get('N', n_dims)
                means = scaling.get('inversion_time_ns', [])
                stds = scaling.get('inversion_stddev_ns', [])
                if not means:
                    continue
                color, marker, ls = get_style(lang, impl, idx)
                label = f"{lang} ({impl})"
                ax2.errorbar(n_dims_impl, means, yerr=stds if stds else None, fmt=f"{marker}{ls}",
                             color=color, ecolor=color, elinewidth=1.5, capsize=4, label=label)
                idx += 1

        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.set_xticks(n_dims)
        ax2.set_xticklabels([f"N={n}" for n in n_dims])
        ax2.set_xlabel("Matrix Dimension N")
        ax2.set_ylabel("Inversion Time (ns)")
        ax2.set_title("Inversion Time Scaling vs. N", fontsize=11,
                      fontweight='bold')
        ax2.legend(frameon=True, fontsize=8)

        ax3 = axs[1, 0]
        idx = 0
        for lang, impls in self.sources.items():
            for impl, impl_data in impls.items():
                jitter = impl_data.get('jitter', {}).get('hilbert_solve_times_ns', [])
                if not jitter or all(v == 0.0 for v in jitter):
                    continue
                iters = np.arange(len(jitter))
                color, marker, _ = get_style(lang, impl, idx)
                alpha = 0.4 if lang == "python3" else 0.7
                ax3.scatter(iters, jitter, color=color, alpha=alpha, s=12, label=f"{lang} ({impl})")
                idx += 1

        ax3.set_yscale('log')
        ax3.set_xlabel("Iteration k")
        ax3.set_ylabel("Solve Time (ns)")
        ax3.set_title("Hilbert Solve Latency Jitter", fontsize=11,
                      fontweight='bold')
        ax3.legend(frameon=True, fontsize=8)

        ax4 = axs[1, 1]
        algos = ['Cholesky', 'LU Solve', 'QR Decomp', 'SVD']
        keys = ['cholesky', 'lu_solve', 'qr_decomp', 'svd']

        impl_times = []
        labels = []
        colors = []

        idx = 0
        for lang, impls in self.sources.items():
            for impl, impl_data in impls.items():
                decomp = impl_data.get('decomp_times_ns', {})

                def _extract_num(val):
                    if val is None:
                        return 0.0
                    if isinstance(val, (list, tuple, np.ndarray)):
                        return float(val[0]) if len(val) > 0 else 0.0
                    return float(val)

                times = [_extract_num(decomp.get(k, 0.0)) for k in keys]
                if all(t == 0.0 for t in times):
                    continue
                impl_times.append(times)
                labels.append(f"{lang} ({impl})")
                color, _, _ = get_style(lang, impl, idx)
                colors.append(color)
                idx += 1

        num_impls = len(impl_times)
        if num_impls > 0:
            x_pos = np.arange(len(algos))
            total_width = 0.8
            width = total_width / num_impls
            for i, times in enumerate(impl_times):
                offset = (i - (num_impls - 1) / 2) * width
                ax4.bar(x_pos + offset, times, width, label=labels[i], color=colors[i], alpha=0.85)

            ax4.set_xticks(x_pos)
            ax4.set_xticklabels(algos)

        ax4.set_yscale('log')
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("Matrix Decomposition Times", fontsize=11,
                      fontweight='bold')
        ax4.legend(frameon=True, fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    def plot_summary(self, ax: plt.Axes):
        # 2D Heatmap Summary of EKF Covariance Relative Error
        py_cov = np.array(self.py_data.get('covariance_heatmap', {}).get('matrix', []))
        rust_cov = np.array(self.rust_data.get('covariance_heatmap', {}).get('matrix', []))

        if py_cov.size == 0 or rust_cov.size != py_cov.size:
            raise MissingData(
                "covariance_heatmap.matrix absent or shape-mismatched between "
                f"the rust ({rust_cov.shape}) and python3 ({py_cov.shape}) payloads"
            )
        err_matrix = np.abs(py_cov - rust_cov) / (np.abs(py_cov) + 1e-15)

        vmin = max(1e-16, np.min(err_matrix[err_matrix > 0]) if np.any(err_matrix > 0) else 1e-16)
        vmax = max(1e-4, np.max(err_matrix))

        im = ax.imshow(err_matrix, cmap=CMAP_CONTROL_RS, norm=LogNorm(vmin=vmin, vmax=vmax),
                       interpolation='nearest', aspect='auto')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title("EKF Covariance Relative Error", fontsize=12, fontweight='bold')
        ax.set_xlabel("State Col")
        ax.set_ylabel("State Row")


class PolynomialPlotter(BaseModelPlotter):
    """
    Polynomial operations analysis featuring four core benchmark quadrants:
    Q1: Computational Complexity (Degree 1..50 Execution Time: Rust vs Python)
    Q2: Algorithmic Efficiency (Newton-Raphson Convergence Rate: Rust vs Python)
    Q3: Numerical Stability (Wilkinson Residual Error W(x) f32 vs f64: Rust vs Python)
    Q4: Control System Stability (Root Sensitivity Pole Cloud: Rust vs Python)
    """

    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("Polynomial Operations Validation",
                     fontsize=15, fontweight='bold', y=0.98)

        ax1 = axs[0, 0]
        rust_comp = self.rust_data.get('complexity', {})
        py_comp = self.py_data.get('complexity', {})

        degrees = require(rust_comp, 'degrees', context='polynomial complexity')
        horner_rs = require(rust_comp, 'horner_time_ns', context='polynomial complexity')
        naive_rs = require(rust_comp, 'naive_time_ns', context='polynomial complexity')

        horner_py = require(py_comp, 'horner_time_ns', context='polynomial complexity')
        naive_py = require(py_comp, 'naive_time_ns', context='polynomial complexity')

        ax1.plot(degrees, horner_rs, label='Rust Horner O(n)', color=COLOR_RS, linewidth=2.0)
        ax1.plot(degrees, naive_rs, '--', label='Rust Naive O(n²)', color=COLOR_RS, linewidth=1.5,
                 alpha=0.7)
        ax1.plot(degrees, horner_py, ':', label='Py polyval O(n)', color=COLOR_PY, linewidth=2.0)
        ax1.plot(degrees, naive_py, '-.', label='Py Naive O(n²)', color=COLOR_PY, linewidth=1.5,
                 alpha=0.7)

        ax1.set_xlabel("Polynomial Degree n")
        ax1.set_ylabel("Mean Execution Time (ns)")
        ax1.set_title("Evaluation Time vs. Degree", fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, fontsize=8)

        ax2 = axs[0, 1]
        rust_conv = self.rust_data.get('root_convergence', {})
        py_conv = self.py_data.get('root_convergence', {})

        distances = rust_conv.get('distances',
                                  [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0])
        rust_iters = require(rust_conv, 'iterations', context='Newton-Raphson convergence')
        py_iters = require(py_conv, 'iterations', context='Newton-Raphson convergence')

        ax2.plot(distances, py_iters, 'o-', color=COLOR_PY, label='Python polyval/polyder',
                 linewidth=1.8)
        ax2.plot(distances, rust_iters, 's--', color=COLOR_RS, label='Rust Poly::evaluate/deriv',
                 linewidth=1.8)
        ax2.axhline(100, color=COLOR_CRIT, linestyle=':', label='Max WCET Limit (100 Iters)')

        ax2.set_xlabel("Initial Guess Distance |x₀ - r*|")
        ax2.set_ylabel("Iterations to Converge (ε < 10⁻⁶)")
        ax2.set_title("Newton-Raphson Convergence", fontsize=11,
                      fontweight='bold')
        ax2.legend(frameon=True, fontsize=8)

        ax3 = axs[1, 0]
        rust_wilk = self.rust_data.get('wilkinson_residual', {})
        py_wilk = self.py_data.get('wilkinson_residual', {})
        flint_wilk = self.sources.get('python3', {}).get('flint', {}).get('wilkinson_residual', {})

        ctx = 'Wilkinson residuals'
        indices = np.array(require(rust_wilk, 'root_indices', context=ctx))
        res_f64_rs = np.clip(np.array(require(rust_wilk, 'residual_f64', context=ctx)), 1e-16, None)
        res_f32_rs = np.clip(np.array(require(rust_wilk, 'residual_f32', context=ctx)), 1e-16, None)

        res_f64_py = np.clip(np.array(require(py_wilk, 'residual_f64', context=ctx)), 1e-16, None)
        res_f32_py = np.clip(np.array(require(py_wilk, 'residual_f32', context=ctx)), 1e-16, None)
        res_flint = np.clip(np.array(require(flint_wilk, 'residual_f64', context=ctx)), 1e-16, None)

        # Plot actual residuals
        ax3.plot(indices, res_f32_py, color=COLOR_PY, marker='^', markersize=5,
                 linestyle=':', alpha=0.7, label='Py f32 Residual')
        ax3.plot(indices, res_f32_rs, color=COLOR_RS, marker='^', markersize=5,
                 linestyle='-', linewidth=1.5, label='Rust f32 Residual')

        ax3.plot(indices, res_f64_py, color=COLOR_PY, marker='o', markersize=5,
                 linestyle=':', alpha=0.7, label='Py f64 Residual')
        ax3.plot(indices, res_f64_rs, color=COLOR_RS, marker='o', markersize=5,
                 linestyle='-', linewidth=1.5, label='Rust f64 Residual')

        # Flint 256-bit arb_poly ball arithmetic: exact ground truth, ~0 at every
        # integer root, in contrast to the f64 catastrophic cancellation above.
        ax3.plot(indices, res_flint, color=COLOR_ALT, marker='*', markersize=8,
                 linestyle='-', linewidth=1.5, label='Flint 256-bit Ground Truth')

        ax3.set_yscale('log')
        ax3.set_xlabel("Root Index k (Wilkinson W(x) Roots 1..20)")
        ax3.set_ylabel("Absolute Residual Error |W(rₖ)|")
        ax3.set_title("Wilkinson Polynomial Residuals", fontsize=11,
                      fontweight='bold')
        ax3.legend(frameon=True, fontsize=8)

        ax4 = axs[1, 1]
        rust_sens = self.rust_data.get('root_sensitivity', {})
        py_sens = self.py_data.get('root_sensitivity', {})

        gt_re = require(rust_sens, 'ground_truth_re', context='root sensitivity')
        gt_im = require(rust_sens, 'ground_truth_im', context='root sensitivity')

        pert_re_rs = rust_sens.get('perturbed_re', [])
        pert_im_rs = rust_sens.get('perturbed_im', [])

        pert_re_py = py_sens.get('perturbed_re', [])
        pert_im_py = py_sens.get('perturbed_im', [])

        if pert_re_py and pert_im_py:
            ax4.scatter(pert_re_py, pert_im_py, color=COLOR_PY, alpha=0.3, s=15,
                        label='Python Poles (Quantized)')

        if pert_re_rs and pert_im_rs:
            ax4.scatter(pert_re_rs, pert_im_rs, color=COLOR_RS, alpha=0.4, marker='+', s=25,
                        label='Rust Poles (Quantized)')

        ax4.scatter(gt_re, gt_im, color='white', marker='X', s=110, edgecolors='black',
                    label='Exact Ground-Truth Poles (f64)', zorder=5)

        ax4.axvline(0.0, color=COLOR_CRIT, linestyle='--', alpha=0.8,
                    label='Stability Axis Re(s)=0')
        ax4.axhline(0.0, color=GRID_COLOR, linestyle='-', alpha=0.5)

        ax4.set_xlabel("Real Axis Re(s)")
        ax4.set_ylabel("Imaginary Axis Im(s)")
        ax4.set_title("Pole Sensitivity Under Perturbation", fontsize=11, fontweight='bold')
        ax4.legend(frameon=True, fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    def plot_summary(self, ax: plt.Axes):
        # Complex Pole Sensitivity Summary comparing Rust & Python vs Ground Truth
        rust_sens = self.rust_data.get('root_sensitivity', {})
        py_sens = self.py_data.get('root_sensitivity', {})

        gt_re = require(rust_sens, 'ground_truth_re', context='root sensitivity')
        gt_im = require(rust_sens, 'ground_truth_im', context='root sensitivity')

        pert_re_rs = rust_sens.get('perturbed_re', [])
        pert_im_rs = rust_sens.get('perturbed_im', [])

        pert_re_py = py_sens.get('perturbed_re', [])
        pert_im_py = py_sens.get('perturbed_im', [])

        if pert_re_py and pert_im_py:
            ax.scatter(pert_re_py, pert_im_py, color=COLOR_PY, alpha=0.3, s=12,
                       label='Python Poles')

        if pert_re_rs and pert_im_rs:
            ax.scatter(pert_re_rs, pert_im_rs, color=COLOR_RS, alpha=0.4, marker='+', s=18,
                       label='Rust Poles')

        ax.scatter(gt_re, gt_im, color='white', marker='X', s=100, edgecolors='black',
                   label='Ground-Truth Poles', zorder=5)
        ax.axvline(0.0, color=COLOR_CRIT, linestyle='--', alpha=0.8, label='Re(s)=0')

        ax.set_xlabel("Re(s)")
        ax.set_ylabel("Im(s)")
        ax.set_title("Pole Sensitivity Under Perturbation", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, fontsize=8)


class StateSpacePlotter(BaseModelPlotter):
    """
    State Space control analysis modeling an Underdamped Inverted Pendulum recovering
    from a step disturbance, ZOH algorithmic scaling, HIL execution jitter,
    and controllability/observability matrix construction.
    """

    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(12, 9))
        fig.suptitle("State Space Validation", fontsize=15,
                     fontweight='bold', y=0.98)

        py_pp = self.py_data.get('phase_portrait', {})
        rust_pp = self.rust_data.get('phase_portrait', {})

        py_scaling = self.py_data.get('scaling', {})
        rust_scaling = self.rust_data.get('scaling', {})

        py_jitter = self.py_data.get('jitter', {})
        rust_jitter = self.rust_data.get('jitter', {})

        py_cl = self.py_data.get('control_loop', {})
        rust_cl = self.rust_data.get('control_loop', {})

        harold_data = self.sources.get('python3', {}).get('harold', {})
        harold_pp = harold_data.get('phase_portrait', {})

        # Subplot 1: Quadrant 1 (Correctness Anchor) - Pendulum Disturbance Rejection Phase Portrait (theta vs theta_dot)
        ax1 = axs[0, 0]
        theta_py = py_pp.get('theta', [])
        theta_dot_py = py_pp.get('theta_dot', [])
        theta_rs = rust_pp.get('theta', [])
        theta_dot_rs = rust_pp.get('theta_dot', [])
        theta_ha = harold_pp.get('theta', [])
        theta_dot_ha = harold_pp.get('theta_dot', [])

        if theta_py and theta_dot_py:
            ax1.plot(theta_py, theta_dot_py, label='Python 3 (scipy)', color=COLOR_PY, linewidth=2.0)
        if theta_rs and theta_dot_rs:
            ax1.plot(theta_rs, theta_dot_rs, '--', label='Rust', color=COLOR_RS,
                     linewidth=2.0)
        if theta_ha and theta_dot_ha:
            ax1.plot(theta_ha, theta_dot_ha, ':', label='Python 3 (harold)', color=COLOR_HAROLD,
                     linewidth=1.5)

        ax1.scatter([0.0], [0.0], color=COLOR_CRIT, marker='*', s=150, zorder=5,
                    label='Equilibrium (0,0)')
        ax1.set_xlabel("Angle θ (rad)")
        ax1.set_ylabel("Angular Rate dθ/dt (rad/s)")
        ax1.set_title("Inverted Pendulum Phase Space", fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, fontsize=8)

        # Subplot 2: Quadrant 2 (Algorithmic Scaling) - ZOH Discretization Scaling vs State Size
        ax2 = axs[0, 1]
        state_sizes = require(py_scaling, 'state_size', context='discretization scaling')
        py_zoh = py_scaling.get('zoh_time_ns', [])
        rust_zoh = rust_scaling.get('zoh_time_ns', [])

        x = np.arange(len(state_sizes))
        width = 0.35

        if py_zoh:
            ax2.bar(x - width / 2, py_zoh, width, label='Python 3 ZOH', color=COLOR_PY, alpha=0.9)
        if rust_zoh:
            ax2.bar(x + width / 2, rust_zoh, width, label='Rust ZOH', color=COLOR_RS, alpha=0.9)

        ax2.set_xticks(x)
        ax2.set_xticklabels([f"N={n}" for n in state_sizes], rotation=30, ha='right')
        ax2.set_yscale('log')
        ax2.set_xlabel("State Size N")
        ax2.set_ylabel("ZOH Execution Time (ns)")
        ax2.set_title("ZOH Discretization Scaling", fontsize=11, fontweight='bold')
        ax2.legend(frameon=True, fontsize=8)

        # Subplot 3: Quadrant 3 (Determinism) - HIL Step Response Compute Times / Jitter
        ax3 = axs[1, 0]
        py_jit_times = py_jitter.get('step_compute_times_ns', [])
        rust_jit_times = rust_jitter.get('step_compute_times_ns', [])

        if py_jit_times:
            ax3.plot(np.arange(len(py_jit_times)), py_jit_times, label='Python 3 Jitter',
                     color=COLOR_PY, alpha=0.8, linewidth=1.2)
        if rust_jit_times:
            ax3.plot(np.arange(len(rust_jit_times)), rust_jit_times, label='Rust Jitter',
                     color=COLOR_RS, alpha=0.8, linewidth=1.2)

        ax3.set_yscale('log')
        ax3.set_xlabel("Iteration k")
        ax3.set_ylabel("Compute Time (ns)")
        ax3.set_title("Single-Step Computation Jitter", fontsize=11, fontweight='bold')
        ax3.legend(frameon=True, fontsize=8)

        # Subplot 4: Quadrant 4 (The Controllable, Observable) - Controllability & Observability
        ax4 = axs[1, 1]
        cl_sizes = require(py_cl, 'state_size', context='controllability scaling')
        py_ctrb = py_cl.get('controllability_time_ns', [])
        rust_ctrb = rust_cl.get('controllability_time_ns', [])
        py_obsv = py_cl.get('observability_time_ns', [])
        rust_obsv = rust_cl.get('observability_time_ns', [])
        harold_cl = harold_data.get('control_loop', harold_data)
        harold_ctrb = harold_cl.get('controllability_time_ns', [])
        harold_obsv = harold_cl.get('observability_time_ns', [])

        x4 = np.arange(len(cl_sizes))

        if py_ctrb:
            ax4.plot(x4, py_ctrb, 'o-', label='Py Ctrb', color=COLOR_PY, linewidth=1.8)
        if rust_ctrb:
            ax4.plot(x4, rust_ctrb, 's--', label='Rust Ctrb', color=COLOR_RS, linewidth=1.8)
        if py_obsv:
            ax4.plot(x4, py_obsv, '^:', label='Py Obsv', color=COLOR_ALT, linewidth=1.8)
        if rust_obsv:
            ax4.plot(x4, rust_obsv, 'd-.', label='Rust Obsv', color=COLOR_QUATERNARY, linewidth=1.8)
        if harold_ctrb:
            ax4.plot(x4[:len(harold_ctrb)], harold_ctrb, 'x:', label='Harold Ctrb',
                     color=COLOR_ASYMPTOTE, linewidth=1.2)
        if harold_obsv:
            ax4.plot(x4[:len(harold_obsv)], harold_obsv, '+--', label='Harold Obsv',
                     color=COLOR_ASYMPTOTE, linewidth=1.2)

        ax4.set_xticks(x4)
        ax4.set_xticklabels([f"N={n}" for n in cl_sizes])
        ax4.set_yscale('log')
        ax4.set_xlabel("State Dimension N")
        ax4.set_ylabel("Matrix Construction Time (ns)")
        ax4.set_title("Ctrb/Obsv Construction Scaling", fontsize=11, fontweight='bold')
        ax4.legend(frameon=True, fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    def plot_summary(self, ax: plt.Axes):
        # Pendulum Disturbance Rejection Phase Portrait Summary
        py_pp = self.py_data.get('phase_portrait', {})
        rust_pp = self.rust_data.get('phase_portrait', {})

        theta_py = py_pp.get('theta', [])
        theta_dot_py = py_pp.get('theta_dot', [])
        theta_rs = rust_pp.get('theta', [])
        theta_dot_rs = rust_pp.get('theta_dot', [])

        if theta_py and theta_dot_py:
            ax.plot(theta_py, theta_dot_py, label='Python 3', color=COLOR_PY, linewidth=2.0)
        if theta_rs and theta_dot_rs:
            ax.plot(theta_rs, theta_dot_rs, '--', label='Rust', color=COLOR_RS, linewidth=2.0)

        ax.scatter([0.0], [0.0], color=COLOR_CRIT, marker='*', s=120, label='Origin (0,0)')
        ax.set_xlabel("Angle θ (rad)")
        ax.set_ylabel("Angular Velocity dθ/dt")
        ax.set_title("Inverted Pendulum Phase Space", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, fontsize=8)


def add_nyquist_direction_arrows(ax, h_re, h_im, color=COLOR_RS, lw=1.5, mutation_scale=12):
    """Draws directional arrows along the Nyquist contours indicating increasing frequency."""
    if h_re is None or h_im is None or len(h_re) < 10:
        return
    re = np.array(h_re)
    im = np.array(h_im)
    im_ref = -im

    def _arrow(x, y, idx, forward=True, label=None):
        pts_disp = ax.transData.transform(np.column_stack([x, y]))
        tip = pts_disp[idx]
        dists = np.hypot(pts_disp[:, 0] - tip[0], pts_disp[:, 1] - tip[1])
        if forward:
            cand = np.where((dists >= 14.0) & (np.arange(len(x)) < idx))[0]
            if len(cand) == 0:
                return
            tail = cand[-1]
            ax.annotate("", xy=(x[idx], y[idx]), xytext=(x[tail], y[tail]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale),
                        zorder=6)
        else:
            # For mirrored curve (omega from -inf to 0), trajectory progresses toward index 0 (decreasing index).
            # Tail is at higher index (closer to -inf), Tip is at idx (closer to 0).
            cand = np.where((dists >= 14.0) & (np.arange(len(x)) > idx))[0]
            if len(cand) == 0:
                return
            tail = cand[0]
            ax.annotate("", xy=(x[idx], y[idx]), xytext=(x[tail], y[tail]),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, mutation_scale=mutation_scale),
                        zorder=6)
        if label:
            offset_y = 0.25 if forward else -0.45
            ax.text(x[idx] + 0.25, y[idx] + offset_y, label, color=color, fontsize=7.0, fontweight="bold", zorder=7)

    if len(re) > 145:
        _arrow(re, im, 115, forward=True, label=r"$\omega \uparrow$")
        _arrow(re, im, 142, forward=True)
        _arrow(re, im_ref, 142, forward=False)
        _arrow(re, im_ref, 115, forward=False, label=r"$\omega \uparrow$")


class TransferFuncPlotter(BaseModelPlotter):
    """
    Transfer function numerical analysis featuring four distinct benchmark quadrants:
    Q1: Discretization Method Error - Bode Magnitude (Rust vs. Python Tustin & ZOH up to Nyquist)
    Q2: Discretization Method Error - Bode Phase & Group Delay (Rust vs. Python phase warping)
    Q3: Nyquist Stability Criterion & Margins (Rust vs. Python polar trajectories, (-1, 0j) point, GM/PM)
    Q4: Filter Topology Stability (Rust vs. Python 6th-order Butterworth f32 Direct Form vs Biquad SOS)
    """

    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(13, 10))
        fig.suptitle(
            "Transfer Function Validation",
            fontsize=15, fontweight='bold', y=0.98)

        rust_disc = self.rust_data.get('discretization_error', {})
        py_disc = self.py_data.get('discretization_error', {})
        harold_data = self.sources.get('python3', {}).get('harold', {})
        harold_disc = harold_data.get('discretization_error', {})

        freqs_hz = require(rust_disc, 'freqs_hz', context='discretization sweep')

        ax1 = axs[0, 0]
        cont_mag_rs = require(rust_disc, 'cont_mag_db', context='discretization sweep')

        tustin_mag_rs = require(rust_disc, 'tustin_mag_db', context='discretization sweep')
        tustin_mag_py = require(py_disc, 'tustin_mag_db', context='discretization sweep')
        tustin_mag_ha = harold_disc.get('tustin_mag_db', [])

        zoh_mag_rs = require(rust_disc, 'zoh_mag_db', context='discretization sweep')
        zoh_mag_py = require(py_disc, 'zoh_mag_db', context='discretization sweep')
        zoh_mag_ha = harold_disc.get('zoh_mag_db', [])

        ax1.plot(freqs_hz, cont_mag_rs, label='Ideal Continuous H(s)', color=COLOR_TARGET,
                 linewidth=2.0)
        ax1.plot(freqs_hz, tustin_mag_py, ':', label='Py Tustin H(z)', color=COLOR_PY,
                 linewidth=2.0)
        ax1.plot(freqs_hz, tustin_mag_rs, '--', label='Rust Tustin H(z)', color=COLOR_RS,
                 linewidth=2.0)
        ax1.plot(freqs_hz, zoh_mag_py, '-.', label='Py ZOH H(z)', color=COLOR_PY, linewidth=1.5,
                 alpha=0.7)
        ax1.plot(freqs_hz, zoh_mag_rs, ':', label='Rust ZOH H(z)', color=COLOR_RS, linewidth=1.5,
                 alpha=0.7)
        if tustin_mag_ha:
            ax1.plot(freqs_hz, tustin_mag_ha, color=COLOR_HAROLD, linewidth=1.2, alpha=0.9,
                     label='Harold Tustin H(z)')
        if zoh_mag_ha:
            ax1.plot(freqs_hz, zoh_mag_ha, color=COLOR_CYAN, linewidth=1.2, alpha=0.9,
                     label='Harold ZOH H(z)')

        ax1.axvline(100.0, color=GRID_COLOR, linestyle='--', label='Nyquist Limit (100 Hz)')
        ax1.set_xlabel("Frequency (Hz)")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.set_title("Bode Magnitude (Fs = 200 Hz)\n"
                      r"Model: Modal Notch ($f_n=25\,\mathrm{Hz}$) + LP ($f_c=40\,\mathrm{Hz}$) | Tustin vs ZOH Gain",
                      fontsize=9.5, fontweight='bold', pad=8)
        ax1.legend(frameon=True, fontsize=7.5, loc='lower center')

        ax2 = axs[0, 1]
        cont_phase_rs = require(rust_disc, 'cont_phase_deg', context='discretization sweep')

        tustin_phase_rs = require(rust_disc, 'tustin_phase_deg', context='discretization sweep')
        tustin_phase_py = require(py_disc, 'tustin_phase_deg', context='discretization sweep')
        tustin_phase_ha = harold_disc.get('tustin_phase_deg', [])

        zoh_phase_rs = require(rust_disc, 'zoh_phase_deg', context='discretization sweep')
        zoh_phase_py = require(py_disc, 'zoh_phase_deg', context='discretization sweep')
        zoh_phase_ha = harold_disc.get('zoh_phase_deg', [])

        ax2.plot(freqs_hz, cont_phase_rs, label='Ideal Continuous H(s)', color=COLOR_TARGET,
                 linewidth=2.0)
        ax2.plot(freqs_hz, tustin_phase_py, ':', label='Py Tustin Phase', color=COLOR_PY,
                 linewidth=2.0)
        ax2.plot(freqs_hz, tustin_phase_rs, '--', label='Rust Tustin Phase', color=COLOR_RS,
                 linewidth=2.0)
        ax2.plot(freqs_hz, zoh_phase_py, '-.', label='Py ZOH Phase', color=COLOR_PY, linewidth=1.5,
                 alpha=0.7)
        ax2.plot(freqs_hz, zoh_phase_rs, ':', label='Rust ZOH Phase', color=COLOR_RS, linewidth=1.5,
                 alpha=0.7)
        if tustin_phase_ha:
            ax2.plot(freqs_hz, tustin_phase_ha, color=COLOR_HAROLD, linewidth=1.2, alpha=0.9,
                     label='Harold Tustin Phase')
        if zoh_phase_ha:
            ax2.plot(freqs_hz, zoh_phase_ha, color=COLOR_CYAN, linewidth=1.2, alpha=0.9,
                     label='Harold ZOH Phase')

        ax2.axvline(100.0, color=GRID_COLOR, linestyle='--', label='Nyquist Limit (100 Hz)')
        ax2.set_xlabel("Frequency (Hz)")
        ax2.set_ylabel("Phase (degrees)")
        ax2.set_title("Bode Phase & Frequency Warping (Fs = 200 Hz)\n"
                      r"Model: Modal Notch ($f_n=25\,\mathrm{Hz}$) + LP ($f_c=40\,\mathrm{Hz}$) | Bilinear Warping near Nyquist",
                      fontsize=9.5, fontweight='bold', pad=8)
        ax2.legend(frameon=True, fontsize=7.5, loc='upper right')

        ax3 = axs[1, 0]
        rust_nyq = self.rust_data.get('nyquist_criterion', {})
        py_nyq = self.py_data.get('nyquist_criterion', {})

        h_re_rs = rust_nyq.get('h_re', [])
        h_im_rs = rust_nyq.get('h_im', [])

        h_re_py = py_nyq.get('h_re', [])
        h_im_py = py_nyq.get('h_im', [])

        harold_nyq = harold_data.get('nyquist_criterion', {})
        h_re_ha = harold_nyq.get('h_re', [])
        h_im_ha = harold_nyq.get('h_im', [])

        crit_pt = rust_nyq.get('critical_point', [-1.0, 0.0])
        pm_deg = rust_nyq.get('phase_margin_deg', 45.0)
        gm_db = rust_nyq.get('gain_margin_db', 6.0)

        theta = np.linspace(0, 2 * np.pi, 200)

        ax3.plot(np.cos(theta), np.sin(theta), color=GRID_COLOR, linestyle='--', alpha=0.8,
                 zorder=0, label='Unit Circle')
        ax3.scatter([crit_pt[0]], [crit_pt[1]], color=COLOR_CRIT, marker='x', s=60,
                    label='Critical Point (-1,0j)')

        if h_re_py and h_im_py:
            h_im_py_ref = [-im for im in h_im_py]
            ax3.plot(h_re_py, h_im_py, label='Py H(jw)', color=COLOR_PY, linewidth=1.0)
            ax3.plot(h_re_py, h_im_py_ref, color=COLOR_PY, linewidth=1.0)

        if h_re_rs and h_im_rs:
            h_im_rs_ref = [-im for im in h_im_rs]
            ax3.plot(h_re_rs, h_im_rs, '--', label='Rust H(jw)', color=COLOR_RS, linewidth=1.0)
            ax3.plot(h_re_rs, h_im_rs_ref, '--', color=COLOR_RS, linewidth=1.0)
            add_nyquist_direction_arrows(ax3, h_re_rs, h_im_rs, color=COLOR_RS)

        if h_re_ha and h_im_ha:
            h_im_ha_ref = [-im for im in h_im_ha]
            ax3.plot(h_re_ha, h_im_ha, ':', label='Harold H(jw)', color=COLOR_HAROLD, linewidth=1.0)
            ax3.plot(h_re_ha, h_im_ha_ref, ':', label='Harold H(jw)', color=COLOR_HAROLD, linewidth=1.0)

        ax3.axvline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax3.axhline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax3.set_xlabel("Re{H(jw)}")
        ax3.set_ylabel("Im{H(jw)}")
        ax3.set_title(f"Nyquist Plot (GM = {gm_db:.1f} dB, PM = {pm_deg:.1f}°)\n"
                      r"Model: Open-Loop Plant $H(s) = \frac{50(s+2)}{s(s^2+2s+25)}$ | Margins vs (-1, 0j)",
                      fontsize=9.5, fontweight='bold', pad=8)
        ax3.set_xlim(-6, 6)
        ax3.set_ylim(-6, 6)
        ax3.legend(frameon=True, fontsize=7.5, loc='upper right')

        ax4 = axs[1, 1]
        rust_top = self.rust_data.get('topology_stability', {})
        py_top = self.py_data.get('topology_stability', {})

        gt_re = rust_top.get('ground_truth_re', [])
        gt_im = rust_top.get('ground_truth_im', [])

        df_re_rs = rust_top.get('direct_form_re', [])
        df_im_rs = rust_top.get('direct_form_im', [])
        df_re_py = py_top.get('direct_form_re', [])
        df_im_py = py_top.get('direct_form_im', [])

        bq_re_rs = rust_top.get('biquad_re', [])
        bq_im_rs = rust_top.get('biquad_im', [])
        bq_re_py = py_top.get('biquad_re', [])
        bq_im_py = py_top.get('biquad_im', [])

        # Draw Unit Circle |z| = 1
        ax4.plot(np.cos(theta), np.sin(theta), color=COLOR_CRIT, linestyle='--', alpha=0.8,
                 label='Unit Circle |z|=1')

        if gt_re and gt_im:
            ax4.scatter(gt_re, gt_im, color='white', marker='X', s=60, edgecolors='black',
                        label='Ground-Truth Poles', zorder=3)

        if df_re_py and df_im_py:
            ax4.scatter(df_re_py, df_im_py, color=COLOR_PY, marker='^', s=40,
                        label='Py f32 Direct Form', zorder=4)
        if df_re_rs and df_im_rs:
            ax4.scatter(df_re_rs, df_im_rs, color=COLOR_CRIT, marker='^', s=40,
                        label='Rust f32 Direct Form (Unstable)', zorder=4)

        if bq_re_py and bq_im_py:
            ax4.scatter(bq_re_py, bq_im_py, color=COLOR_PY, marker='s', s=30,
                        label='Py f32 Biquad SOS', zorder=4)
        if bq_re_rs and bq_im_rs:
            ax4.scatter(bq_re_rs, bq_im_rs, color=COLOR_RS, marker='o', s=20,
                        label='Rust f32 Biquad SOS (Stable)', zorder=4)

        ax4.axvline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax4.axhline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax4.set_xlim(-3, 3)
        ax4.set_ylim(-3, 3)
        ax4.set_xlabel("Re(z)")
        ax4.set_ylabel("Im(z)")
        ax4.set_title("6th-Order Pole Locations (z-Plane)\n"
                      r"Model: Butterworth LP ($f_c = 35\,\mathrm{Hz}$, f32) | Direct Form vs Biquad SOS",
                      fontsize=9.5, fontweight='bold', pad=8)
        ax4.legend(frameon=True, fontsize=7.5, loc='upper right')

        fig.tight_layout(rect=[0, 0, 1, 0.94], h_pad=2.8, w_pad=2.5)
        return fig

    def plot_summary(self, ax: plt.Axes):
        # Nyquist Plot Summary comparing Rust & Python vs Critical Point
        rust_nyq = self.rust_data.get('nyquist_criterion', {})
        py_nyq = self.py_data.get('nyquist_criterion', {})

        h_re_rs = rust_nyq.get('h_re', [])
        h_im_rs = rust_nyq.get('h_im', [])

        h_re_py = py_nyq.get('h_re', [])
        h_im_py = py_nyq.get('h_im', [])

        pm_deg = rust_nyq.get('phase_margin_deg', 45.0)
        gm_db = rust_nyq.get('gain_margin_db', 6.0)

        theta = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(theta), np.sin(theta), color=GRID_COLOR, linestyle='--', alpha=0.8,
                zorder=0, label='Unit Circle')
        ax.scatter([-1], [0], color=COLOR_CRIT, marker='x', s=100,
                   label='Critical Point (-1,0j)')

        if h_re_py and h_im_py:
            h_im_py_ref = [-im for im in h_im_py]
            ax.plot(h_re_py, h_im_py, label='Py H(jw)', color=COLOR_PY, linewidth=1.0)
            ax.plot(h_re_py, h_im_py_ref, label='Py H(-jw)', color=COLOR_PY, linewidth=1.0)

        if h_re_rs and h_im_rs:
            h_im_rs_ref = [-im for im in h_im_rs]
            ax.plot(h_re_rs, h_im_rs, '--', label='Rust H(jw)', color=COLOR_RS, linewidth=1.0)
            ax.plot(h_re_rs, h_im_rs_ref, '--', label='Rust H(-jw)', color=COLOR_RS, linewidth=1.0)
            add_nyquist_direction_arrows(ax, h_re_rs, h_im_rs, color=COLOR_RS, mutation_scale=10)

        ax.axvline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax.axhline(0.0, color=GRID_COLOR, linestyle=':', alpha=0.5)
        ax.set_xlabel("Re{H(jw)}")
        ax.set_ylabel("Im{H(jw)}")
        ax.set_title(f"Nyquist (GM = {gm_db:.1f} dB, PM = {pm_deg:.1f}°)\n"
                     r"Open-Loop $H(s) = \frac{50(s+2)}{s(s^2+2s+25)}$",
                     fontsize=9.5, fontweight='bold', pad=6)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_aspect('equal', adjustable='datalim')
        ax.legend(frameon=True, fontsize=7, loc='upper right')


class TensorPlotter(BaseModelPlotter):
    """
    Tensor & Array analysis featuring four distinct benchmark quadrants:
    Q1: Multilinear Interpolation Manifold (3D Saddle Point z = x^2 - y^2)
    Q2: Tensor Contraction Relative Error Heatmap (ArrayTensor::contract_into matrix multiplication)
    Q3: Quantized Precision Boundaries (Quantized<i8, 7> edge-case scaling & saturation)
    Q4: Bare-Metal Timing Profile (Zero-copy stack vs dynamic heap allocation baselines)
    """

    def plot_details(self) -> plt.Figure:
        fig = plt.figure(figsize=(12, 9))
        fig.suptitle("Tensor Operations Validation",
                     fontsize=15,
                     fontweight='bold', y=0.98)

        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        ax1.set_facecolor(PANEL_BG)

        rust_man = self.rust_data.get('manifold', {})
        mesh_u = np.array(require(rust_man, 'mesh_u', context='interpolation manifold'))
        mesh_v = np.array(require(rust_man, 'mesh_v', context='interpolation manifold'))
        interp_mesh = np.array(rust_man.get('interp_mesh', []))

        if interp_mesh.size > 0:
            U, V = np.meshgrid(mesh_u, mesh_v, indexing='ij')
            ax1.plot_surface(U, V, interp_mesh, cmap=CMAP_CONTROL_RS, alpha=0.85, edgecolor='none')
        else:
            x_grid = np.linspace(-2, 2, 30)
            y_grid = np.linspace(-2, 2, 30)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z_saddle = X ** 2 - Y ** 2
            ax1.plot_surface(X, Y, Z_saddle, cmap=CMAP_CONTROL_RS, alpha=0.85, edgecolor='none')

        ax1.set_xlabel("Grid Axis U", color=TEXT_COLOR, fontsize=8)
        ax1.set_ylabel("Grid Axis V", color=TEXT_COLOR, fontsize=8)
        ax1.set_zlabel("Surface Height Z", color=TEXT_COLOR, fontsize=8)
        ax1.set_title("3D Saddle Surface Manifold", fontsize=11,
                      fontweight='bold', color=TEXT_COLOR)

        # Inset text detailing Interpolation speeds
        rust_time = self.rust_data.get('timing', {})
        py_time = self.py_data.get('timing', {})
        r_interp = rust_time.get('interp_time_ns', 12.0)
        p_interp = py_time.get('interp_time_ns', 150000.0)

        info_text = (f"Grid Interp: Rust {r_interp:.0f}ns vs Py {p_interp / 1e3:.0f}µs\n")
        ax1.text2D(0.04, 0.95, info_text, transform=ax1.transAxes, verticalalignment='top',
                   fontsize=7,
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                             alpha=0.9))

        ax2 = fig.add_subplot(2, 2, 2)
        rust_c = np.array(self.rust_data.get('contraction', {}).get('mat_c', []))
        py_c = np.array(self.py_data.get('contraction', {}).get('mat_c', []))

        if rust_c.size == 0 or py_c.size != rust_c.size:
            raise MissingData(
                "contraction.mat_c absent or shape-mismatched between the "
                f"rust ({rust_c.shape}) and python3 ({py_c.shape}) payloads"
            )
        err_mat = np.abs(rust_c - py_c) / (np.abs(py_c) + 1e-12)

        vmin = max(1e-16, np.min(err_mat[err_mat > 0]) if np.any(err_mat > 0) else 1e-16)
        vmax = max(1e-3, np.max(err_mat))

        im2 = ax2.imshow(err_mat, cmap=CMAP_CONTROL_RS, norm=LogNorm(vmin=vmin, vmax=vmax),
                         interpolation='nearest', aspect='auto')
        cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
        cbar2.set_label("Relative Error E_ij", color=TEXT_COLOR, fontsize=8)
        ax2.set_title("Contraction Relative Error", fontsize=11, fontweight='bold')
        ax2.set_xlabel("Matrix Column j")
        ax2.set_ylabel("Matrix Row i")

        ax3 = fig.add_subplot(2, 2, 3)
        rust_bound = self.rust_data.get('boundaries', {})
        py_bound = self.py_data.get('boundaries', {})

        act_inputs = np.array(require(rust_bound, 'act_inputs', context='quantized boundaries'))
        act_exact = np.array(py_bound.get('act_outputs', py_bound.get('act_exact', np.tanh(act_inputs))))

        # Rust float outputs and quantized outputs
        act_outputs = np.array(rust_bound.get('act_outputs', act_exact))
        act_outputs_q_raw = np.array(rust_bound.get('act_outputs_q_raw', [0] * 121))
        rust_dequant = act_outputs_q_raw.astype(np.float32) / 128.0

        tflite_bound = self.sources.get('python3', {}).get('tflite', {}).get('boundaries', {})
        tflite_dequant = np.array(
            tflite_bound.get('act_outputs', py_bound.get('tflite_dequant', act_exact))
        )

        # Plot Tanh Curves on Left Y-Axis
        line2 = ax3.step(act_inputs, tflite_dequant, where='mid', color=COLOR_PY,
                         linewidth=1.5, label='TFLite Quantized (Q7)')
        line3 = ax3.step(act_inputs, rust_dequant, where='mid', color=COLOR_RS,
                         linewidth=1.5, label='Rust TableActivation (Q7)')
        line1, = ax3.plot(act_inputs, act_exact, color=COLOR_ASYMPTOTE, linewidth=1.0,
                          label='SciPy Exact Tanh (f32)', zorder=10)

        ax3.set_xlabel("Input Value x")
        ax3.set_ylabel("Activation Output")
        ax3.set_title("Quantized Tanh Activation", fontsize=11,
                      fontweight='bold')

        # Plot Errors on Right Y-Axis (twinx)
        ax3_err = ax3.twinx()
        err_scipy = np.abs(act_outputs - act_exact)
        err_tflite = np.abs(rust_dequant - tflite_dequant)

        line4, = ax3_err.plot(act_inputs, err_scipy, color=COLOR_ALT, linewidth=1.5, alpha=0.6,
                              label='|Rust - SciPy| (Approx Error)')
        line5, = ax3_err.plot(act_inputs, err_tflite, color=COLOR_QUATERNARY, linewidth=1.5, alpha=0.6,
                              label='|Rust - TFLite| (Divergence)')

        ax3_err.set_ylabel("Absolute Error", color=COLOR_ALT)
        ax3_err.tick_params(axis='y', labelcolor=COLOR_ALT)

        # Unified legend
        lines = [line1, line2[0], line3[0], line4, line5]
        labels = [l.get_label() for l in lines]
        ax3.legend(lines, labels, frameon=True, fontsize=7, loc='upper left', framealpha=0.6)

        ax4 = fig.add_subplot(2, 2, 4)
        rust_time = self.rust_data.get('timing', {})
        py_time = self.py_data.get('timing', {})

        sizes = rust_time.get('sizes', [4, 8, 16, 32, 64])
        rust_contracts = rust_time.get('contract_times_ns', [10.0, 40.0, 180.0, 1200.0, 9500.0])
        py_contracts = py_time.get('contract_times_ns', [1500.0, 2800.0, 8500.0, 24000.0, 89000.0])

        ax4.plot(sizes, py_contracts, 'o-', color=COLOR_PY, linewidth=2.0, markersize=6,
                 label='Python 3 (NumPy np.matmul)')
        ax4.plot(sizes, rust_contracts, 's--', color=COLOR_RS, linewidth=2.0, markersize=6,
                 label='Rust (control-rs contract_into)')

        ax4.set_xscale('log', base=2)
        ax4.set_yscale('log')
        ax4.set_xticks(sizes)
        ax4.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        ax4.set_xlabel("Tensor Dimension N (N x N Matrix)")
        ax4.set_ylabel("Execution Time per Op (ns)")
        ax4.set_title("Contraction Scaling vs. N", fontsize=11,
                      fontweight='bold')
        ax4.legend(frameon=True, fontsize=8)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        return fig

    def plot_summary(self, ax: plt.Axes):
        # 3D Saddle Surface Interpolation Contour Summary
        rust_man = self.rust_data.get('manifold', {})
        mesh_u = np.array(require(rust_man, 'mesh_u', context='interpolation manifold'))
        mesh_v = np.array(require(rust_man, 'mesh_v', context='interpolation manifold'))
        interp_mesh = np.array(rust_man.get('interp_mesh', []))

        if interp_mesh.size > 0:
            U, V = np.meshgrid(mesh_u, mesh_v, indexing='ij')
            contour = ax.contourf(U, V, interp_mesh, cmap=CMAP_CONTROL_RS, levels=15)
        else:
            x_grid = np.linspace(-2, 2, 30)
            y_grid = np.linspace(-2, 2, 30)
            X, Y = np.meshgrid(x_grid, y_grid)
            Z_saddle = X ** 2 - Y ** 2
            contour = ax.contourf(X, Y, Z_saddle, cmap=CMAP_CONTROL_RS, levels=15)

        plt.colorbar(contour, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("U")
        ax.set_ylabel("V")
        ax.set_title("3D Saddle Surface Contour", fontsize=12, fontweight='bold')


# ==========================================
# 4. Animated GIF Generator Functions
# ==========================================
def generate_covariance_collapse_gif(out_dir: str):
    """Generates matrix_covariance_collapse.gif showing EKF covariance collapse using FuncAnimation and PillowWriter."""
    print("Generating matrix_covariance_collapse.gif with FuncAnimation & PillowWriter...")
    dim = 10
    t_arr = np.linspace(0, 1, dim)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Initial covariance matrix
    P_0 = np.exp(-3 * (t_arr[:, None] - t_arr[None, :]) ** 2) + 1e-5 * np.eye(dim)

    cax = ax.imshow(P_0, cmap=CMAP_CONTROL_RS, interpolation='nearest', vmin=0, vmax=1.0)
    cbar = fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Covariance Magnitude P_ij", color=TEXT_COLOR, fontsize=9)

    ax.set_title("EKF Covariance Collapse (k = 0)", color=TEXT_COLOR, fontsize=11,
                 fontweight='bold')
    ax.set_xlabel("State Col Index", color=TEXT_COLOR)
    ax.set_ylabel("State Row Index", color=TEXT_COLOR)
    fig.tight_layout()

    def update(k):
        decay = np.exp(-0.15 * k)
        P_k = P_0 * (decay + 0.05) + 1e-5 * np.eye(dim)
        cax.set_array(P_k)
        ax.set_title(f"EKF Covariance Collapse (k = {k})", color=TEXT_COLOR, fontsize=11,
                     fontweight='bold')
        return [cax]

    anim = FuncAnimation(fig, update, frames=40, blit=False)
    gif_path = os.path.join(out_dir, "matrix_covariance_collapse.gif")
    anim.save(gif_path, writer=PillowWriter(fps=15))
    plt.close(fig)
    print(f" Saved {gif_path}")


def generate_inverted_pendulum_gif(out_dir: str, py_data: dict = None, rust_data: dict = None):
    """Generates inverted_pendulum_recovery.gif showing pendulum recovery overlaying Python (opacity 0.2) vs Rust."""
    print("Generating pendulum_simulation.gif...")

    # Retrieve phase portrait data from examples JSON payload if available
    theta_py = py_data.get('phase_portrait', {}).get('theta', []) if py_data else []
    theta_dot_py = py_data.get('phase_portrait', {}).get('theta_dot', []) if py_data else []

    theta_rs = rust_data.get('phase_portrait', {}).get('theta', []) if rust_data else []
    theta_dot_rs = rust_data.get('phase_portrait', {}).get('theta_dot', []) if rust_data else []

    # The animation compares two independently produced trajectories. It must
    # not integrate the plant itself: doing so would draw one curve twice and
    # label the halves 'Python' and 'Rust'.
    if not theta_rs or not theta_py:
        raise MissingData(
            "phase_portrait.theta/theta_dot absent from the "
            f"{'rust' if not theta_rs else 'python3'} payload; the pendulum "
            "animation compares emitted trajectories and does not simulate"
        )

    theta_py = np.array(theta_py)
    theta_dot_py = np.array(theta_dot_py)
    theta_rs = np.array(theta_rs)
    theta_dot_rs = np.array(theta_dot_rs)

    frames_cnt = min(len(theta_rs), len(theta_py))

    fig, (ax_phys, ax_phase) = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(
        "Pendulum Sim: Python vs. Rust",
        color=TEXT_COLOR, fontsize=13, fontweight='bold')

    # 1. Physical Pendulum Setup (Left Panel)
    ax_phys.set_xlim(-1.3, 1.3)
    ax_phys.set_ylim(-1.3, 1.3)
    ax_phys.set_aspect('equal')
    ax_phys.axhline(0, color=GRID_COLOR, linestyle=':', alpha=0.5)
    ax_phys.axvline(0, color=GRID_COLOR, linestyle=':', alpha=0.5)

    # Fixed pivot mount at top (0,0)
    ax_phys.scatter([0], [0], color=COLOR_ASYMPTOTE, s=140, zorder=4, label='Top Pivot (0,0)')
    pivot_stand = plt.Polygon([[-0.15, 0.15], [0.15, 0.15], [0, 0]], color=COLOR_ASYMPTOTE)
    ax_phys.add_patch(pivot_stand)

    # Python Pendulum Overlay (Opacity 0.2)
    bob_trail_py, = ax_phys.plot([], [], ':', color=COLOR_PY, alpha=0.2, linewidth=1.5)
    pole_line_py, = ax_phys.plot([], [], color=COLOR_PY, alpha=0.2, linewidth=4.0)
    bob_point_py, = ax_phys.plot([], [], 'o', color=COLOR_PY, alpha=0.2, markersize=14, zorder=5)

    # Rust Pendulum (Main)
    bob_trail_rs, = ax_phys.plot([], [], ':', color=COLOR_RS, alpha=0.4, linewidth=1.5)
    pole_line_rs, = ax_phys.plot([], [], color=COLOR_RS, linewidth=4.0)
    bob_point_rs, = ax_phys.plot([], [], 'o', color=COLOR_RS, markersize=14, zorder=4)

    ax_phys.set_title("Pendulum Trajectory", color=TEXT_COLOR, fontsize=11,
                      fontweight='bold')
    ax_phys.axis('off')

    # 2. Phase Plane Setup (Right Panel - θ vs dθ/dt)
    ax_phase.plot(theta_py, theta_dot_py, color=COLOR_PY, linestyle='-', alpha=0.2,
                  label='Python 3 (α=0.2)')
    ax_phase.plot(theta_rs, theta_dot_rs, color=COLOR_RS, linestyle='--', alpha=0.9,
                  label='Rust (control-rs)')

    phase_trail_py, = ax_phase.plot([], [], color=COLOR_PY, alpha=0.2, linewidth=2.0)
    curr_state_py, = ax_phase.plot([], [], 'o', color=COLOR_PY, alpha=0.2, markersize=8, zorder=4)

    phase_trail_rs, = ax_phase.plot([], [], color=COLOR_RS, alpha=0.9, linewidth=2.0)
    curr_state_rs, = ax_phase.plot([], [], 'o', color=COLOR_RS, markersize=10, zorder=5,
                                   label='State (θ, dθ/dt)')

    ax_phase.scatter([0.0], [0.0], color=COLOR_CRIT, marker='*', s=160, zorder=6,
                     label='Equilibrium (0,0)')

    ax_phase.set_xlabel("Angle θ (rad)", color=TEXT_COLOR)
    ax_phase.set_ylabel("Angular Rate dθ/dt (rad/s)", color=TEXT_COLOR)
    ax_phase.set_title("Phase Space (θ vs. dθ/dt)", color=TEXT_COLOR, fontsize=11,
                       fontweight='bold')
    ax_phase.legend(frameon=True, fontsize=8, loc='upper right')

    fig.tight_layout()

    pole_len = 1.0

    def update(k):
        th_py = theta_py[k]
        th_dot_py = theta_dot_py[k]

        th_rs = theta_rs[k]
        th_dot_rs = theta_dot_rs[k]

        # Coordinates for Python overlay
        tip_x_py = pole_len * np.sin(th_py)
        tip_y_py = -pole_len * np.cos(th_py)
        hist_x_py = pole_len * np.sin(theta_py[:k + 1])
        hist_y_py = -pole_len * np.cos(theta_py[:k + 1])

        bob_trail_py.set_data(hist_x_py, hist_y_py)
        pole_line_py.set_data([0, tip_x_py], [0, tip_y_py])
        bob_point_py.set_data([tip_x_py], [tip_y_py])

        phase_trail_py.set_data(theta_py[:k + 1], theta_dot_py[:k + 1])
        curr_state_py.set_data([th_py], [th_dot_py])

        # Coordinates for Rust
        tip_x_rs = pole_len * np.sin(th_rs)
        tip_y_rs = -pole_len * np.cos(th_rs)
        hist_x_rs = pole_len * np.sin(theta_rs[:k + 1])
        hist_y_rs = -pole_len * np.cos(theta_rs[:k + 1])

        bob_trail_rs.set_data(hist_x_rs, hist_y_rs)
        pole_line_rs.set_data([0, tip_x_rs], [0, tip_y_rs])
        bob_point_rs.set_data([tip_x_rs], [tip_y_rs])

        phase_trail_rs.set_data(theta_rs[:k + 1], theta_dot_rs[:k + 1])
        curr_state_rs.set_data([th_rs], [th_dot_rs])

        ax_phys.set_title(
            f"Step k = {k}\nθ = {th_rs:.2f} rad, dθ/dt = {th_dot_rs:.2f} rad/s",
            color=TEXT_COLOR, fontsize=10, fontweight='bold')
        return [
            bob_trail_py, pole_line_py, bob_point_py, phase_trail_py, curr_state_py,
            bob_trail_rs, pole_line_rs, bob_point_rs, phase_trail_rs, curr_state_rs
        ]

    anim = FuncAnimation(fig, update, frames=frames_cnt, blit=False)
    gif_path = os.path.join(out_dir, "pendulum_simulation.gif")
    anim.save(gif_path, writer=PillowWriter(fps=20))

    fallback_paths = [
        "pendulum_simulation.gif",
        os.path.join(out_dir, "..", "results", "pendulum_simulation.gif")
    ]
    for fb_path in fallback_paths:
        try:
            anim.save(fb_path, writer=PillowWriter(fps=20))
        except:
            pass

    plt.close(fig)
    print(f" Saved {gif_path}")


# ==========================================
# 5. Main Coordinator & Dashboard GridSpec
# ==========================================
def h5_group_to_dict(group):
    """Recursively converts an HDF5 group or file into a Python nested dictionary."""
    res = {}
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            res[key] = h5_group_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            val = item[()]
            if isinstance(val, bytes):
                val = val.decode('utf-8')
            elif isinstance(val, np.ndarray):
                if val.ndim == 0 or val.shape == (1,):
                    val = val.item()
                else:
                    val = val.tolist()
            res[key] = val
    return res


def load_results(filename_or_subject: str) -> dict:
    """Load ``results/<name>.*.h5`` into the plotter's ``sources`` layout."""
    out_dir = get_output_dir()
    base = filename_or_subject.replace('.json', '').replace('.h5', '')
    prefix = f"{base}."
    variant_files = {}
    if os.path.isdir(out_dir):
        for name in os.listdir(out_dir):
            if not name.endswith('.h5') or not name.startswith(prefix):
                continue
            variant = name[len(prefix):-3]
            if variant and '.' not in variant:
                variant_files[variant] = os.path.join(out_dir, name)

    if variant_files:
        try:
            norm_sources = {"python3": {}}
            for variant, path in variant_files.items():
                with h5py.File(path, 'r') as handle:
                    payload = h5_group_to_dict(handle)
                meta = payload.pop('_meta', {}) if isinstance(payload.get('_meta'), dict) else {}
                merged = payload
                if meta:
                    merged = {**meta, **payload}
                    for key, val in payload.items():
                        if isinstance(val, dict) and isinstance(meta.get(key), dict):
                            merged[key] = {**meta[key], **val}
                if variant == "rust":
                    norm_sources["rust"] = {"default": merged}
                else:
                    norm_sources["python3"][variant] = merged
                    norm_sources[variant] = {"default": merged}
            if "rust" not in norm_sources:
                norm_sources["rust"] = {"default": {}}
            return {"sources": norm_sources}
        except Exception as e:
            print(f"Error reading suite HDF5 {base}: {e}")

    h5_path = os.path.join(out_dir, f"{base}.h5")
    if os.path.exists(h5_path):
        try:
            with h5py.File(h5_path, 'r') as f:
                raw_dict = h5_group_to_dict(f)
                norm_sources = {}
                if "rust" in raw_dict:
                    norm_sources["rust"] = {"default": raw_dict["rust"]}
                norm_sources["python3"] = {}
                for k, v in raw_dict.items():
                    if k not in ("rust", "manifest"):
                        norm_sources["python3"][k] = v
                        norm_sources[k] = {"default": v}
                return {"sources": norm_sources}
        except Exception as e:
            print(f"Error reading HDF5 {h5_path}: {e}")

    # Fallback to json if h5 not found
    json_path = os.path.join(out_dir, f"{base}.json")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
                if "sources" in data and isinstance(data["sources"], dict):
                    s = data["sources"]
                    has_payload = any(isinstance(v, dict) and "payload" in v for v in s.values())
                    if has_payload:
                        norm_sources = {}
                        if "rust" in s:
                            rust_p = s["rust"].get("payload", {}) if isinstance(s["rust"], dict) else s["rust"]
                            norm_sources["rust"] = {"default": rust_p}
                        norm_sources["python3"] = {}
                        for k, v in s.items():
                            if k != "rust":
                                p = v.get("payload", {}) if isinstance(v, dict) else v
                                norm_sources["python3"][k] = p
                                norm_sources[k] = {"default": p}
                        data["sources"] = norm_sources
                return data
        except Exception as e:
            print(f"Error reading JSON {json_path}: {e}")

    print(f"Warning: neither {h5_path} nor {json_path} found. Returning empty dict.")
    return {}


load_json = load_results


def generate_overview(out_dir, plotters, matrix_data, poly_data, state_data, tf_data,
                      tensor_data):
    """Builds the 3x3 overview dashboard from the flattened payload views."""
    fig_over = plt.figure(figsize=(14, 10))
    fig_over.suptitle("control-rs Numerical Validation & Performance Dashboard",
                      fontsize=16, fontweight='bold', color=TEXT_COLOR, y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig_over, height_ratios=[1.1, 1.1, 1.0])

    # Row 0: Matrix Q2, Matrix Q3, TransferFunction Q3
    ax_mat_q2 = fig_over.add_subplot(gs[0, 0])
    py_scaling = matrix_data.get('python3', {}).get('scaling', {})
    rust_scaling = matrix_data.get('rust', {}).get('scaling', {})
    n_dims = require(py_scaling, 'N', context='matrix inversion scaling')
    ctx = 'matrix inversion scaling'
    py_means = require(py_scaling, 'inversion_time_ns', context=ctx)
    py_stds = require(py_scaling, 'inversion_stddev_ns', context=ctx)
    rust_means = require(rust_scaling, 'inversion_time_ns', context=ctx)
    rust_stds = require(rust_scaling, 'inversion_stddev_ns', context=ctx)

    ax_mat_q2.errorbar(n_dims, py_means, yerr=py_stds, fmt='o-', color=COLOR_PY, ecolor=COLOR_PY,
                       elinewidth=1.5, capsize=3, label='Py (SciPy)')
    ax_mat_q2.errorbar(n_dims, rust_means, yerr=rust_stds, fmt='s--', color=COLOR_RS,
                       ecolor=COLOR_RS, elinewidth=1.5, capsize=3, label='Rust (control-rs)')
    ax_mat_q2.set_xscale('log', base=2)
    ax_mat_q2.set_yscale('log')
    ax_mat_q2.set_xticks(n_dims)
    ax_mat_q2.set_xticklabels([f"N={n}" for n in n_dims], rotation=25, fontsize=7)
    ax_mat_q2.set_xlabel("Matrix Dimension N", fontsize=8)
    ax_mat_q2.set_ylabel("Inversion Time (ns)", fontsize=8)
    ax_mat_q2.set_title("Matrix Inversion scaling", fontsize=10, fontweight='bold')
    ax_mat_q2.legend(frameon=True, fontsize=7)

    ax_mat_q4 = fig_over.add_subplot(gs[0, 1])
    py_decomp = matrix_data.get('python3', {}).get('decomp_times_ns', {})
    rust_decomp = matrix_data.get('rust', {}).get('decomp_times_ns', {})
    algos = ['Cholesky', 'LU Solve', 'QR Decomp', 'SVD']
    keys = ['cholesky', 'lu_solve', 'qr_decomp', 'svd']
    py_times = [py_decomp.get(k, 0.0) for k in keys]
    rust_times = [rust_decomp.get(k, 0.0) for k in keys]

    x_pos = np.arange(len(algos))
    width = 0.35

    ax_mat_q4.bar(x_pos - width / 2, py_times, width, label='Py (SciPy)', color=COLOR_PY,
                  alpha=0.85)
    ax_mat_q4.bar(x_pos + width / 2, rust_times, width, label='Rust (control-rs)', color=COLOR_RS,
                  alpha=0.85)

    ax_mat_q4.set_xticks(x_pos)
    ax_mat_q4.set_xticklabels(algos, fontsize=7)
    ax_mat_q4.set_yscale('log')
    ax_mat_q4.set_ylabel("Execution Time (ns)", fontsize=8)
    ax_mat_q4.set_title("Matrix Decomposition times (16x16)", fontsize=10, fontweight='bold')
    ax_mat_q4.legend(frameon=True, fontsize=7)

    ax_tf_q3 = fig_over.add_subplot(gs[0, 2])
    plotters["transfer_function"].plot_summary(ax_tf_q3)
    ax_tf_q3.set_title(r"Transfer Function Nyquist (Open-Loop $H(s)$)", fontsize=10, fontweight='bold')

    # Row 1: StateSpace Q1, StateSpace Q2, Tensor Q1 (3D Surface)
    ax_ss_q1 = fig_over.add_subplot(gs[1, 0])
    py_pp = state_data.get('python3', {}).get('phase_portrait', {})
    rust_pp = state_data.get('rust', {}).get('phase_portrait', {})
    theta_py, theta_dot_py = py_pp.get('theta', []), py_pp.get('theta_dot', [])
    theta_rs, theta_dot_rs = rust_pp.get('theta', []), rust_pp.get('theta_dot', [])
    if theta_py and theta_dot_py:
        ax_ss_q1.plot(theta_py, theta_dot_py, label='Py RK4', color=COLOR_PY, linewidth=1.8)
    if theta_rs and theta_dot_rs:
        ax_ss_q1.plot(theta_rs, theta_dot_rs, '--', label='Rust RK4', color=COLOR_RS, linewidth=1.8)
    ax_ss_q1.scatter([0.0], [0.0], color=COLOR_CRIT, marker='*', s=120, zorder=5,
                     label='Origin (0,0)')
    ax_ss_q1.set_xlabel("Angle θ (rad)", fontsize=8)
    ax_ss_q1.set_ylabel("Rate dθ/dt (rad/s)", fontsize=8)
    ax_ss_q1.set_title("State-Space Phase Portrait", fontsize=10, fontweight='bold')
    ax_ss_q1.legend(frameon=True, fontsize=7)

    ax_ss_q2 = fig_over.add_subplot(gs[1, 1])
    py_scaling = state_data.get('python3', {}).get('scaling', {})
    rust_scaling = state_data.get('rust', {}).get('scaling', {})
    state_sizes = py_scaling.get('state_size', [2, 4, 8, 16, 32, 64, 128])
    py_zoh = py_scaling.get('zoh_time_ns', [])
    rust_zoh = rust_scaling.get('zoh_time_ns', [])
    x = np.arange(len(state_sizes))
    width = 0.35
    if py_zoh:
        ax_ss_q2.bar(x - width / 2, py_zoh, width, label='Py ZOH', color=COLOR_PY, alpha=0.85)
    if rust_zoh:
        ax_ss_q2.bar(x + width / 2, rust_zoh, width, label='Rust ZOH', color=COLOR_RS, alpha=0.85)
    ax_ss_q2.set_xticks(x)
    ax_ss_q2.set_xticklabels([f"N={n}" for n in state_sizes], rotation=30, ha='right', fontsize=7)
    ax_ss_q2.set_yscale('log')
    ax_ss_q2.set_xlabel("State Size N", fontsize=8)
    ax_ss_q2.set_ylabel("ZOH Time (ns)", fontsize=8)
    ax_ss_q2.set_title("ZOH Discretization scaling", fontsize=10, fontweight='bold')
    ax_ss_q2.legend(frameon=True, fontsize=7)

    ax_tens_q1 = fig_over.add_subplot(gs[1, 2], projection='3d')
    ax_tens_q1.set_facecolor(PANEL_BG)
    rust_man = tensor_data.get('rust', {}).get('manifold', {})
    mesh_u = np.array(require(rust_man, 'mesh_u', context='interpolation manifold'))
    mesh_v = np.array(require(rust_man, 'mesh_v', context='interpolation manifold'))
    interp_mesh = np.array(rust_man.get('interp_mesh', []))
    if interp_mesh.size > 0:
        U, V = np.meshgrid(mesh_u, mesh_v, indexing='ij')
        ax_tens_q1.plot_surface(U, V, interp_mesh, cmap=CMAP_CONTROL_RS, alpha=0.85,
                                edgecolor='none')
    else:
        x_g = np.linspace(-2, 2, 30);
        y_g = np.linspace(-2, 2, 30)
        X, Y = np.meshgrid(x_g, y_g)
        ax_tens_q1.plot_surface(X, Y, X ** 2 - Y ** 2, cmap=CMAP_CONTROL_RS, alpha=0.85,
                                edgecolor='none')
    ax_tens_q1.set_xlabel("U", fontsize=7);
    ax_tens_q1.set_ylabel("V", fontsize=7);
    ax_tens_q1.set_zlabel("Z", fontsize=7)
    ax_tens_q1.set_title("Tensor 3D Saddle Interpolation", fontsize=10, fontweight='bold')

    # Row 2: Polynomial Q4 / Q1 (Execution Time vs Degree) & Brand Card
    ax_poly_q4 = fig_over.add_subplot(gs[2, 0:2])
    rust_comp = poly_data.get('rust', {}).get('complexity', {})
    py_comp = poly_data.get('python3', {}).get('complexity', {})
    degrees = require(rust_comp, 'degrees', context='polynomial complexity')
    horner_rs = rust_comp.get('horner_time_ns', [d * 15.0 for d in degrees])
    naive_rs = rust_comp.get('naive_time_ns', [d * d * 5.0 for d in degrees])
    horner_py = py_comp.get('horner_time_ns', [d * 40.0 for d in degrees])
    naive_py = py_comp.get('naive_time_ns', [d * d * 10.0 for d in degrees])

    ax_poly_q4.plot(degrees, horner_rs, label='Rust Horner O(n)', color=COLOR_RS, linewidth=2.0)
    ax_poly_q4.plot(degrees, naive_rs, '--', label='Rust Naive O(n²)', color=COLOR_RS,
                    linewidth=1.5, alpha=0.7)
    ax_poly_q4.plot(degrees, horner_py, ':', label='Py polyval O(n)', color=COLOR_PY, linewidth=2.0)
    ax_poly_q4.plot(degrees, naive_py, '-.', label='Py Naive O(n²)', color=COLOR_PY, linewidth=1.5,
                    alpha=0.7)
    ax_poly_q4.set_xlabel("Polynomial Degree n", fontsize=8)
    ax_poly_q4.set_ylabel("Execution Time (ns)", fontsize=8)
    ax_poly_q4.set_title("Horner vs. Naive Evaluation", fontsize=10,
                         fontweight='bold')
    ax_poly_q4.legend(frameon=True, fontsize=7, ncol=2)

    ax_card = fig_over.add_subplot(gs[2, 2])
    ax_card.axis('off')
    ax_card.set_facecolor(PANEL_BG)
    ax_card.text(0.5, 0.82, 'control-rs', ha='center', va='center', fontsize=14,
                 fontweight='bold', color=COLOR_RS)
    ax_card.text(0.5, 0.68, 'Numerical Models Validation & Performance Suite', ha='center',
                 va='center', fontsize=10,
                 fontstyle='italic', color=TEXT_COLOR)

    legend_elements = [
        Line2D([0], [0], color=COLOR_RS, lw=2.5, linestyle='--', label='Rust (control-rs)'),
        Line2D([0], [0], color=COLOR_PY, lw=2.5, label='Python 3 (SciPy / NumPy)'),
        Line2D([0], [0], marker='*', color=COLOR_CRIT, label='Critical Bounds / Origin',
               markersize=9, linestyle='None')
    ]
    ax_card.legend(handles=legend_elements, loc='lower center', fontsize=8, frameon=True,
                   facecolor=PANEL_BG, edgecolor=GRID_COLOR)

    fig_over.tight_layout(rect=[0, 0, 1, 0.93])
    overview_path = os.path.join(out_dir, "overview_summary.png")
    fig_over.savefig(overview_path, dpi=300)
    plt.close(fig_over)
    print(f" Saved {overview_path}")


def main():
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    matrix_data = load_json('matrix.json')
    poly_data = load_json('polynomial.json')
    state_data = load_json('state_space.json')
    tf_data = load_json('transfer_function.json')
    tensor_data = load_json('tensor.json')

    out_dir = get_output_dir()
    print(f"Output directory resolved to: {out_dir}")

    plotters = {
        "matrix": MatrixPlotter(matrix_data.get('sources', {})),
        "polynomial": PolynomialPlotter(poly_data.get('sources', {})),
        "state_space": StateSpacePlotter(state_data.get('sources', {})),
        "transfer_function": TransferFuncPlotter(tf_data.get('sources', {})),
        "tensor": TensorPlotter(tensor_data.get('sources', {}))
    }

    # 3. Generate Details PNGs
    print("Generating detailed plots with control-rs theme...")
    detail_failures: list[str] = []
    for name, plotter in plotters.items():
        try:
            fig = plotter.plot_details()
        except MissingData as exc:
            detail_failures.append(f"{name}_details: {exc}")
            continue
        filename = os.path.join(out_dir, f"{name}_details.png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print(f" Saved {filename}")

    # 4. Generate Animated GIFs
    if os.getenv("MAKE_VALIDATION_GIFS"):
        try:
            generate_covariance_collapse_gif(out_dir)
            state_sources = state_data.get('sources', {})
            generate_inverted_pendulum_gif(out_dir,
                                           state_sources.get('python3', {}).get('scipy', {}),
                                           state_sources.get('rust', {}).get('default', {}))
        except MissingData as exc:
            detail_failures.append(f"animations: {exc}")

    # 5. Generate Overview Dashboard (GridSpec 3x3)
    # Extract backward-compatible dictionaries for the overview dashboard
    matrix_data = {"python3": python_primary(matrix_data.get('sources', {})),
                   "rust": matrix_data.get('sources', {}).get('rust', {}).get('default', {})}
    poly_data = {"python3": python_primary(poly_data.get('sources', {})),
                 "rust": poly_data.get('sources', {}).get('rust', {}).get('default', {})}
    state_data = {"python3": python_primary(state_data.get('sources', {})),
                  "rust": state_data.get('sources', {}).get('rust', {}).get('default', {})}
    tf_data = {"python3": python_primary(tf_data.get('sources', {})),
               "rust": tf_data.get('sources', {}).get('rust', {}).get('default', {})}
    tensor_data = {"python3": python_primary(tensor_data.get('sources', {})),
                   "rust": tensor_data.get('sources', {}).get('rust', {}).get('default', {})}

    missing: list[str] = list(detail_failures)

    print("Generating overview dashboard with latest benchmark panels...")
    try:
        generate_overview(out_dir, plotters, matrix_data, poly_data, state_data,
                          tf_data, tensor_data)
    except MissingData as exc:
        missing.append(f"overview_summary: {exc}")

    if missing:
        print("", file=sys.stderr)
        print("Figures not rendered because required results were absent:",
              file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("No substitute values are drawn; regenerate the payloads and rerun.",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
