import json
import os
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
import numpy as np

# Set global matplotlib style parameters for clean, professional aesthetics
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['grid.color'] = '#e0e0e0'
plt.rcParams['grid.linestyle'] = '--'
plt.rcParams['grid.alpha'] = 0.6

# Color palette for cross-language validation
COLOR_PY = '#2b5c8f'    # Deep Blue for Python3
COLOR_RUST = '#d9534f'  # Crimson/Orange for Rust
COLOR_ALT = '#2ec4b6'   # Teal accent
COLOR_ALT2 = '#ff9f1c'  # Amber accent


def get_output_dir() -> str:
    """Resolves the output directory for saving generated plots."""
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    candidates = [
        os.path.join(script_dir, '..', 'results'),
        os.path.join(script_dir, 'results'),
        os.path.abspath('examples/numerical-models-validation/results'),
        os.path.abspath('results'),
        '.'
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return '.'


# ==========================================
# 1. Abstract Base Class
# ==========================================
class BaseModelPlotter(ABC):
    """
    Abstract base class enforcing a strict contract for all model plotters.
    """
    def __init__(self, py_data: dict, rust_data: dict):
        self.py_data = py_data
        self.rust_data = rust_data

    @abstractmethod
    def plot_details(self) -> plt.Figure:
        """
        Generates and returns a standalone Figure with a 2D-array of subplots 
        detailing the specific numerical domain.
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
# 2. Concrete Implementations
# ==========================================
class MatrixPlotter(BaseModelPlotter):
    """
    Matrix operations analysis including Relative Error Heatmaps E = |A_py - A_rs| / (|A_py| + eps)
    to reveal precision loss, GEMM error concentration, and topological structure.
    """
    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))
        fig.suptitle("Matrix Operations & Error Topology Analysis", fontsize=15, fontweight='bold', y=0.98)

        # 1. Relative Error Heatmap for Matrix Inversion (A_inv)
        ax1 = axs[0, 0]
        py_inv = np.array(self.py_data.get('inversion', {}).get('a_inv', []))
        rust_inv = np.array(self.rust_data.get('inversion', {}).get('a_inv', []))

        if py_inv.size > 0 and rust_inv.size > 0 and py_inv.shape == rust_inv.shape:
            err_inv = np.abs(py_inv - rust_inv) / (np.abs(py_inv) + 1e-15)
            im1 = ax1.imshow(err_inv, cmap='magma', norm=LogNorm(vmin=max(1e-16, np.min(err_inv[err_inv > 0]) if np.any(err_inv > 0) else 1e-16), vmax=max(1e-12, np.max(err_inv))), aspect='auto')
            cbar1 = fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
            cbar1.set_label("Relative Error E", fontsize=8)
            ax1.set_title("Inversion A⁻¹ Relative Error Topology", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Column Index")
            ax1.set_ylabel("Row Index")
        else:
            ax1.text(0.5, 0.5, "Matrix Inversion Data Unavailable", ha='center', va='center')
            ax1.set_title("Inversion Relative Error Heatmap", fontsize=11, fontweight='bold')

        # 2. Relative Error Heatmap for MatMul Chain Product (64x64 / 8x8)
        ax2 = axs[0, 1]
        py_mm = np.array(self.py_data.get('matmul_chain', {}).get('final_matrix', []))
        rust_mm = np.array(self.rust_data.get('matmul_chain', {}).get('final_matrix', []))

        if py_mm.size > 0 and rust_mm.size > 0:
            dim = int(np.sqrt(py_mm.size))
            py_mm_2d = py_mm.reshape(dim, dim)
            rust_mm_2d = rust_mm.reshape(dim, dim)
            err_mm = np.abs(py_mm_2d - rust_mm_2d) / (np.abs(py_mm_2d) + 1e-15)
            
            vmin_mm = max(1e-16, np.min(err_mm[err_mm > 0]) if np.any(err_mm > 0) else 1e-16)
            vmax_mm = max(1e-10, np.max(err_mm))
            im2 = ax2.imshow(err_mm, cmap='inferno', norm=LogNorm(vmin=vmin_mm, vmax=vmax_mm), aspect='auto')
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Relative Error E", fontsize=8)
            ax2.set_title("MatMul Chain Relative Error Heatmap", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Col Index")
            ax2.set_ylabel("Row Index")
        else:
            ax2.text(0.5, 0.5, "MatMul Chain Data Unavailable", ha='center', va='center')
            ax2.set_title("MatMul Relative Error Heatmap", fontsize=11, fontweight='bold')

        # 3. Hilbert Solve Solution Vector Overlay (x_hat) & Accuracy Metrics
        ax3 = axs[1, 0]
        py_xhat = self.py_data.get('hilbert', {}).get('x_hat', [])
        rust_xhat = self.rust_data.get('hilbert', {}).get('x_hat', [])
        indices = np.arange(len(py_xhat))

        if len(py_xhat) > 0:
            ax3.plot(indices, py_xhat, 'o-', label='Python 3 (x_hat)', color=COLOR_PY, linewidth=1.5)
        if len(rust_xhat) > 0:
            ax3.plot(indices, rust_xhat, 's--', label='Rust (x_hat)', color=COLOR_RUST, linewidth=1.5)
        ax3.axhline(1.0, color='gray', linestyle=':', label='True Value (1.0)')
        ax3.set_xlabel("Vector Element Index")
        ax3.set_ylabel("Solved x_hat Value")
        ax3.set_title("Hilbert System Solve Accuracy", fontsize=11, fontweight='bold')
        ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax3.grid(True)

        # 4. Timing Comparison Across Operations
        ax4 = axs[1, 1]
        ops = ['Hilbert', 'Inversion', 'MatMul Chain']
        py_times = [
            self.py_data.get('hilbert', {}).get('time_ns', 0.0),
            self.py_data.get('inversion', {}).get('time_ns', 0.0),
            self.py_data.get('matmul_chain', {}).get('time_ns', 0.0)
        ]
        rust_times = [
            self.rust_data.get('hilbert', {}).get('time_ns', 0.0),
            self.rust_data.get('inversion', {}).get('time_ns', 0.0),
            self.rust_data.get('matmul_chain', {}).get('time_ns', 0.0)
        ]

        x_ops = np.arange(len(ops))
        width = 0.35
        ax4.bar(x_ops - width / 2, py_times, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        ax4.bar(x_ops + width / 2, rust_times, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("Matrix Subprograms Timing Profile", fontsize=11, fontweight='bold')
        ax4.set_xticks(x_ops)
        ax4.set_xticklabels(ops)
        ax4.set_yscale('log')
        ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax4.grid(True, which='both')

        fig.tight_layout()
        return fig

    def plot_summary(self, ax: plt.Axes):
        # 2D Heatmap Summary of MatMul Chain Error Matrix Topology
        py_mm = np.array(self.py_data.get('matmul_chain', {}).get('final_matrix', []))
        rust_mm = np.array(self.rust_data.get('matmul_chain', {}).get('final_matrix', []))

        if py_mm.size > 0 and rust_mm.size > 0:
            dim = int(np.sqrt(py_mm.size))
            err_mm = np.abs(py_mm.reshape(dim, dim) - rust_mm.reshape(dim, dim)) / (np.abs(py_mm.reshape(dim, dim)) + 1e-15)
            vmin = max(1e-16, np.min(err_mm[err_mm > 0]) if np.any(err_mm > 0) else 1e-16)
            vmax = max(1e-10, np.max(err_mm))
            im = ax.imshow(err_mm, cmap='magma', norm=LogNorm(vmin=vmin, vmax=vmax), aspect='auto')
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.set_title("Matrix: MatMul Relative Error Heatmap", fontsize=12, fontweight='bold')
            ax.set_xlabel("Col Index")
            ax.set_ylabel("Row Index")
        else:
            ax.text(0.5, 0.5, "Heatmap Data Unavailable", ha='center', va='center')
            ax.set_title("Matrix: MatMul Relative Error", fontsize=12, fontweight='bold')


class PolynomialPlotter(BaseModelPlotter):
    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Polynomial Operations & Sensitivity Analysis", fontsize=15, fontweight='bold', y=0.98)

        # 1. Clustered Roots Evaluation Curve (x vs y)
        ax1 = axs[0, 0]
        py_x = self.py_data.get('clustered', {}).get('x', [])
        py_y = self.py_data.get('clustered', {}).get('y', [])
        rust_x = self.rust_data.get('clustered', {}).get('x', [])
        rust_y = self.rust_data.get('clustered', {}).get('y', [])

        if py_x and py_y:
            ax1.plot(py_x, py_y, label='Python 3', color=COLOR_PY, linewidth=1.8)
        if rust_x and rust_y:
            ax1.plot(rust_x, rust_y, '--', label='Rust', color=COLOR_RUST, linewidth=1.8)
        ax1.set_xlabel("x")
        ax1.set_ylabel("Polynomial Evaluation y(x)")
        ax1.set_title("Clustered Root Polynomial Evaluation", fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax1.grid(True)

        # 2. Derivative Coefficients Comparison
        ax2 = axs[0, 1]
        py_deriv = self.py_data.get('tutorial', {}).get('deriv', [])
        rust_deriv = self.rust_data.get('tutorial', {}).get('deriv', [])
        idx_deriv = np.arange(max(len(py_deriv), len(rust_deriv)))
        width = 0.35

        if py_deriv:
            ax2.bar(idx_deriv[:len(py_deriv)] - width / 2, py_deriv, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        if rust_deriv:
            ax2.bar(idx_deriv[:len(rust_deriv)] + width / 2, rust_deriv, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax2.set_xlabel("Degree Index")
        ax2.set_ylabel("Derivative Coeff Value")
        ax2.set_title("Polynomial Derivative Coefficients", fontsize=11, fontweight='bold')
        ax2.set_xticks(idx_deriv)
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax2.grid(True)

        # 3. Integral Coefficients Comparison
        ax3 = axs[1, 0]
        py_integ = self.py_data.get('tutorial', {}).get('integ', [])
        rust_integ = self.rust_data.get('tutorial', {}).get('integ', [])
        idx_integ = np.arange(max(len(py_integ), len(rust_integ)))

        if py_integ:
            ax3.bar(idx_integ[:len(py_integ)] - width / 2, py_integ, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        if rust_integ:
            ax3.bar(idx_integ[:len(rust_integ)] + width / 2, rust_integ, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax3.set_xlabel("Degree Index")
        ax3.set_ylabel("Integral Coeff Value")
        ax3.set_title("Polynomial Integral Coefficients", fontsize=11, fontweight='bold')
        ax3.set_xticks(idx_integ)
        ax3.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax3.grid(True)

        # 4. Polynomial Operations Timing Profile
        ax4 = axs[1, 1]
        ops = ['Eval', 'Deriv', 'Integ', 'Mul', 'Div', 'Companion']
        py_tut = self.py_data.get('tutorial', {})
        rust_tut = self.rust_data.get('tutorial', {})

        py_times = [py_tut.get(f'{op.lower()}_time_ns', 0.0) for op in ops]
        rust_times = [rust_tut.get(f'{op.lower()}_time_ns', 0.0) for op in ops]

        x_ops = np.arange(len(ops))
        ax4.bar(x_ops - width / 2, py_times, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        ax4.bar(x_ops + width / 2, rust_times, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("Polynomial Subprograms Timing", fontsize=11, fontweight='bold')
        ax4.set_xticks(x_ops)
        ax4.set_xticklabels(ops)
        ax4.set_yscale('log')
        ax4.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax4.grid(True, which='both')

        fig.tight_layout()
        return fig

    def plot_summary(self, ax: plt.Axes):
        py_time = self.py_data.get('clustered', {}).get('time_ns', 0.0)
        rust_time = self.rust_data.get('clustered', {}).get('time_ns', 0.0)

        langs = ['Python 3', 'Rust']
        times = [py_time, rust_time]
        colors = [COLOR_PY, COLOR_RUST]

        bars = ax.bar(langs, times, color=colors, width=0.5, alpha=0.85)
        ax.set_ylabel("Time (ns, log scale)")
        ax.set_yscale('log')
        ax.set_title("Polynomial: Clustered Eval Time", fontsize=12, fontweight='bold')
        ax.grid(True, which='both', axis='y')

        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2e} ns',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9, fontweight='bold')


class StateSpacePlotter(BaseModelPlotter):
    """
    State space control analysis featuring Phase Portraits (Time Domain) to expose energy
    conservation, orbital spirals, and ZOH discretization numerical drift.
    """
    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("State Space Phase Portraits & Control Dynamics", fontsize=15, fontweight='bold', y=0.98)

        py_tut = self.py_data.get('tutorial', {})
        rust_tut = self.rust_data.get('tutorial', {})

        # 1. Step Response Phase Portrait (x1 vs x2)
        ax1 = axs[0, 0]
        if py_tut.get('step_x1') and py_tut.get('step_x2'):
            ax1.plot(py_tut['step_x1'], py_tut['step_x2'], label='Python 3 (scipy)', color=COLOR_PY, linewidth=2.0)
            ax1.scatter(py_tut['step_x1'][0], py_tut['step_x2'][0], color=COLOR_PY, marker='o', s=40, label='Py Start')
            ax1.scatter(py_tut['step_x1'][-1], py_tut['step_x2'][-1], color=COLOR_PY, marker='*', s=80, label='Py Equilibrium')
        if rust_tut.get('step_x1') and rust_tut.get('step_x2'):
            ax1.plot(rust_tut['step_x1'], rust_tut['step_x2'], '--', label='Rust (control-rs ZOH)', color=COLOR_RUST, linewidth=2.0)
            ax1.scatter(rust_tut['step_x1'][-1], rust_tut['step_x2'][-1], color=COLOR_RUST, marker='x', s=80, label='Rust End State')
        ax1.set_xlabel("State Position x1(t)")
        ax1.set_ylabel("State Velocity x2(t)")
        ax1.set_title("Step Response Phase Portrait (x1 vs x2)", fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax1.grid(True)

        # 2. Free Initial-State Response Phase Portrait (free_x1 vs free_x2)
        ax2 = axs[0, 1]
        if py_tut.get('free_x1') and py_tut.get('free_x2'):
            ax2.plot(py_tut['free_x1'], py_tut['free_x2'], label='Python 3 Free Orbit', color=COLOR_PY, linewidth=2.0)
            ax2.scatter(py_tut['free_x1'][0], py_tut['free_x2'][0], color=COLOR_PY, marker='o', s=40)
        if rust_tut.get('free_x1') and rust_tut.get('free_x2'):
            ax2.plot(rust_tut['free_x1'], rust_tut['free_x2'], '--', label='Rust Free Orbit', color=COLOR_RUST, linewidth=2.0)
            ax2.scatter(rust_tut['free_x1'][0], rust_tut['free_x2'][0], color=COLOR_RUST, marker='x', s=40)
        ax2.set_xlabel("Free State x1(t)")
        ax2.set_ylabel("Free State x2(t)")
        ax2.set_title("Free Response Phase Orbit (Energy Conservation)", fontsize=11, fontweight='bold')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax2.grid(True)

        # 3. Time Series Overlay (step_x1, step_x2, step_y)
        ax3 = axs[1, 0]
        steps = np.arange(len(py_tut.get('step_y', [])))
        if py_tut.get('step_x1'):
            ax3.plot(steps, py_tut['step_x1'], label='Py step_x1', color=COLOR_PY, linestyle='-')
            ax3.plot(steps, py_tut['step_y'], label='Py output y', color=COLOR_PY, linestyle='--')
        if rust_tut.get('step_x1'):
            ax3.plot(steps, rust_tut['step_x1'], label='Rust step_x1', color=COLOR_RUST, linestyle='-')
            ax3.plot(steps, rust_tut['step_y'], label='Rust output y', color=COLOR_RUST, linestyle='--')
        ax3.set_xlabel("Time Step")
        ax3.set_ylabel("State / Output Value")
        ax3.set_title("Time-Series Step Response Overlay", fontsize=11, fontweight='bold')
        ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax3.grid(True)

        # 4. Compare step_time_ns vs zoh_time_ns and other operations
        ax4 = axs[1, 1]
        ops = ['Step Response', 'ZOH Discretization', 'Similarity Trans.', 'Deriv Eval']
        py_times = [
            py_tut.get('step_time_ns', 0.0),
            py_tut.get('zoh_time_ns', 0.0),
            py_tut.get('similarity_time_ns', 0.0),
            py_tut.get('deriv_time_ns', 0.0)
        ]
        rust_times = [
            rust_tut.get('step_time_ns', 0.0),
            rust_tut.get('zoh_time_ns', 0.0),
            rust_tut.get('similarity_time_ns', 0.0),
            rust_tut.get('deriv_time_ns', 0.0)
        ]

        x_ops = np.arange(len(ops))
        width = 0.35
        ax4.bar(x_ops - width / 2, py_times, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        ax4.bar(x_ops + width / 2, rust_times, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("State Space Benchmark Timing Profile", fontsize=11, fontweight='bold')
        ax4.set_xticks(x_ops)
        ax4.set_xticklabels(ops, rotation=15, ha='right')
        ax4.set_yscale('log')
        ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax4.grid(True, which='both')

        fig.tight_layout()
        return fig

    def plot_summary(self, ax: plt.Axes):
        # Phase Portrait Summary (x1 vs x2)
        py_tut = self.py_data.get('tutorial', {})
        rust_tut = self.rust_data.get('tutorial', {})

        if py_tut.get('step_x1') and py_tut.get('step_x2'):
            ax.plot(py_tut['step_x1'], py_tut['step_x2'], label='Python 3', color=COLOR_PY, linewidth=2.0)
        if rust_tut.get('step_x1') and rust_tut.get('step_x2'):
            ax.plot(rust_tut['step_x1'], rust_tut['step_x2'], '--', label='Rust', color=COLOR_RUST, linewidth=2.0)

        ax.set_xlabel("State x1")
        ax.set_ylabel("State x2")
        ax.set_title("State Space: Phase Portrait (x1 vs x2)", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True)


class TransferFuncPlotter(BaseModelPlotter):
    """
    Transfer function analysis featuring Nyquist Polar Plots (Frequency Domain)
    with explicit (-1, 0j) critical point markers to detect encirclements and stability degradation.
    """
    def plot_details(self) -> plt.Figure:
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 9))
        fig.suptitle("Transfer Function Frequency Domain & Nyquist Analysis", fontsize=15, fontweight='bold', y=0.98)

        py_cp = self.py_data.get('complex_pair', {})
        rust_cp = self.rust_data.get('complex_pair', {})

        py_re = py_cp.get('h_re', [])
        py_im = py_cp.get('h_im', [])
        rust_re = rust_cp.get('h_re', [])
        rust_im = rust_cp.get('h_im', [])

        # 1. Nyquist Plot in Frequency Domain (Re vs Im) with Critical Point (-1, 0j)
        ax1 = axs[0, 0]
        if py_re and py_im:
            ax1.plot(py_re, py_im, label='Python 3 (w > 0)', color=COLOR_PY, linewidth=2.0)
            ax1.plot(py_re, [-y for y in py_im], ':', color=COLOR_PY, alpha=0.5, label='Python 3 (w < 0)')
        if rust_re and rust_im:
            ax1.plot(rust_re, rust_im, '--', label='Rust (w > 0)', color=COLOR_RUST, linewidth=2.0)
            ax1.plot(rust_re, [-y for y in rust_im], ':', color=COLOR_RUST, alpha=0.5, label='Rust (w < 0)')

        # Mark Critical Point (-1, 0j)
        ax1.scatter([-1.0], [0.0], color='red', marker='*', s=160, zorder=5, label='Critical Point (-1, 0j)')
        ax1.axvline(-1.0, color='red', linestyle=':', alpha=0.5)
        ax1.axhline(0.0, color='gray', linestyle=':', alpha=0.5)

        # Draw Unit Circle Boundary |z| = 1
        theta = np.linspace(0, 2 * np.pi, 200)
        ax1.plot(np.cos(theta), np.sin(theta), color='gray', linestyle='--', alpha=0.3, label='Unit Circle |H|=1')

        ax1.set_xlabel("Real Axis Re{H(jw)}")
        ax1.set_ylabel("Imaginary Axis Im{H(jw)}")
        ax1.set_title("Nyquist Polar Contour & Critical Point", fontsize=11, fontweight='bold')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=7, loc='upper left')
        ax1.grid(True)

        # 2. Bode Magnitude Plot (dB vs freq rad/s)
        ax2 = axs[0, 1]
        py_freqs = py_cp.get('freqs', [])
        py_mag = py_cp.get('mag', [])
        rust_freqs = rust_cp.get('freqs', [])
        rust_mag = rust_cp.get('mag', [])

        if py_freqs and py_mag:
            mag_db_py = 20 * np.log10(np.array(py_mag))
            ax2.semilogx(py_freqs, mag_db_py, label='Python 3', color=COLOR_PY, linewidth=2.0)
        if rust_freqs and rust_mag:
            mag_db_rust = 20 * np.log10(np.array(rust_mag))
            ax2.semilogx(rust_freqs, mag_db_rust, '--', label='Rust', color=COLOR_RUST, linewidth=2.0)

        ax2.set_xlabel("Frequency (rad/s)")
        ax2.set_ylabel("Magnitude (dB)")
        ax2.set_title("Bode Magnitude Response", fontsize=11, fontweight='bold')
        ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax2.grid(True, which='both')

        # 3. Bode Phase Plot (degrees vs freq rad/s)
        ax3 = axs[1, 0]
        py_phase = py_cp.get('phase', [])
        rust_phase = rust_cp.get('phase', [])

        if py_freqs and py_phase:
            phase_deg_py = np.rad2deg(np.array(py_phase))
            ax3.semilogx(py_freqs, phase_deg_py, label='Python 3', color=COLOR_PY, linewidth=2.0)
        if rust_freqs and rust_phase:
            phase_deg_rust = np.rad2deg(np.array(rust_phase))
            ax3.semilogx(rust_freqs, phase_deg_rust, '--', label='Rust', color=COLOR_RUST, linewidth=2.0)

        ax3.set_xlabel("Frequency (rad/s)")
        ax3.set_ylabel("Phase (degrees)")
        ax3.set_title("Bode Phase Response", fontsize=11, fontweight='bold')
        ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax3.grid(True, which='both')

        # 4. Transfer Function Operations Timing Profile
        ax4 = axs[1, 1]
        ops = ['Complex Pair Bode', 'Cluster Bode', 'CCF Conv.', 'Series Conv.']
        py_times = [
            py_cp.get('bode_time_ns', 0.0),
            self.py_data.get('clustered', {}).get('cluster_bode_time_ns', 0.0),
            self.py_data.get('ccf', {}).get('ccf_time_ns', 0.0),
            self.py_data.get('series', {}).get('series_time_ns', 0.0)
        ]
        rust_times = [
            rust_cp.get('bode_time_ns', 0.0),
            self.rust_data.get('clustered', {}).get('cluster_bode_time_ns', 0.0),
            self.rust_data.get('ccf', {}).get('ccf_time_ns', 0.0),
            self.rust_data.get('series', {}).get('series_time_ns', 0.0)
        ]

        x_ops = np.arange(len(ops))
        width = 0.35
        ax4.bar(x_ops - width / 2, py_times, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        ax4.bar(x_ops + width / 2, rust_times, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("Transfer Function Benchmark Timing", fontsize=11, fontweight='bold')
        ax4.set_xticks(x_ops)
        ax4.set_xticklabels(ops, rotation=15, ha='right')
        ax4.set_yscale('log')
        ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax4.grid(True, which='both')

        fig.tight_layout()
        return fig

    def plot_summary(self, ax: plt.Axes):
        # Nyquist Plot Summary with (-1, 0j) Critical Point
        py_re = self.py_data.get('complex_pair', {}).get('h_re', [])
        py_im = self.py_data.get('complex_pair', {}).get('h_im', [])
        rust_re = self.rust_data.get('complex_pair', {}).get('h_re', [])
        rust_im = self.rust_data.get('complex_pair', {}).get('h_im', [])

        if py_re and py_im:
            ax.plot(py_re, py_im, label='Python 3', color=COLOR_PY, linewidth=2.0)
        if rust_re and rust_im:
            ax.plot(rust_re, rust_im, '--', label='Rust', color=COLOR_RUST, linewidth=2.0)

        # Mark Critical Point (-1, 0j)
        ax.scatter([-1.0], [0.0], color='red', marker='*', s=140, zorder=5, label='Critical (-1,0j)')
        ax.axvline(-1.0, color='red', linestyle=':', alpha=0.4)
        ax.axhline(0.0, color='gray', linestyle=':', alpha=0.4)

        ax.set_xlabel("Re{H(jw)}")
        ax.set_ylabel("Im{H(jw)}")
        ax.set_title("Transfer Function: Nyquist Polar Curve", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax.grid(True)


class TensorPlotter(BaseModelPlotter):
    """
    Tensor and Array analysis featuring Geometric SE(3) Lie Manifold Mapping to project
    abstract numerical error into physical 3D end-effector pose spatial drift.
    """
    def plot_details(self) -> plt.Figure:
        fig = plt.figure(figsize=(11.5, 9))
        fig.suptitle("Tensor Operations & SE(3) Geometric Manifold Mapping", fontsize=15, fontweight='bold', y=0.98)

        # 1. Geometric & Lie Manifold 3D Spatial Trajectory (SE(3) Pose Drift)
        ax1 = fig.add_subplot(2, 2, 1, projection='3d')
        py_cut_x = np.array(self.py_data.get('curved', {}).get('cut_x', []))
        py_samples = np.array(self.py_data.get('curved', {}).get('samples', []))
        rust_cut_x = np.array(self.rust_data.get('curved', {}).get('cut_x', []))
        rust_samples = np.array(self.rust_data.get('curved', {}).get('samples', []))

        if py_cut_x.size > 0 and py_samples.size > 0:
            t_py = py_cut_x
            x_py = t_py * np.cos(2 * np.pi * t_py)
            y_py = t_py * np.sin(2 * np.pi * t_py)
            z_py = py_samples
            ax1.plot(x_py, y_py, z_py, label='Python 3 Pose Path', color=COLOR_PY, linewidth=2.0)
            # Add spatial end-effector pose quivers along trajectory
            for i in range(0, len(t_py), 12):
                ax1.quiver(x_py[i], y_py[i], z_py[i], 0.05, 0.05, 0.05, color=COLOR_PY, alpha=0.7)

        if rust_cut_x.size > 0 and rust_samples.size > 0:
            t_rs = rust_cut_x
            x_rs = t_rs * np.cos(2 * np.pi * t_rs)
            y_rs = t_rs * np.sin(2 * np.pi * t_rs)
            z_rs = rust_samples
            ax1.plot(x_rs, y_rs, z_rs, '--', label='Rust Pose Path', color=COLOR_RUST, linewidth=2.0)
            for i in range(0, len(t_rs), 12):
                ax1.quiver(x_rs[i], y_rs[i], z_rs[i], -0.05, 0.05, 0.05, color=COLOR_RUST, alpha=0.7)

        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Y (m)")
        ax1.set_zlabel("Z (m)")
        ax1.set_title("SE(3) End-Effector 3D Trajectory Drift", fontsize=10, fontweight='bold')
        ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=7)

        # 2. 2D Curved Tensor Relative Error Heatmap
        ax2 = fig.add_subplot(2, 2, 2)
        py_tbl = np.array(self.py_data.get('curved', {}).get('table', []))
        rust_tbl = np.array(self.rust_data.get('curved', {}).get('table', []))

        if py_tbl.size > 0 and rust_tbl.size > 0 and py_tbl.shape == rust_tbl.shape:
            err_tbl = np.abs(py_tbl - rust_tbl) / (np.abs(py_tbl) + 1e-15)
            vmin_tbl = max(1e-16, np.min(err_tbl[err_tbl > 0]) if np.any(err_tbl > 0) else 1e-16)
            vmax_tbl = max(1e-10, np.max(err_tbl))
            im2 = ax2.imshow(err_tbl, cmap='plasma', norm=LogNorm(vmin=vmin_tbl, vmax=vmax_tbl), aspect='auto')
            cbar2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label("Relative Error E", fontsize=8)
            ax2.set_title("Curved Tensor 2D Table Relative Error", fontsize=10, fontweight='bold')
            ax2.set_xlabel("Dim 2 Index")
            ax2.set_ylabel("Dim 1 Index")
        else:
            ax2.text(0.5, 0.5, "Table Error Data Unavailable", ha='center', va='center')
            ax2.set_title("Tensor Table Error Heatmap", fontsize=10, fontweight='bold')

        # 3. Q7 Quantization Precision & ReQuantization Error
        ax3 = fig.add_subplot(2, 2, 3)
        py_q7 = self.py_data.get('q7', {})
        rust_q7 = self.rust_data.get('q7', {})

        langs = ['Python 3', 'Rust']
        errs = [py_q7.get('quant_err', 0.0), rust_q7.get('quant_err', 0.0)]
        ax3.bar(langs, errs, color=[COLOR_PY, COLOR_RUST], width=0.4, alpha=0.85)
        ax3.set_ylabel("Quantization Error")
        ax3.set_title("Q7 Quantization Error Precision", fontsize=10, fontweight='bold')
        for i, err in enumerate(errs):
            ax3.annotate(f"{err:.4e}", (langs[i], err), textcoords="offset points",
                         xytext=(0, 5), ha='center', fontsize=9, fontweight='bold')
        ax3.grid(True, axis='y')

        # 4. Tensor Operation Execution Times
        ax4 = fig.add_subplot(2, 2, 4)
        ops = ['Affine Interp', 'Curved Interp', 'Q7 Quant.']
        py_times = [
            self.py_data.get('affine', {}).get('affine_interp_time_ns', 0.0),
            self.py_data.get('curved', {}).get('interp_time_ns', 0.0),
            py_q7.get('q7_time_ns', 0.0)
        ]
        rust_times = [
            self.rust_data.get('affine', {}).get('affine_interp_time_ns', 0.0),
            self.rust_data.get('curved', {}).get('interp_time_ns', 0.0),
            rust_q7.get('q7_time_ns', 0.0)
        ]

        x_ops = np.arange(len(ops))
        width = 0.35
        ax4.bar(x_ops - width / 2, py_times, width, label='Python 3', color=COLOR_PY, alpha=0.9)
        ax4.bar(x_ops + width / 2, rust_times, width, label='Rust', color=COLOR_RUST, alpha=0.9)
        ax4.set_ylabel("Execution Time (ns)")
        ax4.set_title("Tensor Benchmark Timing Profile", fontsize=10, fontweight='bold')
        ax4.set_xticks(x_ops)
        ax4.set_xticklabels(ops, rotation=15, ha='right')
        ax4.set_yscale('log')
        ax4.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8)
        ax4.grid(True, which='both')

        fig.tight_layout()
        return fig

    def plot_summary(self, ax: plt.Axes):
        # 2D Curved Interpolation Spatial Fit Overlay
        py_cut_x = self.py_data.get('curved', {}).get('cut_x', [])
        py_samples = self.py_data.get('curved', {}).get('samples', [])
        rust_cut_x = self.rust_data.get('curved', {}).get('cut_x', [])
        rust_samples = self.rust_data.get('curved', {}).get('samples', [])

        if py_cut_x and py_samples:
            ax.plot(py_cut_x, py_samples, label='Python 3', color=COLOR_PY, linewidth=2.0)
        if rust_cut_x and rust_samples:
            ax.plot(rust_cut_x, rust_samples, '--', label='Rust', color=COLOR_RUST, linewidth=2.0)

        ax.set_xlabel("Manifold Coordinate x")
        ax.set_ylabel("SE(3) Spatial Position Fit")
        ax.set_title("Tensor: SE(3) Curved Manifold Fit", fontsize=12, fontweight='bold')
        ax.legend(frameon=True, facecolor='white', framealpha=0.9)
        ax.grid(True)


# ==========================================
# 3. Main Coordinator
# ==========================================
def load_json(filename: str) -> dict:
    """Helper to safely load a JSON file with fallback search paths."""
    script_dir = os.path.dirname(__file__) if '__file__' in globals() else '.'
    search_paths = [
        filename,
        os.path.join(script_dir, filename),
        os.path.join(script_dir, '..', 'results', filename),
        os.path.join(script_dir, 'results', filename),
        os.path.join('examples', 'numerical-models-validation', 'results', filename),
        os.path.join('results', filename)
    ]
    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading {path}: {e}")
    print(f"Warning: {filename} not found in search paths. Returning empty dict.")
    return {}


def main():
    out_dir = get_output_dir()
    print(f"Output directory resolved to: {out_dir}")

    # 1. Ingest Data
    matrix_data = load_json('matrix.json')
    poly_data = load_json('polynomial.json')
    state_data = load_json('state_space.json')
    tf_data = load_json('transfer_function.json')
    tensor_data = load_json('tensor.json')

    # 2. Instantiate Plotters
    plotters = {
        "matrix": MatrixPlotter(matrix_data.get('python3', {}), matrix_data.get('rust', {})),
        "polynomial": PolynomialPlotter(poly_data.get('python3', {}), poly_data.get('rust', {})),
        "state_space": StateSpacePlotter(state_data.get('python3', {}), state_data.get('rust', {})),
        "transfer_function": TransferFuncPlotter(tf_data.get('python3', {}), tf_data.get('rust', {})),
        "tensor": TensorPlotter(tensor_data.get('python3', {}), tensor_data.get('rust', {}))
    }

    # 3. Generate Details
    print("Generating detailed plots with enhanced visual contrast...")
    for name, plotter in plotters.items():
        fig = plotter.plot_details()
        filename = os.path.join(out_dir, f"{name}_details.png")
        fig.savefig(filename, dpi=300)
        plt.close(fig)
        print(f" Saved {filename}")

    # 4. Generate Overview (2x3 Grid)
    print("Generating overview summary plot...")
    fig_over, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig_over.suptitle("Numerical Benchmark Overview: Python vs Rust", fontsize=16, fontweight='bold', y=0.98)

    # Flatten the 2x3 axes array for easy iteration
    axes_flat = axes.flatten()

    # Dispatch axes to plotters
    for i, (name, plotter) in enumerate(plotters.items()):
        plotter.plot_summary(axes_flat[i])

    # Handle the 6th empty axis (Index 5)
    ax_legend = axes_flat[5]
    ax_legend.axis('off')

    # Draw a unified legend and summary statistics in the empty space
    ax_legend.text(0.5, 0.78, 'Numerical Models Validation', ha='center', va='center', fontsize=14, fontweight='bold', color='#333333')
    ax_legend.text(0.5, 0.65, 'Cross-Language Benchmark (Python 3 vs Rust)', ha='center', va='center', fontsize=11, fontstyle='italic', color='#666666')
    ax_legend.text(0.5, 0.48, 'Includes: Phase Portraits, SE(3) Manifolds,\n2D Relative Error Heatmaps & Nyquist Contours', ha='center', va='center', fontsize=9, color='#555555')

    # Custom Legend elements
    legend_elements = [
        Line2D([0], [0], color=COLOR_PY, lw=3, label='Python 3 (SciPy / NumPy)'),
        Line2D([0], [0], color=COLOR_RUST, lw=3, linestyle='--', label='Rust (control-rs)'),
        Line2D([0], [0], marker='*', color='red', label='Critical Point (-1, 0j)', markersize=10, linestyle='None')
    ]
    ax_legend.legend(handles=legend_elements, loc='center', fontsize=10, frameon=True, facecolor='#f8f9fa', edgecolor='#cccccc')

    fig_over.tight_layout()
    overview_path = os.path.join(out_dir, "overview_summary.png")
    fig_over.savefig(overview_path, dpi=300)
    plt.close(fig_over)
    print(f" Saved {overview_path}")


if __name__ == "__main__":
    main()