import warnings
from pathlib import Path
import numpy as np
from scipy.interpolate import RegularGridInterpolator

warnings.filterwarnings("ignore", category=RuntimeWarning)

from h5_writer import get_results_dir, write_h5


def _q7_raw(val: float) -> int:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def _q7_roundtrip(val: float) -> float:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        q = 127
    elif scaled <= -128.0:
        q = -128
    elif scaled >= 0.0:
        q = int(scaled + 0.5)
    else:
        q = int(scaled - 0.5)
    return float(q) / 128.0


def generate_datasets():
    # 1. 2D Interpolation Manifold
    center = 7.5
    scale = 3.75

    ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    x_grid = (ii - center) / scale
    y_grid = (jj - center) / scale
    grid_table = np.asarray(x_grid**2 - y_grid**2, dtype=np.float32)

    axes = (np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32))
    interp_func = RegularGridInterpolator(
        axes, grid_table, method="linear", bounds_error=False, fill_value=None
    )

    eval_n = 20
    mesh_u = np.linspace(0.0, 15.0, eval_n, dtype=np.float32)
    mesh_v = np.linspace(0.0, 15.0, eval_n, dtype=np.float32)
    uu, vv = np.meshgrid(mesh_u, mesh_v, indexing="ij")
    eval_pts = np.stack([uu.ravel(), vv.ravel()], axis=1)
    interp_mesh = interp_func(eval_pts).flatten()

    # 2. Tensor Contraction
    ii16, jj16 = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    mat_a = np.asarray(np.sin(ii16 * 0.5 + jj16 * 0.3) * 10.0, dtype=np.float32)
    mat_b = np.asarray(np.cos(ii16 * 0.3 - jj16 * 0.4) * 5.0, dtype=np.float32)
    mat_c = (mat_a @ mat_b).flatten()

    # 3. Quantized Boundaries
    float_inputs = [
        -1.5, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0, 0.0078125, 0.125,
        0.5, 0.75, 0.9921875, 1.0, 1.5,
    ]
    act_outputs = [_q7_roundtrip(np.tanh(x)) for x in float_inputs]

    q_raw = [float(_q7_raw(x)) for x in float_inputs]

    # 4. TableActivation tanh sweep reference (exact tanh)
    act_inputs = np.asarray(-3.0 + 0.05 * np.arange(121), dtype=np.float32)
    act_exact = np.tanh(act_inputs.astype(np.float64))

    datasets = {
        "boundaries/q_raw": q_raw,
        "activation/act_outputs": act_exact,
        "manifold/interp_mesh": interp_mesh,
        "contraction/mat_c": mat_c,
        "boundaries/act_outputs": act_outputs,
    }

    tolerances = {
        "manifold/interp_mesh": ("abs", 0.05),
        "contraction/mat_c": ("abs", 2e-4),
        "boundaries/act_outputs": ("abs", 0.02),
        "boundaries/q_raw": ("abs", 0.0),
        "activation/act_outputs": ("abs", 1e-3, {"tflite": 0.05}),
    }

    # ONNX Runtime provides the contraction only; TFLite the activation sweep only.
    missing_ok = {
        "manifold/interp_mesh": ["onnx", "tflite"],
        "contraction/mat_c": ["tflite"],
        "boundaries/act_outputs": ["onnx", "tflite"],
        "boundaries/q_raw": ["onnx", "tflite"],
        "activation/act_outputs": ["onnx"],
    }

    return datasets, tolerances, missing_ok


def main():
    datasets, tolerances, missing_ok = generate_datasets()
    out_file = get_results_dir() / "tensor.scipy.h5"
    write_h5(out_file, datasets, tolerances, missing_ok)


if __name__ == "__main__":
    main()
