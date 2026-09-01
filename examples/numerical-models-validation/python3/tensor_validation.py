#!/usr/bin/env python3
"""
python3/tensor_validation.py

Executes NumPy/SciPy/TFLite/ONNX Runtime equivalents for tensor and fixed-point numerical models.
Outputs JSON results to stdout for cross-language validation with Rust.
"""

from __future__ import annotations

import json
import os
import time

# Suppress TensorFlow logging warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from scipy.interpolate import RegularGridInterpolator
import tensorflow as tf
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto

MESH_N = 40


def _q7_raw(val: float) -> int:
    scaled = float(val) * 128.0
    if scaled >= 127.0:
        return 127
    if scaled <= -128.0:
        return -128
    if scaled >= 0.0:
        return int(scaled + 0.5)
    return int(scaled - 0.5)


def benchmark_interpolation_manifold() -> dict:
    center = 7.5
    scale = 3.75

    ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    x_grid = (ii - center) / scale
    y_grid = (jj - center) / scale
    grid_table = np.asarray(x_grid ** 2 - y_grid ** 2, dtype=np.float32)

    axes = (np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32))
    interp_func = RegularGridInterpolator(
        axes, grid_table, method="linear", bounds_error=False, fill_value=None
    )

    mesh_u = np.linspace(0.0, 15.0, MESH_N, dtype=np.float32)
    mesh_v = np.linspace(0.0, 15.0, MESH_N, dtype=np.float32)

    uu, vv = np.meshgrid(mesh_u, mesh_v, indexing="ij")
    eval_pts = np.stack([uu.ravel(), vv.ravel()], axis=1)

    interp_vals = interp_func(eval_pts).reshape(MESH_N, MESH_N).astype(np.float32)

    xx = (uu - center) / scale
    yy = (vv - center) / scale
    exact_vals = np.asarray(xx ** 2 - yy ** 2, dtype=np.float32)

    return {
        "grid_table": grid_table.tolist(),
        "mesh_u": mesh_u.tolist(),
        "mesh_v": mesh_v.tolist(),
        "interp_mesh": interp_vals.tolist(),
        "exact_mesh": exact_vals.tolist(),
    }


def build_onnx_matmul() -> bytes:
    # Build a simple float MatMul ONNX model
    a = helper.make_tensor_value_info('A', TensorProto.FLOAT, [16, 16])
    b = helper.make_tensor_value_info('B', TensorProto.FLOAT, [16, 16])
    c = helper.make_tensor_value_info('C', TensorProto.FLOAT, [16, 16])

    node = helper.make_node('MatMul', inputs=['A', 'B'], outputs=['C'])
    graph = helper.make_graph([node], 'matmul_graph', [a, b], [c])
    model = helper.make_model(graph, producer_name='onnx-test', opset_imports=[helper.make_operatorsetid('', 21)])
    return model.SerializeToString()


def benchmark_tensor_contraction() -> dict:
    ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    i_f = ii.astype(np.float32)
    j_f = jj.astype(np.float32)

    mat_a = np.asarray(np.sin(i_f * 0.5 + j_f * 0.3) * 10.0, dtype=np.float32)
    mat_b = np.asarray(np.cos(i_f * 0.3 - j_f * 0.4) * 5.0, dtype=np.float32)

    # Run matrix multiplication via ONNX Runtime
    model_bytes = build_onnx_matmul()
    session = ort.InferenceSession(model_bytes)
    mat_c = session.run(['C'], {'A': mat_a, 'B': mat_b})[0]

    return {
        "mat_a": mat_a.tolist(),
        "mat_b": mat_b.tolist(),
        "mat_c": mat_c.tolist(),
    }


def _build_quant_tflite_tanh() -> bytes:
    class QuantizedActivationModel(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1], dtype=tf.float32)])
        def __call__(self, x):
            return tf.math.tanh(x)

    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [QuantizedActivationModel().__call__.get_concrete_function()],
        QuantizedActivationModel()
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    def representative_data_gen():
        for _ in range(100):
            yield [np.random.uniform(-3.0, 3.0, size=(1, 1)).astype(np.float32)]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def benchmark_quantized_boundaries() -> dict:
    # 1. Inputs for basic quantization validation
    float_inputs = np.array(
        [
            -1.5, -1.0, -0.75, -0.5, -0.125, -0.0078125, 0.0,
            0.0078125, 0.125, 0.5, 0.75, 0.9921875, 1.0, 1.5,
        ],
        dtype=np.float64,
    )

    q_raw = np.array([_q7_raw(float(val)) for val in float_inputs], dtype=np.int32)
    dequant = q_raw.astype(np.float64) / 128.0
    quant_err = np.abs(float_inputs - dequant)

    # 2. Sweep for TableActivation Tanh validation (61 breakpoints, 121 evaluation points)
    act_inputs = np.linspace(-3.0, 3.0, 121, dtype=np.float32)
    act_exact = np.tanh(act_inputs)

    # Build and run quantized TFLite Tanh model
    tflite_model_bytes = _build_quant_tflite_tanh()
    interpreter = tf.lite.Interpreter(model_content=tflite_model_bytes)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    in_scale, in_zp = input_details['quantization']
    out_scale, out_zp = output_details['quantization']

    tflite_outputs_int8 = []
    for val in act_inputs:
        q_in = np.clip(np.round(val / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details['index'], np.array([[q_in]], dtype=np.int8))
        interpreter.invoke()
        q_out = interpreter.get_tensor(output_details['index'])[0][0]
        tflite_outputs_int8.append(int(q_out))

    tflite_dequant = (
            (np.array(tflite_outputs_int8, dtype=np.float32) - out_zp) * out_scale
    ).tolist()

    return {
        "float_inputs": float_inputs.tolist(),
        "q_raw": q_raw.tolist(),
        "dequant": dequant.tolist(),
        "quant_err": quant_err.tolist(),
        "act_inputs": act_inputs.tolist(),
        "act_exact": act_exact.tolist(),
        "tflite_outputs_int8": tflite_outputs_int8,
        "tflite_dequant": tflite_dequant,
    }


def _benchmark_contract_n_py(n: int, iters: int) -> float:
    mat_a = np.random.randn(n, n).astype(np.float32)
    mat_b = np.random.randn(n, n).astype(np.float32)
    t0 = time.perf_counter_ns()
    for _ in range(iters):
        np.matmul(mat_a, mat_b)
    return float(time.perf_counter_ns() - t0) / iters


def benchmark_timing_profile() -> dict:
    # 1. SciPy Interpolation Timing
    axes = (np.arange(16, dtype=np.float32), np.arange(16, dtype=np.float32))
    grid_table = np.ones((16, 16), dtype=np.float32)
    interp_func = RegularGridInterpolator(axes, grid_table, method="linear")
    eval_pt = np.array([[7.3, 8.1]], dtype=np.float32)

    interp_iters = 100_000
    t0 = time.perf_counter_ns()
    for _ in range(interp_iters):
        interp_func(eval_pt)
    interp_time_ns = float(time.perf_counter_ns() - t0) / interp_iters

    # 2. Python Q7 Quantization Timing
    quant_iters = 100_000
    t0 = time.perf_counter_ns()
    for _ in range(quant_iters):
        _q7_raw(0.7853981633974483)
    quant_time_ns = float(time.perf_counter_ns() - t0) / quant_iters

    # 3. Multi-size NumPy Contraction Scaling Timing
    sizes = [4, 8, 16, 32, 64]
    iters_list = [100_000, 50_000, 20_000, 5_000, 1_000]
    contract_times_ns = [
        _benchmark_contract_n_py(n, it) for n, it in zip(sizes, iters_list)
    ]

    return {
        "interp_time_ns": interp_time_ns,
        "quant_time_ns": quant_time_ns,
        "sizes": sizes,
        "contract_times_ns": contract_times_ns,
    }


def run_tensor_oracle() -> dict:
    manifold = benchmark_interpolation_manifold()
    contraction = benchmark_tensor_contraction()
    boundaries = benchmark_quantized_boundaries()
    timing = benchmark_timing_profile()

    return {
        "manifold": manifold,
        "contraction": contraction,
        "boundaries": boundaries,
        "timing": timing,
    }


if __name__ == "__main__":
    results = run_tensor_oracle()
    print(json.dumps({"scipy": results}))
