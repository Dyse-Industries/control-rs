#!/usr/bin/env python3
"""Tensor reference oracle generating target/verification/tensor.onnx.h5 via ONNX Runtime."""

import numpy as np
import onnxruntime as ort
from onnx import TensorProto, helper

from h5_writer import get_results_dir, write_h5


def _matmul_model() -> bytes:
    spec = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [16, 16]) for n in "ABC"]
    node = helper.make_node("MatMul", inputs=["A", "B"], outputs=["C"])
    graph = helper.make_graph([node], "matmul", spec[:2], spec[2:])
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 21)])
    model.ir_version = 10  # accepted by current ONNX Runtime releases
    return model.SerializeToString()


def generate_datasets():
    ii, jj = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    mat_a = np.asarray(np.sin(ii * 0.5 + jj * 0.3) * 10.0, dtype=np.float32)
    mat_b = np.asarray(np.cos(ii * 0.3 - jj * 0.4) * 5.0, dtype=np.float32)
    session = ort.InferenceSession(_matmul_model())
    mat_c = session.run(["C"], {"A": mat_a, "B": mat_b})[0]
    return {"contraction/mat_c": mat_c.flatten()}


def main():
    write_h5(get_results_dir() / "tensor.onnx.h5", generate_datasets())


if __name__ == "__main__":
    main()
