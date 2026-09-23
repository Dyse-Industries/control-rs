#!/usr/bin/env python3
"""Tensor reference oracle generating target/verification/tensor.tflite.h5 via a TFLite int8 tanh model."""

import numpy as np
import tensorflow as tf

from h5_writer import get_results_dir, write_h5


def _quantized_tanh_model() -> bytes:
    class Tanh(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec(shape=[1, 1], dtype=tf.float32)])
        def __call__(self, x):
            return tf.math.tanh(x)

    module = Tanh()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.__call__.get_concrete_function()], module
    )
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(0)

    def representative_data():
        for _ in range(100):
            yield [rng.uniform(-3.0, 3.0, size=(1, 1)).astype(np.float32)]

    converter.representative_dataset = representative_data
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter.convert()


def generate_datasets():
    interpreter = tf.lite.Interpreter(model_content=_quantized_tanh_model())
    interpreter.allocate_tensors()
    inp = interpreter.get_input_details()[0]
    out = interpreter.get_output_details()[0]
    in_scale, in_zp = inp["quantization"]
    out_scale, out_zp = out["quantization"]

    act_inputs = np.asarray(-3.0 + 0.05 * np.arange(121), dtype=np.float32)
    outputs = []
    for x in act_inputs:
        q_in = np.clip(np.round(x / in_scale) + in_zp, -128, 127).astype(np.int8)
        interpreter.set_tensor(inp["index"], np.array([[q_in]], dtype=np.int8))
        interpreter.invoke()
        q_out = int(interpreter.get_tensor(out["index"])[0][0])
        outputs.append((q_out - out_zp) * out_scale)

    return {"activation/act_outputs": outputs}


def main():
    write_h5(get_results_dir() / "tensor.tflite.h5", generate_datasets())


if __name__ == "__main__":
    main()
