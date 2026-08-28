//! Tensor native artifact and JSON equivalence test.

use control_rs::math::fixed_num::Quantized;
use control_rs::tensor::ArrayTensor;
use serde_json::{Value, json};

type Q7 = Quantized<i8, 7>;

/// Native tensor scenario matching `python/src/tensor.py`.
pub fn native_values() -> Value {
    let grid = ArrayTensor::<f32, 3, 3>::from_raw([
        [0.0, 2.0, 4.0],
        [1.0, 3.0, 5.0],
        [2.0, 4.0, 6.0],
    ]);
    let test_points = [
        [0.0f32, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [0.5, 0.5],
        [1.5, 0.5],
        [0.2, 1.8],
    ];
    let mut samples = [0.0f32; 6];
    for (idx, pt) in test_points.iter().enumerate() {
        samples[idx] = grid.interpolate(pt);
    }

    let float_inputs = [-0.75f32, -0.25, 0.0, 0.25, 0.5, 0.75];
    let mut q_raw = [0i32; 6];
    let mut dequant = [0.0f32; 6];
    let mut relu_raw = [0i32; 6];
    let mut relu_dequant = [0.0f32; 6];
    for (idx, &f_in) in float_inputs.iter().enumerate() {
        let q = Q7::quantize(f64::from(f_in));
        q_raw[idx] = i32::from(q.raw());
        dequant[idx] = q.dequantize() as f32;
        let relu_raw_i8 = q.raw().max(0);
        relu_raw[idx] = i32::from(relu_raw_i8);
        relu_dequant[idx] = Q7::from_raw(relu_raw_i8).dequantize() as f32;
    }

    json!({
        "SAMPLES": samples,
        "Q_RAW": q_raw,
        "DEQUANT": dequant,
        "RELU_RAW": relu_raw,
        "RELU_DEQUANT": relu_dequant,
        "query_x0": test_points.map(|p| p[0]),
    })
}

pub fn native_series(values: &Value) -> Value {
    json!({
        "interp": {
            "x": values["query_x0"],
            "y": values["SAMPLES"],
        }
    })
}

pub fn native_artifact() -> Value {
    let values = native_values();
    json!({
        "slug": "tensor",
        "source": "rust",
        "values": values,
        "series": native_series(&values),
    })
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use serde_json::Value;

    use crate::assert_f32;

    const PYTHON_JSON: &str = "results/tensor/python.json";
    const NATIVE_JSON: &str = "results/tensor/native.json";

    fn load_artifact(rel: &str, hint: &str) -> Value {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(rel);
        let text = std::fs::read_to_string(&path).unwrap_or_else(|_| {
            panic!("missing artifact: {}\nrun: {}", path.display(), hint);
        });
        serde_json::from_str(&text).expect("parse json")
    }

    #[test]
    fn tensor_equiv() {
        let python = load_artifact(PYTHON_JSON, "python3 python/src/tensor.py");
        let native = load_artifact(NATIVE_JSON, "cargo run --example tensor");
        let py = &python["values"];
        let rs = &native["values"];
        for i in 0..6 {
            assert_f32(
                py["SAMPLES"][i].as_f64().unwrap() as f32,
                rs["SAMPLES"][i].as_f64().unwrap() as f32,
                "interpolate",
            );
            assert_eq!(
                py["Q_RAW"][i].as_i64().unwrap() as i32,
                rs["Q_RAW"][i].as_i64().unwrap() as i32,
                "Q7 raw",
            );
            assert_f32(
                py["DEQUANT"][i].as_f64().unwrap() as f32,
                rs["DEQUANT"][i].as_f64().unwrap() as f32,
                "Q7 dequant",
            );
            assert_eq!(
                py["RELU_RAW"][i].as_i64().unwrap() as i32,
                rs["RELU_RAW"][i].as_i64().unwrap() as i32,
                "ReLU raw",
            );
            assert_f32(
                py["RELU_DEQUANT"][i].as_f64().unwrap() as f32,
                rs["RELU_DEQUANT"][i].as_f64().unwrap() as f32,
                "ReLU dequant",
            );
        }
    }
}
