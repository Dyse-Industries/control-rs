# `control-rs-verification`

Rust emitters and Python reference oracles for cross-validating the
numerical models. Each side writes one HDF5 container per model to
`target/verification/`; [`control-rs-compare`](../control-rs-compare/README.md) compares them.
[Design](../documentation/vv/cross-compare-design.md) · [Workspace](../README.md)

| Model | Rust emitter | Oracle |
|:--|:--|:--|
| Matrix | `src/matrix.rs` | `python3/matrix_oracle.py` |
| Polynomial | `src/polynomial.rs` | `python3/polynomial_oracle.py` |
| State-space | `src/state_space.rs` | `python3/state_space_oracle.py` |
| Transfer function | `src/transfer_function.rs` | `python3/transfer_function_oracle.py` |
| Tensor | `src/tensor.rs` | `python3/tensor_oracle.py` |

Suite declarations: [`compare.toml`](compare.toml).

## Usage

From the workspace root, with the Python 3.12 virtualenv active:

```sh
pip install -r control-rs-verification/python3/requirements.txt
cargo compare
```

## License

MIT OR Apache-2.0.
