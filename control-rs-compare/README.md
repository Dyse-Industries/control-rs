# `control-rs-compare`

Cross-Compare Harness & HDF5 Verification Comparison Engine for `control-rs`.

## Prerequisites & System Dependencies

### 1. HDF5 Ingestion & Storage

`control-rs-compare` uses typed pure-Rust HDF5 dataset parsing via `hdf5-pure` for cross-platform compatibility and zero C-toolchain dependencies.

### 2. Python Scientific Runtime

Multi-language reference oracles (NumPy, SciPy, JAX, python-flint, harold) and simulation oracles run within the crate-root Python 3.12 virtual environment (`.venv`):

```bash
# Create venv at crate root if not already present
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r examples/numerical-models-validation/python3/requirements.txt
```

## CLI Usage

### `compare` (`cargo compare`)

```bash
# Run all declared variants and immediately perform multi-method comparison
cargo compare

# Specify custom configuration file and output directory
cargo compare --config compare.toml --results-dir results/

# Skip variant execution and evaluate pre-existing .h5 containers
cargo compare --skip-run

# Run variants only without performing comparison
cargo compare --skip-compare

# Run only a specific suite
cargo compare --run numerical_models --compare numerical_models
```
