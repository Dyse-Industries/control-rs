# `control-rs-compare`

HDF5 cross-comparison engine: runs the variants declared in [`compare.toml`](../compare.toml)
and compares their outputs against a reference oracle.
[Design](../documentation/vv/cross-compare-design.md) · [Workspace](../README.md)

## Prerequisites

No system HDF5 library is required (`hdf5-pure`). Oracles run in the Python
3.12 virtualenv at the workspace root:

```bash
# Create the venv at the workspace root if not already present
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r control-rs-verification/python3/requirements.txt
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
cargo compare --run matrix --compare matrix
```
