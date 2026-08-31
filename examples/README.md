# Examples

Runnable demonstrations, host-side numerical oracles, hardware subprogram
backends, and Embedded Test Server (ETS) firmware. The crate root
[`README.md`](../README.md) is the project overview; this file is the
operator's guide for everything under `examples/`.

None of these packages are workspace members of `control-rs`. The
numerical-model
host crate, subprogram backends, QEMU, and Teensy packages each declare their
own `[workspace]` so their toolchains and link flags stay out of the library
graph.

---

## Directory index

| Path                                                           | Kind              | What it is                                                           | Run from                                 |
|:---------------------------------------------------------------|:------------------|:---------------------------------------------------------------------|:-----------------------------------------|
| [`numerical-models-validation/`](numerical-models-validation/) | Nested host crate | Demo generators + JSON V&V; Python in `python3/`                     | `examples/numerical-models-validation/`  |
| [`subprograms/`](subprograms/)                                 | Standalone crates | Architecture backends that implement `control_rs::math::subprograms` | Inside each crate                        |
| [`qemu/`](qemu/)                                               | Firmware package  | Bare-metal ETS runners (Cortex-M7, RISC-V)                           | `examples/qemu/` or `cargo qemu`         |
| [`teensy4/`](teensy4/)                                         | Firmware package  | Teensy 4.0 ETS over USB CDC                                          | `examples/teensy4/` or `cargo teensy`    |

---

## Prerequisites

**Host numerical models** need a Rust toolchain that can build the workspace
(see [
`documentation/development-guide.md`](../documentation/development-guide.md))
and Python ≥ 3.10 with NumPy/SciPy/matplotlib. The dedicated
[
`.github/workflows/numerical-models.yml`](../.github/workflows/numerical-models.yml)
workflow installs
[
`numerical-models-validation/python3/requirements.txt`](numerical-models-validation/python3/requirements.txt)
and runs V&V. Locally:

```bash
python3.12 -m venv examples/numerical-models-validation/.venv
source examples/numerical-models-validation/.venv/bin/activate
pip install -r examples/numerical-models-validation/python3/requirements.txt
```

**QEMU ETS and the two `no_std` subprogram crates** additionally need:

```bash
rustup target add thumbv7em-none-eabihf thumbv7em-none-eabi \
    riscv32imac-unknown-none-elf riscv64gc-unknown-none-elf
```

Install `qemu-system-arm` and `qemu-system-riscv32` (and `qemu-system-riscv64`
for the 64-bit QEMU ETS binary). The Cortex-M subprogram crate compiles its C
sources with `clang --target=thumbv7em-none-eabihf`.

**Teensy 4.0** needs `thumbv7em-none-eabihf`, `rust-objcopy`, and
`teensy_loader_cli`. Details are in [`teensy4/README.md`](teensy4/README.md).

Cargo aliases such as `cargo qemu` and `cargo teensy` are defined in
[`.cargo/config.toml`](../.cargo/config.toml) and launch the host TUI against
those firmware packages. They are not substitutes for `cargo run` inside a
subprogram crate.

---

## 1. Numerical models

These binaries live in the nested crate [
`numerical-models-validation/`](numerical-models-validation/)
(not root workspace targets). Run them from **`examples/numerical-models-validation/`**.

Each model has a standalone Rust validator binary and Python companion oracle script. Running `cargo run` (or `cargo run --bin validate`) executes all model validations in-process, measures tight nanosecond timings, cross-references Rust and Python outputs via built-in `cross_validate()`, and writes combined payload JSON files under `results/<model>.json`.

```bash
pip install -r python3/requirements.txt
cargo run
# or run a single model binary:
cargo run --bin matrix
cargo run --bin polynomial
cargo run --bin state_space
cargo run --bin transfer_function
cargo run --bin tensor
```

| Command | Demonstrates |
|:------------------------------------|:------------------------------------------------------------------------------|
| `cargo run` | Execute full in-process validation suite for all 5 models |
| `cargo run --bin matrix` | Execute standalone Matrix numerical validator and cross-check Python oracle |
| `cargo run --bin polynomial` | Execute standalone Polynomial numerical validator and cross-check Python oracle |
| `cargo run --bin state_space` | Execute standalone State-Space numerical validator and cross-check Python oracle |
| `cargo run --bin transfer_function` | Execute standalone Transfer Function numerical validator and cross-check Python oracle |
| `cargo run --bin tensor` | Execute standalone Tensor numerical validator and cross-check Python oracle |

### Diagnostic plots

The Python plotting suite (`python3/plot_models.py`) generates high-resolution 4-quadrant benchmark figures under `results/*_details.png` and a multi-panel overview summary under `results/overview_summary.png`:

- **Overview Dashboard**: `results/overview_summary.png`
- **Matrix Benchmarks**: `results/matrix_details.png` (EKF covariance relative error heatmap, $O(N^3)$ inversion scaling, Hilbert solve latency jitter, decomposition speedups)
- **Polynomial Benchmarks**: `results/polynomial_details.png` (Horner vs naive evaluation scaling, Newton-Raphson convergence, Wilkinson polynomial residuals with 256-bit Flint ground truth, complex root perturbation sensitivity)
- **State-Space Benchmarks**: `results/state_space_details.png` (Inverted pendulum phase portrait, ZOH scaling, step computation jitter, controllability/observability construction scaling)
- **Transfer Function Benchmarks**: `results/transfer_function_details.png` (Bode magnitude & phase frequency warping up to Nyquist, Nyquist polar trajectory with gain/phase margins, Butterworth direct form vs biquad SOS topology stability)
- **Tensor Benchmarks**: `results/tensor_details.png` (3D saddle interpolation manifold, tensor contraction relative error heatmap, quantized Tanh activation vs TFLite vs SciPy, contraction scaling)

---

## 2. Subprogram backends

Four standalone crates under [`subprograms/`](subprograms/). Each is a
**copyable reference implementor**: a zero-sized marker type plus trait impls
for one ISA. `src/` of `control-rs` is not modified. Copy the directory that
matches the target; do not add these crates to the root workspace.

| Crate                                                  | Marker         | Default backend                                           |
|:-------------------------------------------------------|:---------------|:----------------------------------------------------------|
| [`subprograms/aarch64/`](subprograms/aarch64/)         | `NeonBlas`     | AArch64 NEON; optional `--features accelerate`            |
| [`subprograms/x86_64/`](subprograms/x86_64/)           | `Avx2Blas`     | AVX2+FMA after CPU detection; optional `--features cblas` |
| [`subprograms/thumbv7em/`](subprograms/thumbv7em/)     | `CmsisDspBlas` | CMSIS-DSP ABI, QEMU MPS2-AN500                            |
| [`subprograms/riscv32imac/`](subprograms/riscv32imac/) | `NmsisDspBlas` | NMSIS-DSP ABI, QEMU `virt`                                |

**Always `cd` into the crate.** Each package has its own `[workspace]` and,
for the `no_std` crates, a `.cargo/config.toml` that sets the target triple
and QEMU runner. From the crate directory:

```bash
cd examples/subprograms/aarch64 && cargo run
cd examples/subprograms/aarch64 && cargo run --features accelerate   # macOS
cd examples/subprograms/x86_64 && cargo run
cd examples/subprograms/thumbv7em && cargo run
cd examples/subprograms/riscv32imac && cargo run
```

A passing run ends with `... PASSED.` How to attach the marker in firmware,
which traits each backend implements, and what to edit after copying are in
[`subprograms/README.md`](subprograms/README.md) and the README inside each
crate.

Do not point QEMU at a hardcoded `target/...` path. `CARGO_TARGET_DIR` and
`--release` move the artifact; the crate runner already passes the kernel to
QEMU.

---

## 3. QEMU ETS firmware

[`qemu/`](qemu/) is validation firmware for the Embedded Test Server, not a
BLAS backend. It does not grow CMSIS/NMSIS link steps.

Interactive (workspace root):

```bash
cargo qemu
```

Headless, from the firmware package (aliases live in
`examples/qemu/.cargo/config.toml`):

```bash
cd examples/qemu
cargo arm-hf      # thumbv7em-none-eabihf, MPS2-AN500
cargo arm-sf      # thumbv7em-none-eabi
cargo riscv32     # riscv32imac-unknown-none-elf, virt
cargo riscv64     # riscv64gc-unknown-none-elf, virt
```

See [`qemu/README.md`](qemu/README.md).

---

## 4. Teensy 4.0 ETS firmware

Physical Cortex-M7 board over USB CDC. Build and flash from
`examples/teensy4/`; drive it from the host with `cargo teensy`. Full wiring,
VID/PID, and loader steps: [`teensy4/README.md`](teensy4/README.md).

```bash
cd examples/teensy4 && cargo build --release
cargo teensy                  # from repo root; optional serial-port argument
```