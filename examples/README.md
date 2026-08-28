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

| Path                                     | Kind              | What it is                                                           | Run from                              |
|:-----------------------------------------|:------------------|:---------------------------------------------------------------------|:--------------------------------------|
| [`numerical-models/`](numerical-models/) | Nested host crate | Demo generators + JSON V&V; Python in `python3/`                     | `examples/numerical-models/`          |
| [`subprograms/`](subprograms/)           | Standalone crates | Architecture backends that implement `control_rs::math::subprograms` | Inside each crate                     |
| [`qemu/`](qemu/)                         | Firmware package  | Bare-metal ETS runners (Cortex-M7, RISC-V)                           | `examples/qemu/` or `cargo qemu`      |
| [`teensy4/`](teensy4/)                   | Firmware package  | Teensy 4.0 ETS over USB CDC                                          | `examples/teensy4/` or `cargo teensy` |

---

## Prerequisites

**Host numerical models** need a Rust toolchain that can build the workspace
(see [
`documentation/development-guide.md`](../documentation/development-guide.md))
and Python ≥ 3.10 with NumPy/SciPy/matplotlib. The dedicated
[
`.github/workflows/numerical-models-vv.yml`](../.github/workflows/numerical-models.yml)
workflow installs
[
`numerical-models/python3/requirements.txt`](numerical-models/python3/requirements.txt)
and runs V&V. Locally:

```bash
python3.12 -m venv examples/numerical-models/.venv
source examples/numerical-models/.venv/bin/activate
pip install -r examples/numerical-models/python3/requirements.txt
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

These binaries live in the nested crate [`numerical-models/`](numerical-models/)
(not root workspace targets). Run them from **`examples/numerical-models/`**.

Python and Rust **write** JSON under `results/<slug>/`. Tests and `report.py`
**read** those files only. Each artifact has `values` (tutorial plus compact
stress payloads), `series`, `metrics` (residual / relative / $\kappa$), and
`timings` (kernel-only min nanoseconds). `cargo test` gates on residual and
relative bounds, not wall-clock. `report.py` writes slug-specific diagnostic
plots (Hilbert relative-error heatmap and SE(3) 3D chain, Horner overlay,
phase portrait, Nyquist contours, tensor surface) and Python vs Rust kernel
times.

```bash
pip install -r python3/requirements.txt

python3 python3/matrix.py
python3 python3/polynomial.py
python3 python3/state_space.py
python3 python3/transfer_function.py
python3 python3/tensor.py

cargo run --release                  # all five native generators (kernel timings)
# or: cargo run --release --bin matrix

cargo test
python3 python3/report.py          # optional plots from existing JSON
python3 python3/report.py --force  # regenerate all JSON, then plot
```

The `numerical-models` GitHub Actions workflow runs the generators, then
`cargo test`. It is not part of `cargo ci`.

| Command | Demonstrates |
|:--------|:-------------|
| `python3 python3/matrix.py` | Writes `results/matrix/python.json` |
| `cargo run --release` | All five native JSON files |
| `cargo run --release --bin matrix` | Transcript + `results/matrix/native.json` |
| `cargo test` | Compares the JSON pair |
| `python3 python3/report.py` | Diagnostic plots (heatmap, phase portrait, Nyquist, 3D) and kernel time bars |
| `python3 python3/report.py --force` | Regenerates all Python and Rust JSON, then plots |

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
