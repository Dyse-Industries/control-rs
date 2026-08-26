# Examples

Runnable demonstrations, host-side numerical oracles, hardware subprogram
backends, and Embedded Test Server (ETS) firmware. The crate root
[`README.md`](../README.md) is the project overview; this file is the
operator's guide for everything under `examples/`.

None of these packages are workspace members of `control-rs`. Host numerical
models register as `[[example]]` targets on the root crate. Subprogram,
QEMU, and Teensy packages each declare their own `[workspace]` so their
toolchains and link flags stay out of the library graph.

---

## Directory index

| Path | Kind | What it is | Run from |
|:-----|:-----|:-----------|:---------|
| [`numerical-models/`](numerical-models/) | Root `[[example]]` | `Matrix`, `Polynomial`, `StateSpace`, `Tensor`, `TransferFunction` demos | Repo root |
| [`prototypes/numerical-models/`](prototypes/numerical-models/) | Python oracles | Same mathematics as the Rust demos, used as a host-side check | Repo root |
| [`subprograms/`](subprograms/) | Standalone crates | Architecture backends that implement `control_rs::math::subprograms` | Inside each crate |
| [`qemu/`](qemu/) | Firmware package | Bare-metal ETS runners (Cortex-M7, RISC-V) | `examples/qemu/` or `cargo qemu` |
| [`teensy4/`](teensy4/) | Firmware package | Teensy 4.0 ETS over USB CDC | `examples/teensy4/` or `cargo teensy` |

---

## Prerequisites

**Host numerical models and prototypes** need a Rust toolchain that can build
the workspace (see [`documentation/development-guide.md`](../documentation/development-guide.md))
and, for the Python oracles, Python 3. The oracles use the standard library
only.

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

These binaries live in `examples/numerical-models/` and are wired in the root
`Cargo.toml` as `[[example]]` targets. Run them from the **repository root**.

| Command | Demonstrates |
|:--------|:-------------|
| `cargo run --example matrix_example` | Dense `Matrix`: LU solve, inverse, Kalman covariance update |
| `cargo run --example polynomial_example` | Horner evaluation, calculus, Euclidean division, companion matrix |
| `cargo run --example state_space_example` | Continuous LTI, Taylor ZOH discretization, step trajectory |
| `cargo run --example tensor_example` | 2-D multilinear interpolation, Q7 quantization, ReLU |
| `cargo run --example transfer_function_example` | Butterworth \(H(s)\), Bode samples, series cascade, controllable canonical form |

Each process prints a titled transcript and exits 0. Source for a given model
is `examples/numerical-models/<name>_example.rs`.

---

## 2. Host-side prototypes

Python oracles under `examples/prototypes/numerical-models/` recompute the same
scenarios as the Rust examples. They are the `/cr-prototype` artifacts for
`numerical-models`, not a second implementation of the crate.

```bash
python3 examples/prototypes/numerical-models/matrix_prototype.py
python3 examples/prototypes/numerical-models/polynomial_prototype.py
python3 examples/prototypes/numerical-models/state_space_prototype.py
python3 examples/prototypes/numerical-models/tensor_prototype.py
python3 examples/prototypes/numerical-models/transfer_function_prototype.py
```

Use them to compare printed values against the matching `cargo run --example`
output. They do not take a `--features` flag and do not link `control-rs`.

---

## 3. Subprogram backends

Four standalone crates under [`subprograms/`](subprograms/). Each is a
**copyable reference implementor**: a zero-sized marker type plus trait impls
for one ISA. `src/` of `control-rs` is not modified. Copy the directory that
matches the target; do not add these crates to the root workspace.

| Crate | Marker | Default backend |
|:------|:-------|:----------------|
| [`subprograms/aarch64/`](subprograms/aarch64/) | `NeonBlas` | AArch64 NEON; optional `--features accelerate` |
| [`subprograms/x86_64/`](subprograms/x86_64/) | `Avx2Blas` | AVX2+FMA after CPU detection; optional `--features cblas` |
| [`subprograms/thumbv7em/`](subprograms/thumbv7em/) | `CmsisDspBlas` | CMSIS-DSP ABI, QEMU MPS2-AN500 |
| [`subprograms/riscv32imac/`](subprograms/riscv32imac/) | `NmsisDspBlas` | NMSIS-DSP ABI, QEMU `virt` |

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

## 4. QEMU ETS firmware

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

## 5. Teensy 4.0 ETS firmware

Physical Cortex-M7 board over USB CDC. Build and flash from
`examples/teensy4/`; drive it from the host with `cargo teensy`. Full wiring,
VID/PID, and loader steps: [`teensy4/README.md`](teensy4/README.md).

```bash
cd examples/teensy4 && cargo build --release
cargo teensy                  # from repo root; optional serial-port argument
```
