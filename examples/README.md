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
`.github/workflows/numerical-models.yml`](../.github/workflows/numerical-models.yml)
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

Suite files under `suites/<slug>.json` hold host inputs, validator argv, and
plot argv. Each validator takes the suite path and prints one result JSON
document on stdout. `cargo run --release --bin validate -- suites/` spawns
those validators, writes `results/<slug>/<source>.json`, compares §6.3
bounds, then runs listed plotters with the results directory.

```bash
pip install -r python3/requirements.txt
cargo build --release
cargo run --release --bin validate -- suites/
# or one slug:
cargo run --release --bin validate -- suites/matrix.json
```

The `numerical-models` GitHub Actions workflow runs `validate`. It is not
part of `cargo ci`. Each run uploads the full `results/` tree as the
`numerical-models-results` artifact (JSON plus all PNGs, 30-day retention).
On `main`, PNGs are also published to the rolling
[`plots`](https://github.com/Dyse-Industries/control-rs/releases/tag/plots)
release as `{slug}-{name}.png`. PNG pixels are not a numeric gate. The
gallery below 404s until the first successful `main` run after that
workflow lands.

| Command | Demonstrates |
|:--------|:-------------|
| `python3 python3/matrix.py suites/matrix.json` | Matrix oracle JSON on stdout |
| `cargo run --release --bin matrix -- suites/matrix.json` | Native matrix JSON on stdout (tutorial on stderr) |
| `cargo run --release --bin validate -- suites/` | Spawn all validators and plotters, compare bounds |
| `python3 python3/plot_matrix.py results/matrix/` | Matrix diagnostic plots (listed in the suite file) |

### Diagnostic plots

#### Matrix

![Hilbert inverse relative error](https://github.com/Dyse-Industries/control-rs/releases/download/plots/matrix-hilbert_inverse.png)

![SE(3) rigid GEMM chain](https://github.com/Dyse-Industries/control-rs/releases/download/plots/matrix-se3_chain.png)

#### Polynomial

![Clustered-root Horner](https://github.com/Dyse-Industries/control-rs/releases/download/plots/polynomial-horner.png)

![Companion heatmap](https://github.com/Dyse-Industries/control-rs/releases/download/plots/polynomial-companion.png)

#### State space

![Free-response phase portrait](https://github.com/Dyse-Industries/control-rs/releases/download/plots/state_space-free_response.png)

![Stiff ZOH](https://github.com/Dyse-Industries/control-rs/releases/download/plots/state_space-stiff_zoh.png)

#### Transfer function

![Underdamped Bode](https://github.com/Dyse-Industries/control-rs/releases/download/plots/transfer_function-bode.png)

![Nyquist complex pair](https://github.com/Dyse-Industries/control-rs/releases/download/plots/transfer_function-nyquist_complex_pair.png)

#### Tensor

![Saddle surface](https://github.com/Dyse-Industries/control-rs/releases/download/plots/tensor-curved_surface.png)

![Saddle cut](https://github.com/Dyse-Industries/control-rs/releases/download/plots/tensor-curved_cut.png)

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
