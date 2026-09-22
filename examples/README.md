# Examples

Runnable demonstrations, host-side numerical oracles, hardware subprogram
backends, and Embedded Test Server (ETS) firmware. The crate root
[`README.md`](../README.md) is the project overview; this file is the
operator's guide for everything under `examples/`.

The standalone subprogram backends, QEMU, and Teensy firmware packages each
declare their own `[workspace]` so their toolchains and link flags stay out of
the library graph. Domain examples (`dc_motor.rs`, `buck_converter.rs`) are built
directly via the root workspace.

---

## Directory index

| Path                                                           | Kind              | What it is                                                           | Run from                                 |
|:---------------------------------------------------------------|:------------------|:---------------------------------------------------------------------|:-----------------------------------------|
| [`dc_motor.rs`](dc_motor.rs)                                   | Example binary    | DC motor state-space modeling, Tustin discretization & simulation    | `cargo run --example dc_motor`           |
| [`buck_converter.rs`](buck_converter.rs)                       | Example binary    | Buck converter small-signal TF, frequency analysis & step response  | `cargo run --example buck_converter`     |
| [`fixed_point_math.rs`](fixed_point_math.rs)                   | Example binary    | Q16.16 fixed-point scalar arithmetic, saturation & IIR filter        | `cargo run --example fixed_point_math`   |
| [`dsp_spectral_analysis.rs`](dsp_spectral_analysis.rs)         | Example binary    | Radix-2 FFT spectral analysis, tone detection & IFFT reconstruction  | `cargo run --example dsp_spectral_analysis` |
| [`subprograms/`](subprograms/)                                 | Standalone crates | Architecture backends that implement `control_rs::math::subprograms` | Inside each crate                        |
| [`qemu/`](qemu/)                                               | Firmware package  | Bare-metal ETS runners (Cortex-M7, RISC-V)                           | `examples/qemu/` or `cargo qemu`         |
| [`teensy4/`](teensy4/)                                         | Firmware package  | Teensy 4.0 ETS over USB CDC                                          | `examples/teensy4/` or `cargo teensy`    |

---

## 1. Domain examples

Pedagogical, standalone demonstrations showing how `control-rs` models, solvers, and utilities operate on physical dynamic systems.

Run directly with Cargo:

```bash
cargo run --example dc_motor
cargo run --example buck_converter
cargo run --example fixed_point_math
cargo run --example dsp_spectral_analysis
```

| Command | Demonstrates |
|:------------------------------------|:------------------------------------------------------------------------------|
| `cargo run --example dc_motor` | Permanent magnet DC motor continuous state-space modeling, controllability matrix $M_c$, Tustin discretization ($T_s = 10\text{ ms}$), Bode frequency evaluation, and $12\text{ V}$ step transient simulation |
| `cargo run --example buck_converter` | Synchronous buck converter small-signal modeling, control-to-output transfer function $G_{vd}(s)$, $LC$ resonance evaluation ($f_0 = 1073\text{ Hz}$), and $100\text{ kHz}$ digital controller step response |
| `cargo run --example fixed_point_math` | Q16.16 fixed-point scalar arithmetic, representation bounds, saturation protection against overflow, and integer-only discrete IIR low-pass filtering |
| `cargo run --example dsp_spectral_analysis` | Forward Radix-2 FFT spectral analysis, complex magnitude spectrum extraction, harmonic tone peak detection, and lossless inverse FFT time-domain signal reconstruction |

> **Note on Verification & Benchmarks**: Algorithmic scaling and latency jitter benchmarks are located under `benches/` (run with `cargo bench`). Cross-language numerical oracle validation against SciPy/Python is located under `../control-rs-verification` (run with `cargo compare`).

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