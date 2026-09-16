# Examples

Pedagogical demos of the public API. Each one is a cargo example of the
`control-rs` package, builds under the workspace lints, and needs no Python,
HDF5, or ngspice:

```bash
cargo run --example matrix
```

Oracle cross-validation lives in [`control-rs-validation/`](../control-rs-validation/)
and latency lives in [`benches/`](../benches/). Neither is demonstrated here:
an example shows what the API does on inputs where the arithmetic is boring.

---

## Directory index

| Path | Kind | What it is | Run from |
|:-----|:-----|:-----------|:---------|
| [`matrix.rs`](matrix.rs) | Cargo example | `Owned` construction, identity, LU, `solve_mut` | repository root |
| [`polynomial.rs`](polynomial.rs) | Cargo example | `ArrayPolynomial` evaluate and `mul_poly` | repository root |
| [`state_space.rs`](state_space.rs) | Cargo example | `ArrayStateSpace` continuous → ZOH → `step` | repository root |
| [`transfer_function.rs`](transfer_function.rs) | Cargo example | `bode_point` and controllable canonical form | repository root |
| [`tensor.rs`](tensor.rs) | Cargo example | `ArrayTensor` interpolate and Q7 quantization | repository root |
| [`dc_motor.rs`](dc_motor.rs) | Cargo example | DC-motor plant, Routh, lead, and a few PID steps | repository root |
| [`buck_converter.rs`](buck_converter.rs) | Cargo example | Buck plant, Routh, margins, and a lead compensator | repository root |
| [`subprograms/`](subprograms/) | Standalone crates | Architecture backends for `control_rs::math::subprograms` | Inside each crate |
| [`qemu/`](qemu/) | Firmware package | Bare-metal ETS runners (Cortex-M7, RISC-V) | `examples/qemu/` or `cargo qemu` |
| [`teensy4/`](teensy4/) | Firmware package | Teensy 4.0 ETS over USB CDC | `examples/teensy4/` or `cargo teensy` |

`subprograms/`, `qemu/` and `teensy4/` are nested crates with their own
`[workspace]`, not cargo examples: they need cross-compilation targets, C
sources, or QEMU. Cargo's example auto-discovery only claims `examples/*.rs`
and `examples/*/main.rs`, so they coexist with the files above.

---

## Cargo examples

```bash
cargo run --example matrix
cargo run --example polynomial
cargo run --example state_space
cargo run --example transfer_function
cargo run --example tensor
cargo run --example dc_motor
cargo run --example buck_converter
```

| Example | Demonstrates |
|:--------|:-------------|
| `matrix` | LU solve of a 2x2 system, identity construction |
| `polynomial` | Horner evaluation and a convolution product |
| `state_space` | Continuous plant, ZOH discretization, one discrete step |
| `transfer_function` | Bode point and controllable canonical realization |
| `tensor` | 2x2 grid interpolation and Q7 quantization |
| `dc_motor` | Armature position plant, Routh (the free integrator at `s = 0` is expected), a lead network targeting 50 rad/s and 50°, then discrete `Pid::step` calls at `T_s = 500 µs` |
| `buck_converter` | Averaged CCM duty-to-output plant, Routh on the LC denominator, margins, a lead network targeting `3 ω_n` and 45°, series loop `L(s) = C(s) G_vd(s)`, and a Tustin `DirectForm2T` |

True-oracle comparison, SciPy/ngspice, and `results/*.h5` for these plants are
in `control-rs-validation/`, not here.

---

## Subprogram backends

Copyable reference implementors for `control_rs::math::subprograms`. Each
crate is a zero-sized marker plus trait impls for one ISA. Copy the
directory that matches the target; do not add these crates to the root
workspace. Details: [`subprograms/README.md`](subprograms/README.md).

```bash
cd examples/subprograms/aarch64 && cargo run
cd examples/subprograms/aarch64 && cargo run --features accelerate   # macOS
cd examples/subprograms/x86_64 && cargo run
cd examples/subprograms/thumbv7em && cargo run
cd examples/subprograms/riscv32imac && cargo run
```

`thumbv7em` and `riscv32imac` need the matching `rustup target`, `clang` for
their C sources, and QEMU (`qemu-system-arm`, `qemu-system-riscv32`).

---

## QEMU ETS firmware

[`qemu/`](qemu/) is Embedded Test Server firmware, not a BLAS backend. From
the repository root: `cargo qemu`. Headless, from the firmware package:

```bash
cd examples/qemu
cargo arm-hf      # thumbv7em-none-eabihf, MPS2-AN500
cargo arm-sf      # thumbv7em-none-eabi
cargo riscv32     # riscv32imac-unknown-none-elf, virt
cargo riscv64     # riscv64gc-unknown-none-elf, virt
```

Targets: `rustup target add thumbv7em-none-eabihf thumbv7em-none-eabi \
riscv32imac-unknown-none-elf riscv64gc-unknown-none-elf`. See
[`qemu/README.md`](qemu/README.md).

---

## Teensy 4.0 ETS firmware

Physical Cortex-M7 board over USB CDC. Build from `examples/teensy4/`; drive
it from the host with `cargo teensy`. Wiring, VID/PID, and loader:
[`teensy4/README.md`](teensy4/README.md).

```bash
cd examples/teensy4 && cargo build --release
cargo teensy                  # from repo root; optional serial-port argument
```
