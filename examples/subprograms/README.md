# Subprogram backend examples

Copyable reference implementors for `control_rs::math::subprograms`. Each
crate is a downstream user of the library: a zero-sized marker, trait impls
for the scalars and layouts that backend supports, and a short `main` that
checks those methods against `DefaultBlas`.

`control-rs` `src/` ships only `DefaultBlas`. These packages are not workspace
members, are excluded from `cargo ci`, and are not a substitute for
[`examples/qemu/`](../qemu/) or [`examples/teensy4/`](../teensy4/).

Pick **one** directory, copy it into the firmware tree, and point
`control-rs` at the crate you already depend on. Do not add a crate feature
on `control-rs` to select a backend.

---

## Which crate to copy

| Directory | Marker | Run on | Acceleration |
|:----------|:-------|:-------|:-------------|
| [`aarch64/`](aarch64/) | `NeonBlas` | Host `aarch64` | NEON (baseline). Opt-in `accelerate` (macOS vecLib) |
| [`x86_64/`](x86_64/) | `Avx2Blas` | Host `x86_64` | AVX2+FMA after `is_x86_feature_detected!`. Opt-in `cblas` |
| [`thumbv7em/`](thumbv7em/) | `CmsisDspBlas` | QEMU MPS2-AN500 / Cortex-M7 | CMSIS-DSP ABI, `f32`, guarded fast path |
| [`riscv32imac/`](riscv32imac/) | `NmsisDspBlas` | QEMU `virt` / `rv32imac` | NMSIS-DSP ABI, `f32`, guarded fast path |

Each directory has its own README with the marker's trait list and
limitations.

---

## Run the smoke binary

`cd` into the crate. Host packages use the host triple. `thumbv7em` and
`riscv32imac` set `build.target` and a QEMU `runner` in `.cargo/config.toml`,
so `cargo run` is sufficient.

```bash
cd examples/subprograms/aarch64
cargo run
cargo run --features accelerate          # Apple Silicon macOS

cd examples/subprograms/x86_64
cargo run
cargo run --features cblas               # needs -lcblas -lblas on the linker path

cd examples/subprograms/thumbv7em
cargo run                                # qemu-system-arm, clang for c_src/

cd examples/subprograms/riscv32imac
cargo run                                # qemu-system-riscv32
```

Success prints a line ending in `PASSED.` Residuals are asserted; a mismatch
aborts.

On an `aarch64` host, `x86_64`'s `Avx2Blas` detects missing AVX2/FMA and
delegates to `DefaultBlas`. That is a passing fallback, not an AVX2 proof.
Run that crate on `x86_64` hardware (or an `x86_64` target under emulation)
to exercise the intrinsic kernels.

### Prerequisites (embedded crates)

```bash
rustup target add thumbv7em-none-eabihf riscv32imac-unknown-none-elf
```

| Crate | Extra tools |
|:------|:------------|
| `thumbv7em` | `qemu-system-arm`, `clang` with `--target=thumbv7em-none-eabihf` |
| `riscv32imac` | `qemu-system-riscv32`. A RISC-V gcc is optional; without it the crate uses the portable Rust ABI stand-ins in `src/nmsis.rs` |

Do not invoke QEMU with a handwritten `-kernel path/to/target/...` unless you
know `CARGO_TARGET_DIR`. The crate runner already receives the artifact
cargo built.

---

## Attach the marker in firmware

The orphan rule is satisfied by a **local** marker type. After copying a
crate (or its `src/*.rs` modules) into the firmware package:

```rust
use aarch64_subprograms::NeonBlas; // Avx2Blas, CmsisDspBlas, or NmsisDspBlas
use control_rs::math::storage::{RowArrayStorage, Trans};
use control_rs::math::subprograms::level3::Gemm;

fn step(
    a: &RowArrayStorage<f32, 4, 4>,
    b: &RowArrayStorage<f32, 4, 4>,
    c: &mut RowArrayStorage<f32, 4, 4>,
) {
    NeonBlas::gemm(Trans::NoTrans, Trans::NoTrans, 1.0, a, b, 0.0, c);
}
```

Dispatch is the marker's associated function, same shape as `DefaultBlas::gemm`.
Unsupported scalars and layouts fail to compile (`E0277`), except where the
impl explicitly delegates to `DefaultBlas` (strided views, `Trans::Trans` on
DSP `Gemm`, `alpha != 1` / `beta != 0` on CMSIS/NMSIS, missing AVX2).

### After copying the directory

1. Edit `Cargo.toml`: `control-rs = { path = "...", default-features = false }`
   must point at **your** `control-rs` checkout, not `../../..`.
2. Keep `[workspace]` if the crate stays a standalone package, or fold it
   into the firmware workspace and drop `[workspace]`.
3. For `thumbv7em` / `riscv32imac`, keep `build.rs`, `c_src/`, and
   `.cargo/config.toml`, or replace the C objects with a vendor CMSIS-DSP /
   NMSIS-DSP static library and the same `extern "C"` names.
4. `main.rs` is only the equivalence harness. Firmware uses the library
   target (`NeonBlas`, …).

---

## What the harness checks

Every `main` runs the implemented traits twice on the same stack fixtures,
once through the marker and once through `DefaultBlas`, and asserts a
bounded residual. Host crates also hit `Trans::Trans` and a non-unit strided
subview so the fallback branch executes. DSP crates do the same for `Gemm`
(`alpha != 1`, `beta != 0`) and a strided `Gemv`.

Each kernel is then timed against `DefaultBlas` (host: `Instant` ns/call;
Cortex-M: DWT `CYCCNT`; RISC-V: `rdcycle`). The printed ratio is
`DefaultBlas / backend` and is a measurement, not a gate. Use `--release`
when comparing numbers. Debug builds inflate them.

Host fixtures are large enough that the SIMD inner loop dominates the
scalar tail: L1 `N = 1024`, GEMV `128×128`, GEMM `64×64`. DSP crates stay
on-stack: L1 `N = 64`, GEMV/GEMM `32×32`, Cholesky/TRSM `8×8`. Those are
still below typical CBLAS cache-blocking sizes. QEMU is not a speedup
claim, and `riscv32imac` is soft-float.
