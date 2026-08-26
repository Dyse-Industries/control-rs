# Cortex-M7 subprogram backend (`CmsisDspBlas`)

`no_std` crate that implements `control_rs::math::subprograms` for
`thumbv7em-none-eabihf` using the CMSIS-DSP C ABI. Default runner is QEMU
`mps2-an500` (Cortex-M7). A Teensy 4 rebuild is a runner and linker-script
change in a **copy** of this crate, not an edit to `examples/teensy4/`.

This directory is a copyable reference implementor. It is not part of the
`control-rs` workspace. How the four backends fit together:
[`../README.md`](../README.md).

## Prerequisites

```bash
rustup target add thumbv7em-none-eabihf
```

`qemu-system-arm` on `PATH`. `build.rs` compiles `c_src/` with
`clang --target=thumbv7em-none-eabihf` (`-mcpu=cortex-m7 -mfloat-abi=hard`).

The C sources are a small Apache-style ABI stand-in (`arm_scale_f32`,
`arm_mat_mult_f32`, …), not a full CMSIS-DSP release. Replace `c_src/` and
the `clang` step with a vendor static library if the firmware already has
one; keep the `extern "C"` names in `src/cmsis.rs`.

## Run

```bash
cd examples/subprograms/thumbv7em
cargo run
```

`.cargo/config.toml` sets `build.target` and the QEMU runner. Do not pass a
handwritten `-kernel` path unless `CARGO_TARGET_DIR` is known. A passing run
prints `All Cortex-M CMSIS-DSP subprogram checks PASSED.`

## Marker

| Item | Value |
|:-----|:------|
| Type | `CmsisDspBlas` |
| Traits | `Scal`, `Dotu`, `Dotc`, `Gemv`, `Gemm`, `Potrf`, `Trsm` |
| Scalars | `f32` (`Dotc`: `Complex<f32>`). `q31`/`q15` are out of scope |
| Fast path | Contiguous row-major, `Trans::NoTrans`, `alpha == 1`, `beta == 0` |
| `Trsm` | `Side::Left`, `UpLo::Upper`, `Diag::NonUnit`, `alpha == 1` only |
| Fallback | Anything outside those predicates calls `DefaultBlas` |

CMSIS-DSP is not BLAS: no `alpha`/`beta`/`trans`/`lda`. The predicate is the
point of the example.

```rust
use control_rs::math::storage::Trans;
use control_rs::math::subprograms::level3::Gemm;
use thumbv7em_subprograms::CmsisDspBlas;

CmsisDspBlas::gemm(Trans::NoTrans, Trans::NoTrans, 1.0, &a, &b, 0.0, &mut c);
```

## After copying

1. Point `control-rs` in `Cargo.toml` at your checkout (`default-features = false`).
2. Keep `build.rs`, `c_src/`, `memory.x` generation, and `.cargo/config.toml`,
   or substitute the board's linker script and runner (Teensy: USB, not QEMU).
3. Firmware uses the lib target. `src/main.rs` is the equivalence harness
   (semihosting `hprintln`).
