# AArch64 subprogram backend (`NeonBlas`)

Standalone crate that implements `control_rs::math::subprograms` with ARM NEON
intrinsics. Optional Apple Accelerate (vecLib CBLAS) behind `--features accelerate`.

This directory is a copyable reference implementor. It is not part of the
`control-rs` workspace. How the four backends fit together:
[`../README.md`](../README.md).

## Run

Requires an `aarch64` host (Apple Silicon or `aarch64-unknown-linux-gnu`).

```bash
cd examples/subprograms/aarch64
cargo run
cargo run --features accelerate    # macOS only; links `-framework Accelerate`
```

A passing run prints `All aarch64 subprogram backend equivalence checks PASSED.`

On `x86_64`, the NEON kernels are compiled out and every call delegates to
`DefaultBlas`. That does not test NEON.

## Marker

| Item | Value |
|:-----|:------|
| Type | `NeonBlas` |
| Optional | `AccelerateBlas` (`--features accelerate`, `target_vendor = "apple"`) |
| Traits | `Axpy`, `Scal`, `Dotu`, `Nrm2`, `Gemv`, `Gemm` |
| Scalars | `f32`, `f64` |
| Fast path | Contiguous vectors / row-major matrices; remainder tail is scalar |
| Fallback | Non-unit stride, `Trans::Trans` `Gemv`/`Gemm` as implemented, or non-`aarch64` |

```rust
use aarch64_subprograms::NeonBlas;
use control_rs::math::subprograms::level1::Axpy;

NeonBlas::axpy(2.5, &x, &mut y);
```

NEON is baseline on every `aarch64-*` target; there is no runtime feature
gate. `AccelerateBlas` is an FFI substitute in this crate, not a fifth
architecture.

## After copying

Set `control-rs` in `Cargo.toml` to your checkout (`default-features = false`).
Keep `accelerate` off unless the firmware links the Accelerate framework.
`src/main.rs` is the harness; firmware depends on the lib target only.
