# x86_64 subprogram backend (`Avx2Blas`)

Standalone crate that implements `control_rs::math::subprograms` with AVX2+FMA
intrinsics. Optional Netlib-ABI CBLAS behind `--features cblas`.

This directory is a copyable reference implementor. It is not part of the
`control-rs` workspace. How the four backends fit together:
[`../README.md`](../README.md).

## Run

```bash
cd examples/subprograms/x86_64
cargo run
cargo run --features cblas    # needs libcblas and libblas on the linker path
```

A passing run prints `All x86_64 subprogram backend equivalence checks PASSED.`

AVX2 and FMA are not in the x86-64 baseline. `Avx2Blas` calls
`std::arch::is_x86_feature_detected!` and runs `DefaultBlas` only when either
feature is missing. On an `aarch64` host that path always delegates: the
binary is honest, it is not an AVX2 test. Use an `x86_64` machine (or
`cargo run --target x86_64-apple-darwin` under Rosetta) to exercise the
kernels.

`--features cblas` links `-lcblas`. OpenBLAS or BLIS are link-time
substitutions; this crate does not vendor them.

## Marker

| Item | Value |
|:-----|:------|
| Type | `Avx2Blas` |
| Optional | `CblasBlas` (`--features cblas`) |
| Traits | `Axpy`, `Scal`, `Dotu`, `Nrm2`, `Gemv`, `Gemm` |
| Scalars | `f32`, `f64` |
| Fast path | Contiguous layout, 8-wide `f32` / 4-wide `f64`, plus a scalar tail |
| Fallback | Missing AVX2/FMA, non-unit stride, or `Trans` not handled by the kernel |

```rust
use control_rs::math::storage::Trans;
use control_rs::math::subprograms::level3::Gemm;
use x86_64_subprograms::Avx2Blas;

Avx2Blas::gemm(Trans::NoTrans, Trans::NoTrans, alpha, &a, &b, beta, &mut c);
```

This example is `std` because CPU detection has no `core` equivalent.

## After copying

Set `control-rs` in `Cargo.toml` to your checkout (`default-features = false`).
Leave `cblas` off unless the firmware already links a Netlib-ABI BLAS.
`src/main.rs` is the harness; firmware depends on the lib target only.
