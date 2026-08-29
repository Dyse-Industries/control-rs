# RISC-V 32 subprogram backend (`NmsisDspBlas`)

`no_std` crate that implements `control_rs::math::subprograms` for
`riscv32imac-unknown-none-elf` using the NMSIS-DSP C ABI. Default runner is
QEMU `virt` / `rv32`.

This directory is a copyable reference implementor. It is not part of the
`control-rs` workspace. How the four backends fit together:
[`../README.md`](../README.md).

`riscv32imac-unknown-none-elf` is `+m,+a,+c` only: `f32` is soft-float.
NMSIS-DSP kernels that assume P-ext or V-ext will not speed this target up.
The example shows that the binding compiles and matches `DefaultBlas`.

## Prerequisites

```bash
rustup target add riscv32imac-unknown-none-elf
```

`qemu-system-riscv32` on `PATH`. `build.rs` links `c_src/` when
`riscv32-unknown-elf-gcc`, `riscv-none-elf-gcc`, or `riscv64-unknown-elf-gcc`
is present. If none is found, `src/nmsis.rs` provides portable
`#[no_mangle]` stand-ins with the same symbols. Do not enable both at once
(duplicate symbols).

The C sources are an ABI stand-in, not a full NMSIS-DSP release. A Nuclei
static library can replace them; keep the `riscv_*` names in `src/nmsis.rs`.

## Run

```bash
cd examples/subprograms/riscv32imac
cargo run
```

`.cargo/config.toml` sets `build.target` and the QEMU runner. A passing run
prints `All RISC-V 32 NMSIS-DSP subprogram checks PASSED.`

## Marker

| Item | Value |
|:-----|:------|
| Type | `NmsisDspBlas` |
| Traits | `Scal`, `Dotu`, `Dotc`, `Gemv`, `Gemm`, `Potrf`, `Trsm` |
| Scalars | `f32` (`Dotc`: `Complex<f32>`) |
| Fast path | Same CMSIS-shaped predicate: row-major, `NoTrans`, `alpha == 1`, `beta == 0` |
| `Trsm` | `Side::Left`, `UpLo::Upper`, `Diag::NonUnit`, `alpha == 1` only |
| Fallback | Anything outside those predicates calls `DefaultBlas` |

```rust
use control_rs::math::storage::Trans;
use control_rs::math::subprograms::level3::Gemm;
use riscv32imac_subprograms::NmsisDspBlas;

NmsisDspBlas::gemm(Trans::NoTrans, Trans::NoTrans, 1.0, &a, &b, 0.0, &mut c);
```

## After copying

1. Point `control-rs` in `Cargo.toml` at your checkout (`default-features = false`).
2. Keep `build.rs`, `c_src/`, and `.cargo/config.toml`, or retarget the runner
   to a P-ext/V-ext board if a speedup measurement is required.
3. Firmware uses the lib target. `src/main.rs` is the equivalence harness.
