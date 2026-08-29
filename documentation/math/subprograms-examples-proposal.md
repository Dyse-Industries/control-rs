# Subprogram Backend Examples (Proposal)

**Date:** August 26, 2026
**Status:** Proposal. Not a pipeline artifact. Does not set `Reviewed` or `Approved`.
**Evidence:** `documentation/math/research/subprograms.json` + `.bib`
**Design:** `documentation/math/subprograms-design.md` §4.5, §5, §6.1.5, §9 Phase 5

---

### 1. Purpose

`subprograms-design.md` §4.5 asserts that the subprogram traits are
library-agnostic and that hardware backends attach by implementing the traits
on a marker type. §5 rejects hardcoding one backend on exactly that ground.
The assertion is currently unexercised: `DefaultBlas` is the only implementor,
and §6.1.5 records CMSIS-DSP / NMSIS-DSP conformance as open until Phase 5
backends exist.

This document proposes one example crate per architecture. Each crate is a
self-contained reference implementor: a local marker, the trait impls, and
the FFI or intrinsic calls for that ISA. An integrator copies the directory
that matches their target. Compiling and running the crate is required so the
reference stays honest; it is not the reason the example exists. A timing
loop against `DefaultBlas` is optional bonus on any arch that can report
one, not a deliverable and not a CI gate.

`examples/qemu/` and `examples/teensy4/` are ETS validation firmware. They
do not grow BLAS backends, CMSIS/NMSIS link steps, or companion binaries.
Those crates already demonstrate the build-isolation pattern this proposal
reuses: a separate `[workspace]`, a target-specific `build.rs`, and a thin
runtime. The new examples follow that pattern as new packages, they do not
extend those two.

### 2. Why examples rather than crate features

1. **Coherence permits it.** Each example declares a local marker type
   (`NeonBlas`, `Avx2Blas`, `CmsisDspBlas`, `NmsisDspBlas`, …) and implements
   the foreign trait `control_rs::math::subprograms::level3::Gemm` for it. A
   local self type satisfies the orphan rule, so no `src/` change and no crate
   feature flag is required for a downstream integrator to attach a backend.
   An example that compiles is therefore direct evidence for the §4.5
   library-agnostic claim, in the same position as an external user.
2. **Dependency budget.** `CLAUDE.md` requires minimizing dependencies.
   Examples declare `unsafe extern "C"` blocks directly and take no
   dependency on `blas-sys`, `cblas-sys` or `blas`. The `blas` crate is cited
   as an ABI pattern (blas-lapack-rs, 2026), not consumed.
3. **Build isolation.** Each architecture needs a different C toolchain,
   linker script, and `-l` directive. Those belong in that example's own
   package, which declares `[workspace]` so they stay out of the root crate's
   graph (C-4, NFR-1c). This is the same reason `examples/qemu/` and
   `examples/teensy4/` are not workspace members. Host intrinsic examples
   still sit in their own packages so an integrator copies one directory
   rather than a root-crate `[[example]]` entangled with `std` features.

### 3. What is shared and what is not

The shared *pattern* is four items, restated in each crate so the directory
is copyable:

1. A zero-sized marker.
2. Trait impls only for the scalars and layouts the backend supports.
3. A predicate that takes the accelerated path or delegates to `DefaultBlas`.
4. A `pub type ArchBlas = ...;` alias naming that crate's selected marker.
5. A short `main` that runs the implemented methods on small fixtures and
   checks a residual (or bit equality) against `DefaultBlas`.

Nothing is shared by `#[path]` include or vendoring. A CBLAS `lda`/`incX`
bridge is host C ABI and does not belong in a CMSIS or NMSIS crate. A
contiguity predicate (`one stride == 1`, else delegate) is three lines;
duplicate it. The `ArchBlas` alias is one line; duplicate it too. Restating
it per crate is exactly what lets item 5 be the same source everywhere
without a shared module.

Layout translation lives next to the ABI that needs it:

| Backend | Layout contract | Basis |
|:--|:--|:--|
| CBLAS / Accelerate | `ORDER` → `CblasRowMajor=101` / `CblasColMajor=102`; `lda` from the non-unit stride; `incX` from the vector stride; flags `111..142`; real scalars by value, complex by `*const c_void` | (Anderson et al., 1999; Lawson et al., 1979; Dongarra et al., 1990; Apple Developer, 2026b) |
| CMSIS-DSP / NMSIS-DSP | contiguous row-major `*_matrix_instance_f32` only; no `alpha`/`beta`/`trans`/`lda` | (Arm Software CMSIS-DSP, 2026; Nuclei Software NMSIS-DSP, 2026) |
| NEON / AVX2 | `ContiguousStorage` or a `DefaultBlas` tail for the unaligned remainder | (Rust Project, 2026; Arm Architecture, 2026) |

Scalar dispatch on the C ABI is per concrete type, not generic over `T`.
Real CBLAS takes `const float alpha`; complex CBLAS takes `const void *alpha`
(`cblas_saxpy` vs `cblas_caxpy`). Emit one impl per (`f32`, `f64`,
`Complex32`, `Complex64`) in the host crates that link that ABI (ndarray,
2026c).

### 4. Proposed examples

One package per architecture, all under `examples/subprograms/`. Each
declares `[workspace]`, depends on `control-rs` with `default-features =
false`, and is excluded from `cargo ci`.

| Example | Path | Marker | Default backend | Target |
|:--|:--|:--|:--|:--|
| E-aarch64 | `examples/subprograms/aarch64/` | `NeonBlas` | `core::arch::aarch64` | host `aarch64` |
| E-x86_64 | `examples/subprograms/x86_64/` | `Avx2Blas` | `core::arch::x86_64` | host `x86_64` |
| E-thumbv7em | `examples/subprograms/thumbv7em/` | `CmsisDspBlas` | CMSIS-DSP static lib, Apache-2.0 | `thumbv7em-none-eabihf` |
| E-riscv32 | `examples/subprograms/riscv32imac/` | `NmsisDspBlas` | NMSIS-DSP static lib, Apache-2.0 | `riscv32imac-unknown-none-elf` |

The primary artifact in each crate is the marker module. `src/main.rs` is a
thin runtime: `println` on host, semihosting on the two `no_std` crates. The
`no_std` crates carry their own `memory.x`, panic handler, and
`.cargo/config.toml` runner (`qemu-system-arm` / `qemu-system-riscv32`).
They do not depend on `control-rs-ets` or `control-rs-macros`.

#### The `ArchBlas` alias

Each crate exports one name for its selected backend and `main` uses only
that name. Every `cfg` lives in the alias, never in the driver:

```rust
// examples/subprograms/aarch64/src/backend.rs
#[cfg(feature = "accelerate")]
pub type ArchBlas = AccelerateBlas;
#[cfg(not(feature = "accelerate"))]
pub type ArchBlas = NeonBlas;
```

```rust
// src/main.rs, the same source in every crate
fn main() {
    // `B` is bound on the traits that crate implements.
    equivalence::<ArchBlas>();
}
```

A crate with no opt-in arm needs no `cfg` at all: `pub type ArchBlas =
CmsisDspBlas;`. That is the whole of the generic-implementor requirement.
The alias resolves at compile time, so the call site monomorphizes to the
selected backend and costs nothing at runtime, which is the argument design
§5 already uses to reject runtime backend dispatch.

Build system and source therefore separate cleanly, the same way
`examples/qemu/` and `examples/teensy4/` build differently and run the same
suite code: the manifest, `build.rs`, linker script and runner differ per
crate, while the driver does not.

#### E-aarch64 `NeonBlas`

| Field | Value |
|:--|:--|
| Traits | `Axpy`, `Scal`, `Dotu`, `Nrm2` (L1); `Gemv` (L2); `Gemm` (L3) |
| Scalars | `f32`, `f64` |
| Default | NEON is baseline on every `aarch64-*` target, so no runtime detection (Rust Project, 2026; Arm Architecture, 2026) |
| Opt-in | `--features accelerate` on `target_vendor = "apple"`: `AccelerateBlas` via `-framework Accelerate` (Apple Developer, 2026a, 2026b) |

Vector loops over `float32x4_t` / `float64x2_t` plus a `DefaultBlas` tail
for lengths not divisible by the vector width. That tail is the same fallback
seam the DSP examples need for a different reason. A blocked 4×4 micro-kernel
is not required of a reference example.

`accelerate` is an FFI substitution in the same crate, not a fifth
architecture. It is off by default so the directory copies without a C
framework.

#### E-x86_64 `Avx2Blas`

| Field | Value |
|:--|:--|
| Traits | Same set as E-aarch64 |
| Scalars | `f32`, `f64` |
| Default | AVX2+FMA via `#[target_feature(enable = "avx2,fma")]` after `std::arch::is_x86_feature_detected!` |
| Opt-in | `--features cblas`: `CblasBlas` via `-lcblas -lblas` (OpenBLAS, 2026; BLIS, 2026a) |

AVX2 and FMA are not in the x86-64 baseline. The detection check has no
`core` equivalent, which is why this example is `std` while E-thumbv7em and
E-riscv32 are not. If detection fails, print that fact and run `DefaultBlas`
only; do not execute the intrinsic kernel.

`cblas` is the Netlib-ABI arm (Netlib, OpenBLAS, BLIS are link-time
substitutions). Complex scalars (`Complex32`, `Complex64`) appear on this
arm only. Off by default.

#### E-thumbv7em `CmsisDspBlas`

| Field | Value |
|:--|:--|
| Gate | `build.rs` compiles or links CMSIS-DSP |
| Traits | `Scal`, `Dotu`, `Dotc`, `Gemv`, `Gemm`, `Potrf`, `Trsm` |
| Scalars | `f32` first; `q31`/`q15` once fixed-point storage lands |

CMSIS-DSP is not a BLAS. It has no `alpha`, no `beta`, no `trans` and no
`lda`: the matrix argument is `arm_matrix_instance_f32 { numRows, numCols,
pData }` with `pData[i*numCols + j]`, contiguous row-major only
(Arm Software CMSIS-DSP, 2026). Every impl is a guarded fast path. The
predicate is the teaching point:

```rust
fn gemm(ta: Trans, tb: Trans, alpha: f32, a: &A, b: &B, beta: f32, c: &mut C) {
    // A::ORDER == RowMajor is a compile-time bound, not a runtime test.
    if ta == Trans::NoTrans && tb == Trans::NoTrans && alpha == 1.0 && beta == 0.0 {
        // arm_mat_mult_f32(&a_inst, &b_inst, &mut c_inst)
    } else {
        DefaultBlas::gemm(ta, tb, alpha, a, b, beta, c);
    }
}
```

Also convert `arm_status` → `LinAlgError` for routines that return one
(`arm_mat_mult_f32`, `arm_mat_solve_*`, `ARM_MATH_SINGULAR`)
(Arm Software CMSIS-DSP, 2026; Arm Software, 2026d).

Runnable default: QEMU `mps2-an500` / Cortex-M7 via this crate's own
`.cargo/config.toml`. A Teensy 4 rebuild is a runner and linker-script
change in a copy of this crate, not an edit to `examples/teensy4/`.

#### E-riscv32 `NmsisDspBlas`

| Field | Value |
|:--|:--|
| Gate | `build.rs` compiles or links NMSIS-DSP |
| Traits | Same set as E-thumbv7em |
| Scalars | `f32` first |

NMSIS-DSP is a CMSIS-DSP port with an identical struct shape
(`riscv_matrix_instance_f32`) and a `riscv_` prefix
(Nuclei Software NMSIS-DSP, 2026; Nuclei Software, 2026e, 2026f, 2026g).
The predicate and the fallback are the same; the example exists so a RISC-V
integrator copies RISC-V symbols rather than renaming ARM ones.

Two constraints belong in the marker's header comment:

- `riscv32imac-unknown-none-elf` declares `features: "+m,+a,+c"`
  (Rust Project, 2026b). No F or D extension, so `f32` arithmetic is
  soft-float on this target.
- NMSIS-DSP implementations are optimized for cores with P-ext 0.5.4 +
  N1/N2/N3 or V-ext present (Nuclei Software, 2026b).

Runnable default: QEMU `virt` / `rv32` via this crate's own runner. That
demonstrates that the binding compiles and matches `DefaultBlas`. It is not
a speedup claim. A speedup claim needs a P-ext or V-ext core.

### 5. Verification and measurement

The required check is that each crate builds for its target and that `main`
exercises the marker against `DefaultBlas`. Timing is bonus.

1. **Equivalence (required).** Run every implemented trait method twice on
   identical fixtures, once through `ArchBlas` and once through
   `DefaultBlas`. Floating-point paths assert a bounded `O(N·EPS)` residual
   per design §8; integer and fixed-point paths assert bit equality per
   §6.1.5. Small compile-time `Const<N>` shapes (`N` in `{4, 8, 16}` on
   `no_std`; host may go larger). On-target operands stay on the stack.
2. **Fallback coverage (required).** Fixtures include `alpha != 1`,
   `beta != 0`, `Trans::Trans`, and a non-unit two-sided strided view, so
   the delegation branch executes.
3. **Cost (bonus).** Host crates may print `Instant` elapsed time and a
   GFLOP/s ratio against `DefaultBlas`. Cortex-M may read `DWT->CYCCNT`
   (Arm Software, 2026b). RISC-V may read `nmsis_bench.h` HPM helpers
   (Nuclei Software, 2026c). Per design §6.1.4 these are measurements, not
   gates. Omit them rather than take a dependency (`criterion` is out) or
   claim speedup on QEMU / soft-float `riscv32imac`.

   If a host crate does print a ratio, two bounds travel with the numbers:
   control-relevant `N` sits far below the sizes a general C BLAS is tuned
   for, so below roughly `N = 64` the library can lose to `DefaultBlas`; and
   `DefaultBlas` is inlined over `Const<N>` while an FFI call is opaque, which
   is the choice the integrator actually faces.

This is the concrete form of the §6.1.5 item currently recorded as open. It
is not an ETS suite and not a CI gate. `examples/qemu/` and
`examples/teensy4/` continue to run `DefaultBlas` only.

### 6. Corrections to design §4.5 surfaced by this proposal

The §4.5 mapping table does not survive the collected prototypes. Three rows
need revision before an example can implement them:

| §4.5 row | Problem | Evidence |
|:--|:--|:--|
| `Axpy` / `Scal` → `arm_scale_f32` | `arm_scale_f32` computes `pDst[n] = pSrc[n] * scale`, which is SCAL. It is not AXPY: no accumulation into `y`. AXPY needs `arm_scale_f32` followed by `arm_add_f32`, or the `DefaultBlas` loop. | (Arm Software, 2026c) |
| `Nrm2` → `arm_cmplx_mag_f32` | `arm_cmplx_mag_f32(pSrc, pDst, numSamples)` writes an element-wise magnitude vector. NRM2 returns one scalar `‖x‖₂`. The mapping needs a sum-of-squares reduction plus `sqrt`, not `cmplx_mag`. | (Nuclei Software, 2026i) |
| `Trsm` / `Trsv` → `arm_mat_solve_upper_triangular_f32` | Solves `UT · X = A` with a matrix right-hand side and no `alpha`. It covers TRSM at `Side::Left`, `UpLo::Upper`, `Trans::NoTrans`, `Diag::NonUnit`, `alpha = 1` only, and does not cover TRSV. | (Arm Software, 2026d; Nuclei Software, 2026l) |

Fix the table before Phase 5 rather than during it, so the examples implement
a mapping the design states correctly.

Placement is already stated in design §4.5.1 (implementors live under
`examples/subprograms/`, copied rather than depended on) and §6.1.5
(example conformance is not an ETS gate). Phase 5 must not be discharged by
editing `examples/qemu/` or `examples/teensy4/`.

### 7. Open items

- Fixed-point (`q31`, `q15`) trait impls in E-thumbv7em and E-riscv32 depend
  on the fixed-point scalar type, which is out of scope here. Propose `f32`
  first.
- CMSIS-DSP and NMSIS-DSP expose no GER, SYRK or TRSM entry points in the
  evidence collected so far. `subprograms.json` records this as an open
  query. Until it resolves, those traits delegate unconditionally.
- The cost of issuing a rank-1 update as GEMM with `k = 1` versus native GER
  is an open, unevidenced query in `subprograms.json`. No example asserts a
  figure for it.
- All four crates are excluded from `cargo ci`. The default host builds take
  no system BLAS. CMSIS-DSP and NMSIS-DSP stay inside their own `build.rs`.
- `Avx2Blas` stops at AVX2 and FMA. AVX-512 changes the register width and
  is absent or downclocked on many hosts. SVE raises the same question on
  aarch64 with a vector length not known at compile time. Record both as
  open rather than adding a third intrinsic crate now.
- A single `ArchBlas` type in one shared module, `cfg`-dispatching inside
  its trait impls, was considered and rejected. It puts NEON code paths
  behind `cfg` inside the CMSIS crate and forces one dependency set to cover
  every ISA, which is what the per-package split exists to avoid. The alias
  gets the same call-site cleanliness with no cross-architecture
  contamination.
- A binary built on one host and run on another reports the implementor it
  was built with. If `main` prints a header, it prints the build triple and
  detected features so the mismatch is visible.

### 8. Sequencing

| Step | Content | Blocks |
|:--|:--|:--|
| 1 | §6 corrections to `subprograms-design.md` §4.5 | E-thumbv7em, E-riscv32 |
| 2 | E-aarch64 (`NeonBlas` marker, `ArchBlas` alias, smoke `main`) | none |
| 3 | E-x86_64 (`Avx2Blas` marker + smoke `main`) | none |
| 4 | E-thumbv7em (`CmsisDspBlas`, CMSIS-DSP in `build.rs`) | §6.1.5 closure (Arm) |
| 5 | E-riscv32 (`NmsisDspBlas`, NMSIS-DSP in `build.rs`) | §6.1.5 closure (RISC-V) |
| 6 | Optional: `accelerate` / `cblas` features and timing loops | none |

Step 2 is load-bearing. If E-aarch64 attaches to the traits without an
`src/` edit, the §4.5 claim holds and the remaining crates are substitutions
of ISA, ABI and link directive over the same driver. Timing loops, if added, land in step 6.

---

## References

Bibliographic entries live in
`documentation/math/research/subprograms.bib`. The `Avx2Blas` crate has no
key yet: `core::arch::x86_64` and the AVX2 / FMA intrinsic reference need
one collected before that crate is written. Keys used above:
`anderson1999`, `appleaccelerate2026a`, `appleaccelerate2026b`,
`armcmsisdsp2026`, `armneon2026a`, `armsoftware2026a`, `armsoftware2026b`,
`armsoftware2026c`, `armsoftware2026d`, `blaslapackrs2026`, `blis2026a`,
`dongarra1988`, `dongarra1990`, `lawson1979`, `ndarray2026c`,
`nucleinmsisdsp2026`, `nucleisoftware2026b`, `nucleisoftware2026c`,
`nucleisoftware2026e`, `nucleisoftware2026f`, `nucleisoftware2026g`,
`nucleisoftware2026i`, `nucleisoftware2026l`, `openblas2026`,
`rustneon2026`, `rustproject2026b`.
