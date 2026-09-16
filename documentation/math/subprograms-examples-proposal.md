# Subprogram Backend Examples (Proposal)

![Date Badge](https://img.shields.io/badge/Date-September_10,_2026-blue)
![Type Badge](https://img.shields.io/badge/Type-Proposal-lightgrey)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

This document is a proposal. It does not follow
`documentation/design-template.md` and carries no design status badge.
**Evidence:** `documentation/math/research/subprograms.json` + `.bib`
**Design:** `documentation/math/subprograms-design.md` §4.5, §5, §6.7, §9 Phase 5

---

### 1. Purpose

`subprograms-design.md` §4.5 asserts that the subprogram traits are
library-agnostic and that hardware backends attach by implementing the traits
on a marker type. §5 rejects hardcoding one backend on exactly that ground.
The assertion is currently unexercised: `DefaultBlas` is the only implementor,
and §6.7 records CMSIS-DSP / NMSIS-DSP conformance as open until Phase 5
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
2. **Dependency budget.** Standing crate policy is to minimize dependencies.
   Examples declare `unsafe extern "C"` blocks directly and take no
   dependency on `blas-sys`, `cblas-sys` or `blas`. The `blas` crate is cited
   as an ABI pattern [1], not consumed.
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
| CBLAS / Accelerate | `ORDER` → `CblasRowMajor=101` / `CblasColMajor=102`; `lda` from the non-unit stride; `incX` from the vector stride; flags `111..142`; real scalars by value, complex by `*const c_void` | [2]–[5] |
| CMSIS-DSP / NMSIS-DSP | contiguous row-major `*_matrix_instance_f32` only; no `alpha`/`beta`/`trans`/`lda` | [6], [7] |
| NEON / AVX2 | `ContiguousStorage` or a `DefaultBlas` tail for the unaligned remainder | [8], [9] |

Scalar dispatch on the C ABI is per concrete type, not generic over `T`.
Real CBLAS takes `const float alpha`; complex CBLAS takes `const void *alpha`
(`cblas_saxpy` vs `cblas_caxpy`). Emit one impl per (`f32`, `f64`,
`Complex32`, `Complex64`) in the host crates that link that ABI [10].

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
| Default | NEON is baseline on every `aarch64-*` target, so no runtime detection [8], [9] |
| Opt-in | `--features accelerate` on `target_vendor = "apple"`: `AccelerateBlas` via `-framework Accelerate` [5], [11] |

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
| Opt-in | `--features cblas`: `CblasBlas` via `-lcblas -lblas` [12], [13] |

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
[6]. Every impl is a guarded fast path. The
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
[6], [14].

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
[7], [15]–[17].
The predicate and the fallback are the same; the example exists so a RISC-V
integrator copies RISC-V symbols rather than renaming ARM ones.

Two constraints belong in the marker's header comment:

- `riscv32imac-unknown-none-elf` declares `features: "+m,+a,+c"`
  [18]. No F or D extension, so `f32` arithmetic is
  soft-float on this target.
- NMSIS-DSP implementations are optimized for cores with P-ext 0.5.4 +
  N1/N2/N3 or V-ext present [19].

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
   §6.3. Small compile-time `Const<N>` shapes (`N` in `{4, 8, 16}` on
   `no_std`; host may go larger). On-target operands stay on the stack.
2. **Fallback coverage (required).** Fixtures include `alpha != 1`,
   `beta != 0`, `Trans::Trans`, and a non-unit two-sided strided view, so
   the delegation branch executes.
3. **Cost (bonus).** Host crates may print `Instant` elapsed time and a
   GFLOP/s ratio against `DefaultBlas`. Cortex-M may read `DWT->CYCCNT`
   [20]. RISC-V may read `nmsis_bench.h` HPM helpers
   [21]. Per design §6.3 these are measurements, not
   gates. Omit them rather than take a dependency (`criterion` is out) or
   claim speedup on QEMU / soft-float `riscv32imac`.

   If a host crate does print a ratio, two bounds travel with the numbers:
   control-relevant `N` sits far below the sizes a general C BLAS is tuned
   for, so below roughly `N = 64` the library can lose to `DefaultBlas`; and
   `DefaultBlas` is inlined over `Const<N>` while an FFI call is opaque, which
   is the choice the integrator actually faces.

This is the concrete form of the §6.7 item currently recorded as open. It
is not an ETS suite and not a CI gate. `examples/qemu/` and
`examples/teensy4/` continue to run `DefaultBlas` only.

### 6. Corrections to design §4.5 surfaced by this proposal

The §4.5 mapping table does not survive the collected prototypes. Three rows
need revision before an example can implement them:

| §4.5 row | Problem | Evidence |
|:--|:--|:--|
| `Axpy` / `Scal` → `arm_scale_f32` | `arm_scale_f32` computes `pDst[n] = pSrc[n] * scale`, which is SCAL. It is not AXPY: no accumulation into `y`. AXPY needs `arm_scale_f32` followed by `arm_add_f32`, or the `DefaultBlas` loop. | [22] |
| `Nrm2` → `arm_cmplx_mag_f32` | `arm_cmplx_mag_f32(pSrc, pDst, numSamples)` writes an element-wise magnitude vector. NRM2 returns one scalar `‖x‖₂`. The mapping needs a sum-of-squares reduction plus `sqrt`, not `cmplx_mag`. | [23] |
| `Trsm` / `Trsv` → `arm_mat_solve_upper_triangular_f32` | Solves `UT · X = A` with a matrix right-hand side and no `alpha`. It covers TRSM at `Side::Left`, `UpLo::Upper`, `Trans::NoTrans`, `Diag::NonUnit`, `alpha = 1` only, and does not cover TRSV. | [14], [24] |

Fix the table before Phase 5 rather than during it, so the examples implement
a mapping the design states correctly.

Placement is already stated in design §4.5.1 (implementors live under
`examples/subprograms/`, copied rather than depended on) and §6.7
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
| 4 | E-thumbv7em (`CmsisDspBlas`, CMSIS-DSP in `build.rs`) | §6.7 closure (Arm) |
| 5 | E-riscv32 (`NmsisDspBlas`, NMSIS-DSP in `build.rs`) | §6.7 closure (RISC-V) |
| 6 | Optional: `accelerate` / `cblas` features and timing loops | none |

Step 2 is load-bearing. If E-aarch64 attaches to the traits without an
`src/` edit, the §4.5 claim holds and the remaining crates are substitutions
of ISA, ABI and link directive over the same driver. Timing loops, if added, land in step 6.

---

## References

[1] blas-lapack-rs, "src/lib.rs," in *blas-lapack-rs/blas*, 2026. [Online]. Available: https://raw.githubusercontent.com/blas-lapack-rs/blas/master/src/lib.rs. Accessed: Aug. 24, 2026.

[2] E. Anderson, Z. Bai, C. Bischof, S. Blackford, J. Demmel, J. Dongarra, J. Du Croz, A. Greenbaum, S. Hammarling, A. McKenney, and D. Sorensen, *LAPACK Users' Guide*, 3rd ed., Philadelphia, PA: SIAM, 1999. [Online]. Available: https://www.netlib.org/lapack/lug/. Accessed: Aug. 24, 2026.

[3] C. L. Lawson, R. J. Hanson, D. R. Kincaid, and F. T. Krogh, "Basic Linear Algebra Subprograms for Fortran Usage," *ACM Trans. Math. Softw.*, vol. 5, no. 3, pp. 308–323, Sep. 1979, doi: 10.1145/355841.355847.

[4] J. J. Dongarra, J. Du Croz, I. S. Duff, and S. Hammarling, "A Set of Level 3 Basic Linear Algebra Subprograms," *ACM Trans. Math. Softw.*, vol. 16, no. 1, pp. 1–17, Mar. 1990, doi: 10.1145/77626.79170.

[5] Apple Inc., "BLAS," in *Apple Developer Documentation (Accelerate Framework)*, 2026. [Online]. Available: https://developer.apple.com/documentation/accelerate/blas-library. Accessed: Aug. 24, 2026.

[6] Arm Software, "Include/dsp/matrix_functions.h," in *ARM-software/CMSIS-DSP*, 2026. [Online]. Available: https://raw.githubusercontent.com/ARM-software/CMSIS-DSP/main/Include/dsp/matrix_functions.h. Accessed: Aug. 24, 2026.

[7] Nuclei Software, "NMSIS DSP Matrix Functions," in *doc.nucleisys.com (NMSIS 1.6.0 documentation)*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupmatrix/api_matrixmult.html. Accessed: Aug. 24, 2026.

[8] Rust Project, "Module core::arch::aarch64," in *The Rust Standard Library (core)*, 2026. [Online]. Available: https://doc.rust-lang.org/core/arch/aarch64/index.html. Accessed: Aug. 24, 2026.

[9] Arm Limited, "Arm C Language Extensions (ACLE)," in *Arm Developer Documentation*, 2026. [Online]. Available: https://developer.arm.com/architectures/instruction-sets/intrinsics/. Accessed: Aug. 24, 2026.

[10] bluss and ndarray developers, "src/linalg/impl_linalg.rs," in *rust-ndarray/ndarray*, 2026. [Online]. Available: https://raw.githubusercontent.com/rust-ndarray/ndarray/master/src/linalg/impl_linalg.rs. Accessed: Aug. 24, 2026.

[11] Apple Inc., "Accelerate," in *Apple Developer Documentation*, 2026. [Online]. Available: https://developer.apple.com/documentation/accelerate. Accessed: Aug. 24, 2026.

[12] OpenMathLib, "README.md," in *OpenMathLib/OpenBLAS*, 2026. [Online]. Available: https://raw.githubusercontent.com/OpenMathLib/OpenBLAS/develop/README.md. Accessed: Aug. 24, 2026.

[13] flame (Field G. Van Zee et al.), "README.md," in *flame/blis*, 2026. [Online]. Available: https://raw.githubusercontent.com/flame/blis/master/README.md. Accessed: Aug. 24, 2026.

[14] Arm Software, "CMSIS-DSP: Matrix Inverse," in *arm-software.github.io*, 2026. [Online]. Available: https://arm-software.github.io/CMSIS-DSP/main/group__MatrixInv.html. Accessed: Aug. 24, 2026.

[15] Nuclei Software, "Matrix Multiplication," in *NMSIS DSP API*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupmatrix/api_matrixmult.html. Accessed: Aug. 24, 2026.

[16] Nuclei Software, "Matrix Vector Multiplication," in *NMSIS DSP API*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupmatrix/api_matrixvectmult.html. Accessed: Aug. 24, 2026.

[17] Nuclei Software, "Vector Scale," in *NMSIS DSP API*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupmath/api_basicscale.html. Accessed: Aug. 24, 2026.

[18] Rust Project, "riscv32imac_unknown_none_elf.rs," in *rust-lang/rust*, 2026. [Online]. Available: https://doc.rust-lang.org/nightly/nightly-rustc/src/rustc_target/spec/targets/riscv32imac_unknown_none_elf.rs.html. Accessed: Aug. 24, 2026.

[19] Nuclei Software, "README.md," in *Nuclei-Software/NMSIS*, 2026. [Online]. Available: https://raw.githubusercontent.com/Nuclei-Software/NMSIS/master/README.md. Accessed: Aug. 24, 2026.

[20] Arm Software, "CMSIS-Core (Cortex-M): DWT_Type Struct Reference," in *arm-software.github.io (CMSIS_6 v6.0.0)*, 2026. [Online]. Available: https://arm-software.github.io/CMSIS_6/v6.0.0/Core/structDWT__Type.html. Accessed: Aug. 24, 2026.

[21] Nuclei Software, "Changelog," in *doc.nucleisys.com (NMSIS 1.6.0 documentation)*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/changelog.html. Accessed: Aug. 24, 2026.

[22] Arm Software, "CMSIS-DSP: Vector Scale," in *arm-software.github.io*, 2026. [Online]. Available: https://arm-software.github.io/CMSIS-DSP/main/group__BasicScale.html. Accessed: Aug. 24, 2026.

[23] Nuclei Software, "Complex Magnitude," in *NMSIS DSP API*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupcmplxmath/api_cmplx_mag.html. Accessed: Aug. 24, 2026.

[24] Nuclei Software, "Matrix Inverse," in *NMSIS DSP API*, 2026. [Online]. Available: https://doc.nucleisys.com/nmsis/dsp/api/groupmatrix/api_matrixinv.html. Accessed: Aug. 24, 2026.
