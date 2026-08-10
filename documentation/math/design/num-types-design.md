# Numeric Types (`math::num_types`) (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_7,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@mitchelldscott-blueviolet)

---

### 1. Introduction

Every numerical model in the crate (matrices, polynomials, state-space
systems, tensors, transfer functions) is parameterized over fixed dimensions
that must be known and checked before the program runs. Catching a dimension
mismatch — multiplying a 3×4 matrix by a 5×2 matrix — at compile time rather
than as a runtime panic is a correctness goal shared across every consumer of
these types.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Base Dimension Trait**: Define a single trait for "a compile-time
  dimension," exposing a runtime `usize` value and a pattern-matchable
  type-level representation; concrete types must be
  `Clone + Copy + PartialEq + Eq`.
- **FR-2 — Type-Level Arithmetic**: Define one trait per arithmetic operation (
  addition, subtraction, multiplication, maximum, minimum), each producing an
  output satisfying the base dimension trait (FR-1).
- **FR-3 — Canonical Unary Representation**: Represent every dimension value as
  a recursively-defined Peano successor chain; all arithmetic traits (FR-2) are
  implemented by structural recursion over it.
- **FR-4 — Const-Generic Ergonomics Bridge**: Provide a const-generic wrapper
  letting call sites write an ordinary integer literal that resolves to the
  canonical representation (FR-3).
- **FR-5 — Friendly Aliases**: Generate named aliases (`U0`, `U1`, ... up to a
  fixed ceiling) for the canonical representation of each small dimension.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Memory Footprint**: Every dimension type must occupy zero bytes
  at runtime and not inflate the size of any struct carrying it as a type
  parameter.
- **NFR-2 — `no_std` Compatibility**: The module must depend only on `core`,
  with no allocation and no OS or std-only feature dependency.

#### 2.3 Constraints

- **C-1 — Recursion-Bounded Dimension Ceiling**: Friendly aliases (FR-5) stop at
  a fixed ceiling (`U32`) because the unary canonical representation (FR-3)
  requires trait-solver recursion proportional to dimension size.

---

### 3. Technical Overview

The module has no runtime component — it lowers entirely to zero-sized marker
types and associated-type bookkeeping resolved during compilation.

---

### 4. Architecture

**Base representation.** `Z` is the zero case; `S<N: Dim>(PhantomData<N>)`
is the successor of `N`. Both are zero-sized, satisfying NFR-1. `Dim::DIM` is
computed structurally (`S<N>::DIM = N::DIM + 1`), giving every dimension type a
`usize` value without storing one in the binary.

**Arithmetic by structural recursion.** `DimAdd`, `DimSub`, `DimMul`,
`DimMax`, and `DimMin` are each implemented as a base case over `Z` and an
inductive case over `S<N>`, mirroring textbook Peano-arithmetic recursion
(e.g., `(N+1) + M = (N + M) + 1` for `DimAdd`, `(N+1) * M = M + (N * M)` for
`DimMul`). The unary form was chosen here for the simplicity of a single
recursive case per operation, at the cost of the recursion depth ceiling in
C-1 — the same tradeoff the `peano` crate's own documentation names
explicitly when it recommends `typenum` for anything beyond small numbers (
peano, 2016).

**Const-generic bridge.** `Const<const N: usize>` is a friendly front end
onto the canonical representation, not an independent arithmetic
implementation. The aliasing macro gives `Const<N>` (for `N` in `0..=32`) an
associated `PeanoTypeNum` pointing at the matching `U`-alias, and `Const<N>`'s
`DimAdd` impls resolve by first converting through that associated type and
then reusing the canonical `S<N>`/`Z` arithmetic.

**Friendly aliases.** `generate_peano_aliases!` emits `U0..U32`, each a type
alias for the corresponding `S<S<...<Z>>>` chain.

---

### 5. Alternatives

1. **Binary (typenum-style) encoding instead of Peano/unary.**
   *Considered*: representing dimensions as a bit sequence
   (`UInt<U, B>`/`UTerm`), as `typenum` does (typenum, 2026), where each
   successive value adds one bit rather than one full recursive level, so
   the recursion depth needed to represent a given dimension grows far more
   slowly than in the unary form here.
   *Rejected*: it requires two mutually-recursive base representations
   (`UInt`/`UTerm`) and per-bit carry logic for every arithmetic operation,
   against a single base case each for the unary form here. Given this
   crate's dimensions describe matrix/tensor shapes rather than arbitrary
   integers, the practical ceiling this trades away (`U32`, C-1) was judged
   an acceptable cost for the simpler recursive definitions in §4.

2. **Native const-generic expressions instead of a `Dim` trait hierarchy.**
   *Considered*: using bare `const N: usize` parameters and ordinary
   `usize` arithmetic (`N + M`) directly in type signatures, skipping the
   `Dim`/`DimAdd`/... traits entirely.
   *Rejected*: RFC 2000 does not unify abstract const expressions beyond
   literal AST identity, so `N + M` and `M + N` are different, non-unifying
   types to the compiler even though they denote the same value (RFC 2000,
   2017). A caller who receives `Const<N + M>` from one function
   cannot pass it to another expecting `Const<M + N>` without an explicit
   conversion. Encoding dimensions as `Dim`-bound associated types instead
   makes commutativity (and the other arithmetic identities) provable
   through trait implementations rather than unavailable compiler
   unification — the existing `test_num_type_addition_commutativity` test
   depends on exactly this.

3. **Sealing the arithmetic traits.**
   *Considered*: sealing `DimAdd`/`DimSub`/`DimMul`/`DimMax`/`DimMin` so
   only this module can implement them for new dimension types, following
   the pattern `generic-array`'s `ArrayLength` uses to keep its own
   typenum-backed trait closed to downstream implementors (Kamiński and
   Trent, 2026).
   *Rejected*: `storage-trait-design.md` (§5) leaves its own storage-backend
   traits open specifically so downstream code can add custom
   implementations; sealing the dimension traits here would block any
   future custom `Dim` implementor (e.g. a hardware-specific fixed-size
   type) from participating in the same compile-time arithmetic that
   `Storage` and its consumers rely on.

---

### 6. Verification & Validation

#### 6.1 Verification

1. **Zero-footprint assertion**: `test_num_type_zero_byte_footprint`
   asserts `size_of::<Z>() == 0` and `size_of::<S<Z>>() == 0` /
   `size_of::<S<S<Z>>>() == 0` directly, machine-checking NFR-1 rather than
   relying on the language's general zero-sized-type guarantees for
   `PhantomData`-only structs (Rust Standard Library, 2026; Rust Reference,
   2026a).
2. **Structural arithmetic tests**: `num_type_tests.rs` covers addition
   (including commutativity and the `Z` identity), subtraction (including a
   `compile_fail` doctest for underflow), multiplication, maximum, and
   minimum, each checked both as a type-level assertion (`let _: U5 = ...`)
   and as a runtime `Dim::DIM` value check.
3. **Recursion-limit regression test**:
   `test_num_type_multiplication_recursion_limit`
   exercises arithmetic at the `U32` ceiling (C-1) directly, so a future
   change that increases per-operation recursion depth surfaces as a build
   failure here before it does anywhere downstream.
4. **HIL suite wrapping**: the test module is wrapped with
   `#[cfg_attr(not(test), control_rs_macros::hil_suite)]`, consistent with
   this crate's convention of running math test suites on-target; for a
   compile-time-only module the only on-target-relevant claim is the
   zero-footprint assertion in item 1, since the arithmetic itself has no
   runtime component to diverge across targets.

#### 6.2 Validation

Validation is deferred to the consumers that do not yet exist
(`Storage`, `Matrix`, `Polynomial`, `StateSpace`, `Tensor`,
`TransferFunction`). `storage-trait-design.md` already commits to using this
module's `U0..U32` aliases as `Storage`'s dimension parameter; once that
type lands, its own test suite becomes the first external validation that
`Dim`'s arithmetic composes correctly across a real consumer.

---

### 7. Performance & Resource Considerations

- **Compile time, not runtime**: every trait in this module is resolved
  during compilation; there is no generated code or data at runtime beyond
  what a consumer's own fields require.
- **Zero memory footprint**: `Z`, `S<N>`, and `Const<N>` are zero-sized
  (NFR-1), matching the general Rust guarantee that `PhantomData<T>` and
  types built only from it occupy no space (Rust Standard Library, 2026).
- **Recursion depth scales with dimension size**: because arithmetic is
  unary (§4), a `DimAdd`/`DimMul` on operands near the `U32` ceiling
  recurses close to 32 trait-solver steps; this is the cost C-1 accepts in
  exchange for the simpler recursive definitions over a binary encoding.

---

### 8. Risks & Open Questions

1. **`Const<N>` arithmetic is not O(1).** Unlike nalgebra's `Const<T>`,
   which computes `Dim::value()` directly from its `usize` parameter with
   no type-level detour (Crozet, 2026), this module's `Const<N>` resolves
   `DimAdd` by first mapping to `PeanoTypeNum` and then recursing over the
   canonical `S<N>`/`Z` chain (§4). Compile time for `Const`-expressed
   arithmetic therefore scales with the operands' Peano depth even when
   both sides are given as plain literals. Whether this matters in practice
   depends on how consumers actually call it; not yet measured.
2. **Ceiling extension path is unresolved.** If a future consumer needs
   dimensions above 32, C-1's options are raising the recursion limit,
   switching to a binary encoding (Alternative 1), or introducing a second,
   non-Peano dimension type reserved for large sizes. This document does
   not pick one; it is flagged here for whoever hits the ceiling first.
3. **No type-level division or ordering beyond max/min.** The module
   defines `DimMax`/`DimMin` but no `DimDiv` and no general `DimOrd`. Open
   until a consumer (e.g. a decomposition algorithm needing a ratio of
   dimensions) demonstrates a concrete need.
4. **Overflow behavior above the tested ceiling is uncharacterized.** The
   only `compile_fail` regression test covers subtraction underflow
   (`U2 - U5`); there is no equivalent test demonstrating what happens if
   arithmetic is attempted past `U32` (a recursion-limit compiler error, but
   its exact shape and readability are unverified).

---

### 9. Development Plan

| Phase / Feature                                    | Description                                                                                                                                                                                                                                                             | Estimated Effort |
|:---------------------------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Encoding & Arithmetic (complete)** | `Dim`, `DimAdd`/`DimSub`/`DimMul`/`DimMax`/`DimMin`, `Z`/`S<N>`, `Const<N>` bridge, and the `U0..U32` alias macro already exist in `src/math/num_types.rs` with passing unit tests.                                                                                     | —                |
| **Phase 2: Storage/Matrix Integration**            | Wire `Dim` into the `Storage<T, R, C>` trait per `storage-trait-design.md` FR-2, confirming the `U32` ceiling (C-1) is enforced consistently at the first real call site.                                                                                               | Medium           |
| **Phase 3: Verification Hardening**                | Add a `compile_fail` (or equivalent) regression test characterizing behavior past the `U32` ceiling (Risk 4); consider `proptest`-based coverage of arithmetic identities as more `Dim`-generic consumers land, per this crate's invariant-heavy-code testing standard. | Small            |
| **Phase 4: Extended Arithmetic (conditional)**     | Evaluate `DimDiv`/`DimOrd` (Risk 3) once a concrete consumer needs them; not scheduled otherwise.                                                                                                                                                                       | Small            |

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                               |
|:---------|:---------------|:----------------|:----------------------------------------------------------------------------------------------------------|
| 1.0      | August 7, 2026 | @MitchellDScott | Initial draft backfilling design rationale for the existing `num_types.rs` module from research findings. |
| 1.1      | August 9, 2026 | @MitchellDScott | Review and corrections.                                                                                   |

---

### 11. References

[1] paholg, "typenum," docs.rs, v1.20.1, 2026. [Online].
Available: https://docs.rs/typenum/latest/typenum/

[2] paholg, "peano," docs.rs, v1.0.2, 2016. [Online].
Available: https://docs.rs/peano/latest/peano/

[3] Rust RFC Book, "RFC 2000: Const Generics," Rust Project, 2017. [Online].
Available: https://rust-lang.github.io/rfcs/2000-const-generics.html

[4] S. Crozet, "src/base/dimension.rs," dimforge/nalgebra, 2026. [Online].
Available: https://github.com/dimforge/nalgebra/blob/main/src/base/dimension.rs

[5] B. Kamiński and A. Trent, "ArrayLength," generic_array, docs.rs, v1.4.4,

2026. [Online].
      Available: https://docs.rs/generic-array/latest/generic_array/trait.ArrayLength.html

[6] Rust Project, "Glossary — Zero-sized type (ZST)," The Rust Reference,

2026. [Online].
      Available: https://doc.rust-lang.org/reference/glossary.html#zero-sized-type-zst

[7] Rust Project, "Type layout — The Rust representation," The Rust Reference,

2026. [Online].
      Available: https://doc.rust-lang.org/reference/type-layout.html#the-rust-representation

[8] Rust Project, "PhantomData," std::marker, doc.rust-lang.org, v1.97.1,

2026. [Online].
      Available: https://doc.rust-lang.org/std/marker/struct.PhantomData.html

[9] Rust Project, "Version 1.51.0 (2021-03-25) — RELEASES.md," rust-lang/rust,

2021. [Online].
      Available: https://raw.githubusercontent.com/rust-lang/rust/1.51.0/RELEASES.md
