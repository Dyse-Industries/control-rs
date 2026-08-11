# Numeric Types (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_9,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@mitchelldscott-blueviolet)

---

### 1. Introduction

Every numerical model in the crate (matrices, polynomials, state-space
systems, tensors, transfer functions) is parameterized over fixed dimensions
that must be known and checked before the program runs. Catching a dimension
mismatch (i.e. multiplying a 3×4 matrix by a 5×2 matrix) at compile time rather
than as a runtime panic.

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
- **FR-3 — Const-Generic Bridge**: Provide a const-generic wrapper
  letting call sites write an ordinary integer literal that resolves to the
  canonical representation.
- **FR-4 — Aliases**: Generate named aliases (`U0`, `U1`, ... up to a
  fixed ceiling) for the canonical representation of each small dimension.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero Memory Footprint**: Every dimension type must occupy zero bytes
  at runtime and not inflate the size of any struct carrying it as a type
  parameter.
- **NFR-2 — `no_std` Compatibility**: The module must depend only on `core`,
  with no allocation and no OS or std-only feature dependency.

#### 2.3 Constraints

- **C-1 — Recursion-Bounded Dimension Ceiling**: Friendly aliases (FR-4) stop at
  a fixed ceiling (`U127`) because the unary canonical representation
  requires trait-solver recursion proportional to dimension size, and rustc's
  default `#![recursion_limit]` is 128 (Rust Reference, 2026c) — one Peano level
  above the deepest chain a single `Dim` value can resolve to.
- **C-2 — Ceiling Does Not Bound Pairwise Arithmetic**: Raising C-1 to `U127`
  only guarantees that a single dimension up to that size resolves; it does
  not guarantee `DimAdd`/`DimSub`/`DimMax`/`DimMin` succeeds for every pair of
  aliases below the ceiling. The structural-recursion where-clauses in §4 can
  require resolving `Dim` for an operand more than once per step, so the safe
  envelope for a pair is smaller than, and asymmetric with respect to, `U127`
  (§6.1, item 5).

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
implementation. The aliasing macro gives `Const<N>` (for `N` in `0..=127`) an
associated `PeanoTypeNum` pointing at the matching `U`-alias, and `Const<N>`'s
`DimAdd` impls resolve by first converting through that associated type and
then reusing the canonical `S<N>`/`Z` arithmetic.

**Friendly aliases.** `generate_peano_aliases!` emits `U0..U127`, each a type
alias for the corresponding `S<S<...<Z>>>` chain. The macro itself is
unchanged from Phase 1 (§9); it still consumes a literal, comma-separated
identifier list rather than deriving `U33..U127` from a numeric range,
because `macro_rules!` cannot synthesize identifiers from an expression
without token-pasting support the crate does not otherwise need (§5, item 4).
The extended list is generated once by a throwaway script and checked in as
static source text, identical in kind to the existing `U0..U32` list — this
is a one-time authoring step, not a build-time or proc-macro code generation
concern.

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
   integers, the practical ceiling this trades away (`U127`, C-1) was judged
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

4. **Proc-macro or `paste`-based identifier synthesis for the `U33..U127`
   alias extension, instead of a checked-in static list.**
   *Considered*: adding a `generate_dims!(127)` entry point to the
   in-workspace `control_rs_macros` proc-macro crate (already used for
   `hil_suite`), or pulling in the `paste` crate so `macro_rules!` could
   token-paste `U` + a counted literal directly, avoiding a hand-authored
   identifier list.
   *Rejected*: the alias ceiling is a fixed constant chosen once (C-1), not
   a value that varies per build or per call site, so there is nothing for
   a proc macro or token-pasting crate to compute at compile time that a
   one-time text-generation step cannot equally produce ahead of time. Both
   options add a dependency or expand the surface of an in-tree proc-macro
   crate to solve a problem that resolves to the same static token list
   either way, against this crate's "minimize dependency use" convention.

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
3. **Recursion-limit test**:
   `test_num_type_multiplication_recursion_limit` pins `U125` as the
   largest operand for which `DimMul<U1>` resolves — the per-pair envelope
   stops two levels short of the `U127` ceiling because each recursive step
   re-resolves `Dim` on the multiplier (C-2). A future change that
   increases per-operation recursion depth surfaces as a build failure here
   before it does anywhere downstream.
4. **HIL suite wrapping**: the test module is wrapped with
   `#[cfg_attr(not(test), control_rs_macros::hil_suite)]`, consistent with
   this crate's convention of running math test suites on-target; for a
   compile-time-only module the only on-target-relevant claim is the
   zero-footprint assertion in item 1, since the arithmetic itself has no
   runtime component to diverge across targets.
5. **Arithmetic boundary tests (C-2)**: A single `Dim` value resolves up to
   depth 127 and fails at 128, matching C-1 exactly. Both sides of the boundary
   are tested by unit tests: (`U125 * U1`, `U63 + U64`) and failing cases as
   `compile_fail` doctests in the `num_types` module
   (`U127 * U2`, `U126 * U1`, `U1 + U126`).

#### 6.2 Validation

Validation is deferred to the consumers that do not yet exist
(`Storage`, `Matrix`, `Polynomial`, `StateSpace`, `Tensor`,
`TransferFunction`).

---

### 7. Performance & Resource Considerations

- **Compile time, not runtime**: every trait in this module is resolved
  during compilation; there is no generated code or data at runtime beyond
  what a consumer's own fields require.
- **Zero memory footprint**: `Z`, `S<N>`, and `Const<N>` are zero-sized
  (NFR-1), matching the general Rust guarantee that `PhantomData<T>` and
  types built only from it occupy no space (Rust Standard Library, 2026).
- **Recursion depth scales with dimension size**: because arithmetic is
  unary (§4), a `DimAdd`/`DimMul` on operands near the `U127` ceiling
  recurses close to the trait-solver's 128-step default budget (C-1); this
  is the cost C-1 accepts in exchange for the simpler recursive definitions
  over a binary encoding.
- **~95 additional `Dim`/`DimAdd<Z>` impls**: extending the alias list from
  `U0..U32` to `U0..U127` adds one `impl Dim for Const<N>` and one
  `impl DimAdd<Z> for Const<N>` per new value (§4). Each impl is checked
  once at definition time regardless of use, so this is a fixed, one-time
  compile-time cost rather than one that scales with how many aliases a
  downstream consumer actually instantiates; not yet measured against the
  crate's overall build time.

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
2. **Ceiling extension resolved to `U127`; beyond it remains unresolved.**
   C-1 now stops exactly at the trait-solver's single-value resolution
   limit under the default `#![recursion_limit]` (§6.1, item 5); there is
   no headroom left to push further without one of the options this
   document still declines to pick: raising the recursion limit crate-wide,
   switching to a binary encoding (Alternative 1), or introducing a second,
   non-Peano dimension type for large sizes. Flagged here for whoever needs
   dimensions above 127.
3. **No type-level division or ordering beyond max/min.** The module
   defines `DimMax`/`DimMin` but no `DimDiv` and no general `DimOrd`. Open
   until a consumer (e.g. a decomposition algorithm needing a ratio of
   dimensions) demonstrates a concrete need.
4. **Pairwise-arithmetic boundary is pinned only at sampled points (C-2).**
   The §6.1 item 5 tests fix the envelope at specific operand pairs; no
   closed-form rule (e.g. "sum ≤ 127") predicts whether an arbitrary pair
   below the ceiling compiles. A consumer combining two large dimensions
   still only learns the answer when that combination is first
   instantiated — the failure mode is a compile error, never a
   miscompilation.

---

### 9. Development Plan

| Phase / Feature                                     | Description                                                                                                                                                                                                                                         | Estimated Effort |
|:----------------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Encoding & Arithmetic (complete)**  | `Dim`, `DimAdd`/`DimSub`/`DimMul`/`DimMax`/`DimMin`, `Z`/`S<N>`, `Const<N>` bridge, and the `U0..U32` alias macro already exist in `src/math/num_types.rs` with passing unit tests.                                                                 | —                |
| **Phase 2: Ceiling Extension to `U127` (complete)** | `U33..U127` aliases generated and appended to the `generate_peano_aliases!` invocation; module doc ceiling description, dimension-value assertions, and the recursion-limit test (pinned at `U125`, §6.1 item 3) moved to the new ceiling.          | —                |
| **Phase 3: Storage/Matrix Integration**             | Wire `Dim` into the `Storage<T, R, C>` trait per `storage-trait-design.md` FR-2, confirming the `U127` ceiling (C-1) and the pairwise-arithmetic boundary (C-2) are documented consistently at the first real call site.                            | Medium           |
| **Phase 4: Verification Hardening (complete)**      | Asymmetric-pair boundary tests landed — passing sides as unit tests, failing sides as `compile_fail` doctests (§6.1, item 5) — closing Risk 4; `proptest`-based coverage of arithmetic identities deferred until more `Dim`-generic consumers land. | —                |
| **Phase 5: Extended Arithmetic (conditional)**      | Evaluate `DimDiv`/`DimOrd` (Risk 3) once a concrete consumer needs them; not scheduled otherwise.                                                                                                                                                   | Small            |

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                              |
|:---------|:----------------|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 7, 2026  | @MitchellDScott | Initial draft backfilling design rationale for the existing `num_types.rs` module from research findings.                                                |
| 1.1      | August 9, 2026  | @MitchellDScott | Review and corrections.                                                                                                                                  |
| 1.2      | August 9, 2026  | @MitchellDScott | Raised the friendly-alias ceiling (C-1) from `U32` to `U127`                                                                                             |
| 1.3      | August 10, 2026 | @MitchellDScott | Aligned §6.1 with the shipped `U125` `DimMul` pin; added pairwise boundary tests (unit + `compile_fail`) closing Risk 4; marked Phases 2 and 4 complete. |

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

[10] Rust Project, "recursion_limit," The Rust Reference — Attributes,

doc.rust-lang.org, 2026. [Online].
Available: https://doc.rust-lang.org/reference/attributes/limits.html
