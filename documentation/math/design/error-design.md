# Crate-Wide Error Module (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

> **Pipeline note**: this document has **not** been through `/cr-research`.
> No query file exists at `documentation/math/research/queries/error.json`
> and no results file exists at
> `documentation/math/research/results/error.json`. Per the project's
> gated-pipeline rule (`CLAUDE.md`), a design doc is normally unblocked by a
> completed research phase; this document is an exception made to close an
> immediate, concrete cross-document inconsistency (`ConversionError` being
> defined inside `matrix-design.md` despite being reused, undeclared, by
> `polynomial-design.md` and `tensor-design.md`). Its scope is therefore
> deliberately narrow — it records only what is already validated by
> existing, shipped code and by the numerical-models design docs' own prior
> content. **Any elaboration beyond §4 requires a `/cr-research` pass
> first**; do not add new error variants, taxonomy, or citations to this
> document without one.

---

### 1. Introduction

`control-rs` has two error types that are shared across module boundaries
rather than owned by a single module:

- `ArithmeticError` (`src/math/mod.rs`) — already crate-wide in practice: it
  is produced by `math::ops`'s `Try*` traits and consumed by `complex_num.rs`
  and `assert.rs`, none of which "own" it individually.
- `ConversionError` — specified inside `matrix-design.md` (§4.9.2, prior to
  this document) but reused, without restatement, by `polynomial-design.md`
  (§4.4.1, the `Polynomial → Matrix` companion-matrix conversion) and
  `tensor-design.md` (§4.10, the `Tensor ↔ Matrix`/`Polynomial`
  conversions). None of those three documents state where the type is
  canonically defined, unlike the project's established pattern for shared
  conventions (e.g. `polynomial-design.md` §2.3 explicitly designates
  itself the canonical statement of coefficient ordering, and other
  documents cross-reference it rather than restate it).

This document exists to give `ConversionError` — and, in the future, any
other error type shared by two or more sibling modules — one canonical
home, following the same cross-reference discipline already used elsewhere
in this document family.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **Single Definition for Shared Error Types**: An error type consumed by
  more than one sibling module (`Matrix`, `Polynomial`, `Tensor`,
  `StateSpace`, `TransferFunction`) must be defined once, here, rather than
  inside whichever module happened to need it first.
- **No Redefinition Downstream**: Consuming design docs reference this
  document's variants by name (e.g. `error::ConversionError::LayoutMismatch`)
  rather than restating the enum.

#### 2.2 Non-Functional Requirements

- Follows the crate-wide `thiserror`-enum convention already established by
  `matrix-design.md` §4.9.2 and `state-space-design.md` §4.4 — no behavior
  change, relocation only.

#### 2.3 Constraints

- **Scope Boundary**: This document does **not** claim ownership of
  module-local error types that are not currently shared —
  `LinAlgError` (`matrix-design.md` §4.9.2), `DivisionError`
  (`polynomial-design.md` §4.3), `ContractionError` (`tensor-design.md`
  §4.6), `StateSpaceError` (`state-space-design.md` §4.4), and
  `TransferFunctionError` (`transfer-function-design.md` §4.5) all stay
  where they are. `state-space-design.md` already correctly cross-references
  `matrix-design.md` for its reuse of `LinAlgError::SingularMatrix` — that
  is the pattern this document generalizes for `ConversionError`, not a
  mandate to centralize every error type in the crate.
- **No New Variants**: Per the pipeline note above, this revision only
  relocates the three `ConversionError` variants already specified in
  `matrix-design.md`'s prior revisions. It does not add, rename, or remove
  variants.

---

### 3. Technical Overview

A single module (path to be fixed by implementation — see §8) exposing
error enums that are genuinely cross-cutting. `ArithmeticError` already
lives in `src/math/mod.rs` and is left there for now (§8); `ConversionError`
moves here from `matrix-design.md`.

---

### 4. Core Architecture

```rust
/// Representation and layout conversion errors, shared across `Matrix`,
/// `Polynomial`, and `Tensor` conversions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConversionError {
    /// Rank or coordinate dimensions do not align between Matrix/Tensor.
    LayoutMismatch,
    /// The polynomial is not monic (leading coefficient is not ONE), preventing companion matrix construction.
    NonMonicPolynomial,
    /// Dimension or capacity overflow/underflow during calculations.
    DimensionMismatch,
}
```

This is unchanged from `matrix-design.md`'s prior §4.9.2 — moved, not
redesigned. Consumers:

- `Matrix ↔ Polynomial` (Faddeev–LeVerrier / companion matrix):
  `matrix-design.md` §4.8.1, `polynomial-design.md` §4.4.1.
- `Matrix ↔ Tensor`, `Polynomial ↔ Tensor`: `tensor-design.md` §4.10.

---

### 5. Alternatives

- **Leave `ConversionError` in `matrix-design.md` (status quo, rejected)**:
  Cheapest short-term, but keeps an undeclared dependency from
  `polynomial-design.md` and `tensor-design.md` onto a document that does
  not otherwise sit upstream of them architecturally (`Matrix` does not own
  `Polynomial` or `Tensor`; the three are peers per `matrix-design.md` §5.3/
  `transfer-function-design.md` §6's "avoid wrapping the natural lower-level
  peer type" reasoning). Rejected because peer modules should not silently
  own each other's shared types.
- **Fold `ConversionError` into `ArithmeticError` (considered, deferred)**:
  Would reduce the crate's error-type count further, but `ArithmeticError`'s
  existing variants (`DivisionByZero`, `Overflow`, `DomainViolation`,
  `PrecisionLoss`, `Underflow`, `Saturation`) are scalar-arithmetic-shaped,
  not layout/rank-shaped; merging them would force every `ArithmeticError`
  match arm to also handle layout mismatches that cannot occur in scalar
  code. Deferred to the `/cr-research` pass this document requires before
  any taxonomy change.

---

### 6. Verification & Validation

No new verification surface — `ConversionError`'s existing test
obligations (already implied by `matrix-design.md` §6.1's property-based
testing of conversions) move with the type. Not re-specified here to avoid
duplicating `matrix-design.md` §6.

---

### 7. Performance & Resource Considerations

None — a `#[derive(Debug, Clone, Copy, PartialEq, Eq)]` enum with no
payload has no runtime cost regardless of which module defines it.

---

### 8. Risks & Open Questions

> [!IMPORTANT]
> **This document requires a `/cr-research` pass before further growth.**
> Everything below is an open question this stub deliberately leaves
> unresolved rather than guessing:
> - **Module path**: `crate::error` (new top-level module) vs.
>   `crate::math::error` (keeps it under `math`, consistent with
>   `ArithmeticError`'s current location) vs. leaving `ConversionError` in
>   `crate::math` alongside `ArithmeticError` directly without a dedicated
>   submodule. Not decided here.
> - **Should `ArithmeticError` move here too?** It is already crate-wide in
>   practice (§1) but relocating a type with existing call sites in
>   `ops.rs`, `complex_num.rs`, and `assert.rs` is a larger, code-touching
>   change than this design-only pass is scoped to evaluate.
> - **Should `LinAlgError`, `StateSpaceError`, `TransferFunctionError`,
>   `DivisionError`, or `ContractionError` eventually fold in?** §2.3
>   explicitly scopes this revision to *not* do this. Whether any of them
>   are, in fact, shared (rather than module-local) is a question for a
>   future pass once each module's own design doc is closer to
>   implementation.
> - **`thiserror` vs. hand-rolled `Display`**: `matrix-design.md`'s
>   `LinAlgError`/`ConversionError` and `state-space-design.md`'s
>   `StateSpaceError` all assume `thiserror`, but `ArithmeticError`
>   (`src/math/mod.rs`) currently hand-rolls `core::error::Error` +
>   `core::fmt::Display` directly. Whether that's a pre-`thiserror`-adoption
>   artifact or a deliberate `no_std`-minimalism choice needs research
>   before this document prescribes one convention crate-wide.

---

### 9. Development Plan

| Task / Feature | Description | Estimated Effort |
|:----------------|:-------------|:-------------------|
| Step 0: Research | Run `/cr-research error` to resolve §8's open questions (module path, `ArithmeticError` relocation, `thiserror` convention) before continuing this design. | — |
| Step 1: Relocate `ConversionError` | Move the enum (§4) to its resolved module path; update `Matrix`/`Polynomial`/`Tensor` `TryFrom` impls to reference it. | 0.5 Day (blocked on Step 0) |

---

### 10. Revision History

| Date | Author | Description |
|:-----|:-------|:-------------|
| 2026-08-02 | @MitchellDScott | Initial stub. Relocates `ConversionError` out of `matrix-design.md` §4.9.2 in response to review feedback identifying it as an undeclared shared dependency of `polynomial-design.md` and `tensor-design.md`. Deliberately minimal pending `/cr-research`. |
