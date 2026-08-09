# Crate-Wide Error Module (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Reviewed-yellow)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

`control-rs::math` has two main errors:

- `ArithmeticError` (`src/math/mod.rs`) — Produced by `math::ops`'s `Try*`
  traits and consumed by `complex_num.rs` and `assert.rs`.
- `ConversionError` — Returned by numerical models during conversion.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **Single Definition for Shared Error Types**: An error type consumed by
  more than one sibling module (`Matrix`, `Polynomial`, `Tensor`,
  `StateSpace`, `TransferFunction`) must be defined once, here, rather than
  inside whichever module happened to need it first.

#### 2.2 Non-Functional Requirements

- Follows the crate-wide `thiserror`-enum convention already established by
  `matrix-design.md` and `state-space-design.md` — no behavior
  change, relocation only.

---

### 3. Technical Overview

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

---

### 5. Alternatives

- **Fold `ConversionError` into `ArithmeticError` (considered, deferred)**:
  `ArithmeticError`'s existing variants (`DivisionByZero`, `Overflow`,
  `DomainViolation`, `PrecisionLoss`, `Underflow`, `Saturation`) are
  scalar-arithmetic-shaped, not layout/rank-shaped.

---

---

### 9. Development Plan

| Task / Feature                     | Description                                                                                                            | Estimated Effort |
|:-----------------------------------|:-----------------------------------------------------------------------------------------------------------------------|:-----------------|
| Step 1: Relocate `ConversionError` | Move the enum (§4) to its resolved module path; update `Matrix`/`Polynomial`/`Tensor` `TryFrom` impls to reference it. | 0.5 Day          |

---

### 10. Revision History

| Date       | Author          | Description                                                                                                                                                                                                                                                |
|:-----------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 2026-08-02 | @MitchellDScott | Initial stub. Relocates `ConversionError` out of `matrix-design.md` §4.9.2 in response to review feedback identifying it as an undeclared shared dependency of `polynomial-design.md` and `tensor-design.md`. Deliberately minimal pending `/cr-research`. |
| 2026-08-09 | @MitchellDScott | Review and corrections                                                                                                                                                                                                                                     |
