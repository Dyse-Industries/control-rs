# Numeric Trait Hierarchy (num-traits)

![Date Badge](https://img.shields.io/badge/Date-October_5,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `num_traits` module provides a numerical abstraction designed around
**hardware behavior** rather than abstract mathematical theory. It enables
control algorithms to implement generic code over primitive numerical types,
while giving developers compile-time constraints and overflow behavior (
wrapping vs. saturating).

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Overflow Mode Disambiguation**: Primitive integer types explicitly
  partition overflow semantics into wrapping or saturating execution modes at
  compile time (num-traits, 2024a; num-traits, 2024b; Spiteri, 2026a).
- **FR-2 — Unsigned Primitive Ring Bound**: Unsigned integer primitives
  (`u8`, `u16`, `u32`, `u64`, `usize`) satisfy `Zero + One + Sub + Mul`.
  Linear-algebra kernels consume that bound via `T: Scalar`
  (`subprograms-design.md` FR-1); this document does not specify BLAS-loop
  monomorphization (num-traits, 2024a).
- **FR-3 — Reflexive & Complex Conjugation**: `Conjugate` is a `Scalar`
  super trait exposing `fn conj(self) -> Self`. Real scalars (integer
  primitives, `f32`, `f64`, `Quantized<Repr, SHIFT>`) implement it as the
  identity; `Complex<T>` implements it as imaginary-component negation. A
  value is real iff `self == self.conj()`.
- **FR-4 — Real Projection**: Every `Scalar` exposes
  `type Real: Scalar<Real = Self> + PartialOrd` plus `re()`, `im()`,
  `from_real()`, and `abs2()` (`re² + im²`, no square root; Proposal; not in
  evidence). Real types
  set `Real = Self`; `Complex<T>` sets `Real = T`.
- **FR-5 — Implementor Partition**: `Scalar`, `Float`, `Complex<T>`, and
  `Quantized` occupy distinct implementor sets; the partition is the §4.3
  table.
- **FR-6 — Total Arithmetic Contract**: Every `Scalar` implements
  `SaturatingAdd`, `SaturatingSub` and `SaturatingMul`; `Signed` adds
  `SaturatingNeg` and `Float` adds `SaturatingDiv`. Each operation is total:
  integer and `Quantized` results clamp to `[MIN, MAX]`, floats follow
  IEEE-754. Library arithmetic on a generic `T` calls these methods, never the
  `core::ops` operators, so no kernel panics or wraps on overflow for any
  implementor (§4.4). `clippy::arithmetic_side_effects` is `deny` with no
  suppression (num-traits, 2024b).
- **FR-7 — Multiply Accumulate**: An open trait `MulAcc` gives a type an
  accumulator type `Acc`, `to_acc(self) -> Acc`, `mac(acc, a, b) -> Acc`
  equal to $\text{acc} + a \cdot b$, and `from_acc(acc) -> Self`, which
  rounds and saturates once under the FR-6 contract. The crate implements
  it for primitives, `Quantized` (`fixed-num-design.md` FR-8) and
  `Complex<T>`; types outside the crate may implement it with
  target-specific arithmetic.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero-Cost Abstraction**: Trait calls, zero/one constants, and
  saturating/wrapping operations compile to direct primitive FPU/ALU
  instructions with zero runtime overhead or function call trampolines.

#### 2.3 Constraints

- **C-1 — `#![no_std]` Compatibility**: Numerical traits operate without
  standard library dependencies or dynamic allocation.

---

### 3. Technical Overview

```mermaid
classDiagram
    direction TB

    class Zero {
        <<trait>>
        +ZERO Self
        +is_zero() bool
        +zero() Self
    }

    class One {
        <<trait>>
        +ONE Self
        +is_one() bool
        +one() Self
    }

    class Conjugate {
        <<trait>>
        +conj() Self
    }

    class AdditiveGroup {
        <<trait>>
    }

    class Signed {
        <<trait>>
        +abs() Self
        +is_sign_negative() bool
        +is_sign_positive() bool
    }

    class Integer {
        <<trait>>
        +MAX Self
        +MIN Self
        +MIN_POSITIVE Self
        +TWO Self
    }

    class SaturatingInteger {
        <<trait>>
    }

    class Unsigned {
        <<markertrait>>
    }

    class Scalar {
        <<trait>>
        +Real Scalar
        +re() Real
        +im() Real
        +from_real(re: Real) Self
        +abs2() Real
        +clamp(min: Self, max: Self) Self
        +signum() Self
    }

    class Radical {
        <<trait>>
        +sqrt() Self
        +hypot(y: Self) Self
    }

    class Exponential {
        <<trait>>
        +E Self
        +exp() Self
        +ln() Self
        +log10() Self
        +pow(n: Self) Self
    }

    class Trig {
        <<trait>>
        +PI Self
        +sin() Self
        +cos() Self
        +tan() Self
        +asin() Self
        +acos() Self
        +atan() Self
    }

    class Float {
        <<trait>>
        +epsilon() Self
        +atan2(x: Self) Self
    }

    class Complex~T~ {
        +re T
        +im T
        +Real = T
    }

    Zero <|-- AdditiveGroup
    AdditiveGroup <|-- Signed
    Zero <|-- Integer
    One <|-- Integer
    Zero <|-- SaturatingInteger
    One <|-- SaturatingInteger
    Zero <|-- Scalar
    One <|-- Scalar
    Conjugate <|-- Scalar
    Scalar <|-- Float
    Signed <|-- Float
    Radical <|-- Float
    Exponential <|-- Float
    Trig <|-- Float
    Scalar <|.. Complex
    AdditiveGroup <|.. Complex
```

_Figure 1: UML hierarchy for the numeric trait tower. Solid arrows are
super trait bounds; dashed arrows are type realizations. `Complex<T>`
realizes `Scalar` (`Real = T`) and `AdditiveGroup`; it does not
realize `Float`, `Signed`, or the analytic traits. `Unsigned` is a `Sized`-only
marker with no super trait bounds._

---

### 4. Architecture

#### 4.1 Architectural Layers

1. **Identity Tier (`Zero`, `One`)**:
    - `Zero` requires `Clone + PartialEq + Add<Output = Self>`
      and associated constant `ZERO`.
    - `One` requires `Clone + PartialEq + Mul<Output = Self>`
      and associated constant `ONE`.
    - `PartialOrd` is not a super trait of either. `Complex<T>` therefore
      implements `Zero`/`One`/`Scalar` without a total or partial order.
    - `clamp` / `signum` on `Scalar` (Proposal; not in evidence) are bounded
      `where Self: PartialOrd`; kernels clip through `T::Real`.
2. **Conjugation Tier (`Conjugate`)**:
    - `Conjugate` exposes `fn conj(self) -> Self` and is a `Scalar`
      super trait (FR-3).
    - Identity on every real `Scalar`; imaginary negation on `Complex<T>`
      (`T: Neg`).
    - Realness predicate: `self == self.conj()`. Hermitian diagonal writes
      use this test; they do not require an `.im()` check on a separate
      complex trait.
3. **Subtraction Tier (`AdditiveGroup`)**:
    - Opt-in trait binding `Zero` and `Sub<Output = Self>`.
    - Marks types where `a - b` is underflow-free for ordinary values (
      signed integers, floats, `Complex<T>` when `T: AdditiveGroup`).
4. **Hardware Integer Tier (`Integer`, `SaturatingInteger`, `Unsigned`)**:
    - `Integer` and `SaturatingInteger` expose the wrap and saturate ALU
      behaviors respectively (num-traits, 2024a; num-traits, 2024b), plus range
      constants (`MAX`, `MIN`,
      `MIN_POSITIVE`, `TWO`). Both are implemented by every integer
      primitive, signed and unsigned.
    - `Quantized<Repr, SHIFT>` implements `SaturatingInteger` when `Repr`
      does; Q-format DSP types (`q15`, `q31`) follow the same saturating
      contract (systemonchips.com, 2025).
    - `Unsigned` remains a `Sized`-only marker distinguishing unsigned
      primitives from the `AdditiveGroup`/`Signed`/`Float` branch.
5. **Scalar Tier (`Scalar`)**:
    - `Scalar` requires `Zero + One + Sub + Mul + SaturatingAdd +
      SaturatingSub + SaturatingMul + Conjugate` and adds `clamp()` /
      `signum()` (each `where Self: PartialOrd`), without `Div` (FR-2, FR-6,
      Alternative 3).
    - Associated type `Real: Scalar<Real = Self> + PartialOrd` with `re()`,
      `im()`, `from_real()`, and `abs2()` (FR-4). `im()` returns `Real::ZERO`
      on real types. `abs2()` is `re² + im²` (equals `self.saturating_mul(self)`
      on reals).
      Ordering and clipping of complex values go through `T::Real`.
    - `signum()`'s negative branch is unreachable for unsigned types.
6. **Signed & Analytic Tier (`Signed`, `Float`, `Radical`, `Exponential`,
   `Trig`)**:
    - `Signed` extends `AdditiveGroup` with
      `Neg<Output = Self> + SaturatingNeg + PartialOrd`,
      providing `abs()` and sign predicates. Withheld from `Complex<T>`: BLAS
      1-norms and `Iamax` project through `T::Real` / `abs2()`, not
      `Signed::abs` returning `Complex`.
    - `Float` requires
      `Scalar + Signed + Radical + Exponential + Trig + Div<Output = Self> +
      SaturatingDiv` plus `epsilon()`. Implemented only by `f32` and `f64`
      (FR-5). Division on `Float` follows IEEE-754 (`inf`/`NaN`).
    - `Radical` requires `SaturatingAdd + SaturatingMul`; the default
      `hypot()` is `(x.saturating_mul(x)).saturating_add(y.saturating_mul(y)).sqrt()`.
    - `Complex<T>` implements `Div` and `SaturatingDiv` when
      `T: SaturatingDiv` without implementing `Float`. Integer and `Quantized`
      division use `SaturatingDiv` (total) or `TryDiv` (error-reporting).

#### 4.2 Macro Code Generation

To prevent boilerplate duplication across primitive types, implementation blocks
are generated using internal declarative macros:

- `impl_int!`: Emits `Zero`, `One`, `Integer`, and `SaturatingInteger`
  implementations for all integer primitives (signed and unsigned).
- `impl_additive_group!`: Emits `AdditiveGroup` and `Signed` implementations
  for signed integer primitives and `f32`/`f64`.
- `impl_scalar!`: Emits `Conjugate` (identity) and `Scalar` (`Real = Self`,
  `re`/`from_real` identity, `im` returns `ZERO`, `abs2` is `self * self`)
  for every integer primitive and for `f32`/`f64`. Emitting `Conjugate`
  strictly within `impl_scalar!` prevents duplicate trait implementations.
- `impl_float!`: Emits `Float`, `Radical`, `Exponential` and `Trig`
  implementations for `f32` and `f64`. `Float: Scalar` is already satisfied
  by the preceding `impl_scalar!` invocation.
- `saturating_div_signed_impl!` / `saturating_div_unsigned_impl!` /
  `saturating_float_impl!` (`math::ops`): emit `SaturatingDiv` (and
  `SaturatingNeg` for signed integers) with the §4.4 semantics, and all five
  saturating traits for `f32`/`f64` as the IEEE-754 operators.
- `Complex<T>`: handwritten `Conjugate` (imaginary negation,
  `T: SaturatingNeg`), `Scalar` (`Real = T`,
  `T: Scalar<Real = T> + SaturatingNeg + PartialOrd`), `AdditiveGroup` (when
  `T: AdditiveGroup`), the five saturating traits component-wise, and `Div`
  (when `T: SaturatingDiv`). The `core::ops` operators on `Complex<T>` are
  implemented with the component saturating methods. Does not receive
  `impl_float!`, `Signed`, `Radical`, `Exponential`, or `Trig`.
- `Quantized<Repr, SHIFT>`: implements `Scalar` / `Conjugate` (identity) in
  the quantized-scalar module, not via these macros. `Scalar` requires `1`
  to be representable at `SHIFT` (`fixed-num-design.md` FR-7): signed
  `SHIFT <= BITS - 2`, unsigned `SHIFT <= BITS - 1`. The interchange formats
  `Q7`, `Q15`, `Q31` and `Q63` are therefore not `Scalar`; `UQ7` and the
  other unsigned formats are.

#### 4.3 Implementor Partition (FR-5)

| Type                                                       | `Scalar` | `Real` | `Float` | `Integer` / `SaturatingInteger` | `AdditiveGroup` / `Signed` |
|:-----------------------------------------------------------|:--------:|:------:|:-------:|:-------------------------------:|:--------------------------:|
| signed integers                                            |   yes    | `Self` |   no    |              both               |            both            |
| unsigned integers                                          |   yes    | `Self` |   no    |              both               |          neither           |
| `f32`, `f64`                                               |   yes    | `Self` |   yes   |               no                |            both            |
| `Complex<T>` where `T: Scalar<Real = T> + SaturatingNeg`   |   yes    |  `T`   |   no    |               no                |    `AdditiveGroup` only    |
| `Quantized<Repr, SHIFT>` where `Repr: Scalar<Real = Repr>` and `1` is representable at `SHIFT` |   yes    | `Self` |   no    |    saturating when `Repr` is    |       follows `Repr`       |

`Div` is not a `Scalar` super trait. `Float` requires it. `Complex<T>`
implements `Div` when `T: SaturatingDiv`. Division kernels bound
`T: Scalar + SaturatingDiv`, which every row of the table satisfies.

#### 4.4 Total Arithmetic Contract (FR-6)

The saturating traits (`math::ops`) take both operands by reference and
return a value. Generic code uses the trait form `a.saturating_add(&b)`;
concrete integer code resolves to the inherent `a.saturating_add(b)`.

| Operation            | Signed integer                                             | Unsigned integer                  | `Quantized`                   | `f32`, `f64`      | `Complex<T>`                       |
|:---------------------|:-----------------------------------------------------------|:----------------------------------|:------------------------------|:------------------|:-----------------------------------|
| `saturating_add/sub` | clamp to `[MIN, MAX]`                                      | clamp to `[0, MAX]`               | clamp raw to `[MIN, MAX]`     | IEEE-754          | component-wise                     |
| `saturating_mul`     | clamp to `[MIN, MAX]`                                      | clamp to `[0, MAX]`               | widened product, clamp        | IEEE-754          | `(ac - bd, ad + bc)`, each clamped |
| `saturating_div`     | `x/0` is `MAX` (`x > 0`), `MIN` (`x < 0`), `0` (`x = 0`); `MIN/-1` is `MAX` | `x/0` is `MAX` (`x > 0`), `0/0` is `0` | `round((a << SHIFT) / b)` ties to even, clamp; `x/0` as the `Repr` integer | IEEE-754 (`±inf`, `NaN`) | `((ac + bd) + (bc - ad)i) / (c² + d²)`, each step saturating |
| `saturating_neg`     | `-MIN` is `MAX`                                            | not implemented (not `Signed`)    | `-MIN` is `MAX`               | sign-bit flip     | component-wise                     |

Consequences:

- Saturating integer addition is not associative once an intermediate
  clamps, so a reduction's result depends on summation order at the bounds.
  Kernels fix their loop order (`subprograms-design.md`), so results are
  deterministic per build.
- Within range every operation equals the corresponding operator, so float
  results are bit-identical to the operator form and integer results differ
  only where the operator would have panicked (debug) or wrapped (release).
- `core::ops` impls on library types (`Complex<T>`, `Fixed`, `Matrix`)
  delegate to the saturating methods, so operator syntax in user code carries
  the same contract.

#### 4.5 Multiply Accumulate (FR-7)

`MulAcc` is the accumulation primitive for recurrences such as filter and
controller realizations (`classical-control-design.md` §4.9). A chain
`from_acc(mac(mac(to_acc(c), a_1, b_1), a_2, b_2))` narrows once, however
many terms it holds, so the accumulator type decides both headroom and
rounding count:

```rust
pub trait MulAcc: Copy {
    type Acc: Copy;
    fn to_acc(self) -> Self::Acc;
    fn mac(acc: Self::Acc, a: Self, b: Self) -> Self::Acc;
    fn from_acc(acc: Self::Acc) -> Self;
}
```

| Implementor | `Acc` | `mac` | `from_acc` |
|:--|:--|:--|:--|
| `f32`, `f64` | `Self` | `saturating_mul` then `saturating_add` (two IEEE roundings) | identity |
| Signed and unsigned integers | doubled width (`i8` to `i16`, ..., `i64` to `i128`) | exact product, saturating add in `Acc` | clamp to `[MIN, MAX]` once |
| `Quantized` | `FixedRepr::Acc` at scale $2\,\text{SHIFT}$ | `fixed-num-design.md` FR-8 | one ties-to-even rescale, saturating narrow |
| `Complex<T>` (`T: Zero`, `T::Acc: SaturatingSub + Signed`) | `Complex<T::Acc>` | $\text{acc}_r + a_r b_r - a_i b_i$ via `T::mac` then `Acc` subtract of the exact $a_i b_i$ product; $\text{acc}_i + a_r b_i + a_i b_r$ as two `T::mac`. Unsigned `Acc` is excluded: saturating subtract would floor a negative real intermediate at zero. | component-wise `T::from_acc` |

The float implementation is the unfused operator pair. It adds no
dependency and runs at the cost of a multiply and an add on every target.
It also gives the same result on host and target provided the compiler
does not contract `a * b + c` into a fused operation; that contraction
policy is an assumption to verify (§8). A fused multiply-add computes
$(x \cdot a) + b$ with one rounding error and is specified by IEEE 754 as
`fusedMultiplyAdd` (Rust Project, 2026), but it is faster only where the
target has an `fma` instruction (Rust Project, 2026). The trait is open, so
that choice belongs to an accelerated type outside the crate, the pattern
`subprograms-design.md` §4.5 uses for BLAS backends: a newtype over `f32`
whose `mac` issues the target's fused instruction, shipped beside
`CmsisDspBlas` in `examples/subprograms/thumbv7em/` (§9). `MulAcc` is a
separate trait rather than a `Scalar` supertrait, so existing `Scalar`
implementors outside the crate do not break; consumers bound
`T: Scalar + MulAcc`.

---

### 5. Alternatives

1. **Full Abstract Algebra Taxonomy**:
    - _Considered_: Implementing a granular algebraic hierarchy matching formal
      abstract algebra, of the kind the `noether` crate ships (`Magma`,
      `Semigroup`, `Monoid`, `Group`, `Ring`, `Field`) (warlock-labs, 2025).
    - _Rejected_: Too complex for practical control systems engineering. Rust's
      trait solver overhead and complex bound signatures outweigh the benefits
      — `noether`'s own documentation cautions that "extensive use of dispatch
      ... may incur some runtime cost" (warlock-labs, 2025; secondary,
      uncorroborated claim), a risk this design avoids entirely by not
      building a comparably deep tower. The pragmatic tiering (`Zero`,
      `Conjugate`, `AdditiveGroup`, `Integer`, `SaturatingInteger`, `Scalar`,
      `Float`) provides the exact boundaries required by numerical algorithms.
2. **Blanket Derivation of `AdditiveGroup` from `Zero + Sub`**:
    - _Considered_: Adding
      `impl<T: Zero + Sub<Output = Self>> AdditiveGroup for T {}`.
    - _Rejected_: Standard library unsigned integers already implement
      `core::ops::Sub`. A blanket implementation would automatically grant
      `AdditiveGroup` to unsigned types, defeating the safety goal. Explicit
      per-type opt-in is required.
3. **Requiring `Div` on the Unified `Scalar` Trait**:
    - _Considered_: Giving `Scalar` a `Div<Output = Self>` bound directly, so
      one trait covers every arithmetic operator a control loop might need.
    - _Rejected_: Integer division is not total (`/0` panics, `i32::MIN / -1`
      overflows), so requiring it on every `Scalar` implementor — including
      plain signed integers and `Quantized` — reintroduces exactly the panic
      surface this hierarchy exists to remove. `Div` stays off `Scalar`.
      `Float` requires it (IEEE-754 `inf`/`NaN`). Division kernels bound
      `T: Scalar + SaturatingDiv` instead, which is total on every
      implementor (§4.4). `TryDiv` remains for callers that need the error.
4. **Wrapper-Type Semantics (`fixed`-crate Pattern) Instead of Method-Level
   Traits**:
    - _Considered_: Expressing wrapping/saturating behavior through a new type
      wrapper — analogous to the `fixed` crate's `Saturating<F>`, which
      "provides saturating arithmetic on fixed-point numbers" by overloading
      operators on the wrapper rather than exposing named methods on the
      underlying type (Spiteri, 2026b) — instead of `Integer`/
      `SaturatingInteger` method calls (`wrapping_add()`, `saturating_add()`)
      on the scalar type itself.
    - _Rejected_: Requiring callers to convert into and out of a wrapper type at
      each boundary adds friction the trait-method approach avoids.
5. **`ComplexField` / `RealField` Tower**:
    - _Considered_: A second trait pair (`ComplexField` with associated
      `RealField`) in the style of nalgebra/simba, separate from `Scalar` (
      Crozet, 2020).
    - _Rejected_: `Scalar::Real` plus `Conjugate` as a `Scalar` super trait
      gives ring kernels a single bound (`T: Scalar`) and projects norms,
      real α, and Givens cosine onto `T::Real` without a parallel algebra
      tower (Alternative 1).
6. **`Conjugate` as a Separate Bound, Not a `Scalar` Super trait**:
    - _Considered_: Leaving `Conjugate` independent so ring kernels write
      `T: Scalar + Conjugate`.
    - _Rejected_: Every `Scalar` that participates in `Trans::ConjTrans` or
      Hermitian reflection needs `conj`. On reals the operation is identity
      and monomorphizes away, so the extra bound adds noise without excluding
      any intended implementor.
7. **`Float` for `Complex<T>`**:
    - _Considered_: `impl<T: Float> Float for Complex<T>` so one `T: Float`
      bound covers real and complex analytic kernels.
    - _Rejected_: BLAS `Nrm2`/`Asum` return a real (`SCNRM2`/`DZNRM2`), not
      a complex. `Float` also pulls in `Trig`/`Exponential`/`epsilon` that
      complex LAPACK does not need on `T` itself. Analytic operations run on
      `T::Real`. `complex_num.rs` does not implement `Float`, `Signed`,
      `Radical`, `Trig`, or `Exponential` for `Complex<T>`; `Complex<T>` is
      `Scalar` with `Real = T`.
8. **Operators on Generic `T` with Lint Suppression or Wrapping Semantics**:
    - _Considered_: Keeping `a + b` on generic `T` and either suppressing
      `clippy::arithmetic_side_effects` per module or binding kernels to
      `WrappingAdd`/`WrappingMul`.
    - _Rejected_: Suppression leaves `Matrix<i32>` kernels that panic in debug
      and wrap in release, a silent sign flip in a control loop. Wrapping is
      total but maps overflow to the opposite bound, the worse failure for a
      feedback signal. Saturation bounds the error and matches Q-format DSP
      practice (systemonchips.com, 2025). Rewriting operators as
      `core::ops::Add::add` calls silences the lint without changing the
      semantics and is also rejected.

---

### 6. Verification & Validation

#### 6.1 Verification

Each condition below is discharged by the named targets; the numbered
list after the table describes the checks in prose.

| Condition | Requirement | Method | Target | Criterion |
|:----------|:------------|:-------|:-------|:----------|
| VC-1.1 | FR-1 | `libtest` | `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_add_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_add_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_add_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_add_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_sub_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_sub_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_sub_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_sub_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_mul_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_mul_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_mul_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_wrapping_mul_unsigned_integers` | Integer primitives expose wrapping and saturating addition, subtraction and multiplication as separate methods, each correct at `MIN` and `MAX` |
| VC-2.1 | FR-2 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_unsigned_integer_markers` | Unsigned primitives implement `Zero`, `One`, `Unsigned`, `Integer` and `SaturatingInteger` and withhold `AdditiveGroup` |
| VC-3.1 | FR-3 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_conjugate_and_scalar_projections` | `conj` is the identity on real scalars and negates the imaginary part of `Complex<T>` |
| VC-4.1 | FR-4 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_conjugate_and_scalar_projections`, `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_scalar_properties` | `re`, `im`, `from_real` and `abs2` agree with the component definitions for real scalars and `Complex<T>` |
| VC-5.1 | FR-5 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_scalar_markers`, `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_num_trait_unsigned_integer_markers` | `Scalar` holds for every integer and float primitive and the partition marker bounds compile |
| VC-5.2 | FR-5 | `inspection` | — | Negative bounds (`Complex<f64>: Float`, `Complex<u8>: Scalar`, unsigned `AdditiveGroup`) are `compile_fail` doctests on the `num_traits` module docs |
| VC-6.1 | FR-6 | `libtest` | `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_add_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_add_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_sub_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_sub_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_mul_signed_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_mul_unsigned_integers`, `control_rs::math::tests::op_tests::op_test_suite::test_ops_saturating_div_neg_signed_integers` | Saturating add, subtract, multiply, divide and negate clamp to `[MIN, MAX]` without panicking or wrapping |
| VC-6.2 | FR-6 | `review` | — | `clippy::arithmetic_side_effects` is `deny` and library arithmetic on a generic `T` calls the saturating methods, not `core::ops` operators |
| VC-7.1 | FR-7 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_float_unfused` | Float `mac` equals the unfused `a * b + c` bit for bit |
| VC-7.2 | FR-7 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_integer_exact_chain` | An integer chain whose intermediate sum leaves `[MIN, MAX]` and returns inside it yields the exact final sum, clamped once |
| VC-7.3 | FR-7 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_complex_formula` | `Complex<T>` components equal the four-term formula exactly |
| VC-7.4 | FR-7 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_complex_min_imag`, `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_complex_fixed_full_scale` | `Complex<T>` `mac` keeps the exact doubled-width product when an imaginary operand is `MIN` or `MAX`, for integer and `Fixed` components |
| VC-7.5 | FR-7 | `libtest` | `control_rs::math::tests::num_trait_tests::num_trait_test_suite::test_mul_acc_open_trait` | An external type implementing `MulAcc` compiles against a generic `T: Scalar + MulAcc` consumer |
| VC-8.1 | NFR-1 | `review` | — | Trait calls and constants compile to direct primitive instructions without trampolines |
| VC-9.1 | C-1 | `review` | — | `num_traits` uses `core` only, without heap allocation or `std` |

Verification ensures structural correctness and trait compliance across all
target environments:

1. **Unit Testing & Hardware-Boundary Verification**:
    - Test suites (`num_trait_tests.rs`) validate identity elements, wrapping
      behavior at `MAX`/`MIN` and saturation behavior at `MAX`/`MIN` across
      primitive types and `Complex<T>`.
2. **Compile-Time Marker Assertions**:
    - Negative trait bounds (`unsigned: AdditiveGroup`, `Complex<f64>: Float`,
      `Complex<u8>: Scalar`) are rustdoc `compile_fail` doctests on
      `num_traits` module docs. They do not live in `#[ets_suite]` / `cfg(test)`
      modules; rustdoc does not extract doctests from those.
    - Marker tests verify at compile time that `Scalar` is implemented for
      every integer and float primitive (signed and unsigned), that
      `Unsigned + Integer + SaturatingInteger` hold on unsigned primitives
      including `u128` and `usize`,
      and that `AdditiveGroup`/`Signed` are withheld from unsigned types (
      positive checks in `num_trait_tests.rs`).
    - `Complex<T>: Scalar` with `Real = T` when `T: Neg`; `compile_fail` that
      `Complex<f64>` does not implement `Float` or `Signed`, and that
      `Complex<u8>` does not implement `Scalar`.
    - Identity conjugation: `x.conj() == x` for every real primitive;
      `Complex::new(a, b).conj() == Complex::new(a, -b)`.
    - Real projection: `T::Real = T` and `x.re() == x`, `x.im() == T::ZERO`
      for real primitives; `Complex<T>::Real = T`, method `re()`/`im()` match
      the `re`/`im` fields, and
      `z.abs2() == z.re * z.re + z.im * z.im`.
    - `Quantized` / `Fixed` marker `compile_fail`s are specified in
      `fixed-num-design.md` §6.1.5 (`Q15: One`, `Q15: Scalar`,
      `Fixed<i16, 14>: SaturatingInteger` as trait bounds). They are no
      longer deferred to a tensor scalar type.
    - `Complex<T>` does not implement `PartialOrd`. A rustdoc `compile_fail`
      pins that bound. `clamp` / `signum` stay on `T: Scalar + PartialOrd`;
      `src/math/subprograms.rs` and `src/math/dsp.rs` clip through `T::Real`
      (`abs2`, `re`).
3. **Multiply accumulate (FR-7)**:
    - Floats: `from_acc(mac(to_acc(c), a, b))` equals `a * b + c` bit for bit.
    - Integers: a chain whose intermediate sum leaves `[MIN, MAX]` but whose
      final sum returns inside it yields the exact final sum, where a chain
      of `saturating_mul` and `saturating_add` would have clamped.
    - `Complex<T>`: components equal the four-term formula exactly.
    - An external newtype implementing `MulAcc` compiles against a generic
      `T: Scalar + MulAcc` consumer (openness).
4. **Host tests and ETS suite wrap**:
    - Unit tests within `num_traits` are wrapped with the `#[ets_suite]` proc
      macro infrastructure. The ETS wrap covers wrap/saturate **runtime**
      tests only; it does not verify marker absence or type-level bounds
      beyond ZST `size_of`.

#### 6.2 Validation

Validation confirms that high-level toolbox components integrate seamlessly with
the trait hierarchy:

- **Numerical Assertion Integration**: `assert_almost_eq!` and
  `assert_not_almost_eq!` macros operate seamlessly over `T: Float`.
- **DSP & Linear Algebra Integration**: FFT in `dsp.rs` is scoped to
  `T: Float` (trigonometric twiddle factors). BLAS subprograms
  (`subprograms.rs`) validate compile-time ergonomics over `T: Scalar`
  (integers, floats, `Complex<T>`, and later `Quantized`). Field kernels
  (`Nrm2`, `Trsv`, LAPACK) bound `T: Scalar + SaturatingDiv` with
  `T::Real: Radical` / `Trig` as required, not `T: Float` as a stand-in
  for complex.

---

### 7. Performance & Resource Considerations

The `math::num_traits` hierarchy incurs **zero runtime performance overhead**
and **zero memory footprint**:

- **Static Monomorphization**: All trait method calls and constant accesses are
  resolved statically by the Rust compiler.
- **Zero Memory Allocation**: Marker traits (`AdditiveGroup`, `Unsigned`)
  carry no fields or dynamic dispatch tables.
- **Stack & Bare-Metal Friendly**: Operations execute inline without stack frame
  expansion or heap interaction, adhering to strict bare-metal constraints (2–8
  kB stack limits).

---

### 8. Risks & Open Questions

1. **No Order on `Complex<T>`**: `Zero`/`One` do not require `PartialOrd`.
   `Complex<T>` is unordered. Callers that need comparison or clipping bind
   `T: Scalar + PartialOrd` or project through `T::Real`.
2. **Associated Type and Super trait Migration**: Adding `Conjugate` as a
   `Scalar` super trait and `type Real` on `Scalar` is a breaking change for
   existing `T: Scalar` impls (including the shipped `impl<T: Scalar> Scalar
for Complex<T>`). Every implementor must name `Real` and provide
   projections. Call sites that used `T: Float` to accept `Complex<T>` must
   re-bind to `T: Scalar` (ring) or `T: Scalar + SaturatingDiv` with
   `T::Real: …` (field).
3. **`SafeDiv`/`NonZero<T>` Is Not Yet Specified**: This design keeps `Div`
   off `Scalar`; division kernels use `SaturatingDiv` (total) and callers
   that must detect `/0` use `TryDiv` (`math::ops`). A future `NonZero<T>`-gated `SafeDiv` for
   validate-once/divide-many hot loops is out of scope and
   needs its own design pass. Generic `NonZero<T>` itself is stable (Rust
   stabilized `generic_nonzero` after RFC 2307 replaced a single generic
   wrapper with twelve concrete per-primitive types) (Reitermarkus, 2024; RFC
   2307, 2018), but its `Zeroable`/`ZeroablePrimitive` sealing was adopted
   specifically because "it is unclear what happens ... when `T` is some type
   other than a raw pointer or a primitive integer" (RFC 2307, 2018) — a
   future `SafeDiv` needs its own answer for custom scalar types, since it
   cannot rely on `core::num::NonZero<T>` covering them.
4. **Evolution of `const fn` Traits**: When Rust stabilizes `const_trait_impl`,
   associated trait functions (for example, `is_zero()`) can be made `const fn`,
   expanding compile-time evaluation capabilities. As of the 2025H1 Rust
   Project Goals, "the compiler now has a promising implementation of const
   traits ... [but] the feature is still firmly in experimental territory:
   there has never been an RFC describing its syntax," with stabilization
   itself still a stretch goal (Scherer, 2025).
5. **`Complex<T>` trait impls**: `complex_num.rs` implements `Scalar`
   (`Real = T`), `Conjugate`, `AdditiveGroup`, and inherent methods. It does
   not implement `Float`, `Signed`, `Radical`, `Trig`, or `Exponential`
   (FR-5, Alternative 7).
6. **Float contraction (FR-7)**: Host and target agree bit for bit for the
   default float `mac` only if `rustc` never contracts `a * b + c` into a
   fused operation. The evidence base does not state the policy; a
   host-versus-ETS comparison of the §6.1.3 float case checks it.
7. **Fused accelerated types (FR-7)**: A fused `mac` rounds once and differs
   from the default by up to one rounding per term (Rust Project, 2026).
   Cross-check tolerances that compare an accelerated type against the
   host reference must allow that difference. A newtype must also
   implement every `Scalar` supertrait to reach generic consumers; the
   example states that cost.
8. **Const-traits citation**: `documentation/math/research/num-traits.bib`
   contains `scherer2025` (2025H1 web address). Inline cite and [10] remain at
   (Scherer, 2025).

---

### 9. Development Plan

| Phase / Feature                           | Description                                                                                                                                                                       | Estimated Effort |
|:------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core hierarchy**               | `Zero`, `One`, `AdditiveGroup`, `Signed`, `Integer`, `SaturatingInteger`, `Float`, `Scalar` on primitives.                                                                        | Complete         |
| **Phase 2: `Conjugate` + `Scalar::Real`** | `Conjugate` supertrait of `Scalar`; `type Real` with `re`/`im`/`from_real`/`abs2`; `impl_scalar!` emits identity conjugation and `Real = Self`; `Float: Scalar`.                  | Complete         |
| **Phase 3: `Complex<T>` retraction**      | `Complex<T>: Scalar` (`Real = T`) + `Conjugate` + `AdditiveGroup` + `Div`; remove `Float`/`Signed`/`Radical`/`Trig`/`Exponential`.                                                | Complete         |
| **Phase 4: Call-site migration**          | Re-bound `subprograms.rs`, `dsp.rs`, `assert.rs`, and matrix decompositions that used `T: Float` as a complex stand-in.                                                           | Complete         |
| **Phase 5: Verification**                 | Marker tests and `compile_fail` doctests for FR-3–FR-5; `#[ets_suite]` wrap/saturate suite verified. `Quantized` / `Fixed` negative oracles live in `fixed-num-design.md` §6.1.5. | Complete         |
| **Phase 6: Multiply accumulate**          | `MulAcc` (FR-7) for integers, floats, `Complex<T>` and `Quantized` (`fixed-num-design.md` FR-8); §6.1.3 tests.                                                                       | Complete         |
| **Phase 7: Accelerated example type**     | Fused-`mac` `f32` newtype and a CMSIS-style `q31` accumulator type beside `CmsisDspBlas` in `examples/subprograms/thumbv7em/`, run under QEMU MPS2-AN500.                          | Planned          |

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                               |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 1, 2026  | @MitchellDScott | Initial draft introducing numeric trait hierarchy and verification standards.                                                             |
| 1.1      | August 2, 2026  | @MitchellDScott | Hardware-aligned hierarchy: replaced algebraic Ring/Field with `Zero`, `One`, `Integer`, `Float`, and `Scalar`.                           |
| 1.2      | August 22, 2026 | @MitchellDScott | Complex scalar support: added `Conjugate` trait, `Scalar::Real` projection, and retracted `Complex: Float` in favor of `Complex: Scalar`. |
| 1.3      | August 24, 2026 | @MitchellDScott | Comparison decoupling: dropped `PartialOrd` from `Zero`/`One` and `Complex<T>`, restricting ordering to `Signed` and `Scalar::Real`.      |
| 1.4      | August 24, 2026 | @MitchellDScott | Full implementation and verification of numeric traits and complex number primitives.                                                     |
| 1.5      | September 23, 2026 | @MitchellDScott | FR-6 total arithmetic contract: `Scalar` requires the saturating traits, added `SaturatingDiv`/`SaturatingNeg`, §4.4, Alternative 8. `Quantized` implements `SaturatingDiv` (`fixed-num-design.md` §4.3). |
| 1.6      | September 24, 2026 | @MitchellDScott | §4.2 and §4.3: `Quantized` is `Scalar` only where `1` is representable (`fixed-num-design.md` FR-7); `Q7`/`Q15`/`Q31`/`Q63` are not. |
| 2.0      | October 5, 2026 | @MitchellDScott | Adds FR-7 open `MulAcc` trait (§4.5, §6.1.3, §8.6, §8.7, Phases 6 and 7): unfused default for floats, single-narrowing accumulators for integers and `Quantized`, fused arithmetic through accelerated types outside the crate. |

---

## References

[1] rust-num, "WrappingAdd," in *num_traits::ops::wrapping* (Version 0.2.19),

2024. [Online]. Available:
      https://docs.rs/num-traits/latest/num_traits/ops/wrapping/trait.WrappingAdd.html.
      Accessed: Aug. 6, 2026.

[2] rust-num, "Saturating," in *num_traits::ops::saturating* (Version 0.2.19),

2024. [Online]. Available:
      https://docs.rs/num-traits/latest/num_traits/ops/saturating/trait.Saturating.html.
      Accessed: Aug. 6, 2026.

[3] T. Spiteri, *az* (Version 1.3.0), 2026. [Online]. Available:
https://docs.rs/az/latest/az/. Accessed: Aug. 6, 2026.

[4] systemonchips.com, "...and Correctly Using CMSIS-DSP Fixed-Point (Qx)
Functions," 2025. [Online]. Available:
https://www.systemonchips.com/and-correctly-using-cmsis-dsp-fixed-point-qx-functions/.
Accessed: Aug. 6, 2026.

[5] S. Crozet, "Switch to Simba and make the base and geometry modules mostly
SIMD AoSoA friendly (PR #713)," in *dimforge/nalgebra*, 2020. [Online].
Available: https://github.com/dimforge/nalgebra/pull/713. Accessed: Aug. 6,

2026.

[6] warlock-labs, *noether README* (Version 0.3.0), 2025. [Online]. Available:
https://github.com/warlock-labs/noether. Accessed: Aug. 6, 2026.

[7] T. Spiteri, *fixed::Saturating* (Version 1.31.0), 2026. [Online].
Available: https://docs.rs/fixed/latest/fixed/struct.Saturating.html.
Accessed: Aug. 6, 2026.

[8] M. Reitermarkus, "Tracking Issue for generic NonZero (issue #120257)," in
*rust-lang/rust*, 2024. [Online]. Available:
https://github.com/rust-lang/rust/issues/120257. Accessed: Aug. 6, 2026.

[9] Rust Project, "RFC 2307: Concrete NonZero Types," *Rust RFC Book*, 2018.
[Online]. Available:
https://rust-lang.github.io/rfcs/2307-concrete-nonzero-types.html. Accessed:
Aug. 6, 2026.

[10] O. Scherer, "Prepare const traits for stabilization," *Rust Project Goals
(2025H1)*, 2025. [Online]. Available:
https://rust-lang.github.io/rust-project-goals/2025h1/const-trait.html.
Accessed: Aug. 6, 2026.

[11] The Rust Project Developers, "f32::mul_add," *The Rust Standard
Library* (Version 1.99.0). [Online]. Available:
https://doc.rust-lang.org/std/primitive.f32.html#method.mul_add. Accessed:
Oct. 5, 2026.
