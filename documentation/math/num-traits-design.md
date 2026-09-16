# Numeric Trait Hierarchy (Design Document)

![Date Badge](https://img.shields.io/badge/Date-September_10,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-brightgreen)
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

- **FR-1 — Overflow mode disambiguation**: Primitive integer types explicitly
  partition overflow into wrapping or saturating execution at compile time.
- **FR-2 — Unsigned primitive ring bound**: Unsigned integer primitives
  satisfy a ring bound consumed by linear-algebra kernels. This document does
  not specify BLAS-loop monomorphization.
- **FR-3 — Reflexive and complex conjugation**: Every scalar has a conjugate.
  Real scalars implement it as the identity; a value is real iff it equals its
  conjugate.
- **FR-4 — Real component projection**: Every scalar exposes a real part, an
  imaginary part, and a squared modulus without a square root. Real types are
  their own real component.
- **FR-5 — Disjoint implementor sets**: Statically partition general scalars,
  floats, complex numbers, and fixed-point numbers into mutually exclusive
  implementor sets, preventing field-only operations from binding to ring types
  at compile time.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Zero-cost abstraction**: Trait calls, zero/one constants, and
  saturating/wrapping operations compile to direct primitive FPU/ALU
  instructions with zero runtime overhead or function call trampolines.

#### 2.3 Constraints

- **C-1 — `#![no_std]` compatibility**: Numerical traits operate without
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
    - `clamp` / `signum` on `Scalar` are bounded
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
      behaviors respectively [1], [2], plus range
      constants (`MAX`, `MIN`,
      `MIN_POSITIVE`, `TWO`). Both are implemented by every integer
      primitive, signed and unsigned.
    - `Quantized<Repr, SHIFT>` implements `SaturatingInteger` when `Repr`
      does; Q-format DSP types (`q15`, `q31`) follow the same saturating
      contract [4].
    - `Unsigned` remains a `Sized`-only marker distinguishing unsigned
      primitives from the `AdditiveGroup`/`Signed`/`Float` branch.
5. **Scalar Tier (`Scalar`)**:
    - `Scalar` requires `Zero + One + Sub + Mul + Conjugate` and adds
      `clamp()` / `signum()` (each `where Self: PartialOrd`), without `Div`
      (FR-2, Alternative 3).
    - Associated type `Real: Scalar<Real = Self> + PartialOrd` with `re()`,
      `im()`, `from_real()`, and `abs2()` (FR-4). `im()` returns `Real::ZERO`
      on real types. `abs2()` is `re² + im²` (equals `self * self` on reals).
      Ordering and clipping of complex values go through `T::Real`.
    - `signum()`'s negative branch is unreachable for unsigned types.
6. **Signed & Analytic Tier (`Signed`, `Float`, `Radical`, `Exponential`,
   `Trig`)**:
    - `Signed` extends `AdditiveGroup` with `Neg<Output = Self> + PartialOrd`,
      providing `abs()` and sign predicates. Withheld from `Complex<T>`: BLAS
      1-norms and `Iamax` project through `T::Real` / `abs2()`, not
      `Signed::abs` returning `Complex`.
    - `Float` requires
      `Scalar + Signed + Radical + Exponential + Trig + Div<Output = Self>` plus
      `epsilon()`. Implemented only by `f32` and
      `f64` (FR-5). Division on `Float` follows IEEE-754 (`inf`/`NaN`).
    - `Complex<T>` implements `Div` when `T: Div` without implementing
      `Float`. Integer and `Quantized` division remain on `TryDiv`.

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
- `Complex<T>`: handwritten `Conjugate` (imaginary negation, `T: Neg`),
  `Scalar` (`Real = T`, `T: Scalar<Real = T> + Neg`), `AdditiveGroup` (when
  `T: AdditiveGroup`), and `Div` (when `T: Div`). Does not receive
  `impl_float!`, `Signed`, `Radical`, `Exponential`, or `Trig`.
- `Quantized<Repr, SHIFT>`: implements `Scalar` / `Conjugate` (identity) in
  the quantized-scalar module, not via these macros.

#### 4.3 Implementor Partition (FR-5)

| Type                                                       | `Scalar` | `Real` | `Float` | `Integer` / `SaturatingInteger` | `AdditiveGroup` / `Signed` |
|:-----------------------------------------------------------|:--------:|:------:|:-------:|:-------------------------------:|:--------------------------:|
| signed integers                                            |   yes    | `Self` |   no    |              both               |            both            |
| unsigned integers                                          |   yes    | `Self` |   no    |              both               |          neither           |
| `f32`, `f64`                                               |   yes    | `Self` |   yes   |               no                |            both            |
| `Complex<T>` where `T: Scalar<Real = T> + Neg`             |   yes    |  `T`   |   no    |               no                |    `AdditiveGroup` only    |
| `Quantized<Repr, SHIFT>` where `Repr: Scalar<Real = Repr>` |   yes    | `Self` |   no    |    saturating when `Repr` is    |       follows `Repr`       |

`Div` is not a `Scalar` super trait. `Float` requires it. `Complex<T>`
implements `Div` when `T: Div`. Integer and `Quantized` division stay on
`TryDiv`.

---

### 5. Alternatives

1. **Full Abstract Algebra Taxonomy**:
    - _Considered_: Implementing a granular algebraic hierarchy matching formal
      abstract algebra, of the kind the `noether` crate ships (`Magma`,
      `Semigroup`, `Monoid`, `Group`, `Ring`, `Field`) [5].
    - _Rejected_: Too complex for practical control systems engineering. Rust's
      trait solver overhead and complex bound signatures outweigh the benefits
      — `noether`'s own documentation cautions that "extensive use of dispatch
      ... may incur some runtime cost" [5], a risk this design avoids entirely
      by not building a comparably deep tower. The pragmatic tiering (`Zero`,
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
      `Float` requires it (IEEE-754 `inf`/`NaN`). `Complex<T>` implements
      `Div` when `T: Div` without implementing `Float`. Integer and
      `Quantized` division remain on `TryDiv`.
4. **Wrapper-Type Semantics (`fixed`-crate Pattern) Instead of Method-Level
   Traits**:
    - _Considered_: Expressing wrapping/saturating behavior through a new type
      wrapper — analogous to the `fixed` crate's `Saturating<F>`, which
      "provides saturating arithmetic on fixed-point numbers" by overloading
      operators on the wrapper rather than exposing named methods on the
      underlying type [6] — instead of `Integer`/
      `SaturatingInteger` method calls (`wrapping_add()`, `saturating_add()`)
      on the scalar type itself.
    - _Rejected_: Requiring callers to convert into and out of a wrapper type at
      each boundary adds friction the trait-method approach avoids.
5. **`ComplexField` / `RealField` Tower**:
    - _Considered_: A second trait pair (`ComplexField` with associated
      `RealField`) in the style of nalgebra/simba, separate from `Scalar` [7].
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

---

### 6. Verification & Validation

#### 6.1 Approach

The implementation must produce evidence that `num_traits` imposes zero
runtime overhead and zero memory footprint; that integer overflow modes are
strictly disambiguated between wrapping and saturating paths; that conjugation
and real projection satisfy exact algebraic identities; and that illegal type
trait bounds fail at compile time.

| Method | Mechanism |
|:-------|:----------|
| Compile-time shape check | Static trait bounds and rustdoc `compile_fail` doctests |
| Requirements-based test | `#[test]` unit tests covering identity constants, boundary wrapping/saturation, and projections |
| Property-based test | `proptest` over conjugation involution, real projection, and complex modulus squared |
| Static analysis | `cargo clippy-ci`, source inspection for direct ALU lowering and zero trampolines |
| Resource usage evaluation | `size_of` assertions verifying zero-sized marker types |
| On-target execution | `#[ets_suite]` target execution on bare-metal MCU targets for runtime arithmetic |

Target: 95% statement coverage of `src/math/num_traits.rs`, measured via `cargo coverage`.
Excluded: Unreachable panic paths in const assertions and debug formatting strings.

1. **Numerical Assertion Integration**: `assert_almost_eq!` and `assert_not_almost_eq!` macros operate seamlessly over `T: Float`.
2. **DSP & Linear Algebra Integration**: FFT kernels in `dsp.rs` bind to `T: Float` for trigonometric factors, while Level 1–3 BLAS subprograms instantiate cleanly over `T: Scalar`.

#### 6.2 Acceptance

| Claim | Oracle | Measure | Bound |
|:------|:-------|:--------|:------|
| Identity elements | Algebraic ring definitions | `ZERO + x == x`, `ONE * x == x` | $0$, bit-identical |
| Real conjugation identity | Conjugate definition | `x.conj() == x` for real primitives | $0$, bit-identical |
| Complex conjugation involution | Involutive property | `z.conj().conj() == z` | $0$, bit-identical |
| Real component projections | Projection definitions | `z.re()` and `z.im()` match component values | $0$, bit-identical |
| Squared modulus evaluation | Exact norm formula | `z.abs2() == z.re * z.re + z.im * z.im` | $0$, bit-identical |
| Negative trait bounds | Forbidden trait implementations | rustdoc `compile_fail` doctests | Compilation fails on `unsigned: AdditiveGroup`, `Complex: Float` |
| Overflow mode execution | Integer bounds | Wrapping arithmetic wraps mod $2^n$; saturating clamps | $0$, bit-identical |
| Zero-size footprint | Type system specification | `size_of::<AdditiveGroup>()`, `size_of::<Unsigned>()` | Exactly $0$ bytes |

#### 6.3 Limits

- **Unchecked division without divide-by-zero checks**: A future `NonZero<T>`-gated `SafeDiv` trait is deferred and not verified.
- **Partial ordering over `Complex<T>`**: Complex numbers do not implement `PartialOrd`; ordering operations on complex fields are intentionally rejected at compile time.
- **Floating-point overflow trapping**: Float primitives follow standard IEEE 754 infinity and NaN semantics rather than returning custom error enums.

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
   re-bind to `T: Scalar` (ring) or `T: Scalar + Div` with `T::Real: …`
   (field).
3. **`SafeDiv`/`NonZero<T>` Is Not Yet Specified**: This design keeps `Div`
   off `Scalar` and defers integer/`Quantized` division to `TryDiv`
   (`math::ops`). A future `NonZero<T>`-gated `SafeDiv` for
   validate-once/divide-many hot loops is out of scope and
   needs its own design pass. Generic `NonZero<T>` itself is stable (Rust
   stabilized `generic_nonzero` after RFC 2307 replaced a single generic
   wrapper with twelve concrete per-primitive types) [8], [9], but its
   `Zeroable`/`ZeroablePrimitive` sealing was adopted
   specifically because "it is unclear what happens ... when `T` is some type
   other than a raw pointer or a primitive integer" [9] — a
   future `SafeDiv` needs its own answer for custom scalar types, since it
   cannot rely on `core::num::NonZero<T>` covering them.
4. **Evolution of `const fn` Traits**: When Rust stabilizes `const_trait_impl`,
   associated trait functions (e.g., `is_zero()`) can be made `const fn`,
   expanding compile-time evaluation capabilities. As of the 2025H1 Rust
   Project Goals, "the compiler now has a promising implementation of const
   traits ... [but] the feature is still firmly in experimental territory:
   there has never been an RFC describing its syntax," with stabilization
   itself still a stretch goal [10].
5. **`Complex<T>` trait impls**: `complex_num.rs` implements `Scalar`
   (`Real = T`), `Conjugate`, `AdditiveGroup`, and inherent methods. It does
   not implement `Float`, `Signed`, `Radical`, `Trig`, or `Exponential`
   (FR-5, Alternative 7).

---

### 9. Development Plan

| Phase / Feature                           | Description                                                                                                                                                                       | Estimated Effort |
|:------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core hierarchy**               | `Zero`, `One`, `AdditiveGroup`, `Signed`, `Integer`, `SaturatingInteger`, `Float`, `Scalar` on primitives.                                                                        | Complete         |
| **Phase 2: `Conjugate` + `Scalar::Real`** | `Conjugate` supertrait of `Scalar`; `type Real` with `re`/`im`/`from_real`/`abs2`; `impl_scalar!` emits identity conjugation and `Real = Self`; `Float: Scalar`.                  | Complete         |
| **Phase 3: `Complex<T>` retraction**      | `Complex<T>: Scalar` (`Real = T`) + `Conjugate` + `AdditiveGroup` + `Div`; remove `Float`/`Signed`/`Radical`/`Trig`/`Exponential`.                                                | Complete         |
| **Phase 4: Call-site migration**          | Re-bound `subprograms.rs`, `dsp.rs`, `assert.rs`, and matrix decompositions that used `T: Float` as a complex stand-in.                                                           | Complete         |
| **Phase 5: Verification**                 | Marker tests and `compile_fail` doctests for FR-3–FR-5; `#[ets_suite]` wrap/saturate suite verified. `Quantized` / `Fixed` negative oracles live in `fixed-num-design.md` §6.3. | Complete         |

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                               |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 1, 2026  | @MitchellDScott | Initial draft introducing numeric trait hierarchy and verification standards.                                                             |
| 1.1      | August 2, 2026  | @MitchellDScott | Hardware-aligned hierarchy: replaced algebraic Ring/Field with `Zero`, `One`, `Integer`, `Float`, and `Scalar`.                           |
| 1.2      | August 22, 2026 | @MitchellDScott | Complex scalar support: added `Conjugate` trait, `Scalar::Real` projection, and retracted `Complex: Float` in favor of `Complex: Scalar`. |
| 1.3      | August 24, 2026 | @MitchellDScott | Comparison decoupling: dropped `PartialOrd` from `Zero`/`One` and `Complex<T>`, restricting ordering to `Signed` and `Scalar::Real`.      |
| 1.4      | August 24, 2026 | @MitchellDScott | Full implementation and verification of numeric traits and complex number primitives.                                                     |
| 1.5      | September 9, 2026 | @MitchellDScott | Hardening: sentence-case requirement titles, standard IEEE numeric citation callouts [1]–[10], and consecutively ordered references.        |
| 1.6      | September 10, 2026 | @MitchellDScott | Verification grounding & review closure: grounded §6.4 traceability locators to real tests in src/math/tests/num_trait_tests.rs and fixed §9 dead pointer to fixed-num §6.3. |
| 1.7      | September 15, 2026 | @MitchellDScott | Citations and type laundry removed from FR-1..FR-4 bodies. |
| 1.8      | September 16, 2026 | @MitchellDScott | Retired `vv-standards.md`: §6 authoring rules are `design-template.md` §6. |

---

## References

[1] rust-num, "WrappingAdd," in *num_traits::ops::wrapping* (Version 0.2.19), 2024. [Online]. Available: https://docs.rs/num-traits/latest/num_traits/ops/wrapping/trait.WrappingAdd.html. Accessed: Aug. 6, 2026.

[2] rust-num, "Saturating," in *num_traits::ops::saturating* (Version 0.2.19), 2024. [Online]. Available: https://docs.rs/num-traits/latest/num_traits/ops/saturating/trait.Saturating.html. Accessed: Aug. 6, 2026.

[3] T. Spiteri, *az* (Version 1.3.0), 2026. [Online]. Available: https://docs.rs/az/latest/az/. Accessed: Aug. 6, 2026.

[4] systemonchips.com, "...and Correctly Using CMSIS-DSP Fixed-Point (Qx) Functions," 2025. [Online]. Available: https://www.systemonchips.com/and-correctly-using-cmsis-dsp-fixed-point-qx-functions/. Accessed: Aug. 6, 2026.

[5] warlock-labs, *noether README* (Version 0.3.0), 2025. [Online]. Available: https://github.com/warlock-labs/noether. Accessed: Aug. 6, 2026.

[6] T. Spiteri, *fixed::Saturating* (Version 1.31.0), 2026. [Online]. Available: https://docs.rs/fixed/latest/fixed/struct.Saturating.html. Accessed: Aug. 6, 2026.

[7] S. Crozet, "Switch to Simba and make the base and geometry modules mostly SIMD AoSoA friendly (PR #713)," in *dimforge/nalgebra*, 2020. [Online]. Available: https://github.com/dimforge/nalgebra/pull/713. Accessed: Aug. 6, 2026.

[8] M. Reitermarkus, "Tracking Issue for generic NonZero (issue #120257)," in *rust-lang/rust*, 2024. [Online]. Available: https://github.com/rust-lang/rust/issues/120257. Accessed: Aug. 6, 2026.

[9] Rust Project, "RFC 2307: Concrete NonZero Types," *Rust RFC Book*, 2018. [Online]. Available: https://rust-lang.github.io/rfcs/2307-concrete-nonzero-types.html. Accessed: Aug. 6, 2026.

[10] O. Scherer, "Prepare const traits for stabilization," *Rust Project Goals (2025H1)*, 2025. [Online]. Available: https://rust-lang.github.io/rust-project-goals/2025h1/const-trait.html. Accessed: Aug. 6, 2026.
