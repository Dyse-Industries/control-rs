# Fixed-Point Scalar Type (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

Embedded control applications executing on integer microcontroller units (MCUs)
without floating-point hardware require deterministic fixed-point arithmetic
(ARM, 1996; Analog Devices, 2015). Software emulation of floating-point
operations incurs severe execution cycle penalties and substantial code bloat in
high-rate feedback loops (ARM, 1996).

The `src/math/fixed_num.rs` module introduces the canonical fixed-point scalar
type `Fixed<Repr, const SHIFT: i32>` (alongside the `Quantized<Repr, SHIFT>`
type alias), representing numbers in binary Q-format where each value is
$x = \text{raw} \cdot 2^{-\text{SHIFT}}$ with fixed quantization step
$\Delta = 2^{-\text{SHIFT}}$ (ARM, 1996; Spiteri, 2026a).

Key design features include:

- **Compile-Time Scale Parametrization (FR-1, FR-2)**: Zero runtime memory
  footprint overhead; `Fixed<Repr, SHIFT>` occupies the identical size and
  alignment as its underlying integer representation (`Repr`).
- **Exact Widening Product & Convergent Rounding (FR-4)**: Multiplications
  widen intermediate products to prevent precision loss, followed by
  round-ties-to-even (convergent rounding) rescaling and saturating narrowing
  (IEEE, 2019; AMD, 2024; MathWorks, 2026).
- **Total Saturating Arithmetic (FR-3, FR-5)**: Prevents catastrophic
  limit-cycle
  overflow oscillation in feedback control laws by clamping at representation
  extrema.
- **Representability-Gated Trait Realization (FR-6, FR-7)**: Explicitly gates
  implementations of `One`, `Scalar`, and `SaturatingInteger` to scales where
  unity and required algebraic constants are strictly representable.

---

### 2. Requirements

#### 2.1 Functional Requirements

- **FR-1 — Compile-Time Scale**: The scale exponent `SHIFT` is a const generic
  type parameter. It is not stored at runtime.
- **FR-2 — Binary Power-of-Two Scale**: A value with scale `SHIFT` is
  `raw · 2^(−SHIFT)`. Adjacent values differ by a constant `Δ = 2^(−SHIFT)`.
  Decimal (power-of-ten) scales are out of scope.
- **FR-3 — Total Saturating Arithmetic**: `Add`, `Sub`, `Mul` and `Neg` always
  return a
  value of the type. Overflow saturates to min or max; it does not wrap or
  panic. Overflow detection is provided via the `math::ops` `Try*` traits, which
  return `Result`.
- **FR-4 — Exact Product Rescale**: `Mul` forms the full product in a wider
  integer, then rescales to `SHIFT`. A same-width multiply is not used.
- **FR-5 — Scale Conversion**: `rescale` converts `Fixed<Repr, Q>` to
  `Fixed<Repr, R>` by a left shift of `R − Q` or a right shift of
  `Q − R`. It always returns a value of the destination type. Overflow
  saturates to min or max.
- **FR-6 — Numeric Trait Participation**: The type implements `Zero`, `One`,
  `Conjugate` (identity), `Scalar` with `Real = Self`, and
  `SaturatingInteger`, subject to FR-7. It does not implement `Float`,
  `Radical`, `Exponential`, or `Trig`.
- **FR-7 — Representable-Constant Gating**: A trait is implemented only when
  its properties hold. Do not implement `One` when `1` is not representable:
  then `1 * n ≠ n`. `SaturatingInteger` also requires `2`. The `SHIFT`
  bounds are §4.4.

#### 2.2 Non-Functional Requirements

- **NFR-1 — Single-Word Footprint**: `Fixed<Repr, SHIFT>` has the size
  and alignment of `Repr`. `SHIFT` occupies no storage.
- **NFR-2 — Zero-Cost Arithmetic**: Each operator compiles to the integer
  instructions a hand-scaled `Repr` implementation would emit. There is no
  runtime scale bookkeeping and no call trampoline.

#### 2.3 Constraints

- **C-1 — `no_std`**: `core` only. No allocation. No `std`.
- **C-2 — No New Dependencies**: No new crate dependencies. Width and scale
  use const generics and `math::num_types` already in this crate.
- **C-3 — Scale Range**: `SHIFT` is in `0..=BITS`, where `BITS` is the bit
  width of `Repr`.
- **C-4 — No Bare Primitive Arithmetic**: No operator uses a bare primitive
  `+`, `-`, or `*`. Arithmetic goes through `saturating_*` or `checked_*`.

---

### 3. Technical Overview

The fixed-point module is implemented in `src/math/fixed_num.rs`, exposed via
`pub mod fixed_num;` in `src/math/mod.rs`, and tested in
`src/math/tests/fixed_num_tests.rs`.

The module defines the primary struct `Fixed<Repr, const SHIFT: i32>` and the
canonical type alias `Quantized<Repr, SHIFT>`, supported by the sealed trait
`FixedRepr` that parameterizes widening and narrowing behaviors across primitive
integer widths.

```mermaid
classDiagram
    direction TB

    class FixedRepr {
        <<sealedtrait>>
        +BITS u32
        +Wide
        +widen(self) Wide
        +narrow_saturating(w: Wide) Self
    }

    class Fixed~Repr SHIFT~ {
        -raw Repr
        +DELTA Self
        +from_bits(raw: Repr) Self
        +to_bits(self) Repr
        +from_num~F~(val: F) Self
        +to_num~F~(self) F
        +rescale~R~(self) Fixed~ Repr, R~
    }

    class Zero {
        <<trait>>
        +ZERO Self
    }

    class One {
        <<trait>>
        +ONE Self
    }

    class Conjugate {
        <<trait>>
        +conj(self) Self
    }

    class Scalar {
        <<trait>>
        +Real = Self
        +re(self) Real
        +abs2(self) Real
    }

    class SaturatingInteger {
        <<trait>>
        +MAX Self
        +MIN Self
        +MIN_POSITIVE Self
        +TWO Self
    }

    FixedRepr <.. Fixed: Repr bound
    Zero <|.. Fixed
    One <|.. Fixed: SHIFT <= BITS-2 (signed)
    Conjugate <|.. Fixed
    Scalar <|.. Fixed: SHIFT <= BITS-2 (signed)
    SaturatingInteger <|.. Fixed: SHIFT <= BITS-3 (signed)
```

_Figure 1: `Fixed<Repr, SHIFT>` architecture and numeric trait realizations.
Trait realization dashed lines indicate representability gates enforced at
compile time via const assertions. `Float`, `Radical`, `Exponential` and `Trig`
are absent by
FR-6._

---

### 4. Architecture

#### 4.1 Type Representation & Encapsulation

```rust
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Fixed<Repr, const SHIFT: i32> {
    raw: Repr,
}

/// Canonical type alias for downstream numerical models.
pub type Quantized<Repr, const SHIFT: i32> = Fixed<Repr, SHIFT>;
```

The underlying field `raw` is private to preserve scale invariants. Values are
instantiated via explicit raw integer conversions (`from_bits`, `to_bits`) or
host-side float conversions (`from_num`, `to_num`). Const generic validation
ensures `SHIFT` remains within valid
bounds ($0 \le \text{SHIFT} \le \text{BITS}$).

#### 4.2 Sealed Representation Trait (`FixedRepr`)

The sealed trait `FixedRepr` parameterizes primitive integer widths and provides
doubled-width widening and saturating narrowing:

```rust
pub trait FixedRepr: Copy + Eq + Ord + Sized + 'static {
    const BITS: u32;
    type Wide: Copy + Eq + Ord + Sized + 'static;

    fn widen(self) -> Self::Wide;
    fn narrow_saturating(val: Self::Wide) -> Self;
}
```

The trait is implemented for signed (`i8`, `i16`, `i32`, `i64`) and unsigned
(`u8`, `u16`, `u32`, `u64`) primitives:

- `i8` $\to$ `i16`, `i16` $\to$ `i32`, `i32` $\to$ `i64`, `i64` $\to$ `i128`
- `u8` $\to$ `u16`, `u16` $\to$ `u32`, `u32` $\to$ `u64`, `u64` $\to$ `u128`

#### 4.3 Arithmetic Operations

##### Addition, Subtraction, and Negation

Same-scale values operate directly on underlying integers, saturating at
representation bounds (FR-3):

- $\text{Add}(a, b) = \text{saturating\_add}(a_{\text{raw}}, b_{\text{raw}})$
- $\text{Sub}(a, b) = \text{saturating\_sub}(a_{\text{raw}}, b_{\text{raw}})$
- $\text{Neg}(a) = \text{saturating\_neg}(a_{\text{raw}})$

Because operands share identical scaling factors, no rescaling is required (ARM,
1996).

##### Widening Multiplication

Multiplying two numbers with scale factor $2^{-\text{SHIFT}}$ produces an
intermediate product with scale $2^{-2\text{SHIFT}}$ (ARM, 1996). To prevent
overflow and retain precision prior to rescaling, multiplication executes across
four steps:

```mermaid
flowchart LR
    A["a: Repr (Q)"] --> W1["widen() -> Wide"]
    B["b: Repr (Q)"] --> W2["widen() -> Wide"]
    W1 --> M["Wide Product (2Q)"]
    W2 --> M
    M --> R["Convergent Rescale: + (1 << (SHIFT-1)) >> SHIFT"]
    R --> N["narrow_saturating()"]
    N --> C["c: Repr (Q)"]
```

_Figure 2: Four-step widening multiplication path ensuring full intermediate
precision prior to convergent rounding and saturating narrowing._

##### Rescaling & Convergent Rounding

Right-shifting the widened product discards fractional bits. Narrowing applies
**round-ties-to-even** (convergent rounding) to eliminate systematic bias (IEEE,
2019; AMD, 2024; MathWorks, 2026; Spiteri, 2026b). The rescale kernel adds
the half-LSB rounding bias $1 \ll (\text{SHIFT} - 1)$ prior to arithmetic
right-shift, followed by saturating narrowing into the destination width.

#### 4.4 Representability Gating (FR-7)

For a signed integer `Repr` of bit width $n$ and scale exponent $\text{SHIFT}$,
representable values span:
$$\text{MIN} = -\frac{2^{n-1}}{2^{\text{SHIFT}}}, \quad \text{MAX} = \frac{2^{n-1} - 1}{2^{\text{SHIFT}}}$$

Associated constants and trait bounds are gated as follows:

| Constant                          | Raw Value                | Signed Gate                          | Unsigned Gate                        |
|:----------------------------------|:-------------------------|:-------------------------------------|:-------------------------------------|
| `ZERO`                            | `0`                      | $0 \le \text{SHIFT} \le \text{BITS}$ | $0 \le \text{SHIFT} \le \text{BITS}$ |
| `DELTA` / `MIN_POSITIVE`          | `1`                      | $0 \le \text{SHIFT} \le \text{BITS}$ | $0 \le \text{SHIFT} \le \text{BITS}$ |
| `MIN`, `MAX`                      | `Repr::MIN`, `Repr::MAX` | $0 \le \text{SHIFT} \le \text{BITS}$ | $0 \le \text{SHIFT} \le \text{BITS}$ |
| `ONE` (gates `One` & `Scalar`)    | `1 << SHIFT`             | $\text{SHIFT} \le \text{BITS} - 2$   | $\text{SHIFT} \le \text{BITS} - 1$   |
| `TWO` (gates `SaturatingInteger`) | `1 << (SHIFT + 1)`       | $\text{SHIFT} \le \text{BITS} - 3$   | $\text{SHIFT} \le \text{BITS} - 2$   |

##### DSP Interchange Formats vs. Computational Scalars

Canonical DSP interchange formats (e.g. Q15 with $n=16, \text{SHIFT}=15$) span
$[-1.0, 1.0)$, where the maximum representable value
is $(2^{15}-1)/2^{15} \approx 0.999969$.
Because $1.0$ cannot be represented, Q15 implements `Zero` and `Conjugate`, but
withholds `One`, `Scalar`, and `SaturatingInteger`. Signals arriving in Q15
format
rescale into computation-capable
configurations ($\text{SHIFT} \le \text{BITS} - 2$)
before participating in generic linear algebra kernels.

#### 4.5 Numeric Trait Realization

- **`Conjugate`**: Implemented as the identity function (`conj(self) -> Self`).
- **`Scalar`**: Implemented for gated scales ($\text{SHIFT} \le \text{BITS} - 2$
  signed,
  $\text{SHIFT} \le \text{BITS} - 1$ unsigned), setting `Real = Self`,
  `re(self) = self`,
  `im(self) = ZERO`, and `abs2(self) = self * self`.
- **`AdditiveGroup` & `Signed`**: Implemented for signed representations (`i8`,
  `i16`, `i32`, `i64`).
- **`SaturatingInteger`**: Implemented for scales
  satisfying $\text{SHIFT} \le \text{BITS} - 3$ (signed)
  or $\text{SHIFT} \le \text{BITS} - 2$ (unsigned), providing `TWO`,
  `MIN_POSITIVE`, `MIN`, `MAX`.
- **Excluded Traits**: `Float`, `Radical`, `Exponential`, and `Trig` are
  explicitly
  withheld per FR-6.

#### 4.6 Standard Format Aliases

Named aliases correspond to standard Q notation where $Qm.n$ designates a format
with $n$ fractional bits (and implicit sign bit in the Texas Instruments
notation),
yielding resolution $\Delta = 2^{-n}$ (Wikipedia, 2026; secondary,
uncorroborated):

```rust
pub type Q7 = Fixed<i8, 7>;
pub type Q15 = Fixed<i16, 15>;
pub type Q31 = Fixed<i32, 31>;
pub type Q63 = Fixed<i64, 63>;

pub type UQ7 = Fixed<u8, 7>;
pub type UQ15 = Fixed<u16, 15>;
pub type UQ31 = Fixed<u32, 31>;
pub type UQ63 = Fixed<u64, 63>;
```

#### 4.7 File Impact & Repository Placement

| File Path                                          | Description of Changes                                                                                              |
|:---------------------------------------------------|:--------------------------------------------------------------------------------------------------------------------|
| [`fixed_num.rs`](../../src/math/fixed_num.rs)      | New module: `Fixed<Repr, SHIFT>`, `Quantized` alias, `FixedRepr` sealed trait, operators, trait impls.              |
| [`mod.rs`](../../src/math/mod.rs)                  | Register `pub mod <br/><br/><br/><br/><br/><br/><br/>fixed_num;` and re-export `Fixed`, `Quantized`, and Q-aliases. |
| [`tests/fixed_num_tests.rs`](../../src/math/tests) | Comprehensive unit,<br/><br/><br/> <br/><br/><br/><br/>proptest, and `compile_fail` test suites.                    |

---

### 5. Alternatives

1. **Depend on the reference `fixed` crate**:
    - _Considered_: Taking `FixedI8`…`FixedI128`/`FixedU8`…`FixedU128`
      directly (Spiteri, 2026a) instead of defining a type.
    - _Rejected_: It pulls `az` and `typenum` as normal dependencies
      (Spiteri, 2026a), against C-2, and `typenum` duplicates the type-level
      integer tower `num-types-design.md` already specifies. Its
      representation is also a family of twelve concrete types rather than
      one type generic over `Repr`, which does not compose with the crate's
      single-`T` generic kernels. The evidence base for this design is that
      crate's own documentation; the semantics are adopted, the dependency is
      not.
2. **Type-Level `Frac` Instead of a Const Generic**:
    - _Considered_: Parameterizing on a type-level unsigned in the reference
      crate's style, where the fractional-bit count is a `typenum` type
      bounded by a per-width trait such as `LeEqU32`, "implemented for all
      `Unsigned` integers ≤ 32" (Spiteri, 2026c).
    - _Rejected_: That bound encoding predates stable integer const generics
      and exists to express `f ≤ n` in the trait system. A const generic
      expresses the same bound as a const assertion (§4.4) without a second
      type parameter on every signature, and keeps `math::num_types`'s
      type-level tower reserved for dimensions, where it is load-bearing.
3. **Same-Width Multiply**:
    - _Considered_: Multiplying `raw` values directly and shifting, with no
      widening step.
    - _Rejected_: The product is in `2q`-form (ARM, 1996), so the exact
      result does not fit the representation and the high half is lost before
      the rescale can recover it. The alternative to widening is choosing `q`
      as the largest value for which intermediate calculations cannot
      overflow (ARM, 1996), which pushes the analysis onto every call site
      and costs fractional precision everywhere to protect one product.
4. **Runtime Scale Field**:
    - _Considered_: Storing the exponent beside the mantissa so one type
      covers every scale.
    - _Rejected_: An exponent held in a register and unknown at compile time
      is the definition of a floating-point number (ARM, 1996). It also
      breaks NFR-1 and moves every scale check to runtime.
5. **Type Naming (`Fixed` vs. `Quantized`)**:
    - _Considered_: Exclusive naming as `Quantized` (model quantization) versus
      `Fixed` (representation).
    - _Decision_: Adopt `Fixed<Repr, const SHIFT: i32>` as the canonical
      struct name matching arithmetic naming conventions, and export
      `pub type Quantized<Repr, SHIFT> = Fixed<Repr, SHIFT>;` for drop-in
      compatibility with downstream model and tensor specifications.
6. **Decimal Fixed-Point**:
    - _Considered_: A power-of-ten scale, so authored decimal constants are
      exact.
    - _Rejected_: Binary fractions such as `1/2^4` are exactly representable
      and decimal fractions such as `0.001 = 1/10^3` are not (Spiteri,
      2026a); a power-of-ten scale inverts that, replacing every shift with a
      multiply or divide by a power of ten. Control quantities originate at
      converters whose scale is binary.

Overflow policy is not re-litigated here. `num-traits-design.md`
Alternative 4 already evaluated expressing wrapping and saturating behavior
through wrapper types, the pattern the reference implementation uses for its
`Strict` and `Wrapping` structs (Spiteri, 2026a), and rejected it in favor
of method-level traits. This design inherits that decision.

---

### 6. Verification & Validation

#### 6.1 Verification

1. **Constant and Range Unit Tests** (`fixed_num_tests.rs`): `DELTA`
   equals `from_bits(1)` and its `f64` value equals `2^(−SHIFT)`; `MIN` and
   `MAX` equal `from_bits(Repr::MIN)`/`from_bits(Repr::MAX)` and match
   `−2^(n−1)/2^SHIFT` and `(2^(n−1) − 1)/2^SHIFT` (Spiteri, 2026b); `ONE`
   and `TWO` round-trip through `to_num` exactly at every gated `SHIFT`.
2. **Saturation Oracles**: `MAX + ONE == MAX`, `MIN - ONE == MIN`,
   `MAX * TWO == MAX` and `Neg` at `MIN` saturate rather than wrap or panic
   (FR-3). The `Try*` forms return the error arm on the same inputs.
3. **Product Exactness** (proptest, host): for random raw pairs, the
   §4.3 result equals the `f64` reference product rounded to the grid with
   round-ties-to-even, with error bounded by `DELTA/2`. This is the oracle
   that catches a lost high half (FR-4).
4. **Rescale Round-Trip** (proptest, host): `rescale` from `q` to `r` and
   back is the identity when `r ≥ q` and within `DELTA/2` when `r < q`
   (FR-5).
5. **Compile-Time Gates**: rustdoc `compile_fail` doctests on the module
   docs, matching the placement rule in `num-traits-design.md` §6.1.2:
   `SHIFT` outside C-3; `Fixed<i16, 15>: One`; `Fixed<i16, 15>:
   Scalar`; `Fixed<i16, 14>: SaturatingInteger`; `Fixed<i32, 16>:
   Float`. Positive markers assert `Scalar`, `Conjugate` and
   `SaturatingInteger` on gate-satisfying instantiations, and
   `AdditiveGroup`/`Signed` withheld from unsigned `Repr`.
6. **Footprint**: `size_of::<Fixed<Repr, SHIFT>>() ==
   size_of::<Repr>()` and equal alignment, for every `FixedRepr` width
   (NFR-1).
7. **HIL**: the §4.3 product and rescale paths are wrapped in
   `#[hil_suite]` and executed on an FPU-less target. Fixed-point exists
   because integer cores simulate floating-point operations in software
   (ARM, 1996); host execution cannot confirm that the emitted sequence is
   integer-only. The suite covers runtime arithmetic only, not the §6.1.5
   type-level gates.

#### 6.2 Validation

- **Generic Kernel Integration**: a `Matrix` and a `Tensor` instantiated
  over a gate-satisfying `Fixed` compile and run through the
  `T: Scalar` ring kernels of `subprograms-design.md` with no kernel change,
  confirming FR-6 delivers the drop-in property `tensor-design.md` §4.10
  assumes.
- **Interchange Path**: a Q15 sample stream converts through `rescale` into
  a `Scalar`-capable instantiation, runs a filter, and converts back,
  demonstrating the §4.4 interchange-versus-compute split on a realistic
  signal path.
- **Precision Comparison**: the same filter run in `Fixed` and in `f64`,
  with the error reported against the `f64` reference. This documents the
  precision cost the representation trades for integer-hardware execution
  (Analog Devices, 2015) rather than asserting equivalence.

---

### 7. Performance & Resource Considerations

`Fixed<Repr, SHIFT>` is a single-field struct over `Repr` and
monomorphizes to the bare integer (NFR-1). `Add`, `Sub` and `Neg` are one
saturating integer instruction. `Mul` is a widening multiply, a
round-ties-to-even rescale (half-LSB add then
shift per AMD, 2024) and a saturating narrow: more than a floating-point
multiply on a part with an FPU, and far less than the software floating-point
sequence an integer core would otherwise run (ARM, 1996).

---

### 8. Risks & Open Questions

1. **`SaturatingInteger` on Q15/Q31 Interchange Formats**:
   Canonical DSP interchange formats (such as Q15 and Q31) allocate all
   fractional bits such that $\text{SHIFT} = \text{BITS} - 1$, spanning
   $[−1.0, 1.0)$. Because unity ($1.0$) and $2.0$ are outside this span,
   `One`, `Scalar`, and `SaturatingInteger` are withheld at these scales.
   Signal processing workflows must explicitly `rescale` interchange values to
   computational scales ($\text{SHIFT} \le \text{BITS} - 2$) before generic
   kernel computation.
2. **Downstream Rescale Models**: Downstream tensor models specifying
   `Quantized<i8, 7>` on `Scalar`-bound operations must be updated to either
   use computation scales ($\text{SHIFT} \le 5$) or introduce explicit
   interchange-to-computation rescales.
3. **Definition Placement**: `Fixed<Repr, SHIFT>` is canonically placed in
   `src/math/fixed_num.rs` with `Quantized` re-exported, cleanly decoupling
   tensor crates from fixed-point representation internals.
4. **128-Bit Fixed-Point Scaling**: 128-bit fixed-point scalars are excluded
   because
   primitive widening requires 256-bit arithmetic; no control MCU use case
   currently requires this width.
5. **Negative Scale Bounds**: Negative `SHIFT` values (coarser than unity) are
   deferred pending specific hardware encoder sensor requirements.
6. **Proposals (Not in Evidence)**:
    - Sign-aware convergent rounding on signed `Wide` products (§4.3).
    - Accumulator narrowing rules in hardware DSP extensions (ARM CMSIS-DSP,
      RISC-V NMSIS).

---

### 9. Development Plan

| Phase                                    | Description                                                                                                                                                               | Estimated Effort |
|:-----------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:----------------:|
| **Phase 1: Representation & Core Type**  | Implement `Fixed<Repr, SHIFT>`, `Quantized` alias, sealed `FixedRepr` trait for `i8`–`i64` / `u8`–`u64`, `from_bits`/`to_bits`/`from_num`/`to_num`, and Q-format aliases. |      Medium      |
| **Phase 2: Total Saturating Arithmetic** | Implement saturating `Add`, `Sub`, `Neg`, widening `Mul` with convergent rounding, `rescale`, and fallible `Try*` ops.                                                    |      Medium      |
| **Phase 3: Numeric Trait Integration**   | Implement `Zero`, `One`, `Conjugate`, `Scalar`, `Signed`, and `SaturatingInteger` with representability compile-time assertions.                                          |      Medium      |
| **Phase 4: Verification Suite**          | Implement unit tests, proptest oracles, `compile_fail` doctests, memory footprint assertions, and `#[hil_suite]` verification.                                            |      Medium      |
| **Phase 5: Downstream Model Validation** | Validate generic instantiation in matrix and tensor kernels across control toolboxes.                                                                                     |      Small       |

---

### 10. Revision History

| Revision | Date            | Author          | Description                                                                                                                                                            |
|:---------|:----------------|:----------------|:-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | August 24, 2026 | @MitchellDScott | Initial specification for `math::fixed_num`: `Fixed<Repr, SHIFT>` representation, sealed `FixedRepr` trait, widening multiplication, and representability gates.       |
| 1.1      | August 24, 2026 | @MitchellDScott | Grounded convergent rescaling in IEEE 754-2019, AMD, and MathWorks literature; verified representability gates; promoted status to Reviewed.                           |
| 1.2      | August 24, 2026 | @MitchellDScott | Architectural cleanup: established `Fixed` as primary struct with `Quantized` alias; completed Introduction; streamlined section hierarchy; purged revision narrative. |

---

## References

[1] Advanced RISC Machines Limited, "Fixed Point Arithmetic on the ARM,"
Advanced RISC Machines Limited, Cambridge, UK, Rep. no. ARM DAI 0033A, 1996.
[Online]. Available:
https://documentation-service.arm.com/static/5ed0fdc1ca06a95ce53f84b8.
Accessed: Aug. 12, 2026.

[2] Analog Devices, Inc., "Fixed-Point vs. Floating-Point Digital Signal
Processing," *Analog Devices Technical Articles*, 2015. [Online]. Available:
https://www.analog.com/en/resources/technical-articles/fixedpoint-vs-floatingpoint-dsp.html.
Accessed: Aug. 12, 2026.

[3] T. Spiteri, *fixed*: fixed-point numbers (Version 1.31.0). [Online].
Available: https://docs.rs/fixed/latest/fixed/. Accessed: Aug. 12, 2026.

[4] IEEE, "IEEE Standard for Floating-Point Arithmetic," Institute of
Electrical and Electronics Engineers, Standard IEEE Std 754-2019, 2019.
[Online]. Available: https://standards.ieee.org/standard/754-2019.html.
Accessed: Aug. 24, 2026.

[5] AMD, "Rounding," in *Complex Multiplier LogiCORE IP Product Guide*,
Advanced Micro Devices, Product Guide PG104, Version 6.0, 2024. [Online].
Available: https://docs.amd.com/r/en-US/pg104-cmpy/Rounding. Accessed: Aug.
24, 2026.

[6] The MathWorks, Inc., "Rounding Modes," *MATLAB & Simulink Documentation*,

2026. [Online]. Available:
      https://www.mathworks.com/help/fixedpoint/ug/rounding.html. Accessed: Aug.
      24, 2026.

[7] T. Spiteri, "FixedI32," in *fixed::FixedI32* (Version 1.31.0). [Online].
Available: https://docs.rs/fixed/latest/fixed/struct.FixedI32.html.
Accessed: Aug. 12, 2026.

[8] Wikipedia contributors, "Q (number format)," *Wikipedia*. [Online].
Available: https://en.wikipedia.org/wiki/Q_(number_format). Accessed: Aug.
12, 2026.

[9] T. Spiteri, "fixed::types::extra," in *fixed::types::extra* (Version
1.31.0). [Online]. Available:
https://docs.rs/fixed/latest/fixed/types/extra/index.html. Accessed: Aug.
12, 2026.
