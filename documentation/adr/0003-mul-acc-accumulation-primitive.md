# ADR-0003: Accumulate recurrences through the `MulAcc` trait

![Date Badge](https://img.shields.io/badge/Date-October_16,_2026-blue)
![Status Badge](https://img.shields.io/badge/ADR%20Status-Proposed-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

## Context

Control recurrences (filter sections, controller integrators, matrix
products) sum many products and narrow the result once. A chain of
`saturating_mul` and `saturating_add` on a fixed-point or integer type rounds
and clamps every term, so a partial sum that leaves the range clamps a result
that would have returned inside it, and a `Fixed` section output rounds up to
five times where a DSP core rounds once. The `Scalar` contract
(`num-traits-design.md` FR-6) makes every operation total but gives a
recurrence no way to ask for a wider accumulator. Every numeric model and
every later toolbox (`modern_control`, `sysid`, observers) shares this
question. Which trait carries an accumulator, and where does it live?

---

## Decision

Define an open trait `MulAcc` in `math::num_traits`, with an accumulator type
`Acc`, `to_acc`, `mac` and `from_acc`, and require recurrences to accumulate
through it so that each output narrows once (`num-traits-design.md` FR-7).
`MulAcc` is a separate trait, not a `Scalar` supertrait, and consumers bound
`T: Scalar + MulAcc`. The crate implements it for `f32`, `f64`, the integers
`i8` to `i64` and `u8` to `u64`, `Fixed<Repr, SHIFT>` (`fixed-num-design.md`
FR-8) and `Complex<T>`; a type outside the crate may implement it with
target-specific arithmetic.

Two behavior changes ride with it, because the exact-product contract
needs them: `Complex::conj` negates the imaginary part with `SaturatingNeg`,
and the signed-integer `AdditiveGroup::abs` uses `saturating_abs`. Both
replace a panic in debug builds and a wrap in release builds at `MIN` with a
saturated result, as FR-6 requires.

---

## Consequences

- Good: a recurrence rounds once per output on `Fixed` and clamps only at
  its output, and the same generic code runs on host floats and target
  fixed point.
- Good: an accelerated type (fused `f32`, CMSIS-style `q31`) enters through
  the trait without changing the crate.
- Good: existing `Scalar` implementors outside the crate keep compiling.
- Bad: generic consumers carry a second bound, and a type must implement
  `MulAcc` to reach a recurrence.
- Bad: `Complex::conj` now requires `SaturatingNeg` instead of `Neg`, so a
  component type with `Neg` alone no longer compiles.
- Follow-up: `MulAcc` for `i128`, `u128`, `isize` and `usize`, which are
  `Scalar` but cannot yet accumulate, owned by `num-traits-design.md` §8.
- Follow-up: the accelerated example types, owned by `num-traits-design.md`
  Phase 7.

---

## Rejected Options

- A `Scalar::mul_add` method: it rounds every term, which is the problem.
- A `Scalar` supertrait: it breaks every `Scalar` implementor outside the
  crate and forces an accumulator onto types that never accumulate.
- A fused `libm` multiply-add as the float default: software fusion is slower
  than an unfused multiply and add on targets without an `fma` instruction.

---

## References

[1] `control-rs`, "Numeric Trait Hierarchy," `documentation/math/num-traits-design.md`,
FR-6, FR-7 and §4.5.

[2] `control-rs`, "Fixed-Point Scalar Type," `documentation/math/fixed-num-design.md`,
FR-8.

[3] `control-rs`, "Classical Control Toolbox," `documentation/control-toolboxes/classical-control-design.md`,
C-8 and §4.9.
