# Tensor Type & Low-Cost Inference (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_2,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `Tensor` type provides an N-dimensional array parameterized over a
compile-time shape layout and a pluggable memory storage backend, extending
`control-rs`'s linear algebra beyond the 2-dimensional `Matrix` type. It
shares its architectural foundation with `Matrix` — the same `Dim`/Peano
sizing system and `Storage<T, R, C>` trait hierarchy — so a `Tensor` and a
`Matrix` interoperate without copying.

Beyond static N-D storage, this revision extends `Tensor` toward two embedded
target applications identified during retroactive research: high-dimensional
static lookup tables (e.g., flight-control gain-scheduling tables) and
low-cost inference for small, pre-trained neural-network controllers and
classifiers on microcontrollers. Both applications are storage-and-evaluation
problems, not training problems — `Tensor` remains a storage/inference
primitive; on-device training and automatic differentiation are explicitly
out of scope.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Compile-Time Sizing**: Shape and rank are enforced entirely at
  compile time via the `TensorLayout` trait and the crate's `Dim` system.
- **FR-2 — Static Constructors**: Provide `zero`, `from_raw`, `from_fn`,
  `from_storage`, `from_slice`/`from_mut_slice` for stack, ROM-constant and
  zero-copy instantiation.
- **FR-3 — Core Arithmetic**: Operator overloading for element-wise `Add`,
  `Sub` and scalar `Mul`/`Div`.
- **FR-4 — Tensor Operations**: Zero-copy sub-tensor views, Einstein-summation
  contraction and axis permutation.
- **FR-5 — Grid Interpolation**: A point-query evaluation over a `TensorLayout`
  grid using piecewise-multilinear interpolation for gain-scheduled lookup
  tables.
- **FR-6 — Minimal Activation Function Support**: A small operator or trait for
  applying a pointwise nonlinear function (ReLU, tanh/sigmoid) between
  contracted layers.
- **FR-7 — Type-Level Quantized Scalar Representation**: A fixed-point scalar
  type encoding its quantization scale as const generic parameters, with no
  additional runtime metadata.
- **FR-8 — Type Conversions**: `TryFrom` conversions between `Tensor`, `Matrix`,
  and `Polynomial` for compatible ranks and sizes.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Deterministic Execution**: All operations execute within
  predictable, deterministic timeframes with no dynamic branching on unbounded
  loops.
- **NFR-2 — No Excessive Compile-Time Overhead**: `const fn` constructors and
  shape/quantization-scale encoding must not cause outsized compile-time
  increases relative to `Matrix`'s existing `Dim`/Peano system.
- **NFR-3 — Zero-Cost Quantization**: A quantized `Tensor` carries no runtime
  scale/zero-point fields; dequantization compiles down to a bit-shift or single
  integer division.

#### 2.3. Constraints

- **C-1 — No-Std Environment**: Must compile and run in `#![no_std]`
  environments.
- **C-2 — No Dynamic Allocation**: No heap allocator; all memory is static or
  stack-based.
- **C-3 — Memory Footprint**: Total element capacity across all dimensions is
  capped at 1,024 elements, keeping a single stack tensor under 4KB (`f32`).
- **C-4 — Type-Level Quantization Encoding**: Quantization-scale const generic
  parameters must be plain integers, since Rust disallows floating-point const
  generics.
- **C-5 — Out of Scope**: On-device training, automatic differentiation and
  trained-model import (TFLite/PyTorch Mobile/ONNX) are not addressed by this
  design.

---

### 3. Technical Overview

`Tensor<T, Layout: TensorLayout, S: Storage<T, Layout::Size, U1>>` is a
compile-time-shaped N-dimensional array built on the same generic-scalar,
decoupled-storage foundation as `Matrix`. This revision adds three
capabilities motivated by two concrete embedded use cases surfaced during
research: gain-scheduled flight-control lookup tables (grid interpolation)
and low-cost embedded neural-network inference (a minimal activation
primitive and a type-level quantized scalar representation). All three are
additive to the existing shape/storage/contraction architecture and require
no change to `Tensor`'s own generic signature.

---

### 4. Core Architecture

#### 4.1. Generics Foundation & Sizing

A `Tensor` wraps a storage backend `S: Storage<T, Layout::Size, U1>`.
Coordinate mapping is delegated to a type implementing `TensorLayout`:

```rust
pub trait TensorLayout {
    const RANK: usize;
    type Size: Dim;
    fn dims() -> &'static [usize];
}

pub struct Tensor<
    T,
    Layout: TensorLayout,
    S: Storage<T, Layout::Size, U1> = ArrayStorage<T, Layout::Size, U1>,
> {
    storage: S,
    _marker: PhantomData<Layout>,
}

// Type aliases for common storage backends
pub type ArrayTensor<T, Layout> =
Tensor<T, Layout, ArrayStorage<T, <Layout as TensorLayout>::Size, U1>>;
pub type ViewTensor<'a, T, Layout> =
Tensor<T, Layout, MatrixView<'a, T, <Layout as TensorLayout>::Size, U1>>;
pub type ViewMutTensor<'a, T, Layout> =
Tensor<T, Layout, MatrixViewMut<'a, T, <Layout as TensorLayout>::Size, U1>>;
```

Layout sizing verification uses type-level bounds (`Shape2D`, `Shape3D`, ...)
implemented via `DimMul`, guaranteeing shape correctness at compile time.

#### 4.2. Memory Layout & Storage Strategy

Multidimensional coordinates map to a flat 1D index using column-major
strides (first dimension varies fastest): $\text{flat\_index} = i_0 + i_1
\cdot S_1 + i_2 \cdot S_2 + \dots$, where $S_0 = 1$ and $S_m =
\prod_{j=0}^{m-1} D_j$.

- **Decoupled Physical Storage**: Flat element access dispatches through
  `Storage<T, Layout::Size, U1>` (`get_unchecked`, `ptr`, `ptr_mut`).
- **Matrix Interoperability**: Column-major layout matches `Matrix`'s
  layout, so 2D tensors interoperate with matrices without transposition or
  copying.
- **Flat Indexing Efficiency**: Element-wise operations iterate the
  underlying storage directly, bypassing multi-index arithmetic.

#### 4.3. Memory Representation & Slicing

`#[repr(C)]` guarantees a stable layout. Contiguous slice interfaces
(`as_slice`, `as_mut_slice`) are gated behind the `ContiguousStorage`
sub-traits, facilitating zero-copy casting to `&[T]` for BLAS-like
subprogram routing when the storage backend supports it.

#### 4.4. Instantiation & Constructors

- `zero() -> ArrayTensor<T, Layout>` (`T: Zero + Copy`): all-zero stack
  tensor.
- `from_raw(data: [T; Layout::Size::DIM]) -> ArrayTensor<T, Layout>`:
  direct `const fn` initialization from a flat array, the entry point for
  ROM-resident constant tensors.
- `from_fn<F>(f: F) -> ArrayTensor<T, Layout>` (`F: FnMut(&[usize]) -> T`):
  coordinate-mapped construction.
- `from_storage(storage: S) -> Tensor<T, Layout, S>`: wraps any custom
  storage backend.
- `from_slice`/`from_mut_slice`: non-allocating borrowed view construction.

`const fn` initialization on stable Rust requires `T: Zero + One` from
`crate::math::num_traits`, exposing `T::ZERO`/`T::ONE` as associated
constants.

#### 4.5. Operator Overloading

`Add`, `Sub` and scalar `Mul`/`Div` iterate directly over element storage
and return an owning `ArrayTensor<T, Layout>`.

#### 4.6. Core Operations

- **Zero-Copy Sub-Tensor Views**: `as_view`/`slice_inplace` extract
  non-allocating `MatrixView`-backed sub-tensors.
- **`contract_into`**: Einstein-summation contraction along type-level axes
  (`AxisSelf`, `AxisOther`) into a caller-provided result buffer. Axis
  existence and dimension agreement are enforced at compile time via
  `TensorContract`.
- **`contract_into_dynamic`**: Runtime-checked contraction fallback when
  axes are not known at compile time, returning `Result<(), ContractionError>`.
- **`permute`**: Axis permutation via stride transformation into an output
  tensor.

#### 4.7. Grid Interpolation

Motivated by gain-scheduled flight-control lookup tables (Koo & Sands,

2024) and the general N-D table-interpolation problem formalized by Weiser &
      Zarantonello (1988): a query at a fractional coordinate is evaluated as a
      weighted sum over the $2^{\text{RANK}}$ hypercube corner vertices
      surrounding it (the multilinear generalization of bilinear/trilinear
      interpolation), reading directly through `Storage::get_unchecked` with no
      intermediate allocation:

```rust
impl<T, Layout: TensorLayout, S> Tensor<T, Layout, S>
where
    S: Storage<T, Layout::Size, U1>,
    T: Float,
{
    pub fn interpolate(&self, coords: &[T; Layout::RANK]) -> T { /* ... */ }
}
```

`Float` here is `num-traits-design.md`'s current (pivoted) hierarchy, used
deliberately rather than the retired `Real` — this document already
follows the new naming crate-wide (§4.4, §4.9). Per that document's own
status, this remains provisional until it completes its own
`/cr-research`/`/cr-design-doc` pass (§7).

This is a distinct operation from `contract_into` (which contracts against
another whole tensor) and from `as_view` (which extracts a whole sub-tensor
region); `interpolate` evaluates a single fractional-coordinate point.
Evaluation cost is $O(2^{\text{RANK}})$ corner reads plus $O(\text{RANK})$
weight multiplications per corner. This is inexpensive for the target
lookup-table use case (rank 2-3, 4-8 corner evaluations); Weiser &
Zarantonello's alternative piecewise-linear simplex interpolant offers
better $O(\text{RANK} \log \text{RANK})$ asymptotic scaling but is not
adopted here (see §5.3) since no current target application exceeds rank
~4-5.

The out-of-grid-bounds query policy (clamp vs. error) is not finalized in
this revision — see §7.

#### 4.8. Minimal Activation Function Support

CMSIS-NN's reference sigmoid/tanh implementation is itself a table lookup
over a small int8 domain with linear interpolation between adjacent entries
(Lai, Suda, & Chandra, 2018) — the same table-plus-interpolation mechanism
as §4.7. Rather than implement activation functions as an unrelated
feature, a minimal `Activation` trait is layered directly on
`interpolate`:

```rust
pub trait Activation<T> {
    fn apply(&self, x: T) -> T;
}

pub struct Relu;                                  // trivial max(0, x), bypasses the table
pub struct TableActivation<T, const N: usize> {    // tanh/sigmoid-style, table-driven
    breakpoints: [T; N],
}
```

`Relu` requires no table. `TableActivation` reuses §4.7's interpolation
weight computation over a small precomputed 1D breakpoint array, following
CMSIS-NN's table range/resolution precedent. Applying an `Activation` to a
`Tensor` iterates `as_mut_slice()` in place; no new storage backend is
required. This keeps the operator surface minimal, per the goal of
supporting a hand-written small inference network without pulling in a full
ML framework.

#### 4.9. Type-Level Quantized Scalar Representation

Quantization is implemented as a property of the scalar type `T`, not of
`Tensor` itself — `Tensor<T, Layout, S>` already accepts any `T` satisfying
the crate's numeric traits, so a quantized tensor is simply
`Tensor<Quantized<Repr, SHIFT>, Layout, S>` with no structural change to
`Tensor` (see §5.1 for why this was chosen over threading quantization
parameters through `Tensor`'s own generics).

The default (v1) representation is power-of-two Q-format, matching the
`CMSIS-DSP` Q7/Q15/Q31 convention already validated for `Matrix`:

```rust
pub struct Quantized<Repr, const SHIFT: i32> {
    raw: Repr, // e.g. i8, i16, i32
}
```

`SHIFT` is a plain integer const generic — never a float — encoding the
value as `raw * 2^-SHIFT`, mirroring the integer-exponent pattern already
used by existing Rust fixed-point/decimal crates (`decimal-scaled`,
`primitive_fixed_point_decimal`) for type-level scale. `Quantized`
implements the crate's `Zero`/`One`/arithmetic `num_traits` so it plugs into
every existing generic `Tensor`/`Matrix` code path unchanged.

An affine variant (`AffineQuantized<Repr, const SHIFT: i32, const
ZERO_POINT: i32>`) is deferred future work for imported-model
interoperability (§5.2) — not part of this revision's default path.

#### 4.10. Interoperability & Conversions

`ConversionError` is defined once, canonically, in
[`error-design.md`](../math/error-design.md) — shared with
`Matrix` and `Polynomial`'s conversions — not restated here.

- **To `Matrix`**: `TryFrom<Tensor<T, Layout, S>> for Matrix<T, R, C, S>`
  when `Layout::RANK == 2` and dimensions match, preserving storage
  zero-copy. Returns `ConversionError::LayoutMismatch` otherwise.
- **To `Polynomial`**: `TryFrom<Tensor<T, Layout, S>> for Polynomial<T, N, S>`
  when `Layout::RANK == 1` and size matches, preserving storage zero-copy.
  Returns `ConversionError::LayoutMismatch` otherwise.

System-identification outputs (Volterra/NARX kernels, N4SID state-space
matrices) are, in practice, 2D (CP-decomposed factor matrices or state-space
matrices, not dense high-rank tensors — Batselier, Chen, & Wong, 2016), so
this existing 2D conversion path is the relevant interoperability surface
for that target application; no rank-N-specific system-identification API
is required.

#### 4.11. Error Handling & State Management

- **Compile-Time Constraints**: Dimension and rank mismatches in
  contraction, permutation or layout conversion are compile-time type
  errors.
- **Runtime Fallbacks**: Dynamic coordinate access returns
  `Option<&T>`/`Option<&mut T>` via `get`/`get_mut`; `contract_into_dynamic`
  returns `Result<(), ContractionError>`. Neither raises a panic.
- **Interpolation Bounds**: A query coordinate outside the grid's valid
  range is a genuinely open question (§7) — whether it clamps to the
  boundary, extrapolates or returns `Result<T, InterpolationError>` is not
  finalized in this revision.

#### 4.12. Structural Specializations & Future Extensions

- **Sparse Tensor Representations**, **ROM-Backed Static Storage
  Backends** and **Matrix-Free Operators** remain noted future extensions
  from the prior revision.
- **Model-Import Codegen (future work, not implemented in this revision)**:
  A future ahead-of-time tool could import a trained TFLite/ExecuTorch/ONNX
  model as `ArrayTensor::from_raw` constants plus a fixed
  `contract_into`/`Activation` call sequence, following MicroFlow's proven
  pattern of parsing a model at Rust compile time via procedural macro
  rather than shipping a runtime interpreter (Carnelos, Pasti, & Bellotto,
  2024 — chosen over an interpreter specifically to avoid the C++-toolchain
  dependency a runtime `.tflite` parser would require, inconsistent with
  this crate's `no_std` posture). Depends on external tooling components;
  not allocated a development-plan phase here (§8).

---

### 5. Alternatives

#### 5.1. Quantization Metadata: Scalar-Level Type vs. Tensor-Level Const Generics vs. Runtime Fields

- **Runtime Struct Fields (rejected)**: Storing scale/zero-point as ordinary
  struct fields alongside a `Tensor<i8, Layout, S>` was the first
  candidate considered, but conflicts with the requirement that quantized
  and unquantized tensors carry identical runtime footprint and that shape
  information already lives entirely at the type level.
- **`Tensor`-Level Const Generics (considered, rejected)**: Adding
  `const SHIFT: i32` directly to `Tensor<T, Layout, S, SHIFT>` was
  considered, but this conflates two orthogonal concerns — shape (`Layout`)
  and scalar representation (`SHIFT`) — inside one type and would require
  every existing `impl` block (`Add`, `contract_into`, conversions) to
  thread `SHIFT` arithmetic (e.g. rescaling when contracting two tensors
  with different shifts), a significant complexity increase to an
  already-generic type.
- **Scalar-Level Type (selected)**: `Quantized<Repr, SHIFT>` (§4.9) makes
  quantization a property of `T`, requiring zero change to `Tensor`'s
  existing generic signature — `Tensor<T, Layout, S>` already accepts any
  conforming `T`. This mirrors the `fixed` crate's `FixedI32<Frac>` pattern
  and keeps quantization orthogonal to shape.

In all three options, a raw floating-point scale as a const generic
parameter (`const SCALE: f32`) is not implementable on stable Rust
regardless of which type owns it, since float const generics remain
unstabilized (§2.3).

#### 5.2. Power-of-Two Shift (Q-Format) vs. Rational Scale Encoding

- **Power-of-Two `SHIFT` (selected default)**: Dequantization is a single
  bit-shift; matches the `CMSIS-DSP` Q7/Q15/Q31 convention already validated
  for `Matrix`, giving cross-type consistency. Real post-training
  quantization scales computed by external frameworks are not powers of
  two, so importing such a scale requires rounding to the nearest
  representable shift at import time — a bounded, one-time offline cost.
- **Rational `SCALE_NUM`/`SCALE_DENOM` (future extension)**: Recovers the
  imported model's exact scale, at the cost of one integer division per
  dequantization instead of a bit-shift — more expensive on FPU-less
  Cortex-M targets. Deferred; not required until the (future-work)
  model-import path needs full-precision affine scales.

#### 5.3. Multilinear (Hypercube) vs. Simplex/Triangulated Interpolation

- **Multilinear (selected)**: $O(2^{\text{RANK}})$ evaluation; matches
  CMSIS-NN's LUT-based activation precedent (§4.8) and Weiser &
  Zarantonello's formalization. Adequate for the target applications (rank
  2-3 lookup tables, rank-1 activation tables).
- **Simplex/Triangulated (rejected for this revision)**: $O(\text{RANK}
  \log \text{RANK})$ asymptotic scaling, better for rank $>$ ~5-6 per
  Weiser & Zarantonello (1988), but adds triangulation complexity not
  justified by any current target application. Noted as a future extension
  if a higher-rank use case emerges.

#### 5.4. Activation Function: Minimal Trait vs. Ad-Hoc Methods vs. Caller-Side Only

- **Caller-Side Only (rejected)**: Forces every consumer to hand-roll table
  generation and interpolation calls, duplicating logic across every
  inference example.
- **Ad-Hoc Per-Function Methods (considered)**: Inherent methods
  (`relu()`, `tanh_lut()`) are simple but do not compose with a future
  model-import path that must select an activation generically per layer.
- **Minimal Trait (selected)**: `Activation<T>` (§4.8), implemented per
  function and reusing the interpolation primitive for table-driven
  variants, keeps the surface minimal while composing with a future codegen
  path.

#### 5.5. External Tensor Libraries

As with `Matrix`, no existing crate combines `no_std`/no-alloc,
compile-time (const-generic) tensor shapes and interoperability with a
broader `Matrix`/`Polynomial` type system. MicroFlow is the closest analog
but is a single-model NN inference engine with no general `Tensor<T,
Layout, S>` abstraction; `tfmicro` requires a C++ toolchain. Building on
`control-rs`'s existing `Dim`/`Storage` foundation (already justified in
`matrix-design.md` §5.3) remains the only option meeting the audit-footprint
and `const fn`-on-stable-Rust requirements.

---

### 6. Verification & Validation

#### 6.1. Verification Strategy

1. **Compile-Time Verification**: Shape, rank and axis-index mismatches in
   contraction, permutation and layout conversion are rejected by the type
   system, matching `Matrix`'s `Dim`-based verification model.
2. **Property & Unit Testing** (`proptest`, Claessen & Hughes, 2000):
    - Stride index calculations and axis-permutation round trips (existing).
    - Contraction equivalence to matrix multiplication for 2D slices
      (existing).
    - Interpolation correctness: the interpolated value at an exact grid
      coordinate equals the stored value there; interpolated values between
      grid points match the closed-form weighted-sum definition (§4.7).
    - Quantized round-trip: `quantize` → `dequantize` error is bounded by
      half an LSB of the chosen `SHIFT`, mirroring the fixed-point precision
      bound already validated for `Matrix`.
3. **Golden-Value Regression** (per the crate's testing standards for
   invariant-heavy/numerical code): interpolation and quantized-inference
   results checked against SciPy/NumPy or hand-derived closed-form
   references for a small worked example (a 2D gain-scheduling table and a
   toy 2-layer quantized MLP).
4. **Host/Target Test Integration**: `cargo test` on host; QEMU
   cross-compilation across `ArrayStorage`, `MatrixView`, `MatrixViewMut`,
   and the new `Quantized<Repr, SHIFT>` scalar type.
5. **Benchmarks and Quality Reporting**: Contraction, interpolation and
   activation-function cycle counts benchmarked on ARM hardware; binary
   size checks confirm unused shape/activation variants are dead-code
   eliminated.

#### 6.2. Validation Strategy

##### Spatial Heat Distribution Update

A 3D tensor representing a 2D spatial grid over time is updated by
contracting it with a localized transition matrix (Kolda & Bader, 2009),
validating the existing multi-variable-spatial-grid target application:

```rust
pub fn update_thermal_grid<T, Sa, Sx, Sy>(
    transition_matrix: &Tensor<f32, Shape2D<U4, U4>, Sa>,
    current_grid: &Tensor<f32, Shape3D<U4, U2, U2>, Sx>,
    next_grid: &mut Tensor<f32, Shape3D<U4, U2, U2>, Sy>,
)
where
    Sa: Storage<f32, <Shape2D<U4, U4> as TensorLayout>::Size, U1>,
    Sx: Storage<f32, <Shape3D<U4, U2, U2> as TensorLayout>::Size, U1>,
    Sy: StorageMut<f32, <Shape3D<U4, U2, U2> as TensorLayout>::Size, U1>,
{
    transition_matrix.contract_into::<U1, U0, _, _, _, _>(current_grid, next_grid);
}
```

##### Flight-Control Gain-Scheduling Lookup Table

A 2D autopilot gain table indexed by Mach number and angle of attack (Koo &
Sands, 2024), validating §4.7's interpolation primitive against a
grid-in-between query:

```rust
pub fn scheduled_gain(
    gain_table: &ArrayTensor<f32, Shape2D<U8, U8>>,
    mach: f32,
    angle_of_attack: f32,
) -> f32 {
    gain_table.interpolate(&[mach, angle_of_attack])
}
```

##### Small Quantized Inference Network

A hand-written 2-layer dense network (weights as `ArrayTensor::from_raw`
constants, `contract_into` for the dense layer, `Relu` for the activation)
demonstrates the API surface is sufficient for a small embedded NN
controller end-to-end, without requiring the future-work import tooling
(§4.12):

```rust
pub fn predict<Sw1, Sb1, Sw2, Sb2>(
    input: &Tensor<Quantized<i8, 7>, Shape1D<U4>, impl Storage<Quantized<i8, 7>, U4, U1>>,
    w1: &Tensor<Quantized<i8, 7>, Shape2D<U8, U4>, Sw1>,
    b1: &Tensor<Quantized<i8, 7>, Shape1D<U8>, Sb1>,
    w2: &Tensor<Quantized<i8, 7>, Shape2D<U1, U8>, Sw2>,
    b2: &Tensor<Quantized<i8, 7>, Shape1D<U1>, Sb2>,
) -> Quantized<i8, 7> {
    // hidden = relu(w1 . input + b1); output = w2 . hidden + b2
    // (contract_into + Add + Relu::apply, per §4.6/§4.8)
    todo!()
}
```

---

### 7. Risks & Open Questions

- **Interpolation Bounds Policy**: Whether an out-of-grid query clamps,
  extrapolates or returns an error is not finalized (§4.11).
- **Q-Format vs. Rational Scale Selection**: This revision defaults to
  power-of-two `SHIFT` (§5.2); whether/when the rational variant is needed
  depends on the (future, out-of-scope) model-import path's accuracy
  requirements.
- **Activation Table Resolution**: `TableActivation`'s breakpoint count and
  domain range are not chosen in this revision; CMSIS-NN's `[-8, 8]`
  precedent is a starting point, but the accuracy/size trade-off needs a
  concrete target application before finalizing.
- **Const-Generic Compile-Time Complexity**: Adding `SHIFT`/rational const
  generics to the scalar type stacks on top of the compile-time-arithmetic
  concerns already flagged for `Matrix`'s `Dim`/Peano system
  (`matrix-design.md` §7); watch compile times as `Quantized<Repr, SHIFT>`
  usage grows.
- **Model-Import Tooling**: Recorded as future work only (§4.12); no risk
  analysis is performed in this revision since it is not implemented.
- **`num-traits-design.md` Dependency Is Provisional**: `T: Float` (§4.7)
  and the `Zero`/`One` bounds on `Quantized<Repr, SHIFT>` (§4.9) follow
  `num-traits-design.md`'s current hierarchy, which has not yet had its own
  `/cr-research` or `/cr-design-doc` pass and remains Draft pending one —
  matching the caveat `state-space-design.md` §9 and `matrix-design.md` §7
  already carry for the same dependency.

---

### 8. Development Plan

| Task / Feature                               | Description                                                                                                                                                                                                              | Estimated Effort |
|:---------------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Layout & Storage**           | `TensorLayout`, `Dim`-based sizing, `ArrayStorage`/view aliases, column-major stride mapping, `repr(C)` slicing.                                                                                                         | 2.0 Days         |
| **Phase 2: Element Ops & Contraction**       | Operator overloads, `contract_into`/`contract_into_dynamic`, `permute`, `as_view`/`slice_inplace`.                                                                                                                       | 2.5 Days         |
| **Phase 3: Grid Interpolation & Activation** | Multilinear `interpolate`, `Activation` trait, `Relu`, `TableActivation`.                                                                                                                                                | 2.5 Days         |
| **Phase 4: Quantized Scalar Type**           | `Quantized<Repr, SHIFT>` with full `Zero`/`One`/`Scalar` arithmetic `num_traits` impls (correct rounding/saturation semantics), quantize/dequantize, integration across existing generic `T` paths in `Tensor`/`Matrix`. | 3.5 Days         |
| **Phase 5: Verification & Interoperability** | `proptest` suites, golden-value regression against SciPy/NumPy references, ARM hardware benchmarks, `TryFrom` conversions to `Matrix`/`Polynomial`.                                                                      | 3.0 Days         |

---

### 9. References

1. **Kolda, T. G., & Bader, B. W. (2009).** Tensor Decompositions and
   Applications. _SIAM Review_, 51(3), 455–500. — Tensor contraction and
   decomposition computational complexity.
2. **Soro, S. (2021).** TinyML for Ubiquitous Edge AI. MITRE Technical
   Report MTR200519, _arXiv:2102.01255_. — Memory-footprint and
   inference-latency framing for microcontroller-class deployments.
3. **Warden, P., & Situnayake, D. (2019).** _TinyML: Machine Learning with
   TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers_.
   O'Reilly Media. — Weight/activation memory-budgeting guidelines.
4. **David, R., Duke, J., Jain, A., Janapa Reddi, V., Jeffries, N., Li, J.,
   Kreeger, N., Nappier, I., Natraj, M., Regev, S., Rhodes, R., Wang, T., &
   Warden, P. (2021).** TensorFlow Lite Micro: Embedded Machine Learning on
   TinyML Systems. _Proceedings of Machine Learning and Systems (MLSys) 3_.
   — Static arena memory-planning precedent for embedded inference engines.
5. **Lai, L., Suda, N., & Chandra, V. (2018).** CMSIS-NN: Efficient Neural
   Network Kernels for Arm Cortex-M CPUs. _arXiv:1801.06601_. —
   Table-lookup-plus-interpolation activation-function kernel design.
6. **Carnelos, M., Pasti, F., & Bellotto, N. (2024).** MicroFlow: An
   Efficient Rust-Based Inference Engine for TinyML. _arXiv:2409.19432_. —
   Ahead-of-time model-to-static-Rust-code compilation precedent (§4.12).
7. **Weiser, A., & Zarantonello, S. E. (1988).** A Note on Piecewise Linear
   and Multilinear Table Interpolation in Many Dimensions. _Mathematics of
   Computation_, 50(181), 189–196. — Canonical formalization of the
   multilinear grid-interpolation primitive (§4.7).
8. **Koo, S. M., & Sands, T. (2024).** Bilinear Interpolation of
   Three-Dimensional Gain-Scheduled Autopilots. _Sensors_, 24(1), 13. —
   Flight-control gain-scheduling lookup-table precedent.
9. **Batselier, K., Chen, Z., & Wong, N. (2016).** Tensor Network
   alternating linear scheme for MIMO Volterra system identification.
   _arXiv:1607.00127_. — System-identification tensor shape/rank precedent.
10. **Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P. (2020).**
    Integer Quantization for Deep Learning Inference: Principles and
    Empirical Evaluation. _arXiv:2004.09602_. — Affine int8 quantization
    convention used by mainstream trained-model export formats.
11. **Hennessy, J. L., & Patterson, D. A. (2017).** _Computer Architecture:
    A Quantitative Approach_ (6th ed.). Morgan Kaufmann. — Cache and
    memory-hierarchy modeling for stride-based N-D indexing.
12. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool
    for Random Testing of Haskell Programs. _ACM SIGPLAN Notices_, 35(9),
    268–279. — Property-based testing methodology (`proptest`).
13. **Rust Project Developers. (2024).** _The Rustonomicon: The Dark Arts
    of Advanced and Unsafe Rust Programming_. — Unsafe/aliasing rules for
    tensor slice views.
14. **ISO. (2018).** _ISO 26262-6:2018 Road vehicles — Functional safety —
    Part 6: Product development at the software level_.
15. **RTCA / EUROCAE. (2011).** _DO-178C: Software Considerations in
    Airborne Systems and Equipment Certification_.

---

### 10. Revision History

| Revision | Date           | Author          | Description                                                                                                        |
|:---------|:---------------|:----------------|:-------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026  | @MitchellDScott | Initial draft: `Storage` trait hierarchy, zero-copy views, stack `ArrayStorage` and inline citations.              |
| 1.1      | August 2, 2026 | @MitchellDScott | Full overhaul after a research pass: added grid interpolation, an `Activation` trait and a quantized scalar type.  |
| 1.2      | August 2, 2026 | @MitchellDScott | Relocated `ConversionError`; flagged `Float` dependency as provisional; revised development-plan estimates upward. |
