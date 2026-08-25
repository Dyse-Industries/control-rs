# Tensor Type & Low-Cost Inference (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_24,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Draft-orange)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `Tensor` type provides an N-dimensional array parameterized over a
compile-time shape layout and a pluggable memory storage backend, extending
`control-rs`'s linear algebra beyond the 2-dimensional `Matrix` type. It
shares its architectural foundation with `Matrix` — the same `Dim`
sizing system and the same contiguity contract — while operating
independently of the 2-D shape bound `DenseStorage<T>` imposes.

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

- **FR-1 — Const-Generic Rank & Layout Sizing**: Tensors specify rank and
  dimension bounds at compile time via layout parameters without runtime
  metadata overhead.
- **FR-2 — Multilinear Hypercube Grid Interpolation**: Fractional N-D coordinate
  queries evaluate weighted multilinear interpolation over the $2^{\text{RANK}}$
  hypercube corner vertices surrounding the query point.
- **FR-3 — Zero-Copy Matrix Slicing**: Slicing 2D sub-matrices from a tensor
  extracts zero-copy `MatrixSlice` views when coordinates align with layout
  strides.
- **FR-4 — Contraction & Permutation Invariants**: Contraction along matching
  axes produces an output tensor with combined
  rank $\text{Rank}_A + \text{Rank}_B - 2$, validating dimension equality at
  compile time.

#### 2.2. Non-Functional Requirements

- **NFR-1 — Fixed-Memory Footprint Cap**: Total element storage per tensor is
  capped at 1,024 elements / 4 KB stack allocation limit.
- **NFR-2 — Integer-Only Quantized Scale**: Quantized tensor operations perform
  fixed-point integer scaling (`SHIFT`) without floating-point instructions or
  dynamic allocations.

#### 2.3. Constraints

- **C-1 — `#![no_std]` Environment**: Operates in `#![no_std]` without standard
  library dependencies.
- **C-2 — Zero Dynamic Allocation**: All memory allocations are stack-based or
  statically borrowed.
- **C-3 — Memory Footprint**: Total element capacity across all dimensions is
  capped at 1,024 elements (~4KB for `f32`).
- **C-4 — Type-Level Quantization Encoding**: Quantization-scale const generic
  parameters must be plain integers.
- **C-5 — Out of Scope Capabilities**: On-device training, automatic
  differentiation, gradient backpropagation, and external model file parsers (
  TFLite/PyTorch/ONNX) are explicitly out of scope.

---

### 3. Technical Overview

`Tensor<T, Layout: TensorLayout, B: FlatBuffer<T>>` is a
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

A `Tensor` wraps a flat, padding-free storage backend. `storage-design.md`
Rev 1.8 removed the rank-agnostic `Buffer<T>` tier, so the crate-wide
contiguity contract is now `ContiguousStorage<T>` / `ContiguousStorageMut<T>`
(FR-3), whose `as_slice()` / `as_mut_slice()` are exactly what a tensor
needs. Those traits are declared as sub-traits of the 2-D
`DenseStorage<T>`, so `Tensor` names a rank-neutral projection of them:

```rust
/// Rank-neutral flat-buffer contract. Blanket-implemented for every
/// `ContiguousStorage<T>` leaf, so a rank-2 tensor reuses `ArrayStorage`
/// directly (§7).
pub unsafe trait FlatBuffer<T> {
    const LEN: usize;
    fn as_slice(&self) -> &[T];
    fn as_ptr(&self) -> *const T;
}
```

Coordinate mapping is delegated to a type implementing `TensorLayout`:

```rust
pub trait TensorLayout {
    const RANK: usize;
    type Size: Dim;
    fn dims() -> &'static [usize];
}

pub struct Tensor<T, Layout: TensorLayout, B: FlatBuffer<T>> {
    buffer: B,
    _marker: PhantomData<Layout>,
}

// Rank-2 owning tensors reuse `ArrayStorage` (`storage-design.md` FR-4),
// which is contiguous and therefore a `FlatBuffer`.
// `[T; Layout::Size::USIZE]` is not a valid default: that projection
// requires `generic_const_exprs`. Higher-rank owning layouts supply their
// own nested-array buffer; they do not project `Dim::USIZE` into `[T; N]`.
pub type ArrayTensor<T, const R: usize, const C: usize> =
    Tensor<T, Shape2D<Const<R>, Const<C>>, ArrayStorage<T, R, C>>;
pub type ViewTensor<'a, T, Layout> =
    Tensor<T, Layout, FlatView<'a, T>>;
pub type ViewMutTensor<'a, T, Layout> =
    Tensor<T, Layout, FlatViewMut<'a, T>>;
```

Layout sizing verification uses type-level bounds (`Shape2D`, `Shape3D`, ...)
implemented via `DimMul`, guaranteeing shape correctness at compile time.

#### 4.2. Memory Layout & Storage Strategy

Multidimensional coordinates map to a flat 1D index using column-major
strides (first dimension varies fastest): $\text{flat\_index} = i_0 + i_1
\cdot S_1 + i_2 \cdot S_2 + \dots$, where $S_0 = 1$ and $S_m =
\prod_{j=0}^{m-1} D_j$.

- **Decoupled Physical Storage**: Flat element access dispatches through
  `FlatBuffer<T>` / `FlatBufferMut<T>` (`as_slice`, `as_ptr`, `as_mut_slice`,
  `as_mut_ptr`), which every `ContiguousStorage` leaf satisfies.
- **Matrix Interoperability**: Column-major layout matches `ArrayStorage`'s
  ordering ($RS = 1$, $CS = R$), so 2-D tensors interoperate with matrices
  without transposition or copying. A `Matrix` over a strided `ViewStorage`
  has no flat buffer and converts by element copy instead
  (`matrix-design.md` §4.8.2).
- **Flat Indexing Efficiency**: Element-wise operations iterate the
  underlying storage directly, bypassing multi-index arithmetic.

#### 4.3. Memory Representation & Slicing

`#[repr(C)]` guarantees a stable layout. Padding-free slice interfaces
(`as_slice`, `as_mut_slice`) are gated behind `FlatBuffer`/`FlatBufferMut`,
facilitating zero-copy casting to `&[T]` for subprogram routing when the
storage backend supports it. Tensor contraction reaching a 2-D kernel
rewraps the slice as a `ViewStorage` and calls `Gemm`
(`subprograms-design.md` FR-4) rather than defining its own inner loop.

#### 4.4. Instantiation & Constructors

- `zero() -> ArrayTensor<T, R, C>` (`T: Zero + Copy`): all-zero rank-2 stack
  tensor.
- `from_raw(data: [[T; R]; C]) -> ArrayTensor<T, R, C>`:
  direct `const fn` initialization from `Array2`'s nested array, the entry
  point for ROM-resident constant tensors.
- `from_fn<F>(f: F) -> ArrayTensor<T, R, C>` (`F: FnMut(&[usize]) -> T`):
  coordinate-mapped construction.
- `from_storage(storage: B) -> Tensor<T, Layout, B>`: wraps any custom
  storage backend.
- Borrowed views: FR-6 constructors on an owning tensor (`view()` /
  `view_mut()`), not `from_slice(&[T])` paired with an independent `Layout`.

`const fn` initialization on stable Rust requires `T: Zero + One` from
`crate::math::num_traits`, exposing `T::ZERO`/`T::ONE` as associated
constants.

#### 4.5. Operator Overloading

`Add`, `Sub` and scalar `Mul`/`Div` iterate directly over element storage
and return an owning rank-2 `ArrayTensor<T, R, C>` when both operands are
rank-2 owning tensors. Dim-generic `Tensor<T, Layout, B>` elementwise ops
return `Tensor<T, Layout, B>` with the same buffer family.

#### 4.6. Core Operations

- **Zero-Copy Sub-Tensor Views**: `as_view`/`slice_inplace` extract
  non-allocating `Ref`-backed sub-tensors.
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
      interpolation), reading directly through `FlatBuffer::as_slice` with no
      intermediate allocation:

```rust
impl<T, Layout: TensorLayout, B> Tensor<T, Layout, B>
where
    B: FlatBuffer<T>,
    T: Float,
{
    pub fn interpolate(&self, coords: &[T; Layout::RANK]) -> T { /* ... */ }
}
```

#### 4.8. Matrix Slicing & Extraction

To support zero-copy matrix operations and gain-scheduled model extraction,
`Tensor` provides matrix slicing methods:

- **Exact Matrix Slicing**:
  `slice_matrix<const R: usize, const C: usize>(&self, fixed_indices: &[usize]) -> MatrixSlice<'_, T, R, C>`
  extracts a 2D zero-copy `Matrix` view along selected free axes.
- **Interpolated Matrix Extraction**:
  `slice_matrix_interpolated<const R: usize, const C: usize>(&self, query_coords: &[T]) -> ArrayMatrix<T, R, C>`
  performs multilinear N-D interpolation across non-grid query coordinates to
  yield an evaluated 2D `Matrix`.

*Note*: Detailed N-D storage layout optimizations and tensor contraction
subprogram acceleration warrant a dedicated `storage/tensor` research pass (
`/cr-research math/storage-subprograms`).

`Float` here is [
`num-traits-design.md`](../math/num-traits-design.md)'s
hardware-aligned hierarchy (`Signed + One + Radical + Exponential + Trig +
Div`), used deliberately rather than the retired `Real` — this document
already follows that naming crate-wide (§4.4, §4.9). That dependency is no
longer provisional: `num-traits-design.md` has completed its own
`/cr-research`/`/cr-design-doc` pass and carries an `Approved` status badge
as of its revision 1.4.

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

#### 4.9. Minimal Activation Function Support

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

#### 4.10. Type-Level Quantized Scalar Representation

Quantization is implemented as a property of the scalar type `T`, not of
`Tensor` itself — `Tensor<T, Layout, B>` already accepts any `T` satisfying
the crate's numeric traits, so a quantized tensor is simply
`Tensor<Quantized<Repr, SHIFT>, Layout, B>` with no structural change to
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
implements the crate's `Zero`, `One`, `Conjugate` and `Scalar`
(`Zero + One + Sub + Mul + Conjugate`) traits —
[`num-traits-design.md`](../math/num-traits-design.md)'s unified arithmetic
target for generic `Tensor`/`Matrix` code paths — so it plugs into every
existing generic call site unchanged. `Conjugate` is the identity on
`Quantized`, which is real, and `Scalar::Real = Self` with `im()` returning
`Real::ZERO` (`num-traits-design.md` FR-3, FR-4, §4.3). `Scalar`
deliberately excludes `Div`; `Quantized`'s dequantization is a bit-shift,
not a division, so this omission is not a gap, and integer division stays
on `TryDiv`.

`Quantized` also implements `SaturatingInteger` when `Repr` does
(`num-traits-design.md` §4.1), which is the contract that makes Q-format
overflow saturate rather than wrap. `num-traits-design.md` §3 states that
`Quantized<Repr, SHIFT>` is defined with the tensor scalar type — that is,
here — while the trait contract it must satisfy is specified there.

An affine variant (`AffineQuantized<Repr, const SHIFT: i32, const
ZERO_POINT: i32>`) is deferred future work for imported-model
interoperability (§5.2) — not part of this revision's default path.

#### 4.11. Interoperability & Conversions

`ConversionError` is defined once, canonically, in
[`error-design.md`](../math/error-design.md) (`DimensionMismatch`, `NonMonicPolynomial`).
Because rank and shape dimensions are verified statically at compile time
via `TensorLayout<Size = ...>`, cross-model layout conversions are
infallible compile-time operations:

- **To `Matrix`**: `From<Tensor<T, Layout, B>> for Matrix<T, R, C, Dense<T, R, C, B>>`
  when `Layout: TensorLayout<Size = <R as DimMul<C>>::Output>` and `Layout::RANK == 2`,
  preserving storage zero-copy.
- **To `Polynomial`**: `From<Tensor<T, Layout, B>> for Polynomial<T, N, Dense<T, N, U1, B>>`
  when `Layout: TensorLayout<Size = N>` and `Layout::RANK == 1`,
  preserving storage zero-copy.

System-identification outputs (Volterra/NARX kernels, N4SID state-space
matrices) are, in practice, 2D (CP-decomposed factor matrices or state-space
matrices, not dense high-rank tensors — Batselier, Chen, & Wong, 2016), so
this existing 2D conversion path is the relevant interoperability surface
for that target application; no rank-N-specific system-identification API
is required.

#### 4.12. Error Handling & State Management

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

#### 4.13. Structural Specializations & Future Extensions

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
  struct fields alongside a `Tensor<i8, Layout, B>` was the first
  candidate considered, but conflicts with the requirement that quantized
  and unquantized tensors carry identical runtime footprint and that shape
  information already lives entirely at the type level.
- **`Tensor`-Level Const Generics (considered, rejected)**: Adding
  `const SHIFT: i32` directly to `Tensor<T, Layout, B, SHIFT>` was
  considered, but this conflates two orthogonal concerns — shape (`Layout`)
  and scalar representation (`SHIFT`) — inside one type and would require
  every existing `impl` block (`Add`, `contract_into`, conversions) to
  thread `SHIFT` arithmetic (e.g. rescaling when contracting two tensors
  with different shifts), a significant complexity increase to an
  already-generic type.
- **Scalar-Level Type (selected)**: `Quantized<Repr, SHIFT>` (§4.10) makes
  quantization a property of `T`, requiring zero change to `Tensor`'s
  existing generic signature — `Tensor<T, Layout, B>` already accepts any
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
  CMSIS-NN's LUT-based activation precedent (§4.9) and Weiser &
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
- **Minimal Trait (selected)**: `Activation<T>` (§4.9), implemented per
  function and reusing the interpolation primitive for table-driven
  variants, keeps the surface minimal while composing with a future codegen
  path.

#### 5.5. External Tensor Libraries

As with `Matrix`, no existing crate combines `no_std`/no-alloc,
compile-time (const-generic) tensor shapes and interoperability with a
broader `Matrix`/`Polynomial` type system. MicroFlow is the closest analog
but is a single-model NN inference engine with no general `Tensor<T,
Layout, B>` abstraction; `tfmicro` requires a C++ toolchain. Building on
`control-rs`'s existing `Dim` and contiguity foundation remains the only option
meeting the audit-footprint and `const fn`-on-stable-Rust requirements.

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
   cross-compilation across `Array`, `Ref`, `RefMut`,
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
pub fn update_thermal_grid<T, Ba, Bx, By>(
    transition_matrix: &Tensor<f32, Shape2D<U4, U4>, Ba>,
    current_grid: &Tensor<f32, Shape3D<U4, U2, U2>, Bx>,
    next_grid: &mut Tensor<f32, Shape3D<U4, U2, U2>, By>,
)
where
    Ba: FlatBuffer<f32>,
    Bx: FlatBuffer<f32>,
    By: FlatBufferMut<f32>,
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
    gain_table: &ArrayTensor<f32, 8, 8>,
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
(§4.13):

```rust
pub fn predict<Bw1, Bb1, Bw2, Bb2>(
    input: &Tensor<Quantized<i8, 7>, Shape1D<U4>, impl FlatBuffer<Quantized<i8, 7>>>,
    w1: &Tensor<Quantized<i8, 7>, Shape2D<U8, U4>, Bw1>,
    b1: &Tensor<Quantized<i8, 7>, Shape1D<U8>, Bb1>,
    w2: &Tensor<Quantized<i8, 7>, Shape2D<U1, U8>, Bw2>,
    b2: &Tensor<Quantized<i8, 7>, Shape1D<U1>, Bb2>,
) -> Quantized<i8, 7> {
    // hidden = relu(w1 . input + b1); output = w2 . hidden + b2
    // (contract_into + Add + Relu::apply, per §4.6/§4.9)
    todo!()
}
```

---

### 7. Risks & Open Questions

- **Interpolation Bounds Policy**: Whether an out-of-grid query clamps,
  extrapolates or returns an error is not finalized (§4.12).
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
  concerns; watch compile times as `Quantized<Repr, SHIFT>` usage grows.
- **Model-Import Tooling**: Recorded as future work only (§4.13); no risk
  analysis is performed in this revision since it is not implemented.

---

### 8. Development Plan

| Task / Feature                               | Description                                                                                                                                                                                                     | Estimated Effort |
|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Layout & Storage**           | `TensorLayout`, `Dim`-based sizing, `Array` storage/view aliases, column-major stride mapping, `repr(C)` slicing.                                                                                               | 2.0 Days         |
| **Phase 2: Element Ops & Contraction**       | Operator overloads, `contract_into`/`contract_into_dynamic`, `permute`, `as_view`/`slice_inplace`.                                                                                                              | 2.5 Days         |
| **Phase 3: Grid Interpolation & Activation** | Multilinear `interpolate`, `Activation` trait, `Relu`, `TableActivation`.                                                                                                                                       | 2.5 Days         |
| **Phase 4: Quantized Scalar Type**           | `Quantized<Repr, SHIFT>` with full `Zero`/`One`/`Scalar` arithmetic `num_traits` impls (correct rounding/saturation semantics), quantize/dequantize, integration across existing generic `T` paths in `Tensor`. | 3.5 Days         |
| **Phase 5: Verification & Interoperability** | `proptest` suites, golden-value regression against SciPy/NumPy references, ARM hardware benchmarks, `TryFrom` conversions to `Matrix`/`Polynomial`.                                                             | 3.0 Days         |

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
   Ahead-of-time model-to-static-Rust-code compilation precedent (§4.13).
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

| Revision | Date            | Author          | Description                                                                                                                                                                                                 |
|:---------|:----------------|:----------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026   | @MitchellDScott | Initial draft: `Storage` trait hierarchy, zero-copy views, stack `ArrayStorage` and inline citations.                                                                                                       |
| 1.1      | August 2, 2026  | @MitchellDScott | Full overhaul after a research pass: added grid interpolation, an `Activation` trait and a quantized scalar type.                                                                                           |
| 1.2      | August 2, 2026  | @MitchellDScott | Relocated `ConversionError`; flagged `Float` dependency as provisional; revised development-plan estimates upward.                                                                                          |
| 1.3      | August 10, 2026 | @MitchellDScott | Synced §4.7's `Float` bound and §4.9's `Quantized` trait list with `num-traits-design.md`'s now-`Approved` hierarchy; removed the resolved provisional-dependency risk from §7.                             |
| 1.4      | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.4 (§1, §3): clarified 2D `BlasStorage` matrix storage bounds vs N-dimensional `Tensor` storage bounds.                                                |
| 1.5      | August 16, 2026 | @mitchelldscott | Harmonized with `storage-subprograms-design.md` Rev 1.5 (§1, §3): aligned 2D `BlasStorage<T, R, C, Stride>` generic parameters with N-D `Tensor` layout abstractions.                                       |
| 1.6      | August 16, 2026 | @mitchelldscott | Reconciled storage model to Tier 0 `Buffer<T>` / `BufferMut<T>` (bypassing 2D `BlasStorage`); added exact and interpolated `Matrix` slicing methods (§4.8); flagged pending `storage/tensor` research pass. |
| 1.7      | August 16, 2026 | @mitchelldscott | Updated `Buffer`/`BufferMut` method names (`as_slice`/`as_ptr`) in §4.2/§4.7 and fixed `ArrayStorage` to `Array` in §6.1 V&V.                                                                               |
| 1.8      | August 16, 2026 | @mitchelldscott | Updated Date and Status badges to Reviewed following final cross-document consistency audit.                                                                                                                  |
| 1.9      | August 18, 2026 | @mitchelldscott | Corrected §4.11 `TryFrom` conversions to `Matrix<T, R, C, Dense<T, R, C, B>>` and `Polynomial<T, N, Dense<T, N, U1, B>>`, wrapping Tier-0 `Buffer` into the Tier-3 `Dense` storage leaf.                     |
| 1.10     | August 18, 2026 | @mitchelldscott | Aligned §4.11 `Tensor` → `Matrix`/`Polynomial` conversions to infallible `From` bounded by `TensorLayout<Size = ...>`, eliminating obsolete `LayoutMismatch` runtime check.                                    |
| 1.11     | August 18, 2026 | @mitchelldscott | Propagated `storage-subprograms-design.md` Rev 1.11–1.12: rank-2 `ArrayTensor<T, const R, const C>` over `Array2`; dropped `Array<T, Size::USIZE>` default; views via FR-6, not `from_slice`. |
| 1.12     | August 18, 2026 | @mitchelldscott | Propagated storage Rev 1.16: `slice_matrix` returns `MatrixSlice<'_, T, R, C>` with bare `const R, C`. |
| 1.13     | August 20, 2026 | @mitchelldscott | Renamed the contrastive `BlasStorage` reference (§1) to `MatrixStorage`, matching `storage-subprograms-design.md` Rev 1.31; no other content affected since `Tensor` bypasses the 2D storage branch entirely. |
| 1.14     | August 22, 2026 | @MitchellDScott | Reverted Doc Status to Draft. Body still cites deleted `storage-subprograms-design.md`; retarget onto `storage-design.md` is a dedicated pass. |
| 1.15     | August 24, 2026 | @mitchelldscott | Retargeted onto `storage-design.md` Rev 1.8, closing the Rev 1.14 note. `storage-design.md` removed the rank-agnostic `Buffer<T>` tier, so §4.1 defines the rank-neutral `FlatBuffer<T>`/`FlatBufferMut<T>` projection, blanket-implemented for every `ContiguousStorage<T>` leaf; rank-2 owning tensors bind `ArrayStorage<T, R, C>` (§4.1, §4.2, §4.3). Recorded that a strided `ViewStorage`-backed `Matrix` has no flat buffer and converts by copy, and that 2-D contraction rewraps onto `Gemm`. Extended §4.10 with `Quantized`'s `Conjugate`/`Scalar::Real` obligations and its `SaturatingInteger` bound (`num-traits-design.md` FR-3, FR-4, §4.1). Status stays Draft. |
