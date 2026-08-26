# Tensor Type & Low-Cost Inference (Design Document)

![Date Badge](https://img.shields.io/badge/Date-August_25,_2026-blue)
![Status Badge](https://img.shields.io/badge/Doc%20Status-Approved-green)
![Author Badge](https://img.shields.io/badge/Author-@MitchellDScott-blueviolet)

---

### 1. Introduction

The `Tensor` module provides compile-time sized, N-dimensional tensor
representations and evaluation primitives for multidimensional lookup tables,
coordinate transformations, and embedded inference.

Primary usage scenarios:

- **Gain-Scheduled Flight Control Tables**: Evaluating N-dimensional aerodynamic
  lookup grids (e.g., angle-of-attack, Mach number, dynamic pressure) via
  multilinear grid interpolation in deterministic time without dynamic
  allocation.
- **Microcontroller Neural Network Inference**: Executing feedforward inference
  for quantized (Q7/Q15/integer) pre-trained neural networks and TinyML state
  estimators using tensor contractions.
- **Multidimensional Physical State Modeling**: Representing higher-rank
  physical tensors (such as rank-4 elasticity tensors or stress/strain fields)
  and performing contractions along arbitrary tensor axes.
- **Matrix Hyperplane Slicing**: Extracting 2-D matrix slices from
  multidimensional tables for linear algebra operations without memory copies.

---

### 2. Requirements

#### 2.1. Functional Requirements

- **FR-1 — Multidimensional Array Representation**: Represents N-dimensional
  tensors parameterized over compile-time rank, dimension extents, and memory
  strides without runtime layout metadata overhead.
- **FR-2 — Multilinear Grid Interpolation**: Evaluates fractional N-dimensional
  coordinate queries across lookup grids via weighted multilinear interpolation
  over the $2^{\text{RANK}}$ surrounding hypercube vertices (Weiser &
  Zarantonello, 1988). Interpolation must execute without heap allocations for
  gain-scheduled flight control tables.
- **FR-3 — Tensor Contraction & Matrix Slicing**: Computes tensor contractions
  along matching axes, producing an output tensor of
  rank $\text{Rank}_A + \text{Rank}_B - 2$. Slicing 2-D hyperplanes from a
  tensor must provide zero-copy `MatrixSlice` views when coordinates align with
  layout strides.
- **FR-4 — Quantized Fixed-Point Inference**: Performs tensor arithmetic and
  activation functions over quantized integer representations (`SHIFT`
  fixed-point scaling) without floating-point emulation or dynamic memory
  allocation (Wu et al., 2020).
- **FR-5 — Nonlinear Activation Functions**: Applies elementwise activation
  functions (such as ReLU and piecewise-linear table activations) in-place over
  tensor buffers for embedded TinyML controller evaluation (Lai et al., 2018).

#### 2.2. Non-Functional Requirements

- **NFR-1 — Bounded Stack Allocation**: Total element storage per tensor is
  capped at 1,024 elements ($\le 4\text{ KB}$ for 32-bit types) to protect
  microcontroller stack bounds.
- **NFR-2 — Real-Time Inference Latency**: Multilinear interpolation and tensor
  contraction execute in bounded cycle counts without dynamic layout branching.

#### 2.3. Constraints

- **C-1 — Out of Scope Capabilities**: On-device training, backpropagation,
  automatic differentiation, and runtime model file parsers (ONNX/TFLite) are
  explicitly out of scope.
- **C-2 — Static Quantization Parameter Encoding**: Quantization-scale const
  generic parameters must be plain integer shifts.
- **C-3 — `#![no_std]` / Zero Heap Allocation**: All tensor representations and
  operations execute strictly on stack-allocated arrays or borrowed memory
  buffers.

---

### 3. Technical Overview

`Tensor<T, Layout: TensorLayout, B: FlatBuffer<T>>` provides a statically sized
N-dimensional array built on `control-rs`'s decoupled storage foundation and
type-level dimension system.

The module provides compile-time rank and stride mapping, multi-axis tensor
contractions (delegating 2D operations to Level 3 `Gemm`), zero-copy 2D
`MatrixSlice` hyperplane extraction, multilinear hypercube grid interpolation
for flight-control gain scheduling tables, and fixed-point quantized neural
network inference with lightweight activation functions.

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

// Rank-2 owning tensors reuse `ArrayStorage` (`storage-design.md` FR-2),
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
  without transposition or copying. A `Matrix` over a strided `StorageView`
  has no flat buffer and converts by element copy instead
  (`matrix-design.md` §4.8.2).
- **Flat Indexing Efficiency**: Element-wise operations iterate the
  underlying storage directly, bypassing multi-index arithmetic.

#### 4.3. Memory Representation & Slicing

`#[repr(C)]` guarantees a stable layout. Padding-free slice interfaces
(`as_slice`, `as_mut_slice`) are gated behind `FlatBuffer`/`FlatBufferMut`,
facilitifying zero-copy casting to `&[T]` for subprogram routing when the
storage backend supports it. Tensor contraction reaching a 2-D kernel
rewraps the slice as a `StorageView` and calls `Gemm`
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

##### 4.4.1. Data-Driven Tensor Factories [Proposal (not in evidence)]

To support high-dimensional calibration tables, nonlinear Volterra system
identification, polytopic LPV modeling, and tensor-valued time-series
estimation, dedicated data-driven **Object Factories** format empirical
datasets into multidimensional tensor structures without polluting `Tensor`
(Batselier et al., 2017; Favier & Kibangou, 2023; Van Eeghem et al., 2017;
Baranyi, 2014; Rogers et al., 2013; Weiser & Zarantonello, 1988):

- **Grid Tensor Factory (`GridTensorFactory`)**: Produces N-D lookup tensors
  from
  gridded aerodynamic or engine calibration measurements, populating dense
  coordinate grids for fast multilinear corner interpolation (Weiser &
  Zarantonello, 1988; Koo & Sands, 2024).
- **Hankel Tensor Factory (`HankelTensorFactory`)**: Assembles higher-order
  Hankel tensors $\mathcal{H} \in \mathbb{R}^{I_1 \times I_2 \times \dots \times
  I_D}$ from multivariable time series for blind system identification and
  canonical polyadic decomposition (Van Eeghem et al., 2017).
- **Matrix Series Factory (`MatrixSeriesTensorFactory`)**: Stacks temporal
  series
  of matrix observations into rank-3 batch tensors for Multilinear Dynamical
  Systems (MLDS) parameter estimation (Rogers et al., 2013).

_Detailed standalone design and API signatures for these factories are
specified in `documentation/control-toolboxes/sysid-design.md`._

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
[`error-design.md`](../math/error-design.md) (`DimensionMismatch`,
`NonMonicPolynomial`).
Because rank and shape dimensions are verified statically at compile time
via `TensorLayout<Size = ...>`, cross-model layout conversions are
infallible compile-time operations:

- **To `Matrix`**:
  `From<Tensor<T, Layout, B>> for Matrix<T, R, C, Dense<T, R, C, B>>`
  when `Layout: TensorLayout<Size = <R as DimMul<C>>::Output>` and
  `Layout::RANK == 2`,
  preserving storage zero-copy.
- **To `Polynomial`**:
  `From<Tensor<T, Layout, B>> for Polynomial<T, N, Dense<T, N, Const<1>, B>>`
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

#### 6.1. Objectives

- Demonstrate compile-time verification of tensor rank, axis extents, and
  contraction index bounds.
- Demonstrate mathematical exactness of multi-dimensional column-major
  coordinate indexing and strided slicing.
- Demonstrate equivalence of 2D tensor contractions against reference BLAS Level
  3 `Gemm`.
- Demonstrate numerical accuracy and interpolation bounds for multilinear grid
  interpolation.
- Demonstrate arithmetic exactness, rounding, and saturation invariants for
  `Quantized<Repr, SHIFT>`.
- Demonstrate zero dynamic heap allocation in `#![no_std]` execution and
  deterministic real-time latency.

#### 6.2. Methods

| Method                    | Mechanism                                                                                  | Requirements discharged  |
|:--------------------------|:-------------------------------------------------------------------------------------------|:-------------------------|
| Compile-time shape check  | Type-level `Dim` rank assertions, `compile_fail` doctests                                  | FR-1, C-1, C-3, C-4      |
| Requirements-based test   | `#[test]` unit tests over grid boundaries, activations, and conversions                    | FR-2, FR-4, FR-5, FR-7   |
| Property-based test       | `proptest` suites verifying axis permutation round-trips and tensor contraction identities | FR-3, FR-6               |
| Doctest                   | Runnable rustdoc examples                                                                  | FR-2, FR-4               |
| Back-to-back comparison   | `/cr-prototype numerical-models/tensor` and SciPy/NumPy golden references                  | FR-3, FR-4, FR-6         |
| Resource usage evaluation | `no_alloc` audit, `size_of` assertions, stack analysis                                     | NFR-1, NFR-2, C-2, C-4   |
| On-target execution       | ETS suites under QEMU and Teensy hardware                                                  | NFR-3                    |
| Coverage measurement      | `cargo coverage` reporting statement and branch metrics                                    | FR-1..FR-7, NFR-1..NFR-3 |

#### 6.3. Acceptance Criteria

| Claim                              | Oracle                                       | Measure                     | Bound                                                                                                  | Justification                                                              |
|:-----------------------------------|:---------------------------------------------|:----------------------------|:-------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------|
| Contraction vs `Gemm` equivalence  | BLAS Level 3 `Gemm` reference                | Relative error / test ratio | $\frac{\|C_{\text{contract}} - C_{\text{gemm}}\|_\infty}{\|A\|_\infty \|B\|_\infty K \epsilon} < 20.0$ | Direct reduction of 2D tensor contraction to matrix product                |
| Exact grid interpolation           | Stored vertex coordinates                    | Exact equality              | $0$ (exact)                                                                                            | Multilinear interpolation vertex consistency (Weiser & Zarantonello, 1988) |
| Interior multilinear interpolation | Analytic piecewise linear model              | Absolute error              | $\|f(x) - \hat{f}(x)\| \le \frac{1}{8} \sum h_i^2 \|\frac{\partial^2 f}{\partial x_i^2}\|_\infty$      | Multilinear interpolation error bound (Weiser & Zarantonello, 1988)        |
| Quantization round-trip            | Exact floating-point vs dequantized integer  | Absolute error              | $\|x - \text{dequantize}(\text{quantize}(x))\| \le 2^{-\text{SHIFT}-1}$                                | Half-LSB rounding bound for fixed-point quantization (Wu et al., 2020)     |
| Permutation invertibility          | Permute and inverse permute axes             | Exact equality              | Inverted tensor == original tensor                                                                     | Exact structural permutation invariant                                     |
| Activation lookup residual         | Exact nonlinear activation ($\tanh, \sigma$) | Absolute error              | $\|\hat{a}(x) - a(x)\|_\infty \le \epsilon_{\text{table}}$                                             | Piecewise linear activation table approximation (Lai et al., 2018)         |
| Zero-allocation execution          | Host memory allocator interception           | Exact equality              | 0 heap allocations                                                                                     | NFR-1 `#![no_std]` invariant                                               |

#### 6.4. Traceability

| Requirement                                         | Method                                           | Artifact                                                |
|:----------------------------------------------------|:-------------------------------------------------|:--------------------------------------------------------|
| FR-1 — Compile-Time Static Multi-Dimensional Sizing | Compile-time shape check                         | `tests/tensor_shape_fail.rs` (`compile_fail` doctests)  |
| FR-2 — Strided Multi-Dimensional Indexing & Slicing | Requirements-based test, Doctest                 | `tests/tensor_indexing.rs::test_strided_access`         |
| FR-3 — Tensor Contraction & Axis Permutation        | Property-based test, Back-to-back comparison     | `tests/tensor_contract.rs::prop_contraction_gemm`       |
| FR-4 — Multilinear Grid Interpolation               | Requirements-based test, Back-to-back comparison | `tests/tensor_interp.rs::test_multilinear_grid`         |
| FR-5 — Activation Functions                         | Requirements-based test                          | `tests/tensor_activation.rs::test_relu_table`           |
| FR-6 — Quantized Fixed-Point Arithmetic             | Property-based test, Back-to-back comparison     | `tests/quantized_arithmetic.rs::prop_quantized_ops`     |
| FR-7 — Linear Model Interoperability                | Requirements-based test                          | `tests/tensor_interop.rs::test_matrix_poly_conversions` |
| NFR-1 — Zero Dynamic Heap Allocation                | Resource usage evaluation                        | `#![no_std]` host allocator audit                       |
| NFR-2 — Bounded Stack Footprint                     | Resource usage evaluation                        | `clippy::large_stack_arrays` CI check                   |
| NFR-3 — Predictable Real-Time Latency               | On-target execution                              | ETS test suite `tensor::bench_contract_latency`         |
| C-1 — Column-Major Storage Layout                   | Compile-time shape check                         | Structural stride calculation assertions                |
| C-2 — `#![no_std]` Compatibility                    | Resource usage evaluation                        | Compilation under `#![no_std]` target triples           |
| C-3 — Stable Rust Toolchain                         | Compile-time shape check                         | Cargo workspace build on `stable` Rust                  |
| C-4 — Static Dimension Ceiling                      | Resource usage evaluation                        | Static size verification tests                          |

#### 6.5. Coverage

- **Target**: $\ge 90\%$ statement coverage, $\ge 85\%$ branch coverage reported
  via `cargo coverage`.
- **Excluded**: Target-specific SIMD micro-kernels and debug formatting
  routines (`core::fmt::Debug`).

#### 6.6. Validation

- **Spatial Heat Distribution Update**: Validation of 3D tensor state
  contraction representing 2D heat diffusion over discrete time steps in
  `examples/thermal_grid.rs`.
- **Flight-Control Gain Scheduling**: 2D gain table lookup indexed by Mach
  number and angle of attack in `examples/gain_scheduling.rs`.
- **TinyML Neural Controller**: 2-layer quantized MLP evaluation with
  `Quantized<i8, 7>` and `Relu` activation in `examples/tinyml_controller.rs`.

#### 6.7. Not Verified

- Dynamic arbitrary-rank tensor contractions with runtime-determined dimensions
  are excluded.
- Automatic deep learning framework graph import (ONNX/TFLite converter tools)
  is deferred to future tooling.

---

### 7. Performance & Resource Considerations

- **Stack Overhead**: Inline tensor capacities are bounded by the total element
  ceiling $S \le 16{,}384$ ($64\text{KB}$ for `f32`, $16\text{KB}$ for
  `Quantized<i8, 7>`).
- **Memory Footprint**: `Quantized<i8, 7>` achieves a $4\times$ reduction in
  weight/activation RAM compared to `f32`.
- **Zero-Copy Views**: `TensorView` and `TensorViewMut` operate over borrowed
  `FlatBuffer` pointers with rank metadata, avoiding tensor allocation.

---

### 8. Risks & Open Questions

- **Interpolation Bounds Policy**: Whether an out-of-grid query clamps,
  extrapolates or returns an error is not finalized (§4.12).
- **Q-Format vs. Rational Scale Selection**: This revision defaults to
  power-of-two `SHIFT` (§5.2); whether/when the rational variant is needed
  depends on the future model-import path's accuracy requirements.
- **Activation Table Resolution**: `TableActivation`'s breakpoint count and
  domain range are not chosen in this revision; CMSIS-NN's `[-8, 8]` precedent
  is a starting point, but the accuracy/size trade-off needs a concrete target
  application before finalizing.
- **Const-Generic Compile-Time Complexity**: Adding `SHIFT`/rational const
  generics to the scalar type stacks on top of the compile-time-arithmetic
  concerns; watch compile times as `Quantized<Repr, SHIFT>` usage grows.
- **Model-Import Tooling**: Recorded as future work only (§4.13); no risk
  analysis is performed in this revision since it is not implemented.

---

### 9. Development Plan

| Task / Feature                               | Description                                                                                                                                                                                                     | Estimated Effort |
|:---------------------------------------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|:-----------------|
| **Phase 1: Core Layout & Storage**           | `TensorLayout`, `Dim`-based sizing, `Array` storage/view aliases, column-major stride mapping, `repr(C)` slicing.                                                                                               | 2.0 Days         |
| **Phase 2: Element Ops & Contraction**       | Operator overloads, `contract_into`/`contract_into_dynamic`, `permute`, `as_view`/`slice_inplace`.                                                                                                              | 2.5 Days         |
| **Phase 3: Grid Interpolation & Activation** | Multilinear `interpolate`, `Activation` trait, `Relu`, `TableActivation`.                                                                                                                                       | 2.5 Days         |
| **Phase 4: Quantized Scalar Type**           | `Quantized<Repr, SHIFT>` with full `Zero`/`One`/`Scalar` arithmetic `num_traits` impls (correct rounding/saturation semantics), quantize/dequantize, integration across existing generic `T` paths in `Tensor`. | 3.5 Days         |
| **Phase 5: Verification & Interoperability** | `proptest` suites, golden-value regression against SciPy/NumPy references, ARM hardware benchmarks, `TryFrom` conversions to `Matrix`/`Polynomial` per `vv-standards.md`.                                       | 3.0 Days         |

---

### 10. References

1. **Kolda, T. G., & Bader, B. W. (2009).** Tensor Decompositions and
   Applications. _SIAM Review_, 51(3), 455–500. — Tensor contraction and
   decomposition computational complexity.
2. **Soro, S. (2021).** TinyML for Ubiquitous Edge AI. MITRE Technical Report
   MTR200519, _arXiv:2102.01255_. — Memory-footprint and inference-latency
   framing for microcontroller-class deployments.
3. **Warden, P., & Situnayake, D. (2019).** _TinyML: Machine Learning with
   TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers_. O'Reilly
   Media. — Weight/activation memory-budgeting guidelines.
4. **David, R., et al. (2021).** TensorFlow Lite Micro: Embedded Machine
   Learning on TinyML Systems. _Proceedings of Machine Learning and Systems (
   MLSys) 3_. — Static arena memory-planning precedent for embedded inference
   engines.
5. **Lai, L., Suda, N., & Chandra, V. (2018).** CMSIS-NN: Efficient Neural
   Network Kernels for Arm Cortex-M CPUs. _arXiv:1801.06601_. —
   Table-lookup-plus-interpolation activation-function kernel design.
6. **Carnelos, M., Pasti, F., & Bellotto, N. (2024).** MicroFlow: An Efficient
   Rust-Based Inference Engine for TinyML. _arXiv:2409.19432_. — Ahead-of-time
   model-to-static-Rust-code compilation precedent (§4.13).
7. **Weiser, A., & Zarantonello, S. E. (1988).** A Note on Piecewise Linear and
   Multilinear Table Interpolation in Many Dimensions. _Mathematics of
   Computation_, 50(181), 189–196. — Canonical formalization of the multilinear
   grid-interpolation primitive (§4.7).
8. **Koo, S. M., & Sands, T. (2024).** Bilinear Interpolation of
   Three-Dimensional Gain-Scheduled Autopilots. _Sensors_, 24(1), 13. —
   Flight-control gain-scheduling lookup-table precedent.
9. **Batselier, K., Chen, Z., & Wong, N. (2016).** Tensor Network alternating
   linear scheme for MIMO Volterra system identification. _arXiv:1607.00127_. —
   System-identification tensor shape/rank precedent.
10. **Wu, H., Judd, P., Zhang, X., Isaev, M., & Micikevicius, P. (2020).**
    Integer Quantization for Deep Learning Inference: Principles and Empirical
    Evaluation. _arXiv:2004.09602_. — Affine int8 quantization convention used
    by mainstream trained-model export formats.
11. **Hennessy, J. L., & Patterson, D. A. (2017).** _Computer Architecture: A
    Quantitative Approach_ (6th ed.). Morgan Kaufmann. — Cache and
    memory-hierarchy modeling for stride-based N-D indexing.
12. **Claessen, K., & Hughes, J. (2000).** QuickCheck: A Lightweight Tool for
    Random Testing of Haskell Programs. _ACM SIGPLAN Notices_, 35(9), 268–279. —
    Property-based testing methodology (`proptest`).
13. **Rust Project Developers. (2024).** _The Rustonomicon: The Dark Arts of
    Advanced and Unsafe Rust Programming_. — Unsafe/aliasing rules for tensor
    slice views.
14. **ISO. (2018).** _ISO 26262-6:2018 Road vehicles — Functional safety — Part
    6: Product development at the software level_.
15. **RTCA / EUROCAE. (2011).** _DO-178C: Software Considerations in Airborne
    Systems and Equipment Certification_.
16. **Batselier, K., Chen, Z., & Wong, N. (2017).** A Tensor Network Alternative
    for ODE/PDE-based System Identification. *IFAC-PapersOnLine*, 50(1),
    11429–11434, doi: 10.1016/j.ifacol.2017.08.1750.
17. **Favier, G., & Kibangou, A. (2023).** Overview of Tensor-Based Models for
    Nonlinear System Identification. *Signals*, 4(4), 664–698, doi:
    10.3390/signals4040036.
18. **Van Eeghem, J., Sørensen, M., & De Lathauwer, L. (2017).** Tensor tools
    for
    blind system identification. In *2017 25th European Signal Processing
    Conference (EUSIPCO)*, Kos, Greece.
19. **Batselier, K., Ko, C.-Y., & Wong, N. (2018).** Tensor Network Algorithms
    for
    Linear and Nonlinear System Identification. In *2018 IEEE 28th International
    Workshop on Machine Learning for Signal Processing (MLSP)*, Aalborg,
    Denmark.
20. **Baranyi, P. (2014).** TP-Model Transformation-Based-Control Design
    Frameworks. *Springer International Publishing*, Cham, Switzerland.
21. **Szollosi, A., & Baranyi, P. (2018).** Influence of Sampling Density on the
    Characteristics of Tensor Product Models. *Electronics*, 7(12), 373.
22. **Rogers, M., Li, L., & Russell, S. (2013).** Multilinear Dynamical Systems
    for Tensor Time Series. In *Advances in Neural Information Processing
    Systems (NeurIPS 2013)*, Lake Tahoe, NV, USA.

---

### 11. Revision History

| Revision | Date            | Author          | Description                                                                                                                           |
|:---------|:----------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------|
| 1.0      | July 26, 2026   | @MitchellDScott | Initial draft: N-dimensional storage layouts, indexing arithmetic, and zero-allocation views.                                          |
| 1.1      | August 2, 2026   | @MitchellDScott | Grid interpolation & quantization: added multilinear lookup tables, `Activation` trait, and fixed-point quantized execution.          |
| 1.2      | August 24, 2026 | @MitchellDScott | Rank-neutral buffer projection: integrated `FlatBuffer`/`FlatBufferMut` projection over `ContiguousStorage` backends.                 |
| 1.3      | August 25, 2026 | @MitchellDScott | V&V standardization: aligned test oracles with multilinear interpolation tolerances and fixed-point quantization error bounds.       |
| 1.4      | August 26, 2026 | @MitchellDScott | Storage view retarget: updated references to `StorageView` and `DenseStorage` traits.                                                 |
