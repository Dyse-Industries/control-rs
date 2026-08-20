# BLAS Interface Codegen Experiment

This experiment evaluates different Rust implementation strategies for Basic Linear Algebra Subprograms (BLAS), specifically Level-2 Matrix-Vector Multiplication ($y = \alpha A x + \beta y$ or $y = A x$), behind an opaque static library boundary. 

It validates how different levels of compiler knowledge (runtime fields, type-level constants, raw pointers, and statically sized nested arrays) affect release-profile assembly (branches, panic paths, instruction count) and execution speed.

## Quick Start

### 1. Correctness Checks
Verify that all variants compute mathematically identical results (within $10^{-6}$):
```bash
cargo run --features "cffi"
```

### 2. Runtime Benchmarks
Measure the execution latency (nanoseconds per call) of the variants:
```bash
cargo run --bin c_call_cost --features "cffi" --release
```

### 3. Disassembly & Codegen Analysis
Inspect the instruction count, branches, and panic paths generated for any target.
```bash
# 1. Add target support (if not already installed)
rustup target add thumbv7em-none-eabihf

# 2. Run measure on host target (default)
cargo run --bin measure

# 3. Run measure for a cross-compiled target (e.g. thumbv7em-none-eabihf)
cargo run --bin measure -- thumbv7em-none-eabihf
```

---

## Codegen & Disassembly Metrics

### Host Target (`x86_64-apple-darwin`)

Compiled under `opt-level=3`, `codegen-units=1`, `panic=abort`:

| Variant | Strategy | Dims / Layout | Storage Type | Instructions | Branches & Calls | Panic Paths |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **[A]** | `gemv_dyn` | Runtime Strides | Slice (`&[f32]`) | 123 | 23 | 7 |
| **[B]** | `gemv_const_4` | Assoc Constants | Slice (`&[f32]`) | 166 | 35 | 21 |
| **[H]** | `gemv_const_xy_4` | Assoc Constants | Slice (`&[f32]`) | 158 | 25 | 13 |
| **[C]** | `gemv_arr_4` | Assoc Constants | Flat Array (`&[f32; 16]`) | 28 | 0 | 0 |
| **[C8]** | `gemv_arr_8` | Assoc Constants | Flat Array (`&[f32; 64]`) | 97 | 0 | 0 |
| **[C16]**| `gemv_arr_16` | Assoc Constants | Flat Array (`&[f32; 256]`) | 281 | 0 | 0 |
| **[D]** | `gemv_ptr_4` | Assoc Constants | Raw Pointer (`*const f32`) | 59 | 0 | 0 |
| **[D8]** | `gemv_ptr_8` | Assoc Constants | Raw Pointer (`*const f32`) | 207 | 0 | 0 |
| **[D16]**| `gemv_ptr_16` | Assoc Constants | Raw Pointer (`*const f32`) | 363 | 3 | 0 |
| **[E]** | `gemv_ptr_ab_4` | Assoc Constants | Raw Pointer (`*const f32`) | 74 | 0 | 0 |
| **[G]** | `gemv_checked_4` | Assoc Constants | Manual Bounds + `get_unchecked` | 28 | 0 | 0 |
| **[I]** | **Design A: `gemv_storage_trait_4`** | Generic Storage Trait | Bounded Slice (`&[f32]`) | **162** | **25** | **13** |
| **[J]** | **Design C: `gemv_generic_fn_4`** | Generic Storage Function | Bounded Slice (`&[f32]`) | **163** | **25** | **13** |
| **[K]** | **Design B: `gemv_nested_4`** (Proposed)| Static Const Strides | **Nested Array (`&[[f32; 4]; 4]`)**| **29** | **0** | **0** |

### Bare-Metal Target (`thumbv7em-none-eabihf`)

Compiled under `opt-level=3`, `codegen-units=1`, `panic=abort`:

| Variant | Strategy | Dims / Layout | Storage Type | Instructions | Branches & Calls | Panic Paths |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **[A]** | `gemv_dyn` | Runtime Strides | Slice (`&[f32]`) | 135 | 26 | 10 |
| **[B]** | `gemv_const_4` | Assoc Constants | Slice (`&[f32]`) | 168 | 34 | 20 |
| **[H]** | `gemv_const_xy_4` | Assoc Constants | Slice (`&[f32]`) | 139 | 24 | 12 |
| **[C]** | `gemv_arr_4` | Assoc Constants | Flat Array (`&[f32; 16]`) | 49 | 0 | 0 |
| **[C8]** | `gemv_arr_8` | Assoc Constants | Flat Array (`&[f32; 64]`) | 175 | 0 | 0 |
| **[C16]**| `gemv_arr_16` | Assoc Constants | Flat Array (`&[f32; 256]`) | 99 | 1 | 0 |
| **[D]** | `gemv_ptr_4` | Assoc Constants | Raw Pointer (`*const f32`) | 61 | 0 | 0 |
| **[D8]** | `gemv_ptr_8` | Assoc Constants | Raw Pointer (`*const f32`) | 213 | 0 | 0 |
| **[D16]**| `gemv_ptr_16` | Assoc Constants | Raw Pointer (`*const f32`) | 113 | 1 | 0 |
| **[E]** | `gemv_ptr_ab_4` | Assoc Constants | Raw Pointer (`*const f32`) | 73 | 0 | 0 |
| **[G]** | `gemv_checked_4` | Assoc Constants | Manual Bounds + `get_unchecked` | 49 | 0 | 0 |
| **[I]** | **Design A: `gemv_storage_trait_4`** | Generic Storage Trait | Bounded Slice (`&[f32]`) | **149** | **24** | **12** |
| **[J]** | **Design C: `gemv_generic_fn_4`** | Generic Storage Function | Bounded Slice (`&[f32]`) | **146** | **24** | **12** |
| **[K]** | **Design B: `gemv_nested_4`** (Proposed)| Static Const Strides | **Nested Array (`&[[f32; 4]; 4]`)**| **57** | **0** | **0** |

---

## Runtime Performance Benchmarks

Executing 1,000,000 calls of 4x4 matrix-vector multiplication in release mode:

| Variant | Implementation Strategy | Latency (ns/call) | Relative Cost |
| :--- | :--- | :---: | :---: |
| **`gemv_nested_4`** | **Design B: Proposed Nested Array ABI** | **1.08 ns** | **1.0x** (Reference) |
| **`gemv_generic_fn_4`** | Design C: Generic Storage Function | **3.34 ns** | **3.1x slower** |
| **`gemv_storage_trait_4`** | Design A: Generic Storage Trait Method | **3.56 ns** | **3.3x slower** |
| **`gemv_c`** | C FFI (`extern "C"`) | **7.18 ns** | **6.6x slower** |
| **`gemv_dyn`** | Rust Runtime Fields (Variant A) | **11.63 ns** | **10.7x slower** |

---

## Key Takeaways

1. **Passing Storage Implementors to Traits is Inefficient (Design A & C)**:
   * When subprograms take a generic `A: MatrixStorage` parameter, they must access memory through the trait interface (`a.get(i, j)`). Even when compiled with `#[inline(always)]`, the trait wraps slice indexing (`self.as_slice()[off]`).
   * Because the slice length is not known to the compiler inside the generic body, LLVM is forced to generate bounds checks for every single access. This introduces **13 conditional panic branches** and **163 instructions**, running **3.3x slower**.
2. **The Statically Sized Nested Array Compute ABI is Highly Optimal (Design B)**:
   * Standardizing the compute ABI on nested arrays (`&[[T; LDA]; N]` and `&[T; N]`) provides the compiler with type-level proof of buffer bounds at compile time.
   * LLVM successfully unrolls the loops, vectorizes the arithmetic, and eliminates bounds checks entirely, resulting in **0 branches, 0 panic paths, and only 29 instructions** (running in **1.08 ns**).
