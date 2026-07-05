# Matrix Storage Design Document

**Implementation Order:** 0  
**Status:** Approved  
**Author:** @MitchellDScott  
**Date:** July 4, 2026

---

## 1. Context & Objective

Matrix operations and algorithms are heavily dependent on matrix dimensions but mostly independent of the actual data layout in memory. The objective of the matrix storage traits is to specify how storage types may implement the required functions to be used as a matrix's storage backend. 

By separating the storage backend from the mathematical operations, a thin wrapper type `Matrix<S>` can be defined. Higher-level algorithms can query storage properties (such as contiguity or stride) at compile time via trait bounds, enabling zero-cost abstractions and target-specific optimizations without exposing raw pointer manipulation or unsafe code to the end user.

---

## 2. Glossary & Storage Vocabularies

To maintain documentation integrity and prevent drift, we standardize on the following canonical terms:

| Term | Category | Description | Physical Representation |
| :--- | :--- | :--- | :--- |
| `MatrixStorage` | Base Trait | Immutable access and dimension queries. | Abstract memory representation. |
| `MatrixStorageMut` | Trait | Extension for mutable element access. | Abstract mutable memory. |
| `ContiguousStorage` | Trait | Guarantees sequential column-major element layout. | Flat slice (`&[T]`) exposure. |
| `StridedStorage` | Trait | Guarantees layout is affine with fixed strides. | Non-contiguous memory (e.g., submatrix). |
| `VolatileMatrixStorage` | Trait | Read/write access bypasses compiler caches. | Hardware-mapped peripheral registers. |
| `ArrayStorage` | Backend | Owned, statically-sized matrix. | Fixed-size array on the stack. |
| `ViewStorage` | Backend | Borrowed immutable view over a submatrix. | Pointer + strides referencing external memory. |
| `ViewStorageMut` | Backend | Borrowed mutable view over a submatrix. | Mutable pointer + strides referencing external memory. |
| `VolatileStorage` | Backend | Memory-mapped hardware registers. | Raw volatile pointer. |

---

## 3. Core Traits & Architecture

The storage layer is partitioned into discrete traits to prevent backends from advertising capabilities they cannot support (e.g., a volatile register-mapped backend exposing contiguous slices).

```rust
use crate::math::ArithmeticResult;

/// Base trait defining logical dimensions and safe element retrieval.
pub trait MatrixStorage {
    /// The scalar type stored in this matrix.
    type Element;

    /// Returns the logical number of rows.
    fn rows(&self) -> usize;

    /// Returns the logical number of columns.
    fn cols(&self) -> usize;

    /// Unsafe element access without bounds checking.
    ///
    /// # Safety
    /// The caller must guarantee that `row < self.rows()` and `col < self.cols()`.
    unsafe fn get_unchecked(&self, row: usize, col: usize) -> &Self::Element;

    /// Safe element access with bounds checking.
    #[inline]
    fn get(&self, row: usize, col: usize) -> Option<&Self::Element> {
        if row < self.rows() && col < self.cols() {
            Some(unsafe { self.get_unchecked(row, col) })
        } else {
            None
        }
    }
}

/// Extension trait for mutable element access.
pub trait MatrixStorageMut: MatrixStorage {
    /// Unsafe mutable element access without bounds checking.
    ///
    /// # Safety
    /// The caller must guarantee that `row < self.rows()` and `col < self.cols()`.
    unsafe fn get_unchecked_mut(&mut self, row: usize, col: usize) -> &mut Self::Element;

    /// Safe mutable element access with bounds checking.
    #[inline]
    fn get_mut(&mut self, row: usize, col: usize) -> Option<&mut Self::Element> {
        if row < self.rows() && col < self.cols() {
            Some(unsafe { self.get_unchecked_mut(row, col) })
        } else {
            None
        }
    }
}
```

### 3.1. Layout-Specific Traits

Dense numerical kernels (like BLAS micro-kernels) require contiguous memory to optimize cache locality. In contrast, general element-wise mappings can work with strided matrices.

```rust
/// Guarantees that elements are laid out contiguously in column-major order.
pub trait ContiguousStorage: MatrixStorage {
    /// Exposes the backend as a contiguous immutable flat slice.
    fn as_slice(&self) -> &[Self::Element];
}

/// Guarantees that mutable elements are laid out contiguously in column-major order.
pub trait ContiguousStorageMut: ContiguousStorage + MatrixStorageMut {
    /// Exposes the backend as a contiguous mutable flat slice.
    fn as_mut_slice(&mut self) -> &mut [Self::Element];
}

/// Defines strides for non-contiguous or submatrix layouts.
pub trait StridedStorage: MatrixStorage {
    /// Step size in elements between consecutive rows.
    fn row_stride(&self) -> usize;

    /// Step size in elements between consecutive columns.
    fn col_stride(&self) -> usize;
}
```

### 3.2. Volatile Storage Trait

Volatile memory (such as peripheral registers) cannot return standard Rust references `&T` or `&mut T` because the compiler may optimize away or reorder accesses. Therefore, volatile backends must implement a specialized copy-based interface.

```rust
/// Base trait for register-mapped matrix hardware interfaces.
pub trait VolatileMatrixStorage {
    /// The scalar type stored in the register.
    type Element: Copy;

    /// Returns the logical number of rows.
    fn rows(&self) -> usize;

    /// Returns the logical number of columns.
    fn cols(&self) -> usize;

    /// Performs a volatile read of the element at the specified index.
    ///
    /// # Safety
    /// The caller must guarantee that `row < self.rows()` and `col < self.cols()`.
    unsafe fn read_volatile_unchecked(&self, row: usize, col: usize) -> Self::Element;

    /// Performs a volatile write to the element at the specified index.
    ///
    /// # Safety
    /// The caller must guarantee that `row < self.rows()` and `col < self.cols()`.
    unsafe fn write_volatile_unchecked(&self, row: usize, col: usize, val: Self::Element);

    /// Safe volatile read.
    #[inline]
    fn read_volatile(&self, row: usize, col: usize) -> Option<Self::Element> {
        if row < self.rows() && col < self.cols() {
            Some(unsafe { self.read_volatile_unchecked(row, col) })
        } else {
            None
        }
    }

    /// Safe volatile write.
    #[inline]
    fn write_volatile(&self, row: usize, col: usize, val: Self::Element) -> ArithmeticResult<()> {
        if row < self.rows() && col < self.cols() {
            unsafe { self.write_volatile_unchecked(row, col, val) };
            Ok(())
        } else {
            Err(crate::math::ArithmeticError::DomainViolation)
        }
    }
}
```

---

## 4. Concrete Storage Implementations

1. **`ArrayStorage<T, const R: usize, const C: usize>`**
   - Owned static allocation of size `R * C`.
   - Layout is column-major contiguity (elements along a column are adjacent in memory).
   - Implements `ContiguousStorage` and `ContiguousStorageMut`.

2. **`ViewStorage<'a, T>`**
   - Borrowed view over an arbitrary region of memory.
   - Initialized with a base pointer, dimensions `(R, C)`, and strides `(row_stride, col_stride)`.
   - Implements `StridedStorage`.

3. **`ViewStorageMut<'a, T>`**
   - Mutably borrowed view over an arbitrary region of memory.
   - Initialized with a mutable base pointer, dimensions `(R, C)`, and strides `(row_stride, col_stride)`.
   - Implements `StridedStorage` and `MatrixStorageMut`.

4. **`VolatileStorage<T>`**
   - References a peripheral memory address (e.g., UART FIFO or DMA control register).
   - Implements `VolatileMatrixStorage`.

---

## 5. Testing & Verification

Verification of the storage backends consists of functional verification (unit testing) and performance validation (hardware benchmarking).

### 5.1. Functional Unit Testing

* **`no_std` Verification**: Compile the crate without `std` link dependencies. Confirm that `ArrayStorage` compiles purely on the stack with zero heap symbols.
* **Stride Correction**: Assert that `ViewStorage` indexing equations correctly account for arbitrary row/column strides. Test boundary conditions (e.g., 1xN slices, Nx1 slices, and overlapping memory representations).
* **Aliasing Invariants**: Ensure that construction of `ViewStorageMut` respects Rust's exclusive reference rules (`&mut`). Use test suites to guarantee that no two overlapping mutable views can be instantiated concurrently.

### 5.2. Bare-Metal Performance Benchmarking (Teensy 4.1 Target)

To prove that the generic storage abstractions compile down to zero-cost assembly, we will benchmark performance on a Cortex-M7 microcontroller.

* **Compilation Configuration**: Profile compiled with `opt-level = 3`, LTO enabled, and single codegen units. Trait accessors (like `get_unchecked`) must be inline-optimized.
* **Measurement setup**: Utilize stack painting (filling the stack with `0xCD` prior to execution to measure high-water marks) and the ARM DWT (Data Watchpoint and Trigger) cycle counter with interrupts disabled.
* **Assembly Verification**: Confirm via objdump disassembly that bounds checks are omitted for `get_unchecked` calls, and that the generic storage wrappers compile down to identical instructions as their raw-pointer equivalents.

#### Benchmark Profiles

1. **Owned Stack Array (`ArrayStorage`)**
   - Type: `Matrix<ArrayStorage<f32, 16, 16>>`
   - Memory profile: 1024 bytes (exactly 1KB for 256 `f32` elements) of stack space.
   - Expected latency: Minimal (cycles reflect straight-line memory access with no dereferencing indirection).

2. **Strided Matrix View (`ViewStorage`)**
   - Type: `Matrix<ViewStorage<'_, f32>>` pointing to a DMA buffer located in DTCM (Data Tightly Coupled Memory).
   - Memory profile: Overhead is limited to the view tracking information (20 bytes on 32-bit architecture: 1 pointer + 4 dimensions/strides).
   - Expected latency: Slightly higher than `ArrayStorage` due to pointer dereferencing and stride math, but still fully deterministic with zero bounds checks.