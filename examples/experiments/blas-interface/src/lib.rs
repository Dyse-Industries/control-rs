//! Codegen experiment: runtime fields vs type-level constants, across an
//! opaque boundary. Every `gemv_*` entry point is `#[no_mangle] extern "C"`
//! and lives in a `staticlib` with no in-crate caller, so LLVM never sees
//! how the argument structs are constructed and cannot inline a variant
//! away.
//!
//! `no_std`: only `core`-level slice/array/f32 operations are used, so this
//! also compiles for bare-metal (`-none-`) targets that have no `std` — see
//! `src/bin/measure.rs` for cross-compiling and disassembling those. `std`
//! is a default-on feature; `measure` builds every target (including
//! hosted ones) with `--no-default-features` for one uniform pipeline, so
//! the `#[panic_handler]` below (required by rustc for any `staticlib`
//! output, hosted or not — it doesn't change the `gemv_*` bodies being
//! measured, just satisfies crate-level lang-item resolution) is always in
//! play there. Plain `cargo build`/`cargo run` (the correctness check on
//! `src/main.rs`) keeps the default `std` feature, so this lib defers to
//! std's own panic handling as normal.
//!
//! See `src/bin/measure.rs` for the disassembly-based codegen measurement
//! and `src/main.rs` for the (std-hosted-target-only) cross-crate
//! correctness check.

#![cfg_attr(not(feature = "std"), no_std)]
// Slice/`&T` arguments on `extern "C"` gemv_* are deliberate: the fat-pointer
// length is an ABI fact LLVM cannot assume. This is not a C-stable ABI.
#![allow(improper_ctypes_definitions)]

#[cfg(not(feature = "std"))]
#[panic_handler]
fn panic(_info: &core::panic::PanicInfo<'_>) -> ! {
    loop {}
}

/// Storage order of a dense matrix buffer.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum MatrixLayout {
    /// Row-major (C) storage: consecutive elements are a row.
    RowMajor = 1,
    /// Column-major (Fortran) storage: consecutive elements are a column.
    ColMajor = 2,
}

// ---------------------------------------------------------------- variant A
// lda and order are runtime fields.

/// Variant A: `lda` and `order` are ordinary runtime struct fields.
pub struct DynDense<'a> {
    /// Backing storage, `lda * cols` (or `lda * rows`) elements long.
    pub buf: &'a [f32],
    /// Row count.
    pub rows: usize,
    /// Column count.
    pub cols: usize,
    /// Leading dimension (stride between columns/rows depending on `order`).
    pub lda: usize,
    /// Storage order of `buf`.
    pub order: MatrixLayout,
}

/// `y = A * x` for a [`DynDense`] matrix.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_dyn(s: &DynDense<'_>, x: &[f32], y: &mut [f32]) {
    for i in 0..s.rows {
        let mut acc = 0.0f32;
        for j in 0..s.cols {
            let off = match s.order {
                MatrixLayout::ColMajor => j * s.lda + i,
                MatrixLayout::RowMajor => i * s.lda + j,
            };
            acc += s.buf[off] * x[j];
        }
        y[i] = acc;
    }
}

// ------------------------------------------------------------- variant F
// Same storage struct as variant A (DynDense, runtime lda/order), but the
// gemv body itself is C, linked from csrc/gemv_c.c (see build.rs). Host-only
// and opt-in ("cffi" feature): unlike A–E / G / H / C8 / C16 / D8 / D16, this
// isn't a codegen-shape comparison for `measure` to cross-compile and
// disassemble -- it exists for src/bin/c_call_cost.rs to time the FFI call
// boundary itself.

#[cfg(feature = "cffi")]
unsafe extern "C" {
    fn gemv_c_dyn(
        buf: *const f32,
        rows: usize,
        cols: usize,
        lda: usize,
        order: i32,
        x: *const f32,
        y: *mut f32,
    );
}

/// `y = A * x` for a [`DynDense`] matrix, computed by the C function
/// `gemv_c_dyn` (`csrc/gemv_c.c`) rather than Rust.
#[cfg(feature = "cffi")]
pub fn gemv_c(s: &DynDense<'_>, x: &[f32], y: &mut [f32]) {
    // Safety: `buf` is `lda * cols` (col-major) or `lda * rows` (row-major)
    // per DynDense; `x`/`y` are `cols`/`rows` long by the same contract
    // `gemv_dyn` relies on, and stay borrowed for the call.
    unsafe {
        gemv_c_dyn(
            s.buf.as_ptr(),
            s.rows,
            s.cols,
            s.lda,
            s.order as i32,
            x.as_ptr(),
            y.as_mut_ptr(),
        );
    }
}

// ---------------------------------------------------------------- variant B
// rows, cols, lda and order are associated constants. Buffer and x/y are
// still slices, so their lengths are ABI arguments LLVM cannot assume.

/// Variant B: dimensions and layout are associated constants; only the
/// buffer is a runtime field.
pub trait ConstDense {
    /// Row count.
    const ROWS: usize;
    /// Column count.
    const COLS: usize;
    /// Leading dimension.
    const LDA: usize;
    /// Storage order of the buffer.
    const ORDER: MatrixLayout;
    /// Backing storage.
    fn buf(&self) -> &[f32];
}

/// Fixed 4x4 [`ConstDense`] matrix backed by a slice.
pub struct Dense4<'a>(pub &'a [f32]);
impl ConstDense for Dense4<'_> {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    const LDA: usize = 4;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn buf(&self) -> &[f32] {
        self.0
    }
}

/// `y = A * x` for any [`ConstDense`] matrix.
#[inline(always)]
pub fn gemv_const<S: ConstDense>(s: &S, x: &[f32], y: &mut [f32]) {
    let b = s.buf();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            acc += b[off] * x[j];
        }
        y[i] = acc;
    }
}

/// `y = A * x` for a [`Dense4`] matrix.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_const_4(s: &Dense4<'_>, x: &[f32], y: &mut [f32]) {
    gemv_const(s, x, y);
}

// ---------------------------------------------------------------- variant H
// Same ConstDense / slice buffer as B, but x and y are `[f32; 4]` so their
// length checks fold the way C's do. Whatever panic paths remain are the
// buffer's `b[off]` checks — B's 21 mixed buf + x + y together.

/// `y = A * x` for any [`ConstDense`] matrix, with fixed-length `x`/`y`.
#[inline(always)]
pub fn gemv_const_xy<S: ConstDense>(s: &S, x: &[f32; 4], y: &mut [f32; 4]) {
    let b = s.buf();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            acc += b[off] * x[j];
        }
        y[i] = acc;
    }
}

/// `y = A * x` for a [`Dense4`] matrix with `[f32; 4]` `x`/`y`.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_const_xy_4(s: &Dense4<'_>, x: &[f32; 4], y: &mut [f32; 4]) {
    gemv_const_xy(s, x, y);
}

// ---------------------------------------------------------------- variant G
// Same ConstDense storage as B, but the bounds check is written by hand --
// `i >= S::ROWS || j >= S::COLS` against the trait's own generic associated
// consts -- instead of relying on `b[off]`'s implicit slice-index check.
// Mirrors `control_rs::math::storage::Storage::get`'s
// `i < R::DIM && j < C::DIM` pattern. `x`/`y` are `[f32; 4]` so leftover
// panic paths cannot be the vector-length checks that used to confound this
// variant (those fold the same way they do in C/H). Isolates one question:
// does an explicit `if`/`panic!` over a generic const fold away as readily
// as the implicit array-index check variant C already showed disappears.

/// `y = A * x` for any [`ConstDense`] matrix, using an explicit runtime
/// bounds check against `S::ROWS`/`S::COLS` before an unchecked buffer
/// access, rather than variant B's implicit `b[off]` check.
///
/// The condition is unreachable by construction (`i`/`j` are drawn from
/// `0..S::ROWS`/`0..S::COLS`). `x`/`y` are arrays so their indexing is not
/// a second source of panic paths; any that remain are this `if`.
#[inline(always)]
pub fn gemv_checked<S: ConstDense>(s: &S, x: &[f32; 4], y: &mut [f32; 4]) {
    let b = s.buf();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            if i >= S::ROWS || j >= S::COLS {
                panic!("gemv_checked: index out of bounds");
            }
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            // Safety: every ConstDense impl in this crate has LDA == ROWS
            // == COLS, so `off < LDA * COLS`. The caller (and Dense4's
            // construction) must have `b.len() >= LDA * COLS`; the opaque
            // boundary means LLVM cannot see that length, which is why this
            // is unchecked rather than `b[off]`.
            acc += unsafe { *b.get_unchecked(off) } * x[j];
        }
        y[i] = acc;
    }
}

/// `y = A * x` for a [`Dense4`] matrix, via [`gemv_checked`].
#[unsafe(no_mangle)]
pub extern "C" fn gemv_checked_4(s: &Dense4<'_>, x: &[f32; 4], y: &mut [f32; 4]) {
    gemv_checked(s, x, y);
}

// ---------------------------------------------------------------- variant C
// Same as B, but the buffers are fixed-size arrays, so lengths are const too
// and every bounds check folds away.

/// Variant C: same as [`ConstDense`], but the buffer length `N` is also
/// const, so slice bounds checks fold away entirely.
pub trait ConstArrayDense<const N: usize> {
    /// Row count.
    const ROWS: usize;
    /// Column count.
    const COLS: usize;
    /// Leading dimension.
    const LDA: usize;
    /// Storage order of the buffer.
    const ORDER: MatrixLayout;
    /// Backing storage.
    fn buf(&self) -> &[f32; N];
}

/// Fixed 4x4 [`ConstArrayDense`] matrix backed by a `&[f32; 16]`.
pub struct Arr4<'a>(pub &'a [f32; 16]);
impl ConstArrayDense<16> for Arr4<'_> {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    const LDA: usize = 4;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn buf(&self) -> &[f32; 16] {
        self.0
    }
}

/// `y = A * x` for any [`ConstArrayDense`] matrix. `R`/`C` are the `x`/`y`
/// lengths and must match `S::ROWS`/`S::COLS` (the `#[no_mangle]` wrappers
/// pin both).
#[inline(always)]
pub fn gemv_arr<const N: usize, const R: usize, const C: usize, S: ConstArrayDense<N>>(
    s: &S,
    x: &[f32; C],
    y: &mut [f32; R],
) {
    let b = s.buf();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            acc += b[off] * x[j];
        }
        y[i] = acc;
    }
}

/// `y = A * x` for an [`Arr4`] matrix.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_arr_4(s: &Arr4<'_>, x: &[f32; 4], y: &mut [f32; 4]) {
    gemv_arr(s, x, y);
}

/// Fixed 8x8 [`ConstArrayDense`] matrix backed by a `&[f32; 64]`.
pub struct Arr8<'a>(pub &'a [f32; 64]);
impl ConstArrayDense<64> for Arr8<'_> {
    const ROWS: usize = 8;
    const COLS: usize = 8;
    const LDA: usize = 8;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn buf(&self) -> &[f32; 64] {
        self.0
    }
}

/// `y = A * x` for an [`Arr8`] matrix.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_arr_8(s: &Arr8<'_>, x: &[f32; 8], y: &mut [f32; 8]) {
    gemv_arr(s, x, y);
}

/// Fixed 16x16 [`ConstArrayDense`] matrix backed by a `&[f32; 256]`.
pub struct Arr16<'a>(pub &'a [f32; 256]);
impl ConstArrayDense<256> for Arr16<'_> {
    const ROWS: usize = 16;
    const COLS: usize = 16;
    const LDA: usize = 16;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn buf(&self) -> &[f32; 256] {
        self.0
    }
}

/// `y = A * x` for an [`Arr16`] matrix.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_arr_16(s: &Arr16<'_>, x: &[f32; 16], y: &mut [f32; 16]) {
    gemv_arr(s, x, y);
}

// ---------------------------------------------------------------- variant D
// Const dims + associated-const LDA/ORDER, but a plain `&[f32]` buffer whose
// length is NOT const. Hot path uses raw-pointer offsets, not slice indexing.

/// Variant D: same constants as [`ConstDense`], but the hot path reaches the
/// buffer through a raw pointer instead of a slice, so there is no bounds
/// check to fold away in the first place.
pub trait ConstPtrDense {
    /// Row count.
    const ROWS: usize;
    /// Column count.
    const COLS: usize;
    /// Leading dimension.
    const LDA: usize;
    /// Storage order of the buffer.
    const ORDER: MatrixLayout;
    /// Pointer to the backing storage.
    ///
    /// # Safety
    /// Valid for `LDA * COLS` elements.
    fn ptr(&self) -> *const f32;
}

/// Fixed 4x4 [`ConstPtrDense`] matrix backed by a slice.
pub struct Ptr4<'a>(pub &'a [f32]);
impl ConstPtrDense for Ptr4<'_> {
    const ROWS: usize = 4;
    const COLS: usize = 4;
    const LDA: usize = 4;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }
}

/// `y = A * x` for any [`ConstPtrDense`] matrix.
///
/// # Safety
/// `x` and `y` must hold `S::COLS` / `S::ROWS` elements.
#[inline(always)]
pub unsafe fn gemv_ptr<S: ConstPtrDense>(s: &S, x: *const f32, y: *mut f32) {
    let a = s.ptr();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            acc += unsafe { *a.add(off) * *x.add(j) };
        }
        unsafe { *y.add(i) = acc };
    }
}

/// `y = A * x` for a [`Ptr4`] matrix.
///
/// # Safety
/// `x` must hold 4 elements and `y` must hold 4 writable elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gemv_ptr_4(s: &Ptr4<'_>, x: *const f32, y: *mut f32) {
    unsafe { gemv_ptr(s, x, y) }
}

/// Fixed 8x8 [`ConstPtrDense`] matrix backed by a slice.
pub struct Ptr8<'a>(pub &'a [f32]);
impl ConstPtrDense for Ptr8<'_> {
    const ROWS: usize = 8;
    const COLS: usize = 8;
    const LDA: usize = 8;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }
}

/// `y = A * x` for a [`Ptr8`] matrix.
///
/// # Safety
/// `x` must hold 8 elements and `y` must hold 8 writable elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gemv_ptr_8(s: &Ptr8<'_>, x: *const f32, y: *mut f32) {
    unsafe { gemv_ptr(s, x, y) }
}

/// Fixed 16x16 [`ConstPtrDense`] matrix backed by a slice.
pub struct Ptr16<'a>(pub &'a [f32]);
impl ConstPtrDense for Ptr16<'_> {
    const ROWS: usize = 16;
    const COLS: usize = 16;
    const LDA: usize = 16;
    const ORDER: MatrixLayout = MatrixLayout::ColMajor;
    fn ptr(&self) -> *const f32 {
        self.0.as_ptr()
    }
}

/// `y = A * x` for a [`Ptr16`] matrix.
///
/// # Safety
/// `x` must hold 16 elements and `y` must hold 16 writable elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gemv_ptr_16(s: &Ptr16<'_>, x: *const f32, y: *mut f32) {
    unsafe { gemv_ptr(s, x, y) }
}

// ---------------------------------------------------------------- variant E
// Same access strategy as variant D (assoc consts + raw-pointer offsets),
// but computes the full BLAS-style update `y = alpha * A * x + beta * y`
// instead of a plain matrix-vector product, to check whether the extra
// scalar multiplies/loads change codegen at all.

/// `y = alpha * A * x + beta * y` for any [`ConstPtrDense`] matrix.
///
/// # Safety
/// `x` must hold `S::COLS` elements and `y` must hold `S::ROWS` read/write
/// elements.
#[inline(always)]
pub unsafe fn gemv_ptr_ab<S: ConstPtrDense>(
    s: &S,
    alpha: f32,
    x: *const f32,
    beta: f32,
    y: *mut f32,
) {
    let a = s.ptr();
    for i in 0..S::ROWS {
        let mut acc = 0.0f32;
        for j in 0..S::COLS {
            let off = match S::ORDER {
                MatrixLayout::ColMajor => j * S::LDA + i,
                MatrixLayout::RowMajor => i * S::LDA + j,
            };
            acc += unsafe { *a.add(off) * *x.add(j) };
        }
        unsafe {
            let yi = y.add(i);
            *yi = alpha * acc + beta * *yi;
        }
    }
}

/// `y = alpha * A * x + beta * y` for a [`Ptr4`] matrix.
///
/// # Safety
/// `x` must hold 4 elements and `y` must hold 4 read/write elements.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn gemv_ptr_ab_4(
    s: &Ptr4<'_>,
    alpha: f32,
    x: *const f32,
    beta: f32,
    y: *mut f32,
) {
    unsafe { gemv_ptr_ab(s, alpha, x, beta, y) }
}

// ------------------------------------------------------------- new variants
// Supporting the storage-subprogram design evaluation.

/// Trait representing general matrix storage.
pub trait MatrixStorage {
    fn rows(&self) -> usize;
    fn cols(&self) -> usize;
    fn lda(&self) -> usize;
    fn order(&self) -> MatrixLayout;
    fn as_slice(&self) -> &[f32];

    #[inline(always)]
    fn get(&self, i: usize, j: usize) -> f32 {
        let off = match self.order() {
            MatrixLayout::ColMajor => j * self.lda() + i,
            MatrixLayout::RowMajor => i * self.lda() + j,
        };
        self.as_slice()[off]
    }
}

impl MatrixStorage for Dense4<'_> {
    #[inline(always)]
    fn rows(&self) -> usize { 4 }
    #[inline(always)]
    fn cols(&self) -> usize { 4 }
    #[inline(always)]
    fn lda(&self) -> usize { 4 }
    #[inline(always)]
    fn order(&self) -> MatrixLayout { MatrixLayout::ColMajor }
    #[inline(always)]
    fn as_slice(&self) -> &[f32] { self.0 }
}

/// Trait representing general readable vector storage.
pub trait VectorStorage {
    fn len(&self) -> usize;
    fn as_slice(&self) -> &[f32];
    #[inline(always)]
    fn get(&self, i: usize) -> f32 {
        self.as_slice()[i]
    }
}

/// Trait representing general writable vector storage.
pub trait VectorStorageMut: VectorStorage {
    fn as_mut_slice(&mut self) -> &mut [f32];
    #[inline(always)]
    fn get_mut(&mut self, i: usize) -> &mut f32 {
        &mut self.as_mut_slice()[i]
    }
}

impl VectorStorage for [f32] {
    #[inline(always)]
    fn len(&self) -> usize { self.len() }
    #[inline(always)]
    fn as_slice(&self) -> &[f32] { self }
}

impl VectorStorage for [f32; 4] {
    #[inline(always)]
    fn len(&self) -> usize { 4 }
    #[inline(always)]
    fn as_slice(&self) -> &[f32] { self }
}

impl VectorStorageMut for [f32; 4] {
    #[inline(always)]
    fn as_mut_slice(&mut self) -> &mut [f32] { self }
}

// Design A: Trait with generic storage parameters.
pub trait GEMVStorage {
    fn gemv<A, X, Y>(alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y)
    where
        A: MatrixStorage,
        X: VectorStorage,
        Y: VectorStorageMut;
}

pub struct TraitGenericSubPrograms;

impl GEMVStorage for TraitGenericSubPrograms {
    #[inline(always)]
    fn gemv<A, X, Y>(alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y)
    where
        A: MatrixStorage,
        X: VectorStorage,
        Y: VectorStorageMut,
    {
        let rows = a.rows();
        let cols = a.cols();
        for i in 0..rows {
            let mut acc = 0.0f32;
            for j in 0..cols {
                acc += a.get(i, j) * x.get(j);
            }
            let yi = y.get_mut(i);
            *yi = alpha * acc + beta * *yi;
        }
    }
}

/// Variant I: `y = alpha * A * x + beta * y` via generic storage trait methods.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_storage_trait_4(s: &Dense4<'_>, x: &[f32; 4], y: &mut [f32; 4]) {
    TraitGenericSubPrograms::gemv(1.0, s, x, 1.0, y);
    // Prevent Identical Code Folding (ICF) with gemv_generic_fn_4
    unsafe {
        let _ = core::ptr::read_volatile(&x[0]);
    }
}

// Design C: Generic storage parameter on a subprogram function directly.
#[inline(always)]
pub fn gemv_generic_fn<A, X, Y>(alpha: f32, a: &A, x: &X, beta: f32, y: &mut Y)
where
    A: MatrixStorage,
    X: VectorStorage,
    Y: VectorStorageMut,
{
    let rows = a.rows();
    let cols = a.cols();
    for i in 0..rows {
        let mut acc = 0.0f32;
        for j in 0..cols {
            acc += a.get(i, j) * x.get(j);
        }
        let yi = y.get_mut(i);
        *yi = alpha * acc + beta * *yi;
    }
}

/// Variant J: `y = alpha * A * x + beta * y` via a generic function directly.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_generic_fn_4(s: &Dense4<'_>, x: &[f32; 4], y: &mut [f32; 4]) {
    gemv_generic_fn(1.0, s, x, 1.0, y);
}

// Design B: Statically sized nested-array compute ABI.
pub trait GEMVNested<const M: usize, const N: usize, const LDA: usize> {
    fn gemv(alpha: f32, a: &[[f32; LDA]; N], x: &[f32; N], beta: f32, y: &mut [f32; M]);
}

pub struct NestedSubPrograms;

impl<const M: usize, const N: usize, const LDA: usize> GEMVNested<M, N, LDA> for NestedSubPrograms {
    #[inline(always)]
    fn gemv(alpha: f32, a: &[[f32; LDA]; N], x: &[f32; N], beta: f32, y: &mut [f32; M]) {
        for i in 0..M {
            let mut acc = 0.0f32;
            for j in 0..N {
                acc += a[j][i] * x[j];
            }
            y[i] = alpha * acc + beta * y[i];
        }
    }
}

/// Variant K: `y = alpha * A * x + beta * y` via statically sized nested arrays.
#[unsafe(no_mangle)]
pub extern "C" fn gemv_nested_4(a: &[[f32; 4]; 4], x: &[f32; 4], y: &mut [f32; 4]) {
    NestedSubPrograms::gemv(1.0, a, x, 1.0, y);
}
