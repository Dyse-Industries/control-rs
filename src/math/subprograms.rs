//! Hardware-accelerable numerical subprograms for linear algebra.
//!
//! Organized according to BLAS and LAPACK conventions:
//! - **BLAS Level 1**: Vector-vector kernels ([`level1`]).
//! - **BLAS Level 2**: Matrix-vector kernels and rank-1/2 updates ([`level2`]).
//! - **Packed BLAS**: Structured packed matrix-vector subprograms ([`packed`]).
//! - **BLAS Level 3**: Matrix-matrix kernels and rank-$k$ updates ([`level3`]).
//! - **Sparse BLAS (SpBLAS)**: Compressed row/column sparse kernels ([`sparse`]).
//! - **LAPACK Direct Solvers**: Factorizations ($LU$, $QR$, Cholesky), triangular solvers, and eigensolvers ([`lapack`]).
//!
//! Provides [`DefaultBlas`] as a zero-cost reference implementation across all BLAS/LAPACK traits
//! parameterized over generic [`DenseStorage`]/[`DenseStorageMut`] operands.
//!
//! ## Infallible Interface
//!
//! Subprogram kernels are infallible at the interface level and do not return [`Result`]:
//!
//! ```compile_fail
//! use control_rs::math::storage::{ArrayStorage, Trans};
//! use control_rs::math::subprograms::DefaultBlas;
//! use control_rs::math::subprograms::level2::Gemv;
//!
//! let a = ArrayStorage::<f32, 2, 2>::zeros();
//! let x = ArrayStorage::<f32, 2, 1>::zeros();
//! let mut y = ArrayStorage::<f32, 2, 1>::zeros();
//! // Calling gemv and attempting to handle it as a Result fails to compile because gemv returns ()
//! let _res: Result<(), ()> = DefaultBlas::gemv(Trans::NoTrans, 1.0, &a, &x, 0.0, &mut y);
//! ```
// Conventional single-character argument names are standard and accepted for BLAS/LAPACK subprograms (e.g. m, n, k, A, B, C, x, y).
#![allow(clippy::many_single_char_names)]
// Direct matrix/vector indexing using brackets (slice[idx]) is used throughout this file for optimal memory layout access, bypassing bounds check branches in performance-critical BLAS loops.
#![allow(clippy::indexing_slicing)]
// Standard floating-point matrix arithmetic operations are unavoidable in generic BLAS/LAPACK implementations.
#![allow(clippy::arithmetic_side_effects)]
// BLAS/LAPACK routines naturally require many arguments (exceeding clippy's default limit of 4), conforming to standard BLAS/LAPACK APIs.
#![allow(clippy::too_many_arguments)]
// Parameter names matching BLAS standards (e.g., lda, ldb, trans_a, trans_b) look similar but are standard.
#![allow(clippy::similar_names)]
// Subprogram traits and implementations are grouped logically by BLAS Level 1/2/3 and LAPACK, rather than alphabetically.
#![allow(clippy::arbitrary_source_item_ordering)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::type_complexity)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::wildcard_imports)]
#![allow(clippy::match_like_matches_macro)]
#![allow(clippy::eq_op)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::use_self)]
#![allow(clippy::needless_pass_by_ref_mut)]

use crate::math::num_traits::{Float, One, Radical, Scalar, Zero};
use crate::math::ops::Div;
use crate::math::storage::{
    CscStorage, CsrStorage, DenseStorage, DenseStorageMut, Diag, PackedStorage,
    PackedStorageMut, Side, SparseVectorStorage, Trans, UpLo,
};
use crate::math::{LinAlgError, LinAlgResult};

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 1: Vector-Vector Operations
////////////////////////////////////////////////////////////////////////////////

/// Level 1 BLAS: Vector-vector subprograms.
pub mod level1 {
    use super::*;

    /// Scaled vector addition: $y \leftarrow \alpha x + y$.
    pub trait Axpy<T: Scalar, X: DenseStorage<T>, Y: DenseStorageMut<T>> {
        /// Computes $y \leftarrow \alpha x + y$.
        fn axpy(alpha: T, x: &X, y: &mut Y);
    }

    /// In-place scalar-vector scaling: $x \leftarrow \alpha x$.
    pub trait Scal<T: Scalar, X: DenseStorageMut<T>> {
        /// Computes $x \leftarrow \alpha x$.
        fn scal(alpha: T, x: &mut X);
    }

    /// In-place scaling of a complex vector by a real scalar: $x \leftarrow \alpha_{real} x$.
    pub trait RealScal<T: Scalar, X: DenseStorageMut<T>> {
        /// Computes $x \leftarrow \alpha x$ where $\alpha \in \mathbb{R}$.
        fn real_scal(alpha: T::Real, x: &mut X);
    }

    /// Unconjugated inner product: $x^T y = \sum_{i} x_i y_i$.
    pub trait Dotu<T: Scalar, X: DenseStorage<T>, Y: DenseStorage<T>> {
        /// Computes $x^T y$.
        fn dotu(x: &X, y: &Y) -> T;
    }

    /// Conjugated inner product: $x^H y = \sum_{i} \overline{x_i} y_i$.
    pub trait Dotc<T: Scalar, X: DenseStorage<T>, Y: DenseStorage<T>> {
        /// Computes $x^H y$.
        fn dotc(x: &X, y: &Y) -> T;
    }

    /// Sum of absolute values of real/imaginary components: $\sum (|re(x_i)| + |im(x_i)|)$.
    pub trait Asum<T: Scalar, X: DenseStorage<T>> {
        /// Computes $\sum (|re(x_i)| + |im(x_i)|)$.
        fn asum(x: &X) -> T::Real;
    }

    /// Index of element with maximum component-wise absolute sum.
    pub trait Iamax<T: Scalar, X: DenseStorage<T>> {
        /// Finds the index of the maximum absolute component element.
        fn iamax(x: &X) -> usize;
    }

    /// Swaps the elements of two vectors in place: $x \leftrightarrow y$.
    pub trait Swap<T, X: DenseStorageMut<T>, Y: DenseStorageMut<T>> {
        /// Swaps vectors $x$ and $y$.
        fn swap(x: &mut X, y: &mut Y);
    }

    /// Euclidean 2-norm: $\|x\|_2 = \sqrt{\sum |x_i|^2}$.
    pub trait Nrm2<T: Scalar, X: DenseStorage<T>>
    where
        T::Real: Radical,
    {
        /// Computes $\|x\|_2$.
        fn nrm2(x: &X) -> T::Real;
    }

    /// Applies a Givens plane rotation:
    /// $\begin{bmatrix} x_i \\ y_i \end{bmatrix} \leftarrow \begin{bmatrix} c & s \\ -\overline{s} & c \end{bmatrix} \begin{bmatrix} x_i \\ y_i \end{bmatrix}$.
    pub trait Rot<T: Scalar, X: DenseStorageMut<T>, Y: DenseStorageMut<T>> {
        /// Applies Givens rotation in place.
        fn rot(x: &mut X, y: &mut Y, c: T::Real, s: T);
    }

    // Backwards-compatibility aliases
    pub use Asum as ASUM;
    pub use Axpy as AXPY;
    pub use Dotc as DOTC;
    pub use Dotu as DOT;
    pub use Dotu as DOTU;
    pub use Iamax as IAMAX;
    pub use Nrm2 as NRM2;
    pub use RealScal as RSCAL;
    pub use Rot as ROT;
    pub use Scal as SCAL;
    pub use Swap as SWAP;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 2: Matrix-Vector Operations & Rank-1/2 Updates
////////////////////////////////////////////////////////////////////////////////

/// Level 2 BLAS: Matrix-vector subprograms.
pub mod level2 {
    use super::*;

    /// General matrix-vector multiplication: $y \leftarrow \alpha \text{op}(A) x + \beta y$.
    pub trait Gemv<
        T: Scalar,
        A: DenseStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha \text{op}(A) x + \beta y$.
        fn gemv(trans: Trans, alpha: T, a: &A, x: &X, beta: T, y: &mut Y);
    }

    /// General rank-1 unconjugated update: $A \leftarrow \alpha x y^T + A$.
    pub trait Geru<
        T: Scalar,
        A: DenseStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A \leftarrow \alpha x y^T + A$.
        fn geru(alpha: T, x: &X, y: &Y, a: &mut A);
    }

    /// General rank-1 conjugated update: $A \leftarrow \alpha x y^H + A$.
    pub trait Gerc<
        T: Scalar,
        A: DenseStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A \leftarrow \alpha x y^H + A$.
        fn gerc(alpha: T, x: &X, y: &Y, a: &mut A);
    }

    /// Symmetric matrix-vector multiplication: $y \leftarrow \alpha A x + \beta y$.
    pub trait Symv<
        T: Scalar,
        A: DenseStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A x + \beta y$ with symmetric $A$.
        fn symv(uplo: UpLo, alpha: T, a: &A, x: &X, beta: T, y: &mut Y);
    }

    /// Hermitian matrix-vector multiplication: $y \leftarrow \alpha A x + \beta y$.
    pub trait Hemv<
        T: Scalar,
        A: DenseStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A x + \beta y$ with Hermitian $A$.
        fn hemv(uplo: UpLo, alpha: T, a: &A, x: &X, beta: T, y: &mut Y);
    }

    /// Symmetric rank-1 update: $A \leftarrow \alpha x x^T + A$.
    pub trait Syr<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>> {
        /// Computes $A \leftarrow \alpha x x^T + A$.
        fn syr(uplo: UpLo, alpha: T, x: &X, a: &mut A);
    }

    /// Symmetric rank-2 update: $A \leftarrow \alpha x y^T + \alpha y x^T + A$.
    pub trait Syr2<
        T: Scalar,
        A: DenseStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A \leftarrow \alpha x y^T + \alpha y x^T + A$.
        fn syr2(uplo: UpLo, alpha: T, x: &X, y: &Y, a: &mut A);
    }

    /// Hermitian rank-1 update: $A \leftarrow \alpha_{real} x x^H + A$.
    pub trait Her<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>> {
        /// Computes $A \leftarrow \alpha x x^H + A$ with real $\alpha$.
        fn her(uplo: UpLo, alpha: T::Real, x: &X, a: &mut A);
    }

    /// Hermitian rank-2 update: $A \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A$.
    pub trait Her2<
        T: Scalar,
        A: DenseStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A$.
        fn her2(uplo: UpLo, alpha: T, x: &X, y: &Y, a: &mut A);
    }

    /// Triangular matrix-vector multiplication: $x \leftarrow \text{op}(A) x$.
    pub trait Trmv<T: Scalar, A: DenseStorage<T>, X: DenseStorageMut<T>> {
        /// Computes $x \leftarrow \text{op}(A) x$.
        fn trmv(uplo: UpLo, trans: Trans, diag: Diag, a: &A, x: &mut X);
    }

    /// Triangular system solve: $\text{op}(A) x = b$ via forward/back substitution.
    pub trait Trsv<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        X: DenseStorageMut<T>,
    >
    {
        /// Solves $\text{op}(A) x = b$ in place.
        fn trsv(
            uplo: UpLo,
            trans: Trans,
            diag: Diag,
            a: &A,
            x: &mut X,
        ) -> LinAlgResult<()>;
    }

    // Backwards-compatibility aliases
    pub use Gemv as GEMV;
    pub use Gerc as GERC;
    pub use Geru as GERU;
    pub use Hemv as HEMV;
    pub use Her as HER;
    pub use Her2 as HER2;
    pub use Symv as SYMV;
    pub use Syr as SYR;
    pub use Syr2 as SYR2;
    pub use Trmv as TRMV;
    pub use Trsv as TRSV;
}

////////////////////////////////////////////////////////////////////////////////
// Packed BLAS: Structured Packed Matrix Operations
////////////////////////////////////////////////////////////////////////////////

/// Packed BLAS subprograms.
pub mod packed {
    use super::*;

    /// Symmetric packed matrix-vector multiplication: $y \leftarrow \alpha A_{pack} x + \beta y$.
    pub trait Spmv<
        T: Scalar,
        AP: PackedStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A_{pack} x + \beta y$.
        fn spmv(uplo: UpLo, alpha: T, ap: &AP, x: &X, beta: T, y: &mut Y);
    }

    /// Hermitian packed matrix-vector multiplication: $y \leftarrow \alpha A_{pack} x + \beta y$.
    pub trait Hpmv<
        T: Scalar,
        HP: PackedStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A_{pack} x + \beta y$.
        fn hpmv(uplo: UpLo, alpha: T, hp: &HP, x: &X, beta: T, y: &mut Y);
    }

    /// Symmetric packed rank-1 update: $A_{pack} \leftarrow \alpha x x^T + A_{pack}$.
    pub trait Spr<T: Scalar, AP: PackedStorageMut<T>, X: DenseStorage<T>> {
        /// Computes $A_{pack} \leftarrow \alpha x x^T + A_{pack}$.
        fn spr(uplo: UpLo, alpha: T, x: &X, ap: &mut AP);
    }

    /// Hermitian packed rank-1 update: $A_{pack} \leftarrow \alpha_{real} x x^H + A_{pack}$.
    pub trait Hpr<T: Scalar, HP: PackedStorageMut<T>, X: DenseStorage<T>> {
        /// Computes $A_{pack} \leftarrow \alpha x x^H + A_{pack}$.
        fn hpr(uplo: UpLo, alpha: T::Real, x: &X, hp: &mut HP);
    }

    /// Symmetric packed rank-2 update: $A_{pack} \leftarrow \alpha x y^T + \alpha y x^T + A_{pack}$.
    pub trait Spr2<
        T: Scalar,
        AP: PackedStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A_{pack} \leftarrow \alpha x y^T + \alpha y x^T + A_{pack}$.
        fn spr2(uplo: UpLo, alpha: T, x: &X, y: &Y, ap: &mut AP);
    }

    /// Hermitian packed rank-2 update: $A_{pack} \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A_{pack}$.
    pub trait Hpr2<
        T: Scalar,
        HP: PackedStorageMut<T>,
        X: DenseStorage<T>,
        Y: DenseStorage<T>,
    >
    {
        /// Computes $A_{pack} \leftarrow \alpha x y^H + \overline{\alpha} y x^H + A_{pack}$.
        fn hpr2(uplo: UpLo, alpha: T, x: &X, y: &Y, hp: &mut HP);
    }

    /// Triangular packed matrix-vector multiplication: $x \leftarrow \text{op}(A_{pack}) x$.
    pub trait Tpmv<T: Scalar, TP: PackedStorage<T>, X: DenseStorageMut<T>> {
        /// Computes $x \leftarrow \text{op}(A_{pack}) x$.
        fn tpmv(uplo: UpLo, trans: Trans, diag: Diag, tp: &TP, x: &mut X);
    }

    /// Triangular packed system solve: $\text{op}(A_{pack}) x = b$.
    pub trait Tpsv<
        T: Scalar + Div<Output = T>,
        TP: PackedStorage<T>,
        X: DenseStorageMut<T>,
    >
    {
        /// Solves $\text{op}(A_{pack}) x = b$ in place.
        fn tpsv(
            uplo: UpLo,
            trans: Trans,
            diag: Diag,
            tp: &TP,
            x: &mut X,
        ) -> LinAlgResult<()>;
    }

    // Backwards-compatibility aliases
    pub use Hpmv as HPMV;
    pub use Hpr as HPR;
    pub use Hpr2 as HPR2;
    pub use Spmv as SPMV;
    pub use Spr as SPR;
    pub use Spr2 as SPR2;
    pub use Tpmv as TPMV;
    pub use Tpsv as TPSV;
}

////////////////////////////////////////////////////////////////////////////////
// BLAS Level 3: Matrix-Matrix Operations & Rank-K Updates
////////////////////////////////////////////////////////////////////////////////

/// Level 3 BLAS: Matrix-matrix subprograms.
pub mod level3 {
    use super::*;

    /// General matrix-matrix multiplication: $C \leftarrow \alpha \text{op}(A) \text{op}(B) + \beta C$.
    pub trait Gemm<
        T: Scalar,
        A: DenseStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha \text{op}(A) \text{op}(B) + \beta C$.
        fn gemm(
            ta: Trans,
            tb: Trans,
            alpha: T,
            a: &A,
            b: &B,
            beta: T,
            c: &mut C,
        );
    }

    /// Symmetric matrix-matrix multiply: $C \leftarrow \alpha A B + \beta C$ or $C \leftarrow \alpha B A + \beta C$.
    pub trait Symm<
        T: Scalar,
        A: DenseStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha A B + \beta C$ or $C \leftarrow \alpha B A + \beta C$ where $A$ is symmetric.
        fn symm(
            side: Side,
            uplo: UpLo,
            alpha: T,
            a: &A,
            b: &B,
            beta: T,
            c: &mut C,
        );
    }

    /// Hermitian matrix-matrix multiply: $C \leftarrow \alpha A B + \beta C$ or $C \leftarrow \alpha B A + \beta C$.
    pub trait Hemm<
        T: Scalar,
        A: DenseStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha A B + \beta C$ or $C \leftarrow \alpha B A + \beta C$ where $A$ is Hermitian.
        fn hemm(
            side: Side,
            uplo: UpLo,
            alpha: T,
            a: &A,
            b: &B,
            beta: T,
            c: &mut C,
        );
    }

    /// Symmetric rank-$k$ update: $C \leftarrow \alpha A A^T + \beta C$ or $C \leftarrow \alpha A^T A + \beta C$.
    pub trait Syrk<T: Scalar, A: DenseStorage<T>, C: DenseStorageMut<T>> {
        /// Computes $C \leftarrow \alpha A A^T + \beta C$ (or $A^T A$).
        fn syrk(uplo: UpLo, trans: Trans, alpha: T, a: &A, beta: T, c: &mut C);
    }

    /// Hermitian rank-$k$ update: $C \leftarrow \alpha A A^H + \beta C$.
    pub trait Herk<T: Scalar, A: DenseStorage<T>, C: DenseStorageMut<T>> {
        /// Computes $C \leftarrow \alpha A A^H + \beta C$ with real $\alpha, \beta$.
        fn herk(
            uplo: UpLo,
            trans: Trans,
            alpha: T::Real,
            a: &A,
            beta: T::Real,
            c: &mut C,
        );
    }

    /// Symmetric rank-$2k$ update: $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$.
    pub trait Syr2k<
        T: Scalar,
        A: DenseStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha A B^T + \alpha B A^T + \beta C$.
        fn syr2k(
            uplo: UpLo,
            trans: Trans,
            alpha: T,
            a: &A,
            b: &B,
            beta: T,
            c: &mut C,
        );
    }

    /// Hermitian rank-$2k$ update: $C \leftarrow \alpha A B^H + \overline{\alpha} B A^H + \beta C$.
    pub trait Her2k<
        T: Scalar,
        A: DenseStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha A B^H + \overline{\alpha} B A^H + \beta C$.
        fn her2k(
            uplo: UpLo,
            trans: Trans,
            alpha: T,
            a: &A,
            b: &B,
            beta: T::Real,
            c: &mut C,
        );
    }

    /// Triangular matrix-matrix multiplication: $B \leftarrow \alpha \text{op}(A) B$ or $B \leftarrow \alpha B \text{op}(A)$.
    pub trait Trmm<T: Scalar, A: DenseStorage<T>, B: DenseStorageMut<T>> {
        /// Computes $B \leftarrow \alpha \text{op}(A) B$ or $B \leftarrow \alpha B \text{op}(A)$.
        fn trmm(
            side: Side,
            uplo: UpLo,
            trans: Trans,
            diag: Diag,
            alpha: T,
            a: &A,
            b: &mut B,
        );
    }

    /// Triangular matrix-matrix solve: $\text{op}(A) X = \alpha B$ or $X \text{op}(A) = \alpha B$.
    pub trait Trsm<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        B: DenseStorageMut<T>,
    >
    {
        /// Solves triangular matrix equations in place over $B$.
        fn trsm(
            side: Side,
            uplo: UpLo,
            trans: Trans,
            diag: Diag,
            alpha: T,
            a: &A,
            b: &mut B,
        ) -> LinAlgResult<()>;
    }

    // Backwards-compatibility aliases
    pub use Gemm as GEMM;
    pub use Hemm as HEMM;
    pub use Her2k as HER2K;
    pub use Herk as HERK;
    pub use Symm as SYMM;
    pub use Syr2k as SYR2K;
    pub use Syrk as SYRK;
    pub use Trmm as TRMM;
    pub use Trsm as TRSM;
}

////////////////////////////////////////////////////////////////////////////////
// Sparse BLAS: Compressed Row/Column Matrix & Sparse Vector Operations
////////////////////////////////////////////////////////////////////////////////

/// Sparse BLAS subprograms.
pub mod sparse {
    use super::*;

    /// CSR matrix-vector multiplication: $y \leftarrow \alpha A_{csr} x + \beta y$.
    pub trait Csrmv<
        T: Scalar,
        A: CsrStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A_{csr} x + \beta y$.
        fn csrmv(alpha: T, a: &A, x: &X, beta: T, y: &mut Y);
    }

    /// CSC matrix-vector multiplication: $y \leftarrow \alpha A_{csc} x + \beta y$.
    pub trait Cscmv<
        T: Scalar,
        A: CscStorage<T>,
        X: DenseStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha A_{csc} x + \beta y$.
        fn cscmv(alpha: T, a: &A, x: &X, beta: T, y: &mut Y);
    }

    /// CSR matrix-dense matrix multiplication: $C \leftarrow \alpha A_{csr} B + \beta C$.
    pub trait Csrmm<
        T: Scalar,
        A: CsrStorage<T>,
        B: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Computes $C \leftarrow \alpha A_{csr} B + \beta C$.
        fn csrmm(alpha: T, a: &A, b: &B, beta: T, c: &mut C);
    }

    /// Sparse vector unconjugated inner product: $x_{sparse}^T y$.
    pub trait SpDotu<T: Scalar, X: SparseVectorStorage<T>, Y: DenseStorage<T>> {
        /// Computes $x_{sparse}^T y$.
        fn sp_dotu(x: &X, y: &Y) -> T;
    }

    /// Sparse vector conjugated inner product: $x_{sparse}^H y$.
    pub trait SpDotc<T: Scalar, X: SparseVectorStorage<T>, Y: DenseStorage<T>> {
        /// Computes $x_{sparse}^H y$.
        fn sp_dotc(x: &X, y: &Y) -> T;
    }

    /// Sparse vector scaled addition: $y \leftarrow \alpha x_{sparse} + y$.
    pub trait SpAxpy<
        T: Scalar,
        X: SparseVectorStorage<T>,
        Y: DenseStorageMut<T>,
    >
    {
        /// Computes $y \leftarrow \alpha x_{sparse} + y$.
        fn sp_axpy(alpha: T, x: &X, y: &mut Y);
    }

    // Backwards-compatibility aliases
    pub use Cscmv as CSCMV;
    pub use Csrmm as CSRMM;
    pub use Csrmv as CSRMV;
    pub use SpAxpy as SPAXPY;
    pub use SpDotc as SPDOTC;
    pub use SpDotu as SPDOTU;
}

////////////////////////////////////////////////////////////////////////////////
// LAPACK: Direct Solvers, Factorizations, and Eigensolvers
////////////////////////////////////////////////////////////////////////////////

/// Selection of eigenvector computation for spectral decompositions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JobZ {
    /// Compute eigenvalues only.
    NoVectors,
    /// Compute eigenvalues and orthonormal eigenvectors.
    Vectors,
}

/// LAPACK direct solvers, factorizations, and spectral algorithms.
pub mod lapack {
    use super::*;

    /// Cholesky factorization: $A = L L^T$ (or $U^T U$) for symmetric/Hermitian positive-definite matrices.
    pub trait Potrf<T: Scalar + Div<Output = T>, A: DenseStorageMut<T>>
    where
        T::Real: Radical,
    {
        /// Computes Cholesky factorization in place.
        fn potrf(uplo: UpLo, a: &mut A) -> LinAlgResult<()>;
    }

    /// Solves $A X = B$ using factored Cholesky form from [`Potrf`].
    pub trait Potrs<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        B: DenseStorageMut<T>,
    >
    {
        /// Solves $A X = B$ in place.
        fn potrs(uplo: UpLo, a: &A, b: &mut B) -> LinAlgResult<()>;
    }

    /// Packed Cholesky factorization: $A_{pack} = L L^T$ (or $U^T U$).
    pub trait Pptrf<T: Scalar + Div<Output = T>, AP: PackedStorageMut<T>>
    where
        T::Real: Radical,
    {
        /// Computes packed Cholesky factorization in place.
        fn pptrf(uplo: UpLo, ap: &mut AP) -> LinAlgResult<()>;
    }

    /// Solves $A_{pack} X = B$ using factored packed Cholesky form from [`Pptrf`].
    pub trait Pptrs<
        T: Scalar + Div<Output = T>,
        AP: PackedStorage<T>,
        B: DenseStorageMut<T>,
    >
    {
        /// Solves $A_{pack} X = B$ in place.
        fn pptrs(uplo: UpLo, ap: &AP, b: &mut B) -> LinAlgResult<()>;
    }

    /// General LU factorization with partial row pivoting: $P A = L U$.
    pub trait Getrf<T: Scalar + Div<Output = T>, A: DenseStorageMut<T>> {
        /// Computes $P A = L U$ in place.
        fn getrf(a: &mut A, ipiv: &mut [usize]) -> LinAlgResult<()>;
    }

    /// Solves $A X = B$ using factored LU form from [`Getrf`].
    pub trait Getrs<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        B: DenseStorageMut<T>,
    >
    {
        /// Solves $A X = B$ in place.
        fn getrs(
            trans: Trans,
            a: &A,
            ipiv: &[usize],
            b: &mut B,
        ) -> LinAlgResult<()>;
    }

    /// Householder QR factorization: $A = Q R$.
    pub trait Geqrf<T: Scalar + Div<Output = T>, A: DenseStorageMut<T>>
    where
        T::Real: Radical,
    {
        /// Computes $A = Q R$ in place.
        fn geqrf(a: &mut A, tau: &mut [T], work: &mut [T]) -> LinAlgResult<()>;
    }

    /// Applies real orthogonal matrix $Q$ from QR factorization: $C \leftarrow \text{op}(Q) C$ or $C \leftarrow C \text{op}(Q)$.
    pub trait Ormqr<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Multiplies by $Q$ or $Q^T$.
        fn ormqr(
            side: Side,
            trans: Trans,
            a: &A,
            tau: &[T],
            c: &mut C,
            work: &mut [T],
        ) -> LinAlgResult<()>;
    }

    /// Applies complex unitary matrix $Q$ from QR factorization: $C \leftarrow \text{op}(Q) C$ or $C \leftarrow C \text{op}(Q)$.
    pub trait Unmqr<
        T: Scalar + Div<Output = T>,
        A: DenseStorage<T>,
        C: DenseStorageMut<T>,
    >
    {
        /// Multiplies by $Q$ or $Q^H$.
        fn unmqr(
            side: Side,
            trans: Trans,
            a: &A,
            tau: &[T],
            c: &mut C,
            work: &mut [T],
        ) -> LinAlgResult<()>;
    }

    /// Real symmetric Jacobi eigensolver: computes all eigenvalues and eigenvectors of $A = A^T$.
    pub trait Syev<T: Scalar + Div<Output = T>, A: DenseStorageMut<T>>
    where
        T::Real: Float,
    {
        /// Computes eigenvalues and optional eigenvectors using classical Jacobi rotations.
        fn syev(
            jobz: JobZ,
            uplo: UpLo,
            a: &mut A,
            w: &mut [T::Real],
            work: &mut [T],
        ) -> LinAlgResult<()>;
    }

    /// Complex Hermitian Jacobi eigensolver: computes all eigenvalues and eigenvectors of $A = A^H$.
    pub trait Heev<T: Scalar + Div<Output = T>, A: DenseStorageMut<T>>
    where
        T::Real: Float,
    {
        /// Computes eigenvalues and optional orthonormal eigenvectors of a complex Hermitian matrix.
        fn heev(
            jobz: JobZ,
            uplo: UpLo,
            a: &mut A,
            w: &mut [T::Real],
            work: &mut [T],
        ) -> LinAlgResult<()>;
    }

    // Backwards-compatibility aliases
    pub use Geqrf as GEQRF;
    pub use Getrf as GETRF;
    pub use Getrs as GETRS;
    pub use Heev as HEEV;
    pub use Ormqr as ORMQR;
    pub use Potrf as POTRF;
    pub use Potrs as POTRS;
    pub use Pptrf as PPTRF;
    pub use Pptrs as PPTRS;
    pub use Syev as SYEV;
    pub use Unmqr as UNMQR;
}

////////////////////////////////////////////////////////////////////////////////
// DefaultBlas Engine: Universal Generic Reference Implementation
////////////////////////////////////////////////////////////////////////////////

/// The standard, zero-dependency reference BLAS/LAPACK engine implementing all subprogram traits.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DefaultBlas;

// --- Level 1 Implementation over DenseStorage & DenseStorageMut ---

impl<T: Scalar, X: DenseStorage<T>, Y: DenseStorageMut<T>> level1::Axpy<T, X, Y>
    for DefaultBlas
{
    #[inline(always)]
    fn axpy(alpha: T, x: &X, y: &mut Y) {
        let rows = x.rows();
        let cols = x.cols();
        debug_assert_eq!(rows, y.rows());
        debug_assert_eq!(cols, y.cols());
        for r in 0..rows {
            for c in 0..cols {
                unsafe {
                    let xi = x.get_unchecked(r, c).clone();
                    let yi = y.get_unchecked(r, c).clone();
                    y.set_unchecked(r, c, yi + (alpha.clone() * xi));
                }
            }
        }
    }
}

impl<T: Scalar, X: DenseStorageMut<T>> level1::Scal<T, X> for DefaultBlas {
    #[inline(always)]
    fn scal(alpha: T, x: &mut X) {
        for r in 0..x.rows() {
            for c in 0..x.cols() {
                unsafe {
                    let xi = x.get_unchecked(r, c).clone();
                    x.set_unchecked(r, c, alpha.clone() * xi);
                }
            }
        }
    }
}

impl<T: Scalar, X: DenseStorageMut<T>> level1::RealScal<T, X> for DefaultBlas {
    #[inline(always)]
    fn real_scal(alpha: T::Real, x: &mut X) {
        let a = T::from_real(alpha);
        for r in 0..x.rows() {
            for c in 0..x.cols() {
                unsafe {
                    let xi = x.get_unchecked(r, c).clone();
                    x.set_unchecked(r, c, a.clone() * xi);
                }
            }
        }
    }
}

impl<T: Scalar, X: DenseStorage<T>, Y: DenseStorage<T>> level1::Dotu<T, X, Y>
    for DefaultBlas
{
    #[inline(always)]
    fn dotu(x: &X, y: &Y) -> T {
        let rows = x.rows();
        let cols = x.cols();
        debug_assert_eq!(rows, y.rows());
        debug_assert_eq!(cols, y.cols());
        let mut acc = T::ZERO;
        for r in 0..rows {
            for c in 0..cols {
                unsafe {
                    acc = acc
                        + (x.get_unchecked(r, c).clone()
                            * y.get_unchecked(r, c).clone());
                }
            }
        }
        acc
    }
}

impl<T: Scalar, X: DenseStorage<T>, Y: DenseStorage<T>> level1::Dotc<T, X, Y>
    for DefaultBlas
{
    #[inline(always)]
    fn dotc(x: &X, y: &Y) -> T {
        let rows = x.rows();
        let cols = x.cols();
        debug_assert_eq!(rows, y.rows());
        debug_assert_eq!(cols, y.cols());
        let mut acc = T::ZERO;
        for r in 0..rows {
            for c in 0..cols {
                unsafe {
                    acc = acc
                        + (x.get_unchecked(r, c).clone().conj()
                            * y.get_unchecked(r, c).clone());
                }
            }
        }
        acc
    }
}

impl<T: Scalar, X: DenseStorage<T>> level1::Asum<T, X> for DefaultBlas {
    #[inline(always)]
    fn asum(x: &X) -> T::Real {
        let mut acc = <T::Real as Zero>::ZERO;
        for r in 0..x.rows() {
            for c in 0..x.cols() {
                unsafe {
                    let elem = x.get_unchecked(r, c);
                    let re = elem.re();
                    let im = elem.im();
                    let re_abs = if re < <T::Real as Zero>::ZERO {
                        <T::Real as Zero>::ZERO - re
                    } else {
                        re
                    };
                    let im_abs = if im < <T::Real as Zero>::ZERO {
                        <T::Real as Zero>::ZERO - im
                    } else {
                        im
                    };
                    acc = acc + re_abs + im_abs;
                }
            }
        }
        acc
    }
}

impl<T: Scalar, X: DenseStorage<T>> level1::Iamax<T, X> for DefaultBlas {
    #[inline(always)]
    fn iamax(x: &X) -> usize {
        let total = x.rows() * x.cols();
        if total == 0 {
            return 0;
        }
        let mut max_val = <T::Real as Zero>::ZERO;
        let mut max_idx = 0;
        let mut idx = 0;
        for c in 0..x.cols() {
            for r in 0..x.rows() {
                unsafe {
                    let elem = x.get_unchecked(r, c);
                    let re = elem.re();
                    let im = elem.im();
                    let re_abs = if re < <T::Real as Zero>::ZERO {
                        <T::Real as Zero>::ZERO - re
                    } else {
                        re
                    };
                    let im_abs = if im < <T::Real as Zero>::ZERO {
                        <T::Real as Zero>::ZERO - im
                    } else {
                        im
                    };
                    let val = re_abs + im_abs;
                    if idx == 0 || val > max_val {
                        max_val = val;
                        max_idx = idx;
                    }
                }
                idx += 1;
            }
        }
        max_idx
    }
}

impl<T: Clone, X: DenseStorageMut<T>, Y: DenseStorageMut<T>>
    level1::Swap<T, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn swap(x: &mut X, y: &mut Y) {
        let rows = x.rows();
        let cols = x.cols();
        debug_assert_eq!(rows, y.rows());
        debug_assert_eq!(cols, y.cols());
        for r in 0..rows {
            for c in 0..cols {
                unsafe {
                    let xv = x.get_unchecked(r, c).clone();
                    let yv = y.get_unchecked(r, c).clone();
                    x.set_unchecked(r, c, yv);
                    y.set_unchecked(r, c, xv);
                }
            }
        }
    }
}

impl<T: Scalar, X: DenseStorage<T>> level1::Nrm2<T, X> for DefaultBlas
where
    T::Real: Radical,
{
    #[inline(always)]
    fn nrm2(x: &X) -> T::Real {
        let mut sum2 = <T::Real as Zero>::ZERO;
        for r in 0..x.rows() {
            for c in 0..x.cols() {
                unsafe {
                    sum2 = sum2 + x.get_unchecked(r, c).abs2();
                }
            }
        }
        sum2.sqrt()
    }
}

impl<T: Scalar, X: DenseStorageMut<T>, Y: DenseStorageMut<T>>
    level1::Rot<T, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn rot(x: &mut X, y: &mut Y, c: T::Real, s: T) {
        let rows = x.rows();
        let cols = x.cols();
        debug_assert_eq!(rows, y.rows());
        debug_assert_eq!(cols, y.cols());
        let c_s = T::from_real(c);
        let s_conj = s.clone().conj();
        for r in 0..rows {
            for col in 0..cols {
                unsafe {
                    let xv = x.get_unchecked(r, col).clone();
                    let yv = y.get_unchecked(r, col).clone();
                    let new_x =
                        (c_s.clone() * xv.clone()) + (s.clone() * yv.clone());
                    let new_y = (c_s.clone() * yv) - (s_conj.clone() * xv);
                    x.set_unchecked(r, col, new_x);
                    y.set_unchecked(r, col, new_y);
                }
            }
        }
    }
}

// --- Level 2 Implementation over DenseStorage & DenseStorageMut ---

impl<T: Scalar, A: DenseStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    level2::Gemv<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn gemv(trans: Trans, alpha: T, a: &A, x: &X, beta: T, y: &mut Y) {
        let (m, n) = match trans {
            Trans::NoTrans => (a.rows(), a.cols()),
            Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
        };
        let x_len = x.rows() * x.cols();
        let y_len = y.rows() * y.cols();
        debug_assert_eq!(n, x_len);
        debug_assert_eq!(m, y_len);

        // NaN-safe beta scaling (C-3)
        if beta.is_zero() {
            for i in 0..m {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for i in 0..m {
            let mut dot = T::ZERO;
            for j in 0..n {
                let (ar, ac) = match trans {
                    Trans::NoTrans => (i, j),
                    Trans::Trans | Trans::ConjTrans => (j, i),
                };
                let a_val = unsafe {
                    let elem = a.get_unchecked(ar, ac).clone();
                    if trans == Trans::ConjTrans {
                        elem.conj()
                    } else {
                        elem
                    }
                };
                let (rx, cx) =
                    if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                dot = dot + (a_val * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * dot));
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    level2::Geru<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn geru(alpha: T, x: &X, y: &Y, a: &mut A) {
        let m = a.rows();
        let n = a.cols();
        for i in 0..m {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xv = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                let yv = unsafe { y.get_unchecked(ry, cy).clone() };
                unsafe {
                    let a_val = a.get_unchecked(i, j).clone();
                    a.set_unchecked(
                        i,
                        j,
                        a_val + (alpha.clone() * xv.clone() * yv),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    level2::Gerc<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn gerc(alpha: T, x: &X, y: &Y, a: &mut A) {
        let m = a.rows();
        let n = a.cols();
        for i in 0..m {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xv = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                let yv = unsafe { y.get_unchecked(ry, cy).clone().conj() };
                unsafe {
                    let a_val = a.get_unchecked(i, j).clone();
                    a.set_unchecked(
                        i,
                        j,
                        a_val + (alpha.clone() * xv.clone() * yv),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    level2::Symv<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn symv(uplo: UpLo, alpha: T, a: &A, x: &X, beta: T, y: &mut Y) {
        let n = a.rows();
        debug_assert_eq!(n, a.cols());

        if beta.is_zero() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for i in 0..n {
            let mut dot = T::ZERO;
            for j in 0..n {
                let (r, c) = match uplo {
                    UpLo::Upper => {
                        if i <= j {
                            (i, j)
                        } else {
                            (j, i)
                        }
                    }
                    UpLo::Lower => {
                        if i >= j {
                            (i, j)
                        } else {
                            (j, i)
                        }
                    }
                };
                let a_val = unsafe { a.get_unchecked(r, c).clone() };
                let (rx, cx) =
                    if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                dot = dot + (a_val * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * dot));
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    level2::Hemv<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn hemv(uplo: UpLo, alpha: T, a: &A, x: &X, beta: T, y: &mut Y) {
        let n = a.rows();
        debug_assert_eq!(n, a.cols());

        if beta.is_zero() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for i in 0..n {
            let mut dot = T::ZERO;
            for j in 0..n {
                let a_val = match uplo {
                    UpLo::Upper => {
                        if i <= j {
                            unsafe { a.get_unchecked(i, j).clone() }
                        } else {
                            unsafe { a.get_unchecked(j, i).clone().conj() }
                        }
                    }
                    UpLo::Lower => {
                        if i >= j {
                            unsafe { a.get_unchecked(i, j).clone() }
                        } else {
                            unsafe { a.get_unchecked(j, i).clone().conj() }
                        }
                    }
                };
                let (rx, cx) =
                    if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                dot = dot + (a_val * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * dot));
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>> level2::Syr<T, A, X>
    for DefaultBlas
{
    #[inline(always)]
    fn syr(uplo: UpLo, alpha: T, x: &X, a: &mut A) {
        let n = a.rows();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (ry, cy) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(ry, cy).clone() };
                    unsafe {
                        let a_val = a.get_unchecked(i, j).clone();
                        a.set_unchecked(
                            i,
                            j,
                            a_val + (alpha.clone() * xi.clone() * xj),
                        );
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    level2::Syr2<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn syr2(uplo: UpLo, alpha: T, x: &X, y: &Y, a: &mut A) {
        let n = a.rows();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            let yi = unsafe { y.get_unchecked(ry, cy).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (rx2, cx2) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rx2, cx2).clone() };
                    let (ry2, cy2) =
                        if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                    let yj = unsafe { y.get_unchecked(ry2, cy2).clone() };
                    unsafe {
                        let a_val = a.get_unchecked(i, j).clone();
                        a.set_unchecked(
                            i,
                            j,
                            a_val
                                + (alpha.clone() * xi.clone() * yj)
                                + (alpha.clone() * yi.clone() * xj),
                        );
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>> level2::Her<T, A, X>
    for DefaultBlas
{
    #[inline(always)]
    fn her(uplo: UpLo, alpha: T::Real, x: &X, a: &mut A) {
        let n = a.rows();
        let a_scalar = T::from_real(alpha);
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (ry, cy) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(ry, cy).clone().conj() };
                    unsafe {
                        let a_val = a.get_unchecked(i, j).clone();
                        a.set_unchecked(
                            i,
                            j,
                            a_val + (a_scalar.clone() * xi.clone() * xj),
                        );
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    level2::Her2<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn her2(uplo: UpLo, alpha: T, x: &X, y: &Y, a: &mut A) {
        let n = a.rows();
        let alpha_conj = alpha.clone().conj();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            let yi = unsafe { y.get_unchecked(ry, cy).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (rx2, cx2) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj =
                        unsafe { x.get_unchecked(rx2, cx2).clone().conj() };
                    let (ry2, cy2) =
                        if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                    let yj =
                        unsafe { y.get_unchecked(ry2, cy2).clone().conj() };
                    unsafe {
                        let a_val = a.get_unchecked(i, j).clone();
                        a.set_unchecked(
                            i,
                            j,
                            a_val
                                + (alpha.clone() * xi.clone() * yj)
                                + (alpha_conj.clone() * yi.clone() * xj),
                        );
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, X: DenseStorageMut<T>> level2::Trmv<T, A, X>
    for DefaultBlas
{
    #[inline(always)]
    fn trmv(uplo: UpLo, trans: Trans, diag: Diag, a: &A, x: &mut X) {
        let n = a.rows();
        let forward = matches!(
            (uplo, trans),
            (UpLo::Upper, Trans::NoTrans)
                | (UpLo::Lower, Trans::Trans | Trans::ConjTrans)
        );
        let mut i = if forward { 0 } else { n };
        while if forward { i < n } else { i > 0 } {
            if !forward {
                i -= 1;
            }
            let mut acc = T::ZERO;
            for j in 0..n {
                let (r, c) = match trans {
                    Trans::NoTrans => (i, j),
                    Trans::Trans | Trans::ConjTrans => (j, i),
                };
                let in_tri = match uplo {
                    UpLo::Upper => r <= c,
                    UpLo::Lower => r >= c,
                };
                if in_tri {
                    let a_val = if diag == Diag::Unit && r == c {
                        T::ONE
                    } else {
                        let elem = unsafe { a.get_unchecked(r, c).clone() };
                        if trans == Trans::ConjTrans {
                            elem.conj()
                        } else {
                            elem
                        }
                    };
                    let (rx, cx) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                    acc = acc + (a_val * xv);
                }
            }
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            unsafe {
                x.set_unchecked(rx, cx, acc);
            }
            if forward {
                i += 1;
            }
        }
    }
}

impl<T: Scalar + Div<Output = T>, A: DenseStorage<T>, X: DenseStorageMut<T>>
    level2::Trsv<T, A, X> for DefaultBlas
{
    #[inline(always)]
    fn trsv(
        uplo: UpLo,
        trans: Trans,
        diag: Diag,
        a: &A,
        x: &mut X,
    ) -> LinAlgResult<()> {
        let n = a.rows();
        let is_upper = match (uplo, trans) {
            (UpLo::Upper, Trans::NoTrans) => true,
            (UpLo::Lower, Trans::Trans | Trans::ConjTrans) => true,
            _ => false,
        };

        if is_upper {
            for k in 0..n {
                let i = n - 1 - k;
                let (rxi, cxi) =
                    if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
                let mut sum = unsafe { x.get_unchecked(rxi, cxi).clone() };
                for j in (i + 1)..n {
                    let (r, c) = match trans {
                        Trans::NoTrans => (i, j),
                        Trans::Trans | Trans::ConjTrans => (j, i),
                    };
                    let elem = unsafe { a.get_unchecked(r, c).clone() };
                    let a_val = if trans == Trans::ConjTrans {
                        elem.conj()
                    } else {
                        elem
                    };
                    let (rxj, cxj) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rxj, cxj).clone() };
                    sum = sum - (a_val * xj);
                }
                if diag == Diag::Unit {
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum);
                    }
                } else {
                    let piv = unsafe { a.get_unchecked(i, i).clone() };
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    let piv_val = if trans == Trans::ConjTrans {
                        piv.conj()
                    } else {
                        piv
                    };
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum / piv_val);
                    }
                }
            }
        } else {
            for i in 0..n {
                let (rxi, cxi) =
                    if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
                let mut sum = unsafe { x.get_unchecked(rxi, cxi).clone() };
                for j in 0..i {
                    let (r, c) = match trans {
                        Trans::NoTrans => (i, j),
                        Trans::Trans | Trans::ConjTrans => (j, i),
                    };
                    let elem = unsafe { a.get_unchecked(r, c).clone() };
                    let a_val = if trans == Trans::ConjTrans {
                        elem.conj()
                    } else {
                        elem
                    };
                    let (rxj, cxj) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rxj, cxj).clone() };
                    sum = sum - (a_val * xj);
                }
                if diag == Diag::Unit {
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum);
                    }
                } else {
                    let piv = unsafe { a.get_unchecked(i, i).clone() };
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    let piv_val = if trans == Trans::ConjTrans {
                        piv.conj()
                    } else {
                        piv
                    };
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum / piv_val);
                    }
                }
            }
        }
        Ok(())
    }
}

// --- Packed BLAS Implementation ---

impl<T: Scalar, AP: PackedStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    packed::Spmv<T, AP, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn spmv(uplo: UpLo, alpha: T, ap: &AP, x: &X, beta: T, y: &mut Y) {
        let n = ap.dim();
        if beta.is_zero() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for i in 0..n {
            let mut acc = T::ZERO;
            for j in 0..n {
                let a_val = ap.value_unchecked(i, j);
                let (rx, cx) =
                    if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                acc = acc + (a_val * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * acc));
            }
        }
        let _ = uplo;
    }
}

impl<T: Scalar, HP: PackedStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    packed::Hpmv<T, HP, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn hpmv(uplo: UpLo, alpha: T, hp: &HP, x: &X, beta: T, y: &mut Y) {
        let n = hp.dim();
        if beta.is_zero() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for i in 0..n {
            let mut acc = T::ZERO;
            for j in 0..n {
                let a_val = hp.value_unchecked(i, j);
                let (rx, cx) =
                    if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                acc = acc + (a_val * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * acc));
            }
        }
        let _ = uplo;
    }
}

impl<T: Scalar, AP: PackedStorageMut<T>, X: DenseStorage<T>>
    packed::Spr<T, AP, X> for DefaultBlas
{
    #[inline(always)]
    fn spr(uplo: UpLo, alpha: T, x: &X, ap: &mut AP) {
        let n = ap.dim();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (ry, cy) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(ry, cy).clone() };
                    let current = ap.value_unchecked(i, j);
                    let _ = ap.set(
                        i,
                        j,
                        current + (alpha.clone() * xi.clone() * xj),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, HP: PackedStorageMut<T>, X: DenseStorage<T>>
    packed::Hpr<T, HP, X> for DefaultBlas
{
    #[inline(always)]
    fn hpr(uplo: UpLo, alpha: T::Real, x: &X, hp: &mut HP) {
        let n = hp.dim();
        let a_scalar = T::from_real(alpha);
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (ry, cy) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(ry, cy).clone().conj() };
                    let current = hp.value_unchecked(i, j);
                    let _ = hp.set(
                        i,
                        j,
                        current + (a_scalar.clone() * xi.clone() * xj),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, AP: PackedStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    packed::Spr2<T, AP, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn spr2(uplo: UpLo, alpha: T, x: &X, y: &Y, ap: &mut AP) {
        let n = ap.dim();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            let yi = unsafe { y.get_unchecked(ry, cy).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (rx2, cx2) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rx2, cx2).clone() };
                    let (ry2, cy2) =
                        if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                    let yj = unsafe { y.get_unchecked(ry2, cy2).clone() };
                    let current = ap.value_unchecked(i, j);
                    let _ = ap.set(
                        i,
                        j,
                        current
                            + (alpha.clone() * xi.clone() * yj)
                            + (alpha.clone() * yi.clone() * xj),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, HP: PackedStorageMut<T>, X: DenseStorage<T>, Y: DenseStorage<T>>
    packed::Hpr2<T, HP, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn hpr2(uplo: UpLo, alpha: T, x: &X, y: &Y, hp: &mut HP) {
        let n = hp.dim();
        let alpha_conj = alpha.clone().conj();
        for i in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            let xi = unsafe { x.get_unchecked(rx, cx).clone() };
            let (ry, cy) = if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
            let yi = unsafe { y.get_unchecked(ry, cy).clone() };
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let (rx2, cx2) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj =
                        unsafe { x.get_unchecked(rx2, cx2).clone().conj() };
                    let (ry2, cy2) =
                        if y.rows() >= y.cols() { (j, 0) } else { (0, j) };
                    let yj =
                        unsafe { y.get_unchecked(ry2, cy2).clone().conj() };
                    let current = hp.value_unchecked(i, j);
                    let _ = hp.set(
                        i,
                        j,
                        current
                            + (alpha.clone() * xi.clone() * yj)
                            + (alpha_conj.clone() * yi.clone() * xj),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, TP: PackedStorage<T>, X: DenseStorageMut<T>>
    packed::Tpmv<T, TP, X> for DefaultBlas
{
    #[inline(always)]
    fn tpmv(uplo: UpLo, trans: Trans, diag: Diag, tp: &TP, x: &mut X) {
        let n = tp.dim();
        let forward = matches!(
            (uplo, trans),
            (UpLo::Upper, Trans::NoTrans)
                | (UpLo::Lower, Trans::Trans | Trans::ConjTrans)
        );
        let mut i = if forward { 0 } else { n };
        while if forward { i < n } else { i > 0 } {
            if !forward {
                i -= 1;
            }
            let mut acc = T::ZERO;
            for j in 0..n {
                let (r, c) = match trans {
                    Trans::NoTrans => (i, j),
                    Trans::Trans | Trans::ConjTrans => (j, i),
                };
                let in_tri = match uplo {
                    UpLo::Upper => r <= c,
                    UpLo::Lower => r >= c,
                };
                if in_tri {
                    let a_val = if diag == Diag::Unit && r == c {
                        T::ONE
                    } else {
                        let elem = tp.value_unchecked(r, c);
                        if trans == Trans::ConjTrans {
                            elem.conj()
                        } else {
                            elem
                        }
                    };
                    let (rx, cx) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                    acc = acc + (a_val * xv);
                }
            }
            let (rx, cx) = if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
            unsafe {
                x.set_unchecked(rx, cx, acc);
            }
            if forward {
                i += 1;
            }
        }
    }
}

impl<T: Scalar + Div<Output = T>, TP: PackedStorage<T>, X: DenseStorageMut<T>>
    packed::Tpsv<T, TP, X> for DefaultBlas
{
    #[inline(always)]
    fn tpsv(
        uplo: UpLo,
        trans: Trans,
        diag: Diag,
        tp: &TP,
        x: &mut X,
    ) -> LinAlgResult<()> {
        let n = tp.dim();
        let is_upper = match (uplo, trans) {
            (UpLo::Upper, Trans::NoTrans) => true,
            (UpLo::Lower, Trans::Trans | Trans::ConjTrans) => true,
            _ => false,
        };

        if is_upper {
            for k in 0..n {
                let i = n - 1 - k;
                let (rxi, cxi) =
                    if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
                let mut sum = unsafe { x.get_unchecked(rxi, cxi).clone() };
                for j in (i + 1)..n {
                    let (r, c) = match trans {
                        Trans::NoTrans => (i, j),
                        Trans::Trans | Trans::ConjTrans => (j, i),
                    };
                    let elem = tp.value_unchecked(r, c);
                    let a_val = if trans == Trans::ConjTrans {
                        elem.conj()
                    } else {
                        elem
                    };
                    let (rxj, cxj) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rxj, cxj).clone() };
                    sum = sum - (a_val * xj);
                }
                if diag == Diag::Unit {
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum);
                    }
                } else {
                    let piv = tp.value_unchecked(i, i);
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    let piv_val = if trans == Trans::ConjTrans {
                        piv.conj()
                    } else {
                        piv
                    };
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum / piv_val);
                    }
                }
            }
        } else {
            for i in 0..n {
                let (rxi, cxi) =
                    if x.rows() >= x.cols() { (i, 0) } else { (0, i) };
                let mut sum = unsafe { x.get_unchecked(rxi, cxi).clone() };
                for j in 0..i {
                    let (r, c) = match trans {
                        Trans::NoTrans => (i, j),
                        Trans::Trans | Trans::ConjTrans => (j, i),
                    };
                    let elem = tp.value_unchecked(r, c);
                    let a_val = if trans == Trans::ConjTrans {
                        elem.conj()
                    } else {
                        elem
                    };
                    let (rxj, cxj) =
                        if x.rows() >= x.cols() { (j, 0) } else { (0, j) };
                    let xj = unsafe { x.get_unchecked(rxj, cxj).clone() };
                    sum = sum - (a_val * xj);
                }
                if diag == Diag::Unit {
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum);
                    }
                } else {
                    let piv = tp.value_unchecked(i, i);
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    let piv_val = if trans == Trans::ConjTrans {
                        piv.conj()
                    } else {
                        piv
                    };
                    unsafe {
                        x.set_unchecked(rxi, cxi, sum / piv_val);
                    }
                }
            }
        }
        Ok(())
    }
}

// --- Level 3 Implementation over DenseStorage & DenseStorageMut ---

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    level3::Gemm<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn gemm(ta: Trans, tb: Trans, alpha: T, a: &A, b: &B, beta: T, c: &mut C) {
        let (m, k_a) = match ta {
            Trans::NoTrans => (a.rows(), a.cols()),
            Trans::Trans | Trans::ConjTrans => (a.cols(), a.rows()),
        };
        let (k_b, n) = match tb {
            Trans::NoTrans => (b.rows(), b.cols()),
            Trans::Trans | Trans::ConjTrans => (b.cols(), b.rows()),
        };
        debug_assert_eq!(k_a, k_b);
        debug_assert_eq!(m, c.rows());
        debug_assert_eq!(n, c.cols());
        let k = k_a;

        // NaN-safe beta scaling (C-3)
        if beta.is_zero() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        c.set_unchecked(i, j, T::ZERO);
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, beta.clone() * cv);
                    }
                }
            }
        }

        for i in 0..m {
            for j in 0..n {
                let mut dot = T::ZERO;
                for p in 0..k {
                    let (ar, ac) = match ta {
                        Trans::NoTrans => (i, p),
                        Trans::Trans | Trans::ConjTrans => (p, i),
                    };
                    let a_elem = unsafe { a.get_unchecked(ar, ac).clone() };
                    let a_val = if ta == Trans::ConjTrans {
                        a_elem.conj()
                    } else {
                        a_elem
                    };

                    let (br, bc) = match tb {
                        Trans::NoTrans => (p, j),
                        Trans::Trans | Trans::ConjTrans => (j, p),
                    };
                    let b_elem = unsafe { b.get_unchecked(br, bc).clone() };
                    let b_val = if tb == Trans::ConjTrans {
                        b_elem.conj()
                    } else {
                        b_elem
                    };

                    dot = dot + (a_val * b_val);
                }
                unsafe {
                    let cv = c.get_unchecked(i, j).clone();
                    c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    level3::Symm<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn symm(
        side: Side,
        uplo: UpLo,
        alpha: T,
        a: &A,
        b: &B,
        beta: T,
        c: &mut C,
    ) {
        let m = c.rows();
        let n = c.cols();

        if beta.is_zero() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        c.set_unchecked(i, j, T::ZERO);
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, beta.clone() * cv);
                    }
                }
            }
        }

        match side {
            Side::Left => {
                for i in 0..m {
                    for j in 0..n {
                        let mut dot = T::ZERO;
                        for k in 0..m {
                            let (ar, ac) = match uplo {
                                UpLo::Upper => {
                                    if i <= k {
                                        (i, k)
                                    } else {
                                        (k, i)
                                    }
                                }
                                UpLo::Lower => {
                                    if i >= k {
                                        (i, k)
                                    } else {
                                        (k, i)
                                    }
                                }
                            };
                            let a_val =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let b_val =
                                unsafe { b.get_unchecked(k, j).clone() };
                            dot = dot + (a_val * b_val);
                        }
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                        }
                    }
                }
            }
            Side::Right => {
                for i in 0..m {
                    for j in 0..n {
                        let mut dot = T::ZERO;
                        for k in 0..n {
                            let (ar, ac) = match uplo {
                                UpLo::Upper => {
                                    if k <= j {
                                        (k, j)
                                    } else {
                                        (j, k)
                                    }
                                }
                                UpLo::Lower => {
                                    if k >= j {
                                        (k, j)
                                    } else {
                                        (j, k)
                                    }
                                }
                            };
                            let a_val =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let b_val =
                                unsafe { b.get_unchecked(i, k).clone() };
                            dot = dot + (b_val * a_val);
                        }
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                        }
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    level3::Hemm<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn hemm(
        side: Side,
        uplo: UpLo,
        alpha: T,
        a: &A,
        b: &B,
        beta: T,
        c: &mut C,
    ) {
        let m = c.rows();
        let n = c.cols();

        if beta.is_zero() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        c.set_unchecked(i, j, T::ZERO);
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, beta.clone() * cv);
                    }
                }
            }
        }

        match side {
            Side::Left => {
                for i in 0..m {
                    for j in 0..n {
                        let mut dot = T::ZERO;
                        for k in 0..m {
                            let a_val = match uplo {
                                UpLo::Upper => {
                                    if i <= k {
                                        unsafe { a.get_unchecked(i, k).clone() }
                                    } else {
                                        unsafe {
                                            a.get_unchecked(k, i).clone().conj()
                                        }
                                    }
                                }
                                UpLo::Lower => {
                                    if i >= k {
                                        unsafe { a.get_unchecked(i, k).clone() }
                                    } else {
                                        unsafe {
                                            a.get_unchecked(k, i).clone().conj()
                                        }
                                    }
                                }
                            };
                            let b_val =
                                unsafe { b.get_unchecked(k, j).clone() };
                            dot = dot + (a_val * b_val);
                        }
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                        }
                    }
                }
            }
            Side::Right => {
                for i in 0..m {
                    for j in 0..n {
                        let mut dot = T::ZERO;
                        for k in 0..n {
                            let a_val = match uplo {
                                UpLo::Upper => {
                                    if k <= j {
                                        unsafe { a.get_unchecked(k, j).clone() }
                                    } else {
                                        unsafe {
                                            a.get_unchecked(j, k).clone().conj()
                                        }
                                    }
                                }
                                UpLo::Lower => {
                                    if k >= j {
                                        unsafe { a.get_unchecked(k, j).clone() }
                                    } else {
                                        unsafe {
                                            a.get_unchecked(j, k).clone().conj()
                                        }
                                    }
                                }
                            };
                            let b_val =
                                unsafe { b.get_unchecked(i, k).clone() };
                            dot = dot + (b_val * a_val);
                        }
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                        }
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, C: DenseStorageMut<T>> level3::Syrk<T, A, C>
    for DefaultBlas
{
    #[inline(always)]
    fn syrk(uplo: UpLo, trans: Trans, alpha: T, a: &A, beta: T, c: &mut C) {
        let n = c.rows();
        let k = match trans {
            Trans::NoTrans => a.cols(),
            _ => a.rows(),
        };

        if beta.is_zero() {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            c.set_unchecked(i, j, T::ZERO);
                        }
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, beta.clone() * cv);
                        }
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let mut dot = T::ZERO;
                    for p in 0..k {
                        let (a1_r, a1_c) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let (a2_r, a2_c) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let v1 = unsafe { a.get_unchecked(a1_r, a1_c).clone() };
                        let v2 = unsafe { a.get_unchecked(a2_r, a2_c).clone() };
                        dot = dot + (v1 * v2);
                    }
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, C: DenseStorageMut<T>> level3::Herk<T, A, C>
    for DefaultBlas
{
    #[inline(always)]
    fn herk(
        uplo: UpLo,
        trans: Trans,
        alpha: T::Real,
        a: &A,
        beta: T::Real,
        c: &mut C,
    ) {
        let n = c.rows();
        let k = match trans {
            Trans::NoTrans => a.cols(),
            _ => a.rows(),
        };
        let a_alpha = T::from_real(alpha);
        let a_beta = T::from_real(beta.clone());

        if beta == <T::Real as Zero>::ZERO {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            c.set_unchecked(i, j, T::ZERO);
                        }
                    }
                }
            }
        } else if beta != <T::Real as One>::ONE {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, a_beta.clone() * cv);
                        }
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let mut dot = T::ZERO;
                    for p in 0..k {
                        let (a1_r, a1_c) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let (a2_r, a2_c) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let v1 = unsafe {
                            let elem = a.get_unchecked(a1_r, a1_c).clone();
                            if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };
                        let v2 = unsafe {
                            let elem = a.get_unchecked(a2_r, a2_c).clone();
                            if trans == Trans::NoTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };
                        dot = dot + (v1 * v2);
                    }
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, cv + (a_alpha.clone() * dot));
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    level3::Syr2k<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn syr2k(
        uplo: UpLo,
        trans: Trans,
        alpha: T,
        a: &A,
        b: &B,
        beta: T,
        c: &mut C,
    ) {
        let n = c.rows();
        let k = match trans {
            Trans::NoTrans => a.cols(),
            _ => a.rows(),
        };

        if beta.is_zero() {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            c.set_unchecked(i, j, T::ZERO);
                        }
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, beta.clone() * cv);
                        }
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let mut dot = T::ZERO;
                    for p in 0..k {
                        let (ar, ac) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let (br, bc) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let a_val = unsafe { a.get_unchecked(ar, ac).clone() };
                        let b_val = unsafe { b.get_unchecked(br, bc).clone() };

                        let (ar2, ac2) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let (br2, bc2) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let a_val2 =
                            unsafe { a.get_unchecked(ar2, ac2).clone() };
                        let b_val2 =
                            unsafe { b.get_unchecked(br2, bc2).clone() };

                        dot = dot + (a_val * b_val) + (b_val2 * a_val2);
                    }
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, cv + (alpha.clone() * dot));
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    level3::Her2k<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn her2k(
        uplo: UpLo,
        trans: Trans,
        alpha: T,
        a: &A,
        b: &B,
        beta: T::Real,
        c: &mut C,
    ) {
        let n = c.rows();
        let k = match trans {
            Trans::NoTrans => a.cols(),
            _ => a.rows(),
        };
        let a_beta = T::from_real(beta.clone());
        let alpha_conj = alpha.clone().conj();

        if beta == <T::Real as Zero>::ZERO {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            c.set_unchecked(i, j, T::ZERO);
                        }
                    }
                }
            }
        } else if beta != <T::Real as One>::ONE {
            for i in 0..n {
                for j in 0..n {
                    let in_tri = match uplo {
                        UpLo::Upper => i <= j,
                        UpLo::Lower => i >= j,
                    };
                    if in_tri {
                        unsafe {
                            let cv = c.get_unchecked(i, j).clone();
                            c.set_unchecked(i, j, a_beta.clone() * cv);
                        }
                    }
                }
            }
        }

        for i in 0..n {
            for j in 0..n {
                let in_tri = match uplo {
                    UpLo::Upper => i <= j,
                    UpLo::Lower => i >= j,
                };
                if in_tri {
                    let mut dot = T::ZERO;
                    for p in 0..k {
                        let (ar, ac) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let (br, bc) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let a_val = unsafe {
                            let elem = a.get_unchecked(ar, ac).clone();
                            if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };
                        let b_val = unsafe {
                            let elem = b.get_unchecked(br, bc).clone();
                            if trans == Trans::NoTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };

                        let (ar2, ac2) = match trans {
                            Trans::NoTrans => (j, p),
                            _ => (p, j),
                        };
                        let (br2, bc2) = match trans {
                            Trans::NoTrans => (i, p),
                            _ => (p, i),
                        };
                        let a_val2 = unsafe {
                            let elem = a.get_unchecked(ar2, ac2).clone();
                            if trans == Trans::NoTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };
                        let b_val2 = unsafe {
                            let elem = b.get_unchecked(br2, bc2).clone();
                            if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            }
                        };

                        dot = dot
                            + (alpha.clone() * a_val * b_val)
                            + (alpha_conj.clone() * b_val2 * a_val2);
                    }
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, cv + dot);
                    }
                }
            }
        }
    }
}

impl<T: Scalar, A: DenseStorage<T>, B: DenseStorageMut<T>> level3::Trmm<T, A, B>
    for DefaultBlas
{
    #[inline(always)]
    fn trmm(
        side: Side,
        uplo: UpLo,
        trans: Trans,
        diag: Diag,
        alpha: T,
        a: &A,
        b: &mut B,
    ) {
        let m = b.rows();
        let n = b.cols();

        match side {
            Side::Left => {
                let forward = matches!(
                    (uplo, trans),
                    (UpLo::Upper, Trans::NoTrans)
                        | (UpLo::Lower, Trans::Trans | Trans::ConjTrans)
                );
                for j in 0..n {
                    let mut i = if forward { 0 } else { m };
                    while if forward { i < m } else { i > 0 } {
                        if !forward {
                            i -= 1;
                        }
                        let mut acc = T::ZERO;
                        for k in 0..m {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (i, k),
                                _ => (k, i),
                            };
                            let in_tri = match uplo {
                                UpLo::Upper => ar <= ac,
                                UpLo::Lower => ar >= ac,
                            };
                            if in_tri {
                                let a_val = if diag == Diag::Unit && ar == ac {
                                    T::ONE
                                } else {
                                    let elem = unsafe {
                                        a.get_unchecked(ar, ac).clone()
                                    };
                                    if trans == Trans::ConjTrans {
                                        elem.conj()
                                    } else {
                                        elem
                                    }
                                };
                                let bv =
                                    unsafe { b.get_unchecked(k, j).clone() };
                                acc = acc + (a_val * bv);
                            }
                        }
                        unsafe {
                            b.set_unchecked(i, j, alpha.clone() * acc);
                        }
                        if forward {
                            i += 1;
                        }
                    }
                }
            }
            Side::Right => {
                let j_forward = matches!(
                    (uplo, trans),
                    (UpLo::Lower, Trans::NoTrans)
                        | (UpLo::Upper, Trans::Trans | Trans::ConjTrans)
                );
                for i in 0..m {
                    let mut j = if j_forward { 0 } else { n };
                    while if j_forward { j < n } else { j > 0 } {
                        if !j_forward {
                            j -= 1;
                        }
                        let mut acc = T::ZERO;
                        for k in 0..n {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (k, j),
                                _ => (j, k),
                            };
                            let in_tri = match uplo {
                                UpLo::Upper => ar <= ac,
                                UpLo::Lower => ar >= ac,
                            };
                            if in_tri {
                                let a_val = if diag == Diag::Unit && ar == ac {
                                    T::ONE
                                } else {
                                    let elem = unsafe {
                                        a.get_unchecked(ar, ac).clone()
                                    };
                                    if trans == Trans::ConjTrans {
                                        elem.conj()
                                    } else {
                                        elem
                                    }
                                };
                                let bv =
                                    unsafe { b.get_unchecked(i, k).clone() };
                                acc = acc + (bv * a_val);
                            }
                        }
                        unsafe {
                            b.set_unchecked(i, j, alpha.clone() * acc);
                        }
                        if j_forward {
                            j += 1;
                        }
                    }
                }
            }
        }
    }
}

impl<T: Scalar + Div<Output = T>, A: DenseStorage<T>, B: DenseStorageMut<T>>
    level3::Trsm<T, A, B> for DefaultBlas
{
    #[inline(always)]
    fn trsm(
        side: Side,
        uplo: UpLo,
        trans: Trans,
        diag: Diag,
        alpha: T,
        a: &A,
        b: &mut B,
    ) -> LinAlgResult<()> {
        let m = b.rows();
        let n = b.cols();

        if side == Side::Left {
            let is_upper = match (uplo, trans) {
                (UpLo::Upper, Trans::NoTrans) => true,
                (UpLo::Lower, Trans::Trans | Trans::ConjTrans) => true,
                _ => false,
            };

            for j in 0..n {
                if is_upper {
                    for k in 0..m {
                        let i = m - 1 - k;
                        let mut sum = alpha.clone()
                            * unsafe { b.get_unchecked(i, j).clone() };
                        for p in (i + 1)..m {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (i, p),
                                _ => (p, i),
                            };
                            let elem =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let a_val = if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            };
                            let bp = unsafe { b.get_unchecked(p, j).clone() };
                            sum = sum - (a_val * bp);
                        }
                        if diag == Diag::Unit {
                            unsafe {
                                b.set_unchecked(i, j, sum);
                            }
                        } else {
                            let piv = unsafe { a.get_unchecked(i, i).clone() };
                            if piv.is_zero() {
                                return Err(LinAlgError::SingularMatrix);
                            }
                            let piv_val = if trans == Trans::ConjTrans {
                                piv.conj()
                            } else {
                                piv
                            };
                            unsafe {
                                b.set_unchecked(i, j, sum / piv_val);
                            }
                        }
                    }
                } else {
                    for i in 0..m {
                        let mut sum = alpha.clone()
                            * unsafe { b.get_unchecked(i, j).clone() };
                        for p in 0..i {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (i, p),
                                _ => (p, i),
                            };
                            let elem =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let a_val = if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            };
                            let bp = unsafe { b.get_unchecked(p, j).clone() };
                            sum = sum - (a_val * bp);
                        }
                        if diag == Diag::Unit {
                            unsafe {
                                b.set_unchecked(i, j, sum);
                            }
                        } else {
                            let piv = unsafe { a.get_unchecked(i, i).clone() };
                            if piv.is_zero() {
                                return Err(LinAlgError::SingularMatrix);
                            }
                            let piv_val = if trans == Trans::ConjTrans {
                                piv.conj()
                            } else {
                                piv
                            };
                            unsafe {
                                b.set_unchecked(i, j, sum / piv_val);
                            }
                        }
                    }
                }
            }
        } else {
            // Side::Right: X op(A) = alpha B
            let is_upper = match (uplo, trans) {
                (UpLo::Upper, Trans::NoTrans) => true,
                (UpLo::Lower, Trans::Trans | Trans::ConjTrans) => true,
                _ => false,
            };
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        let bv = b.get_unchecked(i, j).clone();
                        b.set_unchecked(i, j, alpha.clone() * bv);
                    }
                }
            }
            if is_upper {
                for j in 0..n {
                    for i in 0..m {
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for k in 0..j {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (k, j),
                                _ => (j, k),
                            };
                            let elem =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let a_val = if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            };
                            let bk = unsafe { b.get_unchecked(i, k).clone() };
                            sum = sum - (bk * a_val);
                        }
                        if diag == Diag::Unit {
                            unsafe {
                                b.set_unchecked(i, j, sum);
                            }
                        } else {
                            let piv = unsafe { a.get_unchecked(j, j).clone() };
                            if piv.is_zero() {
                                return Err(LinAlgError::SingularMatrix);
                            }
                            let piv_val = if trans == Trans::ConjTrans {
                                piv.conj()
                            } else {
                                piv
                            };
                            unsafe {
                                b.set_unchecked(i, j, sum / piv_val);
                            }
                        }
                    }
                }
            } else {
                for jj in 0..n {
                    let j = n - 1 - jj;
                    for i in 0..m {
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for k in (j + 1)..n {
                            let (ar, ac) = match trans {
                                Trans::NoTrans => (k, j),
                                _ => (j, k),
                            };
                            let elem =
                                unsafe { a.get_unchecked(ar, ac).clone() };
                            let a_val = if trans == Trans::ConjTrans {
                                elem.conj()
                            } else {
                                elem
                            };
                            let bk = unsafe { b.get_unchecked(i, k).clone() };
                            sum = sum - (bk * a_val);
                        }
                        if diag == Diag::Unit {
                            unsafe {
                                b.set_unchecked(i, j, sum);
                            }
                        } else {
                            let piv = unsafe { a.get_unchecked(j, j).clone() };
                            if piv.is_zero() {
                                return Err(LinAlgError::SingularMatrix);
                            }
                            let piv_val = if trans == Trans::ConjTrans {
                                piv.conj()
                            } else {
                                piv
                            };
                            unsafe {
                                b.set_unchecked(i, j, sum / piv_val);
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

// --- Sparse BLAS Implementation ---

impl<T: Scalar, A: CsrStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    sparse::Csrmv<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn csrmv(alpha: T, a: &A, x: &X, beta: T, y: &mut Y) {
        let m = a.rows();
        let row_offsets = a.row_offsets();
        let col_indices = a.col_indices();
        let values = a.values();

        if beta.is_zero() {
            for i in 0..m {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for r in 0..m {
            let start = row_offsets[r];
            let end = row_offsets[r + 1];
            let mut dot = T::ZERO;
            for idx in start..end {
                let c = col_indices[idx];
                let (rx, cx) =
                    if x.rows() >= x.cols() { (c, 0) } else { (0, c) };
                let xv = unsafe { x.get_unchecked(rx, cx).clone() };
                dot = dot + (values[idx].clone() * xv);
            }
            let (ry, cy) = if y.rows() >= y.cols() { (r, 0) } else { (0, r) };
            unsafe {
                let yi = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yi + (alpha.clone() * dot));
            }
        }
    }
}

impl<T: Scalar, A: CscStorage<T>, X: DenseStorage<T>, Y: DenseStorageMut<T>>
    sparse::Cscmv<T, A, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn cscmv(alpha: T, a: &A, x: &X, beta: T, y: &mut Y) {
        let n = a.cols();
        let col_offsets = a.col_offsets();
        let row_indices = a.row_indices();
        let values = a.values();

        if beta.is_zero() {
            for i in 0..a.rows() {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    y.set_unchecked(ry, cy, T::ZERO);
                }
            }
        } else if !beta.is_one() {
            for i in 0..a.rows() {
                let (ry, cy) =
                    if y.rows() >= y.cols() { (i, 0) } else { (0, i) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(ry, cy, beta.clone() * yi);
                }
            }
        }

        for c in 0..n {
            let (rx, cx) = if x.rows() >= x.cols() { (c, 0) } else { (0, c) };
            let xv = unsafe { x.get_unchecked(rx, cx).clone() };
            let scalar = alpha.clone() * xv;
            let start = col_offsets[c];
            let end = col_offsets[c + 1];
            for idx in start..end {
                let r = row_indices[idx];
                let (ry, cy) =
                    if y.rows() >= y.cols() { (r, 0) } else { (0, r) };
                unsafe {
                    let yi = y.get_unchecked(ry, cy).clone();
                    y.set_unchecked(
                        ry,
                        cy,
                        yi + (scalar.clone() * values[idx].clone()),
                    );
                }
            }
        }
    }
}

impl<T: Scalar, A: CsrStorage<T>, B: DenseStorage<T>, C: DenseStorageMut<T>>
    sparse::Csrmm<T, A, B, C> for DefaultBlas
{
    #[inline(always)]
    fn csrmm(alpha: T, a: &A, b: &B, beta: T, c: &mut C) {
        let m = a.rows();
        let n = b.cols();
        let row_offsets = a.row_offsets();
        let col_indices = a.col_indices();
        let values = a.values();

        if beta.is_zero() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        c.set_unchecked(i, j, T::ZERO);
                    }
                }
            }
        } else if !beta.is_one() {
            for i in 0..m {
                for j in 0..n {
                    unsafe {
                        let cv = c.get_unchecked(i, j).clone();
                        c.set_unchecked(i, j, beta.clone() * cv);
                    }
                }
            }
        }

        for r in 0..m {
            let start = row_offsets[r];
            let end = row_offsets[r + 1];
            for j in 0..n {
                let mut dot = T::ZERO;
                for idx in start..end {
                    let p = col_indices[idx];
                    let bv = unsafe { b.get_unchecked(p, j).clone() };
                    dot = dot + (values[idx].clone() * bv);
                }
                unsafe {
                    let cv = c.get_unchecked(r, j).clone();
                    c.set_unchecked(r, j, cv + (alpha.clone() * dot));
                }
            }
        }
    }
}

impl<T: Scalar, X: SparseVectorStorage<T>, Y: DenseStorage<T>>
    sparse::SpDotu<T, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn sp_dotu(x: &X, y: &Y) -> T {
        let indices = x.indices();
        let values = x.values();
        let mut acc = T::ZERO;
        for (idx, val) in indices.iter().zip(values.iter()) {
            let (ry, cy) = if y.rows() >= y.cols() {
                (*idx, 0)
            } else {
                (0, *idx)
            };
            let yv = unsafe { y.get_unchecked(ry, cy).clone() };
            acc = acc + (val.clone() * yv);
        }
        acc
    }
}

impl<T: Scalar, X: SparseVectorStorage<T>, Y: DenseStorage<T>>
    sparse::SpDotc<T, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn sp_dotc(x: &X, y: &Y) -> T {
        let indices = x.indices();
        let values = x.values();
        let mut acc = T::ZERO;
        for (idx, val) in indices.iter().zip(values.iter()) {
            let (ry, cy) = if y.rows() >= y.cols() {
                (*idx, 0)
            } else {
                (0, *idx)
            };
            let yv = unsafe { y.get_unchecked(ry, cy).clone() };
            acc = acc + (val.clone().conj() * yv);
        }
        acc
    }
}

impl<T: Scalar, X: SparseVectorStorage<T>, Y: DenseStorageMut<T>>
    sparse::SpAxpy<T, X, Y> for DefaultBlas
{
    #[inline(always)]
    fn sp_axpy(alpha: T, x: &X, y: &mut Y) {
        let indices = x.indices();
        let values = x.values();
        for (idx, val) in indices.iter().zip(values.iter()) {
            let (ry, cy) = if y.rows() >= y.cols() {
                (*idx, 0)
            } else {
                (0, *idx)
            };
            unsafe {
                let yv = y.get_unchecked(ry, cy).clone();
                y.set_unchecked(ry, cy, yv + (alpha.clone() * val.clone()));
            }
        }
    }
}

// --- LAPACK Implementation over DenseStorage & DenseStorageMut ---

impl<T, A> lapack::Potrf<T, A> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    T::Real: Radical,
    A: DenseStorageMut<T>,
{
    #[inline(always)]
    fn potrf(uplo: UpLo, a: &mut A) -> LinAlgResult<()> {
        let n = a.rows();
        debug_assert_eq!(n, a.cols());

        for j in 0..n {
            let mut sum2 = <T::Real as Zero>::ZERO;
            for k in 0..j {
                let (r, c) = match uplo {
                    UpLo::Lower => (j, k),
                    UpLo::Upper => (k, j),
                };
                let elem = unsafe { a.get_unchecked(r, c) };
                sum2 = sum2 + elem.abs2();
            }

            let a_jj_re = unsafe { a.get_unchecked(j, j).re() };
            if a_jj_re <= sum2 {
                return Err(LinAlgError::NotPositiveDefinite);
            }
            let l_jj_re = (a_jj_re - sum2).sqrt();
            let l_jj = T::from_real(l_jj_re.clone());
            unsafe {
                a.set_unchecked(j, j, l_jj.clone());
            }

            for i in (j + 1)..n {
                let mut dot = T::ZERO;
                for k in 0..j {
                    let (ik_r, ik_c) = match uplo {
                        UpLo::Lower => (i, k),
                        UpLo::Upper => (k, i),
                    };
                    let (jk_r, jk_c) = match uplo {
                        UpLo::Lower => (j, k),
                        UpLo::Upper => (k, j),
                    };
                    let v_ik = unsafe { a.get_unchecked(ik_r, ik_c).clone() };
                    let v_jk =
                        unsafe { a.get_unchecked(jk_r, jk_c).clone().conj() };
                    dot = dot + (v_ik * v_jk);
                }

                let (target_r, target_c) = match uplo {
                    UpLo::Lower => (i, j),
                    UpLo::Upper => (j, i),
                };
                let target_val =
                    unsafe { a.get_unchecked(target_r, target_c).clone() };
                let val = (target_val - dot) / l_jj.clone();
                unsafe {
                    a.set_unchecked(target_r, target_c, val);
                }
            }
        }
        Ok(())
    }
}

impl<T, A, B> lapack::Potrs<T, A, B> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    A: DenseStorage<T>,
    B: DenseStorageMut<T>,
{
    #[inline(always)]
    fn potrs(uplo: UpLo, a: &A, b: &mut B) -> LinAlgResult<()> {
        match uplo {
            UpLo::Lower => {
                // Solve L Y = B
                <DefaultBlas as level3::Trsm<T, A, B>>::trsm(
                    Side::Left,
                    UpLo::Lower,
                    Trans::NoTrans,
                    Diag::NonUnit,
                    T::ONE,
                    a,
                    b,
                )?;
                // Solve L^H X = Y
                <DefaultBlas as level3::Trsm<T, A, B>>::trsm(
                    Side::Left,
                    UpLo::Lower,
                    Trans::ConjTrans,
                    Diag::NonUnit,
                    T::ONE,
                    a,
                    b,
                )?;
            }
            UpLo::Upper => {
                // Solve U^T Y = B
                <DefaultBlas as level3::Trsm<T, A, B>>::trsm(
                    Side::Left,
                    UpLo::Upper,
                    Trans::ConjTrans,
                    Diag::NonUnit,
                    T::ONE,
                    a,
                    b,
                )?;
                // Solve U X = Y
                <DefaultBlas as level3::Trsm<T, A, B>>::trsm(
                    Side::Left,
                    UpLo::Upper,
                    Trans::NoTrans,
                    Diag::NonUnit,
                    T::ONE,
                    a,
                    b,
                )?;
            }
        }
        Ok(())
    }
}

impl<T, AP> lapack::Pptrf<T, AP> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    T::Real: Radical,
    AP: PackedStorageMut<T>,
{
    #[inline(always)]
    fn pptrf(uplo: UpLo, ap: &mut AP) -> LinAlgResult<()> {
        let n = ap.dim();
        match uplo {
            UpLo::Lower => {
                for j in 0..n {
                    let mut sum2 = <T::Real as Zero>::ZERO;
                    for k in 0..j {
                        let elem = ap.value_unchecked(j, k);
                        sum2 = sum2 + elem.abs2();
                    }

                    let a_jj_re = ap.value_unchecked(j, j).re();
                    if a_jj_re <= sum2 {
                        return Err(LinAlgError::NotPositiveDefinite);
                    }
                    let l_jj_re = (a_jj_re - sum2).sqrt();
                    let l_jj = T::from_real(l_jj_re.clone());
                    let _ = ap.set(j, j, l_jj.clone());

                    for i in (j + 1)..n {
                        let mut dot = T::ZERO;
                        for k in 0..j {
                            let v_ik = ap.value_unchecked(i, k);
                            let v_jk = ap.value_unchecked(j, k).conj();
                            dot = dot + (v_ik * v_jk);
                        }
                        let val =
                            (ap.value_unchecked(i, j) - dot) / l_jj.clone();
                        let _ = ap.set(i, j, val);
                    }
                }
            }
            UpLo::Upper => {
                for j in 0..n {
                    let mut sum2 = <T::Real as Zero>::ZERO;
                    for k in 0..j {
                        let elem = ap.value_unchecked(k, j);
                        sum2 = sum2 + elem.abs2();
                    }

                    let a_jj_re = ap.value_unchecked(j, j).re();
                    if a_jj_re <= sum2 {
                        return Err(LinAlgError::NotPositiveDefinite);
                    }
                    let u_jj_re = (a_jj_re - sum2).sqrt();
                    let u_jj = T::from_real(u_jj_re.clone());
                    let _ = ap.set(j, j, u_jj.clone());

                    for i in (j + 1)..n {
                        let mut dot = T::ZERO;
                        for k in 0..j {
                            let v_kj = ap.value_unchecked(k, j).conj();
                            let v_ki = ap.value_unchecked(k, i);
                            dot = dot + (v_kj * v_ki);
                        }
                        let val =
                            (ap.value_unchecked(j, i) - dot) / u_jj.clone();
                        let _ = ap.set(j, i, val);
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T, AP, B> lapack::Pptrs<T, AP, B> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    AP: PackedStorage<T>,
    B: DenseStorageMut<T>,
{
    #[inline(always)]
    fn pptrs(uplo: UpLo, ap: &AP, b: &mut B) -> LinAlgResult<()> {
        let n = ap.dim();
        let nrhs = b.cols();
        for j in 0..nrhs {
            match uplo {
                UpLo::Lower => {
                    for i in 0..n {
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for k in 0..i {
                            sum = sum
                                - (ap.value_unchecked(i, k)
                                    * unsafe { b.get_unchecked(k, j).clone() });
                        }
                        let piv = ap.value_unchecked(i, i);
                        if piv.is_zero() {
                            return Err(LinAlgError::SingularMatrix);
                        }
                        unsafe {
                            b.set_unchecked(i, j, sum / piv);
                        }
                    }
                    for k in 0..n {
                        let i = n - 1 - k;
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for p in (i + 1)..n {
                            sum = sum
                                - (ap.value_unchecked(p, i).conj()
                                    * unsafe { b.get_unchecked(p, j).clone() });
                        }
                        let piv = ap.value_unchecked(i, i);
                        if piv.is_zero() {
                            return Err(LinAlgError::SingularMatrix);
                        }
                        unsafe {
                            b.set_unchecked(i, j, sum / piv.conj());
                        }
                    }
                }
                UpLo::Upper => {
                    for i in 0..n {
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for k in 0..i {
                            sum = sum
                                - (ap.value_unchecked(k, i).conj()
                                    * unsafe { b.get_unchecked(k, j).clone() });
                        }
                        let piv = ap.value_unchecked(i, i);
                        if piv.is_zero() {
                            return Err(LinAlgError::SingularMatrix);
                        }
                        unsafe {
                            b.set_unchecked(i, j, sum / piv.conj());
                        }
                    }
                    for k in 0..n {
                        let i = n - 1 - k;
                        let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                        for p in (i + 1)..n {
                            sum = sum
                                - (ap.value_unchecked(i, p)
                                    * unsafe { b.get_unchecked(p, j).clone() });
                        }
                        let piv = ap.value_unchecked(i, i);
                        if piv.is_zero() {
                            return Err(LinAlgError::SingularMatrix);
                        }
                        unsafe {
                            b.set_unchecked(i, j, sum / piv);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T, A> lapack::Getrf<T, A> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    A: DenseStorageMut<T>,
{
    #[inline(always)]
    fn getrf(a: &mut A, ipiv: &mut [usize]) -> LinAlgResult<()> {
        let m = a.rows();
        let n = a.cols();
        let min_dim = core::cmp::min(m, n);
        if ipiv.len() < min_dim {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        for k in 0..min_dim {
            let mut max_abs = <T::Real as Zero>::ZERO;
            let mut p = k;
            for i in k..m {
                let abs_val = unsafe { a.get_unchecked(i, k).abs2() };
                if i == k || abs_val > max_abs {
                    max_abs = abs_val;
                    p = i;
                }
            }

            ipiv[k] = p;
            if max_abs.is_zero() {
                return Err(LinAlgError::SingularMatrix);
            }

            if p != k {
                for j in 0..n {
                    unsafe {
                        let vk = a.get_unchecked(k, j).clone();
                        let vp = a.get_unchecked(p, j).clone();
                        a.set_unchecked(k, j, vp);
                        a.set_unchecked(p, j, vk);
                    }
                }
            }

            let pivot = unsafe { a.get_unchecked(k, k).clone() };
            for i in (k + 1)..m {
                let mult =
                    unsafe { a.get_unchecked(i, k).clone() } / pivot.clone();
                unsafe {
                    a.set_unchecked(i, k, mult.clone());
                }

                for j in (k + 1)..n {
                    unsafe {
                        let a_ij = a.get_unchecked(i, j).clone();
                        let a_kj = a.get_unchecked(k, j).clone();
                        a.set_unchecked(i, j, a_ij - (mult.clone() * a_kj));
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T, A, B> lapack::Getrs<T, A, B> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    A: DenseStorage<T>,
    B: DenseStorageMut<T>,
{
    #[inline(always)]
    fn getrs(
        trans: Trans,
        a: &A,
        ipiv: &[usize],
        b: &mut B,
    ) -> LinAlgResult<()> {
        let n = a.rows();
        let nrhs = b.cols();
        if ipiv.len() < n {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        for j in 0..nrhs {
            if trans == Trans::NoTrans {
                for (k, &p) in ipiv.iter().enumerate().take(n) {
                    if p != k {
                        unsafe {
                            let vk = b.get_unchecked(k, j).clone();
                            let vp = b.get_unchecked(p, j).clone();
                            b.set_unchecked(k, j, vp);
                            b.set_unchecked(p, j, vk);
                        }
                    }
                }

                for i in 0..n {
                    let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                    for k in 0..i {
                        sum = sum
                            - (unsafe { a.get_unchecked(i, k).clone() }
                                * unsafe { b.get_unchecked(k, j).clone() });
                    }
                    unsafe {
                        b.set_unchecked(i, j, sum);
                    }
                }

                for k in 0..n {
                    let i = n - 1 - k;
                    let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                    for p in (i + 1)..n {
                        sum = sum
                            - (unsafe { a.get_unchecked(i, p).clone() }
                                * unsafe { b.get_unchecked(p, j).clone() });
                    }
                    let piv = unsafe { a.get_unchecked(i, i).clone() };
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    unsafe {
                        b.set_unchecked(i, j, sum / piv);
                    }
                }
            } else {
                for i in 0..n {
                    let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                    for k in 0..i {
                        let elem = unsafe { a.get_unchecked(k, i).clone() };
                        let a_val = if trans == Trans::ConjTrans {
                            elem.conj()
                        } else {
                            elem
                        };
                        sum = sum
                            - (a_val
                                * unsafe { b.get_unchecked(k, j).clone() });
                    }
                    let piv = unsafe { a.get_unchecked(i, i).clone() };
                    if piv.is_zero() {
                        return Err(LinAlgError::SingularMatrix);
                    }
                    let piv_val = if trans == Trans::ConjTrans {
                        piv.conj()
                    } else {
                        piv
                    };
                    unsafe {
                        b.set_unchecked(i, j, sum / piv_val);
                    }
                }

                for k in 0..n {
                    let i = n - 1 - k;
                    let mut sum = unsafe { b.get_unchecked(i, j).clone() };
                    for p in (i + 1)..n {
                        let elem = unsafe { a.get_unchecked(p, i).clone() };
                        let a_val = if trans == Trans::ConjTrans {
                            elem.conj()
                        } else {
                            elem
                        };
                        sum = sum
                            - (a_val
                                * unsafe { b.get_unchecked(p, j).clone() });
                    }
                    unsafe {
                        b.set_unchecked(i, j, sum);
                    }
                }

                for k in (0..n).rev() {
                    let p = ipiv[k];
                    if p != k {
                        unsafe {
                            let vk = b.get_unchecked(k, j).clone();
                            let vp = b.get_unchecked(p, j).clone();
                            b.set_unchecked(k, j, vp);
                            b.set_unchecked(p, j, vk);
                        }
                    }
                }
            }
        }
        Ok(())
    }
}

impl<T, A> lapack::Geqrf<T, A> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    T::Real: Radical,
    A: DenseStorageMut<T>,
{
    #[inline(always)]
    fn geqrf(a: &mut A, tau: &mut [T], work: &mut [T]) -> LinAlgResult<()> {
        let m = a.rows();
        let n = a.cols();
        let min_dim = core::cmp::min(m, n);
        if tau.len() < min_dim || work.len() < n {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        for k in 0..min_dim {
            let mut sum2 = <T::Real as Zero>::ZERO;
            for i in k..m {
                sum2 = sum2 + unsafe { a.get_unchecked(i, k).abs2() };
            }

            let norm = sum2.sqrt();
            if norm.is_zero() {
                tau[k] = T::ZERO;
                continue;
            }

            let alpha = unsafe { a.get_unchecked(k, k).clone() };
            let beta_re = if alpha.re() >= <T::Real as Zero>::ZERO {
                <T::Real as Zero>::ZERO - norm
            } else {
                norm
            };
            let beta = T::from_real(beta_re.clone());

            let v0 = alpha.clone() - beta.clone();
            tau[k] = (beta.clone() - alpha) / beta.clone();
            unsafe {
                a.set_unchecked(k, k, beta);
            }

            for i in (k + 1)..m {
                let current = unsafe { a.get_unchecked(i, k).clone() };
                unsafe {
                    a.set_unchecked(i, k, current / v0.clone());
                }
            }

            // Apply Householder reflector to remaining submatrix
            for j in (k + 1)..n {
                let mut dot = T::ZERO;
                for i in k..m {
                    let v_i = if i == k {
                        T::ONE
                    } else {
                        unsafe { a.get_unchecked(i, k).clone() }
                    };
                    let a_ij = unsafe { a.get_unchecked(i, j).clone() };
                    dot = dot + (v_i.conj() * a_ij);
                }

                let scalar = tau[k].clone() * dot;
                for i in k..m {
                    let v_i = if i == k {
                        T::ONE
                    } else {
                        unsafe { a.get_unchecked(i, k).clone() }
                    };
                    let a_ij = unsafe { a.get_unchecked(i, j).clone() };
                    unsafe {
                        a.set_unchecked(i, j, a_ij - (v_i * scalar.clone()));
                    }
                }
            }
        }
        let _ = work;
        Ok(())
    }
}

impl<T, A, C> lapack::Ormqr<T, A, C> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    A: DenseStorage<T>,
    C: DenseStorageMut<T>,
{
    #[inline(always)]
    fn ormqr(
        side: Side,
        trans: Trans,
        a: &A,
        tau: &[T],
        c: &mut C,
        work: &mut [T],
    ) -> LinAlgResult<()> {
        let m = c.rows();
        let n = c.cols();
        let k_limit = core::cmp::min(a.rows(), a.cols());
        let min_work = match side {
            Side::Left => n,
            Side::Right => m,
        };
        if tau.len() < k_limit || work.len() < min_work {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        match (side, trans) {
            (Side::Left, Trans::NoTrans) => {
                // Q C = H_0 H_1 ... H_{k-1} C (applied backwards)
                for k in (0..k_limit).rev() {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for j in 0..n {
                        let mut dot = unsafe { c.get_unchecked(k, j).clone() };
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_i * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_kj = unsafe { c.get_unchecked(k, j).clone() };
                        unsafe {
                            c.set_unchecked(k, j, c_kj - scalar.clone());
                        }
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_i * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Left, Trans::Trans | Trans::ConjTrans) => {
                // Q^T C = H_{k-1} ... H_0 C (applied forwards)
                for k in 0..k_limit {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for j in 0..n {
                        let mut dot = unsafe { c.get_unchecked(k, j).clone() };
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_i * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_kj = unsafe { c.get_unchecked(k, j).clone() };
                        unsafe {
                            c.set_unchecked(k, j, c_kj - scalar.clone());
                        }
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_i * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Right, Trans::NoTrans) => {
                for k in 0..k_limit {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for i in 0..m {
                        let mut dot = unsafe { c.get_unchecked(i, k).clone() };
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_j * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_ik = unsafe { c.get_unchecked(i, k).clone() };
                        unsafe {
                            c.set_unchecked(i, k, c_ik - scalar.clone());
                        }
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_j * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Right, Trans::Trans | Trans::ConjTrans) => {
                for k in (0..k_limit).rev() {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for i in 0..m {
                        let mut dot = unsafe { c.get_unchecked(i, k).clone() };
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_j * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_ik = unsafe { c.get_unchecked(i, k).clone() };
                        unsafe {
                            c.set_unchecked(i, k, c_ik - scalar.clone());
                        }
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_j * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
        }
        let _ = work;
        Ok(())
    }
}

impl<T, A, C> lapack::Unmqr<T, A, C> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    A: DenseStorage<T>,
    C: DenseStorageMut<T>,
{
    #[inline(always)]
    fn unmqr(
        side: Side,
        trans: Trans,
        a: &A,
        tau: &[T],
        c: &mut C,
        work: &mut [T],
    ) -> LinAlgResult<()> {
        let m = c.rows();
        let n = c.cols();
        let k_limit = core::cmp::min(a.rows(), a.cols());
        let min_work = match side {
            Side::Left => n,
            Side::Right => m,
        };
        if tau.len() < k_limit || work.len() < min_work {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        match (side, trans) {
            (Side::Left, Trans::NoTrans) => {
                for k in (0..k_limit).rev() {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for j in 0..n {
                        let mut dot = unsafe { c.get_unchecked(k, j).clone() };
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_i.conj() * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_kj = unsafe { c.get_unchecked(k, j).clone() };
                        unsafe {
                            c.set_unchecked(k, j, c_kj - scalar.clone());
                        }
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_i * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Left, Trans::ConjTrans | Trans::Trans) => {
                for k in 0..k_limit {
                    let tau_k = tau[k].clone().conj();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for j in 0..n {
                        let mut dot = unsafe { c.get_unchecked(k, j).clone() };
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_i.conj() * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_kj = unsafe { c.get_unchecked(k, j).clone() };
                        unsafe {
                            c.set_unchecked(k, j, c_kj - scalar.clone());
                        }
                        for i in (k + 1)..m {
                            let v_i = unsafe { a.get_unchecked(i, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_i * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Right, Trans::NoTrans) => {
                for k in 0..k_limit {
                    let tau_k = tau[k].clone();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for i in 0..m {
                        let mut dot = unsafe { c.get_unchecked(i, k).clone() };
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_j.conj() * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_ik = unsafe { c.get_unchecked(i, k).clone() };
                        unsafe {
                            c.set_unchecked(i, k, c_ik - scalar.clone());
                        }
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_j * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
            (Side::Right, Trans::ConjTrans | Trans::Trans) => {
                for k in (0..k_limit).rev() {
                    let tau_k = tau[k].clone().conj();
                    if tau_k.is_zero() {
                        continue;
                    }
                    for i in 0..m {
                        let mut dot = unsafe { c.get_unchecked(i, k).clone() };
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            dot = dot + (v_j.conj() * c_ij);
                        }
                        let scalar = tau_k.clone() * dot;
                        let c_ik = unsafe { c.get_unchecked(i, k).clone() };
                        unsafe {
                            c.set_unchecked(i, k, c_ik - scalar.clone());
                        }
                        for j in (k + 1)..n {
                            let v_j = unsafe { a.get_unchecked(j, k).clone() };
                            let c_ij = unsafe { c.get_unchecked(i, j).clone() };
                            unsafe {
                                c.set_unchecked(
                                    i,
                                    j,
                                    c_ij - (v_j * scalar.clone()),
                                );
                            }
                        }
                    }
                }
            }
        }
        let _ = work;
        Ok(())
    }
}

#[inline(always)]
fn uplo_entry<T: Scalar, A: DenseStorage<T>>(
    a: &A,
    i: usize,
    j: usize,
    uplo: UpLo,
) -> T {
    match uplo {
        UpLo::Upper => {
            if i <= j {
                unsafe { a.get_unchecked(i, j).clone() }
            } else {
                unsafe { a.get_unchecked(j, i).clone().conj() }
            }
        }
        UpLo::Lower => {
            if i >= j {
                unsafe { a.get_unchecked(i, j).clone() }
            } else {
                unsafe { a.get_unchecked(j, i).clone().conj() }
            }
        }
    }
}

/// Jacobi eigendecomposition of a real symmetric operand under an explicit
/// sweep budget.
///
/// Crate-private seam for [`lapack::Syev`]. The public `syev` supplies the
/// default budget $50 n^2$; verification supplies `max_iter = 0` to reach the
/// [`LinAlgError::MaxIterationsReached`] arm on a well-conditioned operand
/// (`subprograms-design.md` §4.3, §6.1.2). The budget travels on the call
/// stack, so no global state participates in the result (NFR-1b).
#[inline(always)]
pub(crate) fn syev_impl<T, A>(
    jobz: JobZ,
    uplo: UpLo,
    a: &mut A,
    w: &mut [T::Real],
    work: &mut [T],
    max_iter: usize,
) -> LinAlgResult<()>
where
    T: Scalar + Div<Output = T>,
    T::Real: Float,
    A: DenseStorageMut<T>,
{
    {
        let n = a.rows();
        debug_assert_eq!(n, a.cols());
        let min_work = if jobz == JobZ::Vectors { n * n } else { n };
        if w.len() < n || work.len() < min_work {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        if jobz == JobZ::Vectors {
            for i in 0..n {
                for j in 0..n {
                    work[i * n + j] = if i == j { T::ONE } else { T::ZERO };
                }
            }
        }

        let mut iter = 0;
        loop {
            let mut max_off = <T::Real as Zero>::ZERO;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = uplo_entry(a, i, j, uplo).abs2();
                    if val > max_off || val != val {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            let eps = <T::Real as Float>::epsilon();
            let converged = max_off <= eps;
            if converged || iter >= max_iter {
                if iter >= max_iter && !converged {
                    return Err(LinAlgError::MaxIterationsReached);
                }
                break;
            }

            let a_pp = uplo_entry(a, p, p, uplo).re();
            let a_qq = uplo_entry(a, q, q, uplo).re();
            let a_pq = uplo_entry(a, p, q, uplo).re();

            let one = <T::Real as One>::ONE;
            let two = one.clone() + one.clone();
            let theta = (a_qq.clone() - a_pp.clone()) / two / a_pq.clone();
            let t = if theta >= <T::Real as Zero>::ZERO {
                one.clone()
                    / (theta.clone()
                        + (one.clone() + theta.clone() * theta).sqrt())
            } else {
                -one.clone()
                    / (-theta.clone()
                        + (one.clone() + theta.clone() * theta).sqrt())
            };
            let c = one.clone() / (one + t.clone() * t.clone()).sqrt();
            let s = t * c.clone();

            let c_val = T::from_real(c.clone());
            let s_val = T::from_real(s.clone());

            for k in 0..n {
                if k != p && k != q {
                    let a_kp = uplo_entry(a, k, p, uplo);
                    let a_kq = uplo_entry(a, k, q, uplo);

                    let new_kp = (c_val.clone() * a_kp.clone())
                        - (s_val.clone() * a_kq.clone());
                    let new_kq =
                        (s_val.clone() * a_kp) + (c_val.clone() * a_kq);

                    unsafe {
                        a.set_unchecked(k, p, new_kp.clone());
                        a.set_unchecked(k, q, new_kq.clone());
                        a.set_unchecked(p, k, new_kp);
                        a.set_unchecked(q, k, new_kq);
                    }
                }
            }

            let two_a_pq = a_pq.clone() + a_pq;
            let c2 = c.clone() * c.clone();
            let s2 = s.clone() * s.clone();
            let cs = c * s;
            let new_pp = (c2.clone() * a_pp.clone())
                - (cs.clone() * two_a_pq.clone())
                + (s2.clone() * a_qq.clone());
            let new_qq = (s2 * a_pp) + (cs * two_a_pq) + (c2 * a_qq);

            unsafe {
                a.set_unchecked(p, p, T::from_real(new_pp));
                a.set_unchecked(q, q, T::from_real(new_qq));
                a.set_unchecked(p, q, T::ZERO);
                a.set_unchecked(q, p, T::ZERO);
            }

            if jobz == JobZ::Vectors {
                for k in 0..n {
                    let v_kp = work[k * n + p].clone();
                    let v_kq = work[k * n + q].clone();
                    work[k * n + p] = (c_val.clone() * v_kp.clone())
                        - (s_val.clone() * v_kq.clone());
                    work[k * n + q] =
                        (s_val.clone() * v_kp) + (c_val.clone() * v_kq);
                }
            }

            iter += 1;
        }

        for i in 0..n {
            w[i] = unsafe { a.get_unchecked(i, i).re() };
        }

        if jobz == JobZ::Vectors {
            for i in 0..n {
                for j in 0..n {
                    unsafe {
                        a.set_unchecked(i, j, work[i * n + j].clone());
                    }
                }
            }
        }

        Ok(())
    }
}

/// Jacobi eigendecomposition of a complex Hermitian operand under an explicit
/// sweep budget.
///
/// Crate-private seam for [`lapack::Heev`]. See [`syev_impl`] for the budget
/// contract.
#[inline(always)]
pub(crate) fn heev_impl<T, A>(
    jobz: JobZ,
    uplo: UpLo,
    a: &mut A,
    w: &mut [T::Real],
    work: &mut [T],
    max_iter: usize,
) -> LinAlgResult<()>
where
    T: Scalar + Div<Output = T>,
    T::Real: Float,
    A: DenseStorageMut<T>,
{
    {
        let n = a.rows();
        debug_assert_eq!(n, a.cols());
        let min_work = if jobz == JobZ::Vectors { n * n } else { n };
        if w.len() < n || work.len() < min_work {
            return Err(LinAlgError::WorkspaceTooSmall);
        }

        if jobz == JobZ::Vectors {
            for i in 0..n {
                for j in 0..n {
                    work[i * n + j] = if i == j { T::ONE } else { T::ZERO };
                }
            }
        }

        let mut iter = 0;
        loop {
            let mut max_off = <T::Real as Zero>::ZERO;
            let mut p = 0;
            let mut q = 1;

            for i in 0..n {
                for j in (i + 1)..n {
                    let val = uplo_entry(a, i, j, uplo).abs2();
                    if val > max_off || val != val {
                        max_off = val;
                        p = i;
                        q = j;
                    }
                }
            }

            let eps = <T::Real as Float>::epsilon();
            let converged = max_off <= eps;
            if converged || iter >= max_iter {
                if iter >= max_iter && !converged {
                    return Err(LinAlgError::MaxIterationsReached);
                }
                break;
            }

            let a_pp = uplo_entry(a, p, p, uplo).re();
            let a_qq = uplo_entry(a, q, q, uplo).re();
            let a_pq = uplo_entry(a, p, q, uplo);
            let abs_a_pq = a_pq.abs2().sqrt();

            if abs_a_pq.is_zero() {
                break;
            }

            let one = <T::Real as One>::ONE;
            let two = one.clone() + one.clone();
            let theta =
                (a_qq.clone() - a_pp.clone()) / (two * abs_a_pq.clone());
            let t_real = if theta >= <T::Real as Zero>::ZERO {
                one.clone()
                    / (theta.clone()
                        + (one.clone() + theta.clone() * theta).sqrt())
            } else {
                -one.clone()
                    / (-theta.clone()
                        + (one.clone() + theta.clone() * theta).sqrt())
            };
            let c_real =
                one.clone() / (one + t_real.clone() * t_real.clone()).sqrt();
            let s_phase = a_pq / T::from_real(abs_a_pq);
            let s = s_phase * T::from_real(t_real * c_real.clone());
            let c = T::from_real(c_real);
            let s_conj = s.clone().conj();

            for k in 0..n {
                if k != p && k != q {
                    let a_kp = uplo_entry(a, k, p, uplo);
                    let a_kq = uplo_entry(a, k, q, uplo);

                    let new_kp =
                        (c.clone() * a_kp.clone()) - (s.clone() * a_kq.clone());
                    let new_kq = (s_conj.clone() * a_kp) + (c.clone() * a_kq);

                    unsafe {
                        a.set_unchecked(k, p, new_kp.clone());
                        a.set_unchecked(k, q, new_kq.clone());
                        a.set_unchecked(p, k, new_kp.conj());
                        a.set_unchecked(q, k, new_kq.conj());
                    }
                }
            }

            let a_pp_val = uplo_entry(a, p, p, uplo);
            let a_qq_val = uplo_entry(a, q, q, uplo);
            let a_pq_val = uplo_entry(a, p, q, uplo);

            let new_pp = (c.clone() * c.clone() * a_pp_val.clone())
                - (c.clone() * s.clone() * a_pq_val.clone().conj())
                - (c.clone() * s_conj.clone() * a_pq_val.clone())
                + (s.clone() * s_conj.clone() * a_qq_val.clone());
            let new_qq = (s.clone() * s_conj.clone() * a_pp_val)
                + (c.clone() * s_conj.clone() * a_pq_val.clone())
                + (c.clone() * s.clone() * a_pq_val.conj())
                + (c.clone() * c.clone() * a_qq_val);

            unsafe {
                a.set_unchecked(p, p, T::from_real(new_pp.re()));
                a.set_unchecked(q, q, T::from_real(new_qq.re()));
                a.set_unchecked(p, q, T::ZERO);
                a.set_unchecked(q, p, T::ZERO);
            }

            if jobz == JobZ::Vectors {
                for k in 0..n {
                    let v_kp = work[k * n + p].clone();
                    let v_kq = work[k * n + q].clone();
                    work[k * n + p] =
                        (c.clone() * v_kp.clone()) - (s.clone() * v_kq.clone());
                    work[k * n + q] =
                        (s_conj.clone() * v_kp) + (c.clone() * v_kq);
                }
            }

            iter += 1;
        }

        for i in 0..n {
            w[i] = unsafe { a.get_unchecked(i, i).re() };
        }

        if jobz == JobZ::Vectors {
            for i in 0..n {
                for j in 0..n {
                    unsafe {
                        a.set_unchecked(i, j, work[i * n + j].clone());
                    }
                }
            }
        }

        Ok(())
    }
}

impl<T, A> lapack::Syev<T, A> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    T::Real: Float,
    A: DenseStorageMut<T>,
{
    #[inline(always)]
    fn syev(
        jobz: JobZ,
        uplo: UpLo,
        a: &mut A,
        w: &mut [T::Real],
        work: &mut [T],
    ) -> LinAlgResult<()> {
        let n = a.rows();
        syev_impl(jobz, uplo, a, w, work, 50 * n * n)
    }
}

impl<T, A> lapack::Heev<T, A> for DefaultBlas
where
    T: Scalar + Div<Output = T>,
    T::Real: Float,
    A: DenseStorageMut<T>,
{
    #[inline(always)]
    fn heev(
        jobz: JobZ,
        uplo: UpLo,
        a: &mut A,
        w: &mut [T::Real],
        work: &mut [T],
    ) -> LinAlgResult<()> {
        let n = a.rows();
        heev_impl(jobz, uplo, a, w, work, 50 * n * n)
    }
}
