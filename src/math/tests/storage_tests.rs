//! # Static Storage Tests
//!
//! ## Functional Requirement Coverage (`storage-trait-design.md`)
//!
//! - **FR-1** (core storage trait — ptr access, index mapping, unchecked
//!   access): `ArrayStorage`/`StorageView` layout and bounds-check tests,
//!   plus the property-based index-mapping suite below.
//! - **FR-2** (type-level dimension encoding): the `Const<_>`/`U2`/`U3`
//!   parameterization throughout and `test_storage_view_length_mismatch`
//!   for the runtime-length assertion.
//! - **FR-3** (owned/borrowed/scratch backend categories): `ArrayStorage`
//!   (owned), `StorageView`/`StorageViewMut` (borrowed), `PivotStorage`
//!   (scratch).
//! - **FR-3a** (stack array storage): `test_array_storage_*`.
//! - **FR-3b** (zero-copy view storage): `test_storage_view_*`.
//! - **FR-4** (mutable vs. contiguous as independent capabilities):
//!   `test_array_storage_bounds_checked_access` and the `ORDER` assertions
//!   in the layout tests.
//! - **FR-5** (safe initialization strategies): `test_array_storage_init_strategies`,
//!   `test_pivot_storage_identity_and_swap`.
//! - **FR-6** (BLAS interoperability): `test_storage_gemv_contiguous_storage_interop`.
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::type_complexity,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::doc_markdown,
    clippy::similar_names
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod storage_test_suite {
    use crate::assert_almost_eq;
    use crate::math::ConversionError;
    use crate::math::complex_num::Complex;
    use crate::math::num_traits::{One, Zero};
    use crate::math::num_types::{Const, U1, U2, U3};
    use crate::math::storage::{
        ArrayCooStorage, ArrayCsrStorage, ArraySparseVector, ArrayStorage,
        ColMajor, ContiguousStorage, DenseStorage, DenseStorageMut, Diag,
        DiagonalStorage, DiagonalView, DiagonalViewMut, FromDenseStorage,
        HermitianPackedStorage, HermitianPackedView, HermitianPackedViewMut,
        MatrixLayout, PackedStorage, PackedStorageMut, PivotStorage,
        RowArrayStorage, RowMajor, SparseStorage, SparseVectorStorage,
        StorageError, StorageInit, StorageView, StorageViewMut,
        SymmetricPackedStorage, SymmetricPackedView, SymmetricPackedViewMut,
        ToCsrStorage, ToDenseStorage, TriangularPackedStorage,
        TriangularPackedView, TriangularPackedViewMut, UpLo, ViewStorage,
        ViewStorageMut, array_from_iterator, reverse_array,
    };
    use crate::math::subprograms::level2::Gemv;

    #[cfg_attr(test, test)]
    /// Verifies reversing an odd-length static array.
    fn test_storage_reverse_array_odd_length() {
        let array = [1, 2, 3, 4, 5];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [5, 4, 3, 2, 1]);
    }

    #[cfg_attr(test, test)]
    /// Verifies reversing an even-length static array.
    fn test_storage_reverse_array_even_length() {
        let array = [1, 2, 3, 4];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [4, 3, 2, 1]);
    }

    #[cfg_attr(test, test)]
    /// Verifies reversing a single-element array (identity behavior).
    fn test_storage_reverse_array_single_element() {
        let array = [1];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [1]);
    }

    #[cfg_attr(test, test)]
    /// Verifies populating a static array from an iterator with exact element count matching the array size.
    fn test_storage_array_from_iterator_exact() {
        let array: [i32; 5] = unsafe { array_from_iterator(0..5) };
        assert_eq!(array, [0, 1, 2, 3, 4]);
    }

    #[cfg(test)]
    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn _test_array_from_iterator_too_few() {
        let _array: [i32; 5] = unsafe { array_from_iterator(0..3) };
    }

    #[cfg_attr(test, test)]
    /// Verifies populating a static array from an iterator that produces more elements than the array size (extra items ignored).
    fn test_storage_array_from_iterator_excess_elements() {
        let array: [i32; 5] = unsafe { array_from_iterator(0..10) };
        assert_eq!(array, [0, 1, 2, 3, 4]);
    }

    // --- ArrayStorage ---

    #[cfg_attr(test, test)]
    /// Verifies `ArrayStorage::from_array` lays elements out column-major,
    /// matching `ContiguousStorage::ORDER` and `Storage::offset` (FR-1 +
    /// FR-3a of `storage-trait-design.md`).
    fn test_array_storage_column_major_layout() {
        // 2 rows x 3 cols: columns are [1,2], [3,4], [5,6]
        let storage = ArrayStorage::from_array([[1, 2], [3, 4], [5, 6]]);

        assert_eq!(
            <ArrayStorage<i32, 2, 3> as ContiguousStorage<i32>>::ORDER,
            MatrixLayout::ColMajor
        );
        assert_eq!(storage.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(storage.get(0, 0), Some(&1));
        assert_eq!(storage.get(1, 0), Some(&2));
        assert_eq!(storage.get(0, 1), Some(&3));
        assert_eq!(storage.get(1, 2), Some(&6));
    }

    #[cfg_attr(test, test)]
    /// Verifies `get`/`get_mut` return `None` outside `[0, rows) x [0, cols)`
    /// (FR-1 + FR-4 of `storage-trait-design.md`).
    fn test_array_storage_bounds_checked_access() {
        let mut storage: ArrayStorage<i32, 2, 2> =
            ArrayStorage::from_array([[1, 2], [3, 4]]);

        assert_eq!(storage.get(2, 0), None);
        assert_eq!(storage.get(0, 2), None);
        assert_eq!(storage.get_mut(2, 0), None);

        if let Some(elem) = storage.get_mut(1, 1) {
            *elem = 42;
        }
        assert_eq!(storage.as_slice(), &[1, 2, 3, 42]);
    }

    #[cfg_attr(test, test)]
    /// Verifies `StorageInit`'s four safe construction strategies (FR-5 of
    /// `storage-trait-design.md`).
    fn test_array_storage_init_strategies() {
        let from_fn: ArrayStorage<i32, 2, 2> =
            StorageInit::<i32, Const<2>, Const<2>>::from_fn(|i, j| {
                (i * 10 + j) as i32
            });
        assert_eq!(from_fn.as_slice(), &[0, 10, 1, 11]);

        let from_element: ArrayStorage<i32, 2, 2> =
            StorageInit::<i32, Const<2>, Const<2>>::from_element(7);
        assert_eq!(from_element.as_slice(), &[7, 7, 7, 7]);

        let zeros: ArrayStorage<i32, 2, 2> =
            StorageInit::<i32, Const<2>, Const<2>>::zeros();
        assert_eq!(zeros.as_slice(), &[0, 0, 0, 0]);

        let identity: ArrayStorage<i32, 2, 2> =
            StorageInit::<i32, Const<2>, Const<2>>::identity();
        assert_eq!(identity.as_slice(), &[1, 0, 0, 1]);
    }

    #[cfg_attr(test, test)]
    /// Verifies `as_mut_slice` observes writes made through `get_mut` and
    /// vice versa (both paths address the same backing memory) (FR-3a of
    /// `storage-trait-design.md`).
    fn test_array_storage_mutation_round_trip() {
        let mut storage: ArrayStorage<i32, 2, 2> =
            StorageInit::<i32, Const<2>, Const<2>>::zeros();
        storage.as_mut_slice()[2] = 9;
        assert_eq!(storage.get(0, 1), Some(&9));
    }

    // --- StorageView / StorageViewMut ---

    #[cfg_attr(test, test)]
    /// Verifies `StorageView` reproduces both row-major and column-major
    /// index mappings over the same backing slice without copying (FR-2 +
    /// FR-3b of `storage-trait-design.md`).
    fn test_storage_view_layout_selection() {
        let data = [1, 2, 3, 4, 5, 6];

        let col_major: StorageView<'_, i32, U2, U3, ColMajor> =
            StorageView::new(&data).unwrap();
        assert_eq!(col_major.get(1, 2), Some(&6));
        assert_eq!(
            <StorageView<'_, i32, U2, U3, ColMajor> as ContiguousStorage<
                i32,
            >>::ORDER,
            MatrixLayout::ColMajor
        );

        let row_major: StorageView<'_, i32, U2, U3, RowMajor> =
            StorageView::new(&data).unwrap();
        assert_eq!(row_major.get(1, 2), Some(&6));
        assert_eq!(row_major.get(0, 1), Some(&2));
        assert_eq!(
            <StorageView<'_, i32, U2, U3, RowMajor> as ContiguousStorage<
                i32,
            >>::ORDER,
            MatrixLayout::RowMajor
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies both view constructors reject a backing slice whose length
    /// does not match `R::USIZE * C::USIZE` (FR-2 of `storage-trait-design.md`).
    fn test_storage_view_length_mismatch() {
        let data = [1, 2, 3];
        assert!(matches!(
            StorageView::<'_, i32, U2, U3, ColMajor>::new(&data),
            Err(ConversionError::DimensionMismatch)
        ));

        let mut data_mut = [1, 2, 3];
        assert!(matches!(
            StorageViewMut::<'_, i32, U2, U3, ColMajor>::new(&mut data_mut),
            Err(ConversionError::DimensionMismatch)
        ));
    }

    #[cfg_attr(test, test)]
    /// Verifies `StorageViewMut` writes are visible through the original
    /// borrowed slice once the view is dropped (FR-3b of
    /// `storage-trait-design.md`).
    fn test_storage_view_mut_writes_through() {
        let mut data = [0; 4];
        {
            let mut view: StorageViewMut<'_, i32, U2, U2, ColMajor> =
                StorageViewMut::new(&mut data).unwrap();
            if let Some(elem) = view.get_mut(1, 1) {
                *elem = 5;
            }
        }
        assert_eq!(data, [0, 0, 0, 5]);
    }

    // --- BLAS Interoperability (FR-6) ---

    #[cfg_attr(test, test)]
    /// Verifies a `ContiguousStorage` backend's `as_slice()`/`as_mut_slice()`
    /// feed directly into a BLAS subprogram trait (`GEMV`) with no
    /// intermediate copy and that `ORDER` matches the layout `GEMV`'s
    /// default implementation assumes (row-major: `a.chunks_exact(cols)`)
    /// (FR-6 of `storage-trait-design.md`).
    fn test_storage_gemv_contiguous_storage_interop() {
        // A = [[1, 2], [3, 4]], row-major.
        let a_data = [1.0f32, 2.0, 3.0, 4.0];
        let a: StorageView<'_, f32, U2, U2, RowMajor> =
            StorageView::new(&a_data).unwrap();
        assert_eq!(
            <StorageView<'_, f32, U2, U2, RowMajor> as ContiguousStorage<
                f32,
            >>::ORDER,
            MatrixLayout::RowMajor
        );

        let x_data = [1.0f32, 1.0];
        let x: StorageView<'_, f32, U2, U1, RowMajor> =
            StorageView::new(&x_data).unwrap();
        let mut y_data = [0.0f32; 2];
        let mut y: StorageViewMut<'_, f32, U2, U1, RowMajor> =
            StorageViewMut::new(&mut y_data).unwrap();

        // y = 1.0 * A * x + 0.0 * y
        crate::math::subprograms::DefaultBlas::gemv(
            crate::math::storage::Trans::NoTrans,
            1.0,
            &a,
            &x,
            0.0,
            &mut y,
        );

        crate::assert_almost_eq!(y_data[0], 3.0);
        crate::assert_almost_eq!(y_data[1], 7.0);
    }

    // --- PivotStorage ---

    #[cfg_attr(test, test)]
    /// Verifies `PivotStorage::identity` and `swap` maintain a permutation
    /// (FR-3 + FR-5 of `storage-trait-design.md`, the scratch-data backend
    /// category and its identity initialization).
    fn test_pivot_storage_identity_and_swap() {
        let mut pivots: PivotStorage<4> = PivotStorage::identity();
        assert_eq!(pivots.as_slice(), &[0, 1, 2, 3]);

        pivots.swap(0, 3);
        assert_eq!(pivots.as_slice(), &[3, 1, 2, 0]);
    }

    // --- Packed Structured Storage Tests ---

    #[cfg_attr(test, test)]
    /// Verifies `DiagonalStorage` operations and bounds.
    fn test_diagonal_storage() {
        use crate::math::storage::{
            DiagonalStorage, PackedStorage, PackedStorageMut,
        };

        let mut diag = DiagonalStorage::<f64, 3>::from_array([1.0, 2.0, 3.0]);
        assert_eq!(diag.dim(), 3);
        assert_eq!(diag.value(0, 0), Some(1.0));
        assert_eq!(diag.value(1, 1), Some(2.0));
        assert_eq!(diag.value(2, 2), Some(3.0));
        assert_eq!(diag.value(0, 1), Some(0.0));
        assert_eq!(diag.value(3, 0), None);

        assert!(diag.set(0, 0, 10.0).is_ok());
        assert_eq!(diag.value(0, 0), Some(10.0));
        assert!(diag.set(0, 1, 5.0).is_err());
    }

    #[cfg_attr(test, test)]
    /// Verifies `SymmetricPackedStorage` upper and lower triangle indexing.
    fn test_symmetric_packed_storage() {
        use crate::math::storage::{
            PackedStorage, SymmetricPackedStorage, UpLo,
        };

        // 3x3 symmetric: PACKED_LEN = 3*4/2 = 6
        // Upper: (0,0)=1, (0,1)=2, (1,1)=3, (0,2)=4, (1,2)=5, (2,2)=6
        let sym_upper = SymmetricPackedStorage::<f64, 3, 6>::new(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            UpLo::Upper,
        );
        assert_eq!(sym_upper.value(0, 1), Some(2.0));
        assert_eq!(sym_upper.value(1, 0), Some(2.0)); // Symmetric reflection
        assert_eq!(sym_upper.value(2, 1), Some(5.0));

        // Lower: (0,0)=1, (1,0)=2, (2,0)=3, (1,1)=4, (2,1)=5, (2,2)=6
        let sym_lower = SymmetricPackedStorage::<f64, 3, 6>::new(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            UpLo::Lower,
        );
        assert_eq!(sym_lower.value(1, 0), Some(2.0));
        assert_eq!(sym_lower.value(0, 1), Some(2.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies `HermitianPackedStorage` enforces real diagonal and conjugate reflection.
    fn test_hermitian_packed_storage() {
        use crate::math::complex_num::Complex64;
        use crate::math::storage::{
            HermitianPackedStorage, PackedStorage, PackedStorageMut, UpLo,
        };

        let data = [
            Complex64::new(1.0, 0.0),
            Complex64::new(2.0, 3.0),
            Complex64::new(4.0, 0.0),
        ];
        let mut herm =
            HermitianPackedStorage::<Complex64, 2, 3>::new(data, UpLo::Upper);
        assert_eq!(herm.value(0, 1), Some(Complex64::new(2.0, 3.0)));
        assert_eq!(herm.value(1, 0), Some(Complex64::new(2.0, -3.0))); // Conjugate reflection

        // Reject non-real diagonal write
        assert!(herm.set(0, 0, Complex64::new(1.0, 2.0)).is_err());
        assert!(herm.set(0, 0, Complex64::new(5.0, 0.0)).is_ok());
    }

    #[cfg_attr(test, test)]
    /// Verifies `TriangularPackedStorage` unit diagonal and zero regions.
    fn test_triangular_packed_storage() {
        use crate::math::storage::{
            Diag, PackedStorage, PackedStorageMut, TriangularPackedStorage,
            UpLo,
        };

        let mut tri = TriangularPackedStorage::<f64, 2, 3>::new(
            [9.0, 2.0, 9.0],
            UpLo::Upper,
            Diag::Unit,
        );
        assert_eq!(tri.value(0, 0), Some(1.0)); // Unit diagonal overrides
        assert_eq!(tri.value(0, 1), Some(2.0));
        assert_eq!(tri.value(1, 0), Some(0.0)); // Strictly zero

        // Rejects write to unit diagonal
        assert!(tri.set(0, 0, 5.0).is_err());
    }

    // --- Sparse Storage Tests ---

    #[cfg_attr(test, test)]
    /// Verifies `ArrayCooStorage` push and `ArrayCsrStorage::from_coo` 3-pass assembly.
    fn test_coo_to_csr_assembly() {
        use crate::math::storage::{
            ArrayCooStorage, ArrayCsrStorage, SparseStorage,
        };

        let mut coo = ArrayCooStorage::<f64, 3, 3, 10>::new();
        assert!(coo.push(0, 0, 1.0).is_ok());
        assert!(coo.push(0, 2, 2.0).is_ok());
        assert!(coo.push(1, 1, 3.0).is_ok());
        assert!(coo.push(0, 2, 1.0).is_ok()); // Duplicate (0, 2) to test accumulation

        let csr = ArrayCsrStorage::<f64, 3, 3, 10, 4>::from_coo(&coo).unwrap();
        assert_eq!(csr.nnz(), 3);
        assert_eq!(csr.get(0, 0), Some(1.0));
        assert_eq!(csr.get(0, 2), Some(3.0)); // 2.0 + 1.0 accumulated
        assert_eq!(csr.get(1, 1), Some(3.0));
        assert_eq!(csr.get(0, 1), Some(0.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies `ArraySparseVector` operations.
    fn test_sparse_vector() {
        let mut vec = ArraySparseVector::<f64, 10, 5>::new();
        assert!(vec.push(2, 4.0).is_ok());
        assert!(vec.push(7, 8.0).is_ok());
        assert_eq!(vec.nnz(), 2);
        assert_eq!(vec.indices(), &[2, 7]);
        assert_eq!(vec.values(), &[4.0, 8.0]);
    }

    #[cfg_attr(test, test)]
    /// Verifies `transpose_view`, `reverse_view`, and their mutable counterparts (FR-6 of `storage-design.md`).
    fn test_storage_view_transformations() {
        let mut storage = ArrayStorage::<f64, 2, 3>::from_array([
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]);

        // Transpose view: shape becomes 3x2
        let trans = storage.transpose_view();
        assert_eq!(trans.rows(), 3);
        assert_eq!(trans.cols(), 2);
        assert_eq!(trans.get(0, 0), Some(&1.0));
        assert_eq!(trans.get(0, 1), Some(&2.0));
        assert_eq!(trans.get(1, 0), Some(&3.0));
        assert_eq!(trans.get(2, 1), Some(&6.0));

        // Reverse view: reads backwards from (1, 2)
        let rev = storage.reverse_view();
        assert_eq!(rev.rows(), 2);
        assert_eq!(rev.cols(), 3);
        assert_eq!(rev.get(0, 0), Some(&6.0));
        assert_eq!(rev.get(1, 2), Some(&1.0));

        // Mutable transpose view: updates underlying storage
        {
            let mut trans_mut = storage.transpose_mut_view();
            assert!(trans_mut.set(0, 1, 20.0).is_ok());
        }
        assert_eq!(storage.get(1, 0), Some(&20.0));

        // Mutable reverse view: updates underlying storage
        {
            let mut rev_mut = storage.reverse_mut_view();
            assert!(rev_mut.set(0, 0, 60.0).is_ok());
        }
        assert_eq!(storage.get(1, 2), Some(&60.0));

        // Direct ViewStorage and ViewStorageMut constructors
        let raw = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let vs = ViewStorage::<f64, U2, U3>::new(&raw).unwrap();
        assert_eq!(vs.rows(), 2);
        assert_eq!(vs.cols(), 3);
        assert_eq!(vs.get(1, 0), Some(&2.0));

        let mut raw_mut = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let mut vsm = ViewStorageMut::<f64, U2, U3>::new(&mut raw_mut).unwrap();
        assert!(vsm.set(1, 0, 200.0).is_ok());
        assert_eq!(vsm.get(1, 0), Some(&200.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies non-owning packed structured views (FR-10 of `storage-design.md`).
    fn test_storage_packed_views() {
        // DiagonalView & DiagonalViewMut
        let mut diag_data = [10.0, 20.0, 30.0];
        let diag_view = DiagonalView::<f64, 3>::new(&diag_data).unwrap();
        assert_eq!(diag_view.dim(), 3);
        assert_eq!(diag_view.value(0, 0), Some(10.0));
        assert_eq!(diag_view.value(1, 1), Some(20.0));
        assert_eq!(diag_view.value(0, 1), Some(0.0));

        let mut diag_mut =
            DiagonalViewMut::<f64, 3>::new(&mut diag_data).unwrap();
        assert!(diag_mut.set(1, 1, 25.0).is_ok());
        assert_eq!(diag_mut.value(1, 1), Some(25.0));

        // SymmetricPackedView & SymmetricPackedViewMut
        let mut sym_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // Upper triangle for 3x3: (0,0)=1, (0,1)=2, (1,1)=3, (0,2)=4, (1,2)=5, (2,2)=6
        let sym_view =
            SymmetricPackedView::<f64, 3>::new(&sym_data, UpLo::Upper).unwrap();
        assert_eq!(sym_view.value(0, 1), Some(2.0));
        assert_eq!(sym_view.value(1, 0), Some(2.0)); // Symmetric reflection
        assert_eq!(sym_view.value(0, 2), Some(4.0));
        assert_eq!(sym_view.value(2, 0), Some(4.0));

        let mut sym_mut =
            SymmetricPackedViewMut::<f64, 3>::new(&mut sym_data, UpLo::Upper)
                .unwrap();
        assert!(sym_mut.set(0, 1, 20.0).is_ok());
        assert_eq!(sym_mut.value(1, 0), Some(20.0));

        // HermitianPackedView & HermitianPackedViewMut
        let c1 = Complex::new(1.0, 0.0);
        let c12 = Complex::new(2.0, 3.0);
        let c2 = Complex::new(4.0, 0.0);
        let c13 = Complex::new(5.0, -1.0);
        let c23 = Complex::new(6.0, 2.0);
        let c3 = Complex::new(7.0, 0.0);
        let mut herm_data = [c1, c12, c2, c13, c23, c3];
        let herm_view = HermitianPackedView::<Complex<f64>, 3>::new(
            &herm_data,
            UpLo::Upper,
        )
        .unwrap();
        assert_eq!(herm_view.value(0, 1), Some(Complex::new(2.0, 3.0)));
        assert_eq!(herm_view.value(1, 0), Some(Complex::new(2.0, -3.0))); // Conjugate reflection

        let mut herm_mut = HermitianPackedViewMut::<Complex<f64>, 3>::new(
            &mut herm_data,
            UpLo::Upper,
        )
        .unwrap();
        assert!(herm_mut.set(0, 1, Complex::new(20.0, 30.0)).is_ok());
        assert_eq!(herm_mut.value(1, 0), Some(Complex::new(20.0, -30.0)));

        // TriangularPackedView & TriangularPackedViewMut
        let mut tri_data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let tri_view = TriangularPackedView::<f64, 3>::new(
            &tri_data,
            UpLo::Upper,
            Diag::NonUnit,
        )
        .unwrap();
        assert_eq!(tri_view.value(0, 1), Some(2.0));
        assert_eq!(tri_view.value(1, 0), Some(0.0)); // Strictly zero below diagonal

        let mut tri_mut = TriangularPackedViewMut::<f64, 3>::new(
            &mut tri_data,
            UpLo::Upper,
            Diag::NonUnit,
        )
        .unwrap();
        assert!(tri_mut.set(0, 1, 22.0).is_ok());
        assert_eq!(tri_mut.value(0, 1), Some(22.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies layout conversions between dense, packed, and sparse formats (FR-17 of `storage-design.md`).
    fn test_storage_conversions() {
        // DiagonalStorage <-> ArrayStorage
        let diag = DiagonalStorage::<f64, 3>::from_array([1.0, 2.0, 3.0]);
        let dense: ArrayStorage<f64, 3, 3> = diag.to_dense().unwrap();
        assert_eq!(dense.get(0, 0), Some(&1.0));
        assert_eq!(dense.get(1, 1), Some(&2.0));
        assert_eq!(dense.get(0, 1), Some(&0.0));

        let diag_back = DiagonalStorage::<f64, 3>::from_dense(&dense).unwrap();
        assert_eq!(diag_back.value(2, 2), Some(3.0));

        // SymmetricPackedStorage <-> ArrayStorage
        let sym = SymmetricPackedStorage::<f64, 2, 3>::new(
            [10.0, 20.0, 30.0],
            UpLo::Upper,
        );
        let dense_sym: ArrayStorage<f64, 2, 2> = sym.to_dense().unwrap();
        assert_eq!(dense_sym.get(0, 1), Some(&20.0));
        assert_eq!(dense_sym.get(1, 0), Some(&20.0));

        let sym_back =
            SymmetricPackedStorage::<f64, 2, 3>::from_dense(&dense_sym)
                .unwrap();
        assert_eq!(sym_back.value(0, 1), Some(20.0));

        // TriangularPackedStorage <-> ArrayStorage
        let tri = TriangularPackedStorage::<f64, 2, 3>::new(
            [1.0, 2.0, 3.0],
            UpLo::Upper,
            Diag::NonUnit,
        );
        let dense_tri: ArrayStorage<f64, 2, 2> = tri.to_dense().unwrap();
        assert_eq!(dense_tri.get(0, 1), Some(&2.0));
        assert_eq!(dense_tri.get(1, 0), Some(&0.0));

        let tri_back =
            TriangularPackedStorage::<f64, 2, 3>::from_dense(&dense_tri)
                .unwrap();
        assert_eq!(tri_back.value(0, 1), Some(2.0));

        // ArrayCooStorage -> ArrayStorage & ToCsrStorage
        let mut coo = ArrayCooStorage::<f64, 2, 2, 4>::new();
        coo.push(0, 0, 5.0).unwrap();
        coo.push(1, 1, 7.0).unwrap();
        let dense_coo: ArrayStorage<f64, 2, 2> = coo.to_dense().unwrap();
        assert_eq!(dense_coo.get(0, 0), Some(&5.0));
        assert_eq!(dense_coo.get(1, 1), Some(&7.0));
        assert_eq!(dense_coo.get(0, 1), Some(&0.0));

        let csr: ArrayCsrStorage<f64, 2, 2, 4, 3> = coo.to_csr().unwrap();
        assert_eq!(csr.get(0, 0), Some(5.0));
        let dense_csr: ArrayStorage<f64, 2, 2> = csr.to_dense().unwrap();
        assert_eq!(dense_csr.get(0, 0), Some(&5.0));
    }

    #[cfg_attr(test, test)]
    /// Verifies all `StorageError` failure paths (V&V §6.1 of `storage-design.md`).
    fn test_storage_error_failure_paths() {
        // 1. InvalidHermitianDiagonal: imaginary component on diagonal
        let mut herm = HermitianPackedStorage::<Complex<f64>, 2, 3>::new(
            [
                Complex::new(1.0, 0.0),
                Complex::new(2.0, 1.0),
                Complex::new(3.0, 0.0),
            ],
            UpLo::Upper,
        );
        let err = herm.set(0, 0, Complex::new(5.0, 2.0));
        assert_eq!(err, Err(StorageError::InvalidHermitianDiagonal));

        // 2. ImmutableUnitDiagonal: writing to unit diagonal
        let mut tri = TriangularPackedStorage::<f64, 2, 3>::new(
            [1.0, 2.0, 3.0],
            UpLo::Upper,
            Diag::Unit,
        );
        assert_eq!(
            tri.set(0, 0, 10.0),
            Err(StorageError::ImmutableUnitDiagonal)
        );

        // 3. CapacityExceeded: pushing past MAX_NNZ
        let mut coo = ArrayCooStorage::<f64, 2, 2, 2>::new();
        assert!(coo.push(0, 0, 1.0).is_ok());
        assert!(coo.push(1, 1, 2.0).is_ok());
        assert_eq!(coo.push(0, 1, 3.0), Err(StorageError::CapacityExceeded));

        let mut svec = ArraySparseVector::<f64, 5, 1>::new();
        assert!(svec.push(0, 1.0).is_ok());
        assert_eq!(svec.push(1, 2.0), Err(StorageError::CapacityExceeded));

        // 4. OutOfBounds
        assert_eq!(coo.push(5, 0, 1.0), Err(StorageError::OutOfBounds));
        assert_eq!(svec.push(10, 1.0), Err(StorageError::OutOfBounds));
        assert_eq!(
            herm.set(10, 10, Complex::new(1.0, 0.0)),
            Err(StorageError::OutOfBounds)
        );

        // 5. InvalidStructuralInvariant: writing off-diagonal in DiagonalStorage
        let mut diag = DiagonalStorage::<f64, 2>::from_array([1.0, 2.0]);
        assert_eq!(
            diag.set(0, 1, 5.0),
            Err(StorageError::InvalidStructuralInvariant)
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies Level 1 size_of memory formulas and stride invariants across dense, packed, and sparse backends.
    fn test_storage_level1_size_of_formulas_and_stride_invariants() {
        use core::mem::size_of;
        assert_eq!(size_of::<ArrayStorage<f32, 4, 4>>(), 64);
        assert_eq!(size_of::<ArrayStorage<f32, 16, 16>>(), 1024);
        assert_eq!(size_of::<ArrayStorage<f32, 64, 64>>(), 16384);
        assert_eq!(size_of::<ArrayStorage<Complex<f32>, 16, 16>>(), 2048);
        assert_eq!(size_of::<RowArrayStorage<f32, 4, 4>>(), 64);

        assert_eq!(size_of::<DiagonalStorage<f32, 4>>(), 16);
        assert_eq!(size_of::<DiagonalStorage<f32, 16>>(), 64);
        assert_eq!(size_of::<DiagonalStorage<f32, 64>>(), 256);
        assert_eq!(size_of::<DiagonalStorage<Complex<f32>, 16>>(), 128);

        assert_eq!(size_of::<SymmetricPackedStorage<f32, 4, 10>>(), 44);
        assert_eq!(size_of::<TriangularPackedStorage<f32, 4, 10>>(), 44);
        assert_eq!(
            size_of::<HermitianPackedStorage<Complex<f32>, 4, 10>>(),
            84
        );

        assert_eq!(
            size_of::<ArrayCsrStorage<f32, 4, 4, 12, 5>>(),
            12 * (4 + size_of::<usize>()) + 6 * size_of::<usize>()
        );
        assert_eq!(
            size_of::<ArraySparseVector<f32, 4, 2>>(),
            2 * (4 + size_of::<usize>()) + size_of::<usize>()
        );

        let dense_col = ArrayStorage::<f32, 4, 4>::zeros();
        assert_eq!(dense_col.r_stride(), 1);
        assert_eq!(dense_col.c_stride(), 4);
        let dense_row = RowArrayStorage::<f32, 4, 4>::zeros();
        assert_eq!(dense_row.r_stride(), 4);
        assert_eq!(dense_row.c_stride(), 1);
    }

    #[cfg_attr(test, test)]
    /// Val-1: Multi-Layout State Estimation — packed symmetric covariance with dense state.
    fn test_storage_val1_kalman_packed_covariance() {
        let mut x = ArrayStorage::<f64, 3, 1>::zeros();
        unsafe {
            x.set_unchecked(0, 0, 10.0);
            x.set_unchecked(1, 0, 2.0);
            x.set_unchecked(2, 0, 0.5);
        }
        let p_data = [1.0, 0.1, 2.0, 0.05, 0.2, 0.5];
        let mut p =
            SymmetricPackedStorage::<f64, 3, 6>::new(p_data, UpLo::Upper);
        assert_almost_eq!(p.value(0, 0).unwrap(), 1.0, 1e-10);
        assert_almost_eq!(p.value(0, 1).unwrap(), 0.1, 1e-10);
        assert_almost_eq!(p.value(1, 0).unwrap(), 0.1, 1e-10);
        assert_almost_eq!(p.value(1, 1).unwrap(), 2.0, 1e-10);
        assert_almost_eq!(p.value(2, 0).unwrap(), 0.05, 1e-10);
        assert_almost_eq!(p.value(0, 2).unwrap(), 0.05, 1e-10);

        assert!(p.set(0, 0, 1.2).is_ok());
        assert_almost_eq!(p.value(0, 0).unwrap(), 1.2, 1e-10);
        assert_almost_eq!(unsafe { *x.get_unchecked(0, 0) }, 10.0, 1e-10);
    }

    #[cfg_attr(test, test)]
    /// Val-2: Fixed-Capacity Sparse MPC — condensed horizon trajectory constraints on stack.
    fn test_storage_val2_sparse_mpc_condensed_horizon() {
        let mut coo = ArrayCooStorage::<f64, 4, 4, 8>::new();
        assert!(coo.push(0, 0, 1.0).is_ok());
        assert!(coo.push(0, 1, -0.5).is_ok());
        assert!(coo.push(1, 1, 1.0).is_ok());
        assert!(coo.push(1, 2, -0.5).is_ok());
        assert!(coo.push(2, 2, 1.0).is_ok());
        assert!(coo.push(2, 3, -0.5).is_ok());
        assert!(coo.push(3, 3, 1.0).is_ok());
        assert_eq!(coo.nnz(), 7);
        let csr: ArrayCsrStorage<f64, 4, 4, 8, 5> = coo.to_csr().unwrap();
        assert_eq!(csr.get(0, 0), Some(1.0));
        assert_eq!(csr.get(0, 1), Some(-0.5));
        assert_eq!(csr.get(0, 2), Some(0.0));
        assert_eq!(csr.get(1, 1), Some(1.0));
        assert_eq!(csr.get(3, 3), Some(1.0));
    }

    #[cfg_attr(test, test)]
    /// Val-3: Zero-Copy Windowing — submatrix extraction from coupled block model without copies.
    fn test_storage_val3_zero_copy_windowing() {
        let full = ArrayStorage::<f64, 4, 4>::from_array([
            [1.0, 3.0, 0.1, 0.3],
            [2.0, 4.0, 0.2, 0.4],
            [0.0, 0.0, 5.0, 7.0],
            [0.0, 0.0, 6.0, 8.0],
        ]);
        let slice = full.as_slice();
        let sub_view = unsafe {
            ViewStorage::<f64, Const<2>, Const<2>>::new_with_strides(
                slice.as_ptr(),
                1,
                4,
            )
        };
        assert_eq!(sub_view.get(0, 0), Some(&1.0));
        assert_eq!(sub_view.get(1, 0), Some(&3.0));
        assert_eq!(sub_view.get(0, 1), Some(&2.0));
        assert_eq!(sub_view.get(1, 1), Some(&4.0));
    }

    #[cfg_attr(test, test)]
    /// Val-4: Complex Frequency Response — MIMO G(jw) storage across discrete frequency grid.
    fn test_storage_val4_complex_mimo_frequency_response() {
        let g_jw = ArrayStorage::<Complex<f64>, 2, 2>::from_array([
            [Complex::new(0.5, -0.5), Complex::new(0.2, -0.1)],
            [Complex::ZERO, Complex::ONE],
        ]);
        assert_eq!(g_jw.get(0, 0), Some(&Complex::new(0.5, -0.5)));
        assert_eq!(g_jw.get(0, 1), Some(&Complex::ZERO));
        assert_eq!(g_jw.get(1, 0), Some(&Complex::new(0.2, -0.1)));
        assert_eq!(g_jw.get(1, 1), Some(&Complex::ONE));

        let g_t = g_jw.transpose_view();
        assert_eq!(g_t.get(0, 1), Some(&Complex::new(0.2, -0.1)));
        assert_eq!(g_t.get(1, 0), Some(&Complex::ZERO));
    }
}

// Property-based coverage of `ArrayStorage`/`StorageView` index-mapping
// invariants (FR-1 of `storage-trait-design.md`, §6.1 item 2). Kept outside
// the `#[hil_suite]`-wrapped module above: `proptest` is a host-only
// dev-dependency, unavailable to the `no_std`/on-target `hil` feature build.
#[cfg(test)]
mod storage_property_tests {
    use crate::math::num_types::Const;
    use crate::math::storage::{
        ArrayStorage, ColMajor, DenseStorage, StorageInit, StorageView,
    };
    use proptest::prelude::*;

    proptest! {
        /// Every logical `(i, j)` in a 3x4 `ArrayStorage` reads back the
        /// value it was constructed with and `offset()` matches the
        /// column-major formula exactly (implying, over the full `i, j`
        /// range, that the index mapping is injective).
        #[test]
        fn prop_array_storage_round_trips_every_index(
            vals in proptest::collection::vec(any::<i32>(), 12),
        ) {
            let storage: ArrayStorage<i32, 3, 4> =
                StorageInit::<i32, Const<3>, Const<4>>::from_fn(|i, j| vals[j * 3 + i]);

            for j in 0..4 {
                for i in 0..3 {
                    prop_assert_eq!(
                        storage.get(i, j),
                        Some(&vals[j * 3 + i])
                    );
                    prop_assert_eq!(storage.offset(i, j), (j * 3 + i) as isize);
                }
            }
            prop_assert_eq!(storage.get(3, 0), None);
            prop_assert_eq!(storage.get(0, 4), None);
        }

        /// `StorageView` over an arbitrary slice reproduces the same
        /// column-major index mapping as `ArrayStorage` for the same data.
        #[test]
        fn prop_storage_view_matches_array_storage(
            vals in proptest::collection::vec(any::<i32>(), 6),
        ) {
            let array: ArrayStorage<i32, 2, 3> =
                StorageInit::<i32, Const<2>, Const<3>>::from_fn(|i, j| vals[j * 2 + i]);
            let view: StorageView<'_, i32, Const<2>, Const<3>, ColMajor> =
                StorageView::new(&vals).unwrap();

            for j in 0..3 {
                for i in 0..2 {
                    prop_assert_eq!(
                        array.get(i, j),
                        view.get(i, j)
                    );
                }
            }
        }
    }
}
