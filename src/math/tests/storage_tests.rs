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
    clippy::cast_possible_wrap
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod storage_test_suite {
    use crate::math::ConversionError;
    use crate::math::num_types::{Const, U1, U2, U3};
    use crate::math::storage::{
        ArrayStorage, ColMajor, ContiguousStorage, ContiguousStorageMut,
        MatrixLayout, PivotStorage, RowMajor, Storage, StorageInit, StorageMut,
        StorageView, StorageViewMut, array_from_iterator, reverse_array,
    };
    use crate::math::subprograms::{BasicSubProgramsF32, level2::GEMV};

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
            <ArrayStorage<i32, 2, 3> as ContiguousStorage<
                i32,
                Const<2>,
                Const<3>,
            >>::ORDER,
            MatrixLayout::ColMajor
        );
        assert_eq!(storage.as_slice(), &[1, 2, 3, 4, 5, 6]);
        assert_eq!(
            Storage::<i32, Const<2>, Const<3>>::get(&storage, 0, 0),
            Some(&1)
        );
        assert_eq!(
            Storage::<i32, Const<2>, Const<3>>::get(&storage, 1, 0),
            Some(&2)
        );
        assert_eq!(
            Storage::<i32, Const<2>, Const<3>>::get(&storage, 0, 1),
            Some(&3)
        );
        assert_eq!(
            Storage::<i32, Const<2>, Const<3>>::get(&storage, 1, 2),
            Some(&6)
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies `get`/`get_mut` return `None` outside `[0, rows) x [0, cols)`
    /// (FR-1 + FR-4 of `storage-trait-design.md`).
    fn test_array_storage_bounds_checked_access() {
        let mut storage: ArrayStorage<i32, 2, 2> =
            ArrayStorage::from_array([[1, 2], [3, 4]]);

        assert_eq!(
            Storage::<i32, Const<2>, Const<2>>::get(&storage, 2, 0),
            None
        );
        assert_eq!(
            Storage::<i32, Const<2>, Const<2>>::get(&storage, 0, 2),
            None
        );
        assert_eq!(
            StorageMut::<i32, Const<2>, Const<2>>::get_mut(&mut storage, 2, 0),
            None
        );

        if let Some(elem) =
            StorageMut::<i32, Const<2>, Const<2>>::get_mut(&mut storage, 1, 1)
        {
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
        assert_eq!(
            Storage::<i32, Const<2>, Const<2>>::get(&storage, 0, 1),
            Some(&9)
        );
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
        assert_eq!(Storage::<i32, U2, U3>::get(&col_major, 1, 2), Some(&6));
        assert_eq!(
            <StorageView<'_, i32, U2, U3, ColMajor> as ContiguousStorage<
                i32,
                U2,
                U3,
            >>::ORDER,
            MatrixLayout::ColMajor
        );

        let row_major: StorageView<'_, i32, U2, U3, RowMajor> =
            StorageView::new(&data).unwrap();
        assert_eq!(Storage::<i32, U2, U3>::get(&row_major, 1, 2), Some(&6));
        assert_eq!(Storage::<i32, U2, U3>::get(&row_major, 0, 1), Some(&2));
        assert_eq!(
            <StorageView<'_, i32, U2, U3, RowMajor> as ContiguousStorage<
                i32,
                U2,
                U3,
            >>::ORDER,
            MatrixLayout::RowMajor
        );
    }

    #[cfg_attr(test, test)]
    /// Verifies both view constructors reject a backing slice whose length
    /// does not match `R::DIM * C::DIM` (FR-2 of `storage-trait-design.md`).
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
            if let Some(elem) =
                StorageMut::<i32, U2, U2>::get_mut(&mut view, 1, 1)
            {
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
                U2,
                U2,
            >>::ORDER,
            MatrixLayout::RowMajor
        );

        let x = [1.0f32, 1.0];
        let mut y_data = [0.0f32; 2];
        let mut y: StorageViewMut<'_, f32, U2, U1, RowMajor> =
            StorageViewMut::new(&mut y_data).unwrap();

        // y = 1.0 * A * x + 0.0 * y
        BasicSubProgramsF32::gemv(
            1.0,
            a.as_slice(),
            &x,
            0.0,
            y.as_mut_slice(),
            2,
            2,
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
}

// Property-based coverage of `ArrayStorage`/`StorageView` index-mapping
// invariants (FR-1 of `storage-trait-design.md`, §6.1 item 2). Kept outside
// the `#[hil_suite]`-wrapped module above: `proptest` is a host-only
// dev-dependency, unavailable to the `no_std`/on-target `hil` feature build.
#[cfg(test)]
mod storage_property_tests {
    use crate::math::num_types::Const;
    use crate::math::storage::{
        ArrayStorage, ColMajor, Storage, StorageInit, StorageView,
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
                        Storage::<i32, Const<3>, Const<4>>::get(&storage, i, j),
                        Some(&vals[j * 3 + i])
                    );
                    prop_assert_eq!(storage.offset(i, j), (j * 3 + i) as isize);
                }
            }
            prop_assert_eq!(Storage::<i32, Const<3>, Const<4>>::get(&storage, 3, 0), None);
            prop_assert_eq!(Storage::<i32, Const<3>, Const<4>>::get(&storage, 0, 4), None);
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
                        Storage::<i32, Const<2>, Const<3>>::get(&array, i, j),
                        Storage::<i32, Const<2>, Const<3>>::get(&view, i, j)
                    );
                }
            }
        }
    }
}
