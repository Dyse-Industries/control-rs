//! # Static Storage Tests
#![allow(
    clippy::arithmetic_side_effects,
    clippy::indexing_slicing,
    clippy::type_complexity,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap
)]

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod storage_test_suite {
    use crate::math::num_types::{Const, U2, U3};
    use crate::math::storage::{
        ArrayStorage, ContiguousStorage, ContiguousStorageMut, MatrixLayout,
        PivotStorage, Storage, StorageInit, StorageMut, StorageView,
        StorageViewMut, array_from_iterator, reverse_array,
    };

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
    /// matching `ContiguousStorage::ORDER` and `Storage::offset`.
    fn test_array_storage_column_major_layout() {
        // 2 rows x 3 cols: columns are [1,2], [3,4], [5,6]
        let storage: ArrayStorage<i32, 2, 3> =
            ArrayStorage::from_array([[1, 2], [3, 4], [5, 6]]);

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
    /// Verifies `get`/`get_mut` return `None` outside `[0, rows) x [0, cols)`.
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
    /// Verifies `StorageInit`'s four safe construction strategies.
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
    /// Verifies `as_mut_slice` observes writes made through `get_mut`, and
    /// vice versa (both paths address the same backing memory).
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
    /// index mappings over the same backing slice without copying.
    fn test_storage_view_layout_selection() {
        let data = [1, 2, 3, 4, 5, 6];

        let col_major: StorageView<'_, i32, U2, U3> =
            StorageView::new(&data, MatrixLayout::ColMajor);
        assert_eq!(Storage::<i32, U2, U3>::get(&col_major, 1, 2), Some(&6));

        let row_major: StorageView<'_, i32, U2, U3> =
            StorageView::new(&data, MatrixLayout::RowMajor);
        assert_eq!(Storage::<i32, U2, U3>::get(&row_major, 1, 2), Some(&6));
        assert_eq!(Storage::<i32, U2, U3>::get(&row_major, 0, 1), Some(&2));
    }

    #[cfg_attr(test, test)]
    /// Verifies `StorageViewMut` writes are visible through the original
    /// borrowed slice once the view is dropped.
    fn test_storage_view_mut_writes_through() {
        let mut data = [0; 4];
        {
            let mut view: StorageViewMut<'_, i32, U2, U2> =
                StorageViewMut::new(&mut data, MatrixLayout::ColMajor);
            if let Some(elem) =
                StorageMut::<i32, U2, U2>::get_mut(&mut view, 1, 1)
            {
                *elem = 5;
            }
        }
        assert_eq!(data, [0, 0, 0, 5]);
    }

    // --- PivotStorage ---

    #[cfg_attr(test, test)]
    /// Verifies `PivotStorage::identity` and `swap` maintain a permutation.
    fn test_pivot_storage_identity_and_swap() {
        let mut pivots: PivotStorage<4> = PivotStorage::identity();
        assert_eq!(pivots.as_slice(), &[0, 1, 2, 3]);

        pivots.swap(0, 3);
        assert_eq!(pivots.as_slice(), &[3, 1, 2, 0]);
    }
}

// Property-based coverage of `ArrayStorage`/`StorageView` index-mapping
// invariants (`storage-trait-design.md` §6.1 item 2). Kept outside the
// `#[hil_suite]`-wrapped module above: `proptest` is a host-only
// dev-dependency, unavailable to the `no_std`/on-target `hil` feature build.
#[cfg(test)]
mod storage_property_tests {
    use crate::math::num_types::Const;
    use crate::math::storage::{
        ArrayStorage, MatrixLayout, Storage, StorageInit, StorageView,
    };
    use proptest::prelude::*;

    proptest! {
        /// Every logical `(i, j)` in a 3x4 `ArrayStorage` reads back the
        /// value it was constructed with, and `offset()` matches the
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
            let view: StorageView<'_, i32, Const<2>, Const<3>> =
                StorageView::new(&vals, MatrixLayout::ColMajor);

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
