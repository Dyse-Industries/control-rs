//! Bounds, ordering and stride behavior of dense, sparse and view storage.

use crate::math::num_types::{Const, U1, U2, U3};
use crate::math::storage::{
    ArrayCooStorage, ArrayCscStorage, ArrayCsrStorage, ArraySparseVector,
    ArrayStorage, ColMajor, CscStorage, CsrStorage, DenseStorage,
    DenseStorageMut, Diag, PackedStorage, RowMajor, SparseStorage,
    SparseStorageMut, SparseVectorStorage, StaticStorageView,
    StaticStorageViewMut, StorageError, StorageInit, StorageView,
    TriangularPackedStorage, UpLo,
};

type ColView<'a> = StaticStorageView<'a, i32, U3, U2, ColMajor>;
type RowView<'a> = StaticStorageView<'a, i32, U3, U2, RowMajor>;
type ColViewMut<'a> = StaticStorageViewMut<'a, i32, U3, U2, ColMajor>;
type RowViewMut<'a> = StaticStorageViewMut<'a, i32, U3, U2, RowMajor>;

#[test]
fn dense_access_rejects_the_first_index_past_each_axis() {
    let mut a = ArrayStorage::<f64, 2, 3>::zeros();
    assert!(a.get_mut(1, 2).is_some());
    assert!(a.get_mut(2, 0).is_none());
    assert!(a.get_mut(0, 3).is_none());
    assert_eq!(a.set(2, 0, 1.0), Err(StorageError::OutOfBounds));
    assert_eq!(a.set(0, 3, 1.0), Err(StorageError::OutOfBounds));
    assert!(a.set(1, 2, 1.0).is_ok());
}

/// Entries pushed out of order, with a duplicate that must be summed.
fn unsorted_coo() -> ArrayCooStorage<f64, 3, 3, 9> {
    let mut coo = ArrayCooStorage::<f64, 3, 3, 9>::new();
    for (r, c, v) in [
        (0, 2, 1.0),
        (0, 0, 2.0),
        (0, 1, 3.0),
        (0, 1, 4.0),
        (2, 1, 5.0),
        (2, 0, 6.0),
    ] {
        coo.push(r, c, v).unwrap();
    }
    coo
}

#[test]
fn csr_rows_are_sorted_by_column_and_duplicates_are_summed() {
    let csr =
        ArrayCsrStorage::<f64, 3, 3, 9, 4>::from_coo(&unsorted_coo()).unwrap();
    assert_eq!(csr.nnz(), 5);
    assert_eq!(csr.row_offsets(), [0, 3, 3, 5]);
    assert_eq!(csr.col_indices().get(..5), Some(&[0, 1, 2, 0, 1][..]));
    assert_eq!(csr.values().get(..5), Some(&[2.0, 7.0, 1.0, 6.0, 5.0][..]));
    let (cols, vals) = csr.row_slice(2).unwrap();
    assert_eq!((cols, vals), (&[0, 1][..], &[6.0, 5.0][..]));
    assert!(csr.row_slice(3).is_none(), "row 3 does not exist");
}

#[test]
fn csc_columns_are_sorted_by_row_and_duplicates_are_summed() {
    let csc =
        ArrayCscStorage::<f64, 3, 3, 9, 4>::from_coo(&unsorted_coo()).unwrap();
    assert_eq!(csc.nnz(), 5);
    assert_eq!(csc.col_offsets(), [0, 2, 4, 5]);
    assert_eq!(csc.row_indices().get(..5), Some(&[0, 2, 0, 2, 0][..]));
    assert_eq!(csc.values().get(..5), Some(&[2.0, 6.0, 7.0, 5.0, 1.0][..]));
}

#[test]
fn sparse_writes_report_missing_entries_and_bounds_distinctly() {
    let mut csr =
        ArrayCsrStorage::<f64, 3, 3, 9, 4>::from_coo(&unsorted_coo()).unwrap();
    let mut csc =
        ArrayCscStorage::<f64, 3, 3, 9, 4>::from_coo(&unsorted_coo()).unwrap();

    assert!(csr.set(0, 1, 9.0).is_ok());
    assert_eq!(csr.get(0, 1), Some(9.0));
    assert_eq!(
        csr.set(1, 1, 9.0),
        Err(StorageError::InvalidStructuralInvariant)
    );
    assert_eq!(csr.set(3, 0, 9.0), Err(StorageError::OutOfBounds));
    assert_eq!(csr.set(0, 3, 9.0), Err(StorageError::OutOfBounds));
    assert!(csr.get(0, 3).is_none());
    assert!(csr.get(3, 0).is_none());

    assert!(csc.set(0, 1, 9.0).is_ok());
    assert_eq!(csc.get(0, 1), Some(9.0));
    assert_eq!(
        csc.set(1, 1, 9.0),
        Err(StorageError::InvalidStructuralInvariant)
    );
    assert_eq!(csc.set(3, 0, 9.0), Err(StorageError::OutOfBounds));
    assert_eq!(csc.set(0, 3, 9.0), Err(StorageError::OutOfBounds));
    assert!(csc.get_mut(0, 3).is_none());
    assert!(csc.get_mut(3, 0).is_none());
    assert!(csc.get(0, 3).is_none());
}

#[test]
fn sparse_vectors_report_their_logical_length() {
    let mut v = ArraySparseVector::<f64, 5, 5>::new();
    v.push(1, 2.0).unwrap();
    assert_eq!(v.len(), 5);
    assert_eq!(v.nnz(), 1);
}

#[test]
fn view_strides_follow_the_layout() {
    let data = [1, 2, 3, 4, 5, 6];
    let col: ColView<'_> = StaticStorageView::new(&data).unwrap();
    assert_eq!((col.r_stride(), col.c_stride()), (1, 3));
    let row: RowView<'_> = StaticStorageView::new(&data).unwrap();
    assert_eq!((row.r_stride(), row.c_stride()), (2, 1));

    let mut buf = [1, 2, 3, 4, 5, 6];
    let col: ColViewMut<'_> = StaticStorageViewMut::new(&mut buf).unwrap();
    assert_eq!((col.r_stride(), col.c_stride()), (1, 3));
    let mut buf = [1, 2, 3, 4, 5, 6];
    let row: RowViewMut<'_> = StaticStorageViewMut::new(&mut buf).unwrap();
    assert_eq!((row.r_stride(), row.c_stride()), (2, 1));
    let _: Option<U1> = None;
}

/// Three values at one coordinate whose float sum depends on the order.
fn order_sensitive_coo() -> ArrayCooStorage<f64, 3, 3, 9> {
    let mut coo = ArrayCooStorage::<f64, 3, 3, 9>::new();
    for v in [1.0, 1e16, -1e16] {
        coo.push(1, 1, v).unwrap();
    }
    coo
}

#[test]
fn duplicates_are_summed_in_insertion_order() {
    // (1 + 1e16) - 1e16 is 0; the reverse order gives 1.
    let csr =
        ArrayCsrStorage::<f64, 3, 3, 9, 4>::from_coo(&order_sensitive_coo())
            .unwrap();
    assert_eq!(csr.get(1, 1).map(f64::to_bits), Some(0.0_f64.to_bits()));
    let csc =
        ArrayCscStorage::<f64, 3, 3, 9, 4>::from_coo(&order_sensitive_coo())
            .unwrap();
    assert_eq!(csc.get(1, 1).map(f64::to_bits), Some(0.0_f64.to_bits()));
}

#[test]
fn csc_rows_are_sorted_whatever_the_insertion_order() {
    let mut coo = ArrayCooStorage::<f64, 3, 3, 9>::new();
    for (r, v) in [(2, 1.0), (0, 2.0), (1, 3.0)] {
        coo.push(r, 0, v).unwrap();
    }
    let csc = ArrayCscStorage::<f64, 3, 3, 9, 4>::from_coo(&coo).unwrap();
    assert_eq!(csc.row_indices().get(..3), Some(&[0, 1, 2][..]));
    assert_eq!(csc.values().get(..3), Some(&[2.0, 3.0, 1.0][..]));
}

#[test]
fn an_empty_window_fits_any_buffer() {
    let no_rows =
        StorageView::<f64, Const<0>, Const<3>>::new_with_strides(&[], 1, 1);
    assert!(no_rows.is_ok());
    let no_cols =
        StorageView::<f64, Const<3>, Const<0>>::new_with_strides(&[], 1, 1);
    assert!(no_cols.is_ok());
}

#[test]
fn a_unit_diagonal_projection_copies_only_the_stored_triangle() {
    // Every off-triangle entry is non-zero, so any stray copy is visible.
    let dense = ArrayStorage::<f64, 3, 3>::from_fn(|i, j| {
        f64::from(u8::try_from(i * 3 + j + 1).unwrap())
    });
    let upper = TriangularPackedStorage::<f64, 3, 6>::from_dense_triangle(
        &dense,
        UpLo::Upper,
        Diag::Unit,
    )
    .unwrap();
    assert_eq!(upper.value(0, 1), Some(2.0));
    assert_eq!(upper.value(0, 2), Some(3.0));
    assert_eq!(upper.value(1, 2), Some(6.0));
    // The diagonal slots stay untouched: a unit diagonal is implicit.
    assert_eq!(upper.as_slice(), [0.0, 2.0, 0.0, 3.0, 6.0, 0.0]);
    let lower = TriangularPackedStorage::<f64, 3, 6>::from_dense_triangle(
        &dense,
        UpLo::Lower,
        Diag::Unit,
    )
    .unwrap();
    assert_eq!(lower.value(1, 0), Some(4.0));
    assert_eq!(lower.value(2, 0), Some(7.0));
    assert_eq!(lower.value(2, 1), Some(8.0));
    assert_eq!(lower.as_slice(), [0.0, 4.0, 7.0, 0.0, 8.0, 0.0]);
}
