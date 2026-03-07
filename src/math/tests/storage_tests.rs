//! # Static Storage Tests

use crate::assert_almost_eq;
use crate::math::num_types::{U2, U3};
use crate::math::storage::{
    MatrixLayout, MatrixStorage, StaticArray, StaticStorage,
};

// Helper struct to implement MatrixStorage for testing default methods
struct TestMatrix<T, const N: usize>(StaticArray<T, N>);

impl<T, const N: usize> StaticStorage<T> for TestMatrix<T, N> {
    fn get_mut_ptr(&mut self) -> *mut T {
        self.0.get_mut_ptr()
    }
    fn get_ptr(&self) -> *const T {
        self.0.get_ptr()
    }
}

impl<T> MatrixStorage<T, U2, U3> for TestMatrix<T, 6> {}

#[test]
#[allow(clippy::cast_precision_loss)]
fn static_array_compile() {
    let mut array = TestMatrix(StaticArray([0.0; 6]));
    let mut_ptr = array.get_mut_ptr();
    for i in 0..10 {
        unsafe {
            *mut_ptr.offset(i as isize) = i as f32;
        }
    }

    let ptr = array.get_ptr();
    for i in 0..10 {
        // # Safety
        // The array has 10 elements, indexing in this loop is safe.
        unsafe {
            assert_almost_eq!(*ptr.offset(i as isize), i as f32);
        }
    }
}

#[test]
fn test_matrix_layout_values() {
    assert_eq!(MatrixLayout::RowMajor as i32, 101);
    assert_eq!(MatrixLayout::ColMajor as i32, 102);
}

#[test]
fn test_matrix_storage_dims() {
    let storage = TestMatrix(StaticArray([0.0; 6]));
    assert_eq!(storage.rows(), 2);
    assert_eq!(storage.cols(), 3);
}

#[test]
fn test_linear_index_unchecked_row_major() {
    let storage = TestMatrix(StaticArray([0.0; 6]));
    // 2x3 Matrix:
    // (0,0) -> 0
    // (0,1) -> 1
    // (0,2) -> 2
    // (1,0) -> 3
    // (1,1) -> 4
    // (1,2) -> 5
    assert_eq!(
        storage.linear_index_unchecked(0, 0, MatrixLayout::RowMajor),
        0
    );
    assert_eq!(
        storage.linear_index_unchecked(0, 1, MatrixLayout::RowMajor),
        1
    );
    assert_eq!(
        storage.linear_index_unchecked(0, 2, MatrixLayout::RowMajor),
        2
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 0, MatrixLayout::RowMajor),
        3
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 1, MatrixLayout::RowMajor),
        4
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 2, MatrixLayout::RowMajor),
        5
    );
}

#[test]
fn test_linear_index_unchecked_col_major() {
    let storage = TestMatrix(StaticArray([0.0; 6]));
    // 2x3 Matrix (ColMajor):
    // (0,0) -> 0
    // (1,0) -> 1
    // (0,1) -> 2
    // (1,1) -> 3
    // (0,2) -> 4
    // (1,2) -> 5
    assert_eq!(
        storage.linear_index_unchecked(0, 0, MatrixLayout::ColMajor),
        0
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 0, MatrixLayout::ColMajor),
        1
    );
    assert_eq!(
        storage.linear_index_unchecked(0, 1, MatrixLayout::ColMajor),
        2
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 1, MatrixLayout::ColMajor),
        3
    );
    assert_eq!(
        storage.linear_index_unchecked(0, 2, MatrixLayout::ColMajor),
        4
    );
    assert_eq!(
        storage.linear_index_unchecked(1, 2, MatrixLayout::ColMajor),
        5
    );
}

#[test]
#[should_panic(expected = "Row index out of bounds")]
fn test_linear_index_oob_row() {
    let storage = TestMatrix(StaticArray([0.0; 6]));
    let _ = storage.linear_index_unchecked(2, 0, MatrixLayout::RowMajor);
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_linear_index_oob_col() {
    let storage = TestMatrix(StaticArray([0.0; 6]));
    let _ = storage.linear_index_unchecked(0, 3, MatrixLayout::RowMajor);
}
