//! # Static Storage Tests

use crate::assert_almost_eq;
use crate::math::storage::{
    MatrixLayout, MatrixStorage, StaticArray, StaticMatrix, StaticStorage,
    UninitStaticArray, UninitStaticStorage,
};

#[test]
#[allow(clippy::cast_precision_loss)]
fn static_array() {
    let mut array = StaticArray([0.0; 6]);
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
    // ptr is dropped so it's safe to iter through the elements
    for (i, x) in array.iter().enumerate() {
        assert_almost_eq!(*x, i as f32);
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn uninit_static_array() {
    let array: StaticArray<i32, 10> =
        unsafe { UninitStaticArray::unchecked_from_iterator::<_, 10>(0..10) };
    for i in 0..10i32 {
        unsafe {
            assert_eq!(array.get_ptr().add(i as usize).read(), i);
        }
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
fn uninit_array_from_iter_too_many() {
    let array: StaticArray<i32, 10> =
        unsafe { UninitStaticArray::unchecked_from_iterator::<_, 10>(0..20) };
    for i in 0..10i32 {
        unsafe {
            assert_eq!(array.get_ptr().add(i as usize).read(), i);
        }
    }
}

#[test]
#[allow(clippy::cast_sign_loss)]
#[should_panic(expected = "Incorrect number of elements in iter.")]
fn uninit_from_iter_too_few() {
    let array: StaticArray<i32, 10> =
        unsafe { UninitStaticArray::unchecked_from_iterator::<_, 10>(0..5) };
    for i in 0..10i32 {
        unsafe {
            assert_eq!(array.get_ptr().add(i as usize).read(), i);
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
    let storage: StaticMatrix<f32, 3, 2> = unsafe {
        UninitStaticArray::unchecked_from_iterator::<_, 2>([[0.0; 3], [1.0; 3]])
    };
    assert_eq!(storage.rows(), 3);
    assert_eq!(storage.cols(), 2);
}

#[test]
fn test_linear_index_unchecked_row_major() {
    let storage = StaticArray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
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
    let storage = StaticArray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]);
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
    let storage = StaticArray([[0.0; 3]; 2]);
    let _ = storage.linear_index_unchecked(4, 0, MatrixLayout::RowMajor);
}

#[test]
#[should_panic(expected = "Column index out of bounds")]
fn test_linear_index_oob_col() {
    let storage = StaticArray([[0.0; 3]; 2]);
    let _ = storage.linear_index_unchecked(0, 3, MatrixLayout::RowMajor);
}
