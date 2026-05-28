//! # Static Storage Tests

use crate::math::storage::{array_from_iterator, reverse_array};

#[test]
fn test_reverse_array() {
    let array = [1, 2, 3, 4, 5];
    let reversed = reverse_array(array);
    assert_eq!(reversed, [5, 4, 3, 2, 1]);
}

#[test]
fn test_reverse_array_even() {
    let array = [1, 2, 3, 4];
    let reversed = reverse_array(array);
    assert_eq!(reversed, [4, 3, 2, 1]);
}

#[test]
fn test_reverse_array_single_element() {
    let array = [1];
    let reversed = reverse_array(array);
    assert_eq!(reversed, [1]);
}

#[test]
fn test_array_from_iterator() {
    let array: [i32; 5] = unsafe { array_from_iterator(0..5) };
    assert_eq!(array, [0, 1, 2, 3, 4]);
}

#[test]
#[should_panic]
fn test_array_from_iterator_too_few() {
    let _array: [i32; 5] = unsafe { array_from_iterator(0..3) };
}

#[test]
fn test_array_from_iterator_too_many() {
    let array: [i32; 5] = unsafe { array_from_iterator(0..10) };
    assert_eq!(array, [0, 1, 2, 3, 4]);
}
//
// #[test]
// fn test_array_from_iterator_with_default() {
//     let array: [i32; 5] = array_from_iterator_with_default(0..3, 7);
//     assert_eq!(array, [0, 1, 2, 7, 7]);
// }
//
// #[test]
// fn test_array_from_iterator_with_default_too_many() {
//     let array: [i32; 5] = array_from_iterator_with_default(0..10, 7);
//     assert_eq!(array, [0, 1, 2, 3, 4]);
// }

// #[test]
// fn test_arrays_from_zipped_iterator() {
//     let (arr1, arr2): ([i32; 3], [i32; 3]) =
//         unsafe { arrays_from_zipped_iterator((0..3).zip(3..6)) };
//     assert_eq!(arr1, [0, 1, 2]);
//     assert_eq!(arr2, [3, 4, 5]);
// }
//
// #[test]
// #[should_panic]
// fn test_arrays_from_zipped_iterator_too_few() {
//     let (_arr1, _arr2): ([i32; 5], [i32; 5]) =
//         unsafe { arrays_from_zipped_iterator((0..3).zip(3..6)) };
// }
//
// #[test]
// fn test_arrays_from_zipped_iterator_too_many() {
//     let (arr1, arr2): ([i32; 3], [i32; 3]) =
//         unsafe { arrays_from_zipped_iterator((0..10).zip(10..20)) };
//     assert_eq!(arr1, [0, 1, 2]);
//     assert_eq!(arr2, [10, 11, 12]);
// }
