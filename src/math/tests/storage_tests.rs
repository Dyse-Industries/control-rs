//! # Static Storage Tests

use crate::assert_almost_eq;
use crate::math::storage::{StaticArray, StaticStorage};

#[test]
#[allow(clippy::cast_precision_loss)]
fn static_array_compile() {
    let mut array = StaticArray::<f32, 10>([0.0; 10]);
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
