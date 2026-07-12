//! # Static Storage Tests

#[cfg_attr(not(test), control_rs_macros::hil_suite)]
pub mod storage_test_suite {
    use crate::math::storage::{array_from_iterator, reverse_array};

    #[cfg_attr(test, test)]
    fn test_reverse_array() {
        let array = [1, 2, 3, 4, 5];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [5, 4, 3, 2, 1]);
    }

    #[cfg_attr(test, test)]
    fn test_reverse_array_even() {
        let array = [1, 2, 3, 4];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [4, 3, 2, 1]);
    }

    #[cfg_attr(test, test)]
    fn test_reverse_array_single_element() {
        let array = [1];
        let reversed = reverse_array(array);
        assert_eq!(reversed, [1]);
    }

    #[cfg_attr(test, test)]
    fn test_array_from_iterator() {
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
    fn test_array_from_iterator_too_many() {
        let array: [i32; 5] = unsafe { array_from_iterator(0..10) };
        assert_eq!(array, [0, 1, 2, 3, 4]);
    }
}
