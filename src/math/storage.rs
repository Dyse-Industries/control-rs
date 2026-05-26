//! Utilities to help initialize static arrays.
//!
//! This is inspired by [array-init](https://github.com/Manishearth/array-init/) and [nalgebra].
//! This is meant to eventually become a `static_storage` trait that can provide the same tools
//! for sparse and dense arrays.

#![allow(clippy::indexing_slicing, clippy::arithmetic_side_effects)]

use core::mem::MaybeUninit;

type ZippedArray<T, const N: usize> = ([T; N], [T; N]);
type UninitArray<T, const N: usize> = MaybeUninit<[T; N]>;

/// Helper function to reverse arrays given to `Polynomial::new()`
#[inline]
pub const fn reverse_array<T: Copy, const N: usize>(input: [T; N]) -> [T; N] {
    let mut output = input;
    let mut i = 0;
    while i < N / 2 {
        let tmp = output[i];
        output[i] = output[N - 1 - i];
        output[N - 1 - i] = tmp;
        i += 1;
    }

    output
}

/// Initialize an array from an iterator.
///
/// # Generic Arguments
/// * `I` - Any collection that implements [`IntoIterator<item=T>`].
/// * `T` - Field type of the array.
/// * `N` - Capacity of the array.
///
/// # Arguments
/// * `iterator` - Collection of `T`.
///
/// # Returns
/// * `initialized_array` - An array filled with elements from the iterator.
///
/// # Safety
/// * The iterator must have **at least** `N` elements or this will assume an uninitialized
///   value is initialized (resulting in UB).
///
/// # Panics
/// * This function will panic in debug builds if the safety criterion is not met.
pub(crate) unsafe fn array_from_iterator<I, T, const N: usize>(
    iterator: I,
) -> [T; N]
where
    I: IntoIterator<Item = T>,
{
    let mut maybe_uninit_array: UninitArray<T, N> = MaybeUninit::uninit();
    let arr_ptr = maybe_uninit_array.as_mut_ptr().cast::<T>();
    let mut write_counter = 0;
    for (i, b) in (0..N).zip(iterator.into_iter()) {
        unsafe {
            arr_ptr.add(i).write(b);
        }
        write_counter += 1;
    }
    debug_assert_eq!(write_counter, N);
    unsafe { maybe_uninit_array.assume_init() }
}

/// Initialize an array from an iterator.
///
/// # Generic Arguments
/// * `I` - Any collection that implements [`IntoIterator<item=(T, T)>`].
/// * `T` - Field type of the array.
/// * `N` - Capacity of the array.
///
/// # Arguments
/// * `iterator` - Collection of `(T, T)`.
///
/// # Returns
/// * `initialized_array` - An array filled with elements from the iterator.
///
/// # Safety
/// * The iterator must have **at least** `N` elements or this will assume an uninitialized
///   value is initialized (resulting in UB).
///
/// # Panics
/// * This function will panic in debug builds if the safety criterion is not met.
pub(crate) unsafe fn arrays_from_zipped_iterator<I, T, const N: usize>(
    iterator: I,
) -> ZippedArray<T, N>
where
    I: IntoIterator<Item = (T, T)>,
{
    let mut maybe_uninit_array1: UninitArray<T, N> = MaybeUninit::uninit();
    let mut maybe_uninit_array2: UninitArray<T, N> = MaybeUninit::uninit();
    let arr1_ptr = maybe_uninit_array1.as_mut_ptr().cast::<T>();
    let arr2_ptr = maybe_uninit_array2.as_mut_ptr().cast::<T>();
    let mut write_counter = 0;
    for (i, (a, b)) in (0..N).zip(iterator.into_iter()) {
        unsafe {
            arr1_ptr.add(i).write(a);
            arr2_ptr.add(i).write(b);
        }
        write_counter += 1;
    }
    debug_assert_eq!(write_counter, N);
    unsafe {
        (
            maybe_uninit_array1.assume_init(),
            maybe_uninit_array2.assume_init(),
        )
    }
}
