//! Single-writer state cell.
//!
//! # Description
//!
//! [`StateCell`] holds a `Copy` value that the executive writes in executive context and that the
//! fault path reads after it masks interrupts. It uses volatile loads and stores and no
//! read-modify-write, so it builds on targets without atomic instructions and never needs a
//! compare-and-swap polyfill.
//!
//! # Core Concepts
//!
//! - **Single writer**: Only the executive writes; the fault path reads (or, at a fault, writes
//!   after masking interrupts).
//! - **Ordering**: Each store is followed by a compiler fence so the compiler cannot move it
//!   across a step call.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::cell::StateCell;
//!
//! static PHASE: StateCell<u8> = StateCell::new(0);
//! PHASE.set(2);
//! assert_eq!(PHASE.get(), 2);
//! ```
//!
//! # Limitations
//!
//! - **Contract**: A concurrent writer is a data race; the type's `Sync` implementation relies on
//!   the single-writer rule of the executive.

use core::cell::UnsafeCell;
use core::sync::atomic::{Ordering, compiler_fence};

/// A value written by one context and read by another that cannot interleave with it.
pub struct StateCell<T: Copy>(UnsafeCell<T>);

// SAFETY: The executive is the only writer, and the fault path reads or writes only after it
// masked interrupts, so no two accesses interleave (`ets-executive-design.md` §4.9).
unsafe impl<T: Copy> Sync for StateCell<T> {}

impl<T: Copy> StateCell<T> {
    /// Creates a cell holding `value`.
    #[must_use]
    pub const fn new(value: T) -> Self {
        Self(UnsafeCell::new(value))
    }

    /// Reads the value.
    pub fn get(&self) -> T {
        // SAFETY: The pointer is valid and aligned, and the single-writer rule excludes a
        // concurrent write.
        let value = unsafe { self.0.get().read_volatile() };
        compiler_fence(Ordering::SeqCst);
        value
    }

    /// Writes the value.
    pub fn set(&self, value: T) {
        compiler_fence(Ordering::SeqCst);
        // SAFETY: The pointer is valid and aligned, and the single-writer rule excludes a
        // concurrent access.
        unsafe { self.0.get().write_volatile(value) };
        compiler_fence(Ordering::SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cell_returns_the_last_value_written() {
        let cell = StateCell::new(1_u32);
        assert_eq!(cell.get(), 1);
        cell.set(9);
        assert_eq!(cell.get(), 9);
    }

    #[test]
    fn a_cell_in_a_static_is_usable_from_any_function() {
        type Slot = StateCell<Option<u8>>;
        static CELL: Slot = StateCell::new(None);
        CELL.set(Some(3));
        assert_eq!(CELL.get(), Some(3));
    }
}
