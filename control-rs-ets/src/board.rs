//! Board binding.
//!
//! # Description
//!
//! A [`Board`] connects the executive to board resources it cannot know: the watchdog, a low-power
//! wait and a release source for paced runs. Every hook has a default, so a board without a
//! watchdog, a sleep or a timer works unchanged. The binding is a monomorphic type parameter of the
//! server; the executive starts, configures and disables nothing in it.
//!
//! # Core Concepts
//!
//! - **`feed`**: Called at idle, at each boundary and around each wait; never inside a step.
//! - **`wait`**: Called when nothing is runnable.
//! - **Releases**: Armed with the period (`Periodic`) or without one (`Triggered`) at run start,
//!   read and cleared at each boundary and disarmed at settle. A board that reports no release
//!   source leaves `Periodic` to the profiler clock and cannot run `Triggered`.
//!
//! # Usage
//!
//! ```
//! use control_rs_ets::board::{Board, NoBoard};
//!
//! struct Timer { ticks: u32 }
//! impl Board for Timer {
//!     fn has_release_source(&self) -> bool { true }
//!     fn releases(&mut self) -> u32 { core::mem::take(&mut self.ticks) }
//! }
//!
//! let mut board = Timer { ticks: 3 };
//! assert_eq!(board.releases(), 3);
//! assert_eq!(board.releases(), 0);
//! assert!(!NoBoard.has_release_source());
//! ```
//!
//! # Limitations
//!
//! - **Interrupts**: The binding defines no interrupt. A release interrupt only increments a
//!   counter the user owns; `releases` reads and clears it in executive context.

/// Hooks from the executive to board resources.
pub trait Board {
    /// Reloads the watchdog that board setup started.
    fn feed(&mut self) {}

    /// Waits until something may be runnable, for example by sleeping until an interrupt.
    fn wait(&mut self) {}

    /// Returns whether the board supplies a release source.
    fn has_release_source(&self) -> bool {
        false
    }

    /// Arms the release source, with the period in nanoseconds for a periodic run.
    fn arm(&mut self, _period_ns: Option<u32>) {}

    /// Returns the releases since the previous call and clears the count.
    fn releases(&mut self) -> u32 {
        0
    }

    /// Disarms the release source.
    fn disarm(&mut self) {}
}

/// A board with no watchdog, no sleep and no release source.
#[derive(Debug, Clone, Copy, Default)]
pub struct NoBoard;

impl Board for NoBoard {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_default_binding_does_nothing() {
        let mut board = NoBoard;
        board.feed();
        board.wait();
        board.arm(Some(1));
        assert_eq!(board.releases(), 0);
        board.disarm();
        assert!(!board.has_release_source());
    }
}
