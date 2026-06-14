//! Timekeeping abstractions for Hardware-in-the-Loop tests.

/// A trait for retrieving the current time from target hardware.
///
/// Implementations of this trait encapsulate the hardware-specific details of
/// reading timers or hardware clocks (e.g. SysTick, DWT, or TIM peripherals).
pub trait ClientClock {
    /// Returns the current time in milliseconds.
    fn now_ms(&self) -> u32;

    /// Returns the current time in microseconds.
    fn now_us(&self) -> u64;
}

/// A dummy clock that always returns 0. Useful for tests or architectures
/// without timer hardware configured yet.
pub struct DummyClock;

impl ClientClock for DummyClock {
    fn now_ms(&self) -> u32 {
        0
    }

    fn now_us(&self) -> u64 {
        0
    }
}
