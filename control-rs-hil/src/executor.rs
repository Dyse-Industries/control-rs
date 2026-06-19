//! Hardware execution abstraction for HIL tests.
//!
//! Provides traits to execute test cases and measure hardware performance metrics
//! (like clock cycles and stack space consumption) in a hardware-agnostic manner.

/// Abstraction over hardware-specific test execution.
pub trait TestExecutor {
    /// Executes a test function, measuring and returning (`elapsed_cycles`, `stack_peak_bytes`).
    fn execute(&self, test_fn: fn()) -> (u64, u32);
}

/// Fallback executor that runs the test function without measuring any hardware metrics.
pub struct DummyExecutor;

impl TestExecutor for DummyExecutor {
    fn execute(&self, test_fn: fn()) -> (u64, u32) {
        test_fn();
        (0, 0)
    }
}
